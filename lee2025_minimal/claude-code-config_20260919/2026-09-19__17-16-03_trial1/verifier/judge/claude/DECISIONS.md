# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject's data is stored in a joblib file (extensionless, e.g. `QLAK-CA1-08`) in the data directory. The AI loads each file with `joblib.load()`, extracting a dictionary keyed by animal name containing `trace`, `position`, `envs`, and `blocked` arrays. The 7 animals are hardcoded in an `ANIMALS` list.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position = dat['trace'], dat['position']
envs, blocked = dat['envs'], dat['blocked']
```

iii. The AI explored the data directory and found both `.mat` and extensionless (joblib) files. After inspecting the joblib files and confirming they contain all the needed data (trace, position, envs, blocked), it chose to load them with joblib. The agent verified the data structure by printing shapes and keys before writing the conversion script.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The AI hardcodes the 7 animal names in the `ANIMALS` list and iterates over them. The subject name is the animal identifier (e.g. `QLAK-CA1-08`).

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal_idx, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
```

iii. The AI identified 7 subjects in the data directory by listing files and exploring the data structure.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days. The `trace` array has shape `(n_days, n_cells, n_frames)`. Each day becomes a separate session. The AI iterates `for day in range(n_days)` over all days for each animal.

ii.
```python
n_days = trace.shape[0]
...
for day in range(n_days):
    neural, keep = session_neural(trace[day])
    bins = session_position_bins(position[day], bin_width)
    geometry = blocked_vector(blocked[day])
```

iii. The AI verified that 7 mice × ~31 days = 207 sessions, matching the paper's stated "207 sessions".

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into consecutive, non-overlapping 1-minute trials. After temporal rebinning to 100 ms bins, each trial is 600 bins. The incomplete remainder at the end of the session is dropped.

ii.
```python
TRIAL_BINS = TRIAL_SECONDS * FPS // TEMPORAL_BIN_FRAMES   # 600 bins per trial
...
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. The instructions specify 1-minute trials. Each 40-minute session yields ~39-40 trials of 600 bins (100 ms each). The agent confirmed this produced 8,187 total trials across 207 sessions.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All complete 1-minute segments are kept. Only the incomplete remainder at the end of each session is dropped.

ii. N/A (no trial filtering code)

iii. The agent did not apply trial filtering since the task requires contiguous trials and the paper does not describe trial-level exclusion criteria for the decoding analysis.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib files, which contains the binarized rising-phase calcium transient vectors (the paper's "firing rate"). Shape per day: `(n_cells, n_frames)`.

ii.
```python
trace, position = dat['trace'], dat['position']
...
neural, keep = session_neural(trace[day])
```

iii. The AI identified `trace` as the binarized rising phase of calcium transients by reading the paper's methods section and the reference code.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps following the paper's within-session decoding pipeline (`decode_position_within`/`fit_decoder`):
1. Filter neurons: keep only cells registered on that day (non-NaN) with >5 transients
2. Gaussian smoothing along time with sigma = 3 frames
3. Average pooling into non-overlapping 3-frame (100 ms) bins

ii.
```python
def session_neural(trace_day):
    registered = ~np.isnan(trace_day[:, 0])
    events = np.nansum(trace_day, axis=1)
    keep = registered & (events > MIN_EVENTS)
    trace = trace_day[keep].astype(np.float32)
    smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
    return pool_time(smoothed).astype(np.float32), keep

def pool_time(x):
    n = x.shape[-1] // TEMPORAL_BIN_FRAMES
    return x[..., :n * TEMPORAL_BIN_FRAMES].reshape(
        x.shape[:-1] + (n, TEMPORAL_BIN_FRAMES)).mean(-1)
```

iii. The AI read the paper's code (`utils.py`) and identified the `fit_decoder` function's processing: Gaussian smoothing with sigma=3 followed by average pooling with bin size 3 (producing 100 ms bins). It explicitly modeled these steps to match the paper's decoding analysis.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied per session:
1. Only cells registered on that day (non-NaN first frame) are kept
2. Only cells with more than 5 transients (sum of binary trace > 5) are kept, matching the paper's `cell_threshold=5` for decoding

ii.
```python
MIN_EVENTS = 5
...
registered = ~np.isnan(trace_day[:, 0])
events = np.nansum(trace_day, axis=1)
keep = registered & (events > MIN_EVENTS)
```

iii. The AI found the `cell_threshold` parameter in the paper's decoding code and applied it. The agent noted this drops only 112 of 69,744 cell-sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous free exploration with no discrete trial events. Sessions are simply cut into consecutive 1-minute segments starting from the beginning.

ii. N/A (no alignment code beyond sequential slicing)

iii. The agent notes in metadata: "Sessions are continuous 40-minute free exploration recordings with no discrete trial events, so each session is cut into consecutive non-overlapping 1-minute trials."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins from the native 30 Hz (33.33 ms) to 100 ms bins by average-pooling every 3 frames. This matches the paper's `temporal_bin_size=3` used in their decoding analysis.

ii.
```python
TEMPORAL_BIN_FRAMES = 3       # paper's `temporal_bin_size` for decoding -> 100 ms bins
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100 ms
...
def pool_time(x):
    n = x.shape[-1] // TEMPORAL_BIN_FRAMES
    return x[..., :n * TEMPORAL_BIN_FRAMES].reshape(
        x.shape[:-1] + (n, TEMPORAL_BIN_FRAMES)).mean(-1)
```

iii. The AI identified the 3-frame temporal binning from the paper's decoding code and applied it to produce 100 ms time bins, resulting in 600 bins per 1-minute trial.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the joblib files, which contains the indices of blocked (walled-off) partitions in the 3x3 grid for each session.

ii.
```python
envs, blocked = dat['envs'], dat['blocked']
...
geometry = blocked_vector(blocked[day])
```

iii. The AI explored the `blocked` field and verified it against the `envs` field and the `get_env_mat` function from the paper's code.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted to a 9-dimensional binary vector (one-hot over the 3x3 grid). Blocked positions get 1, unblocked get 0. If no positions are blocked (index is -1), the vector is all zeros. The input is static per trial (same for all trials in a session).

ii.
```python
def blocked_vector(blocked_day):
    idx = np.array(blocked_day[0]).ravel().astype(int)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if idx.size and idx[0] != -1:
        vec[idx] = 1.0
    return vec
...
input_trials.append(geometry.copy())
```

iii. The AI verified the blocked encoding against the paper's `get_env_mat` function and confirmed that the blocked partitions exactly correspond to zero-occupancy spatial bins (0.003% exceptions from tracking jitter at walls).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the joblib files, which contains 2D (x, y) head coordinates in cm at 30 Hz. Shape per day: `(2, n_frames)`.

ii.
```python
trace, position = dat['trace'], dat['position']
...
bins = session_position_bins(position[day], bin_width)
```

iii. The paper states position data is "generated from tracking the head with DeepLabCut pose-estimation software."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is first average-pooled into 100 ms bins (same temporal binning as neural data), then discretized into a 3x3 grid. The bin width is computed per animal as `(nanmax(position) + BUFFER) / 3`, using the maximum position coordinate across all sessions for that animal. This matches the paper's `decode_position_within` approach.

ii.
```python
def session_position_bins(position_day, bin_width):
    pooled = pool_time(position_day.astype(np.float64))
    xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
...
bin_width = (np.nanmax(position) + BUFFER) / N_SPATIAL_BINS
```

iii. The AI read the paper's `decode_position_within` function and replicated its spatial binning approach, where the bin width is derived from each animal's maximum position to ensure the 3x3 grid aligns with the physical partition grid in all geometries.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). The partition index is `3 * (y // bin_width) + (x // bin_width)`, where bin_width is `(nanmax(position) + BUFFER) / 3`. Values are clipped to [0, 2] per axis to handle edge cases.

ii.
```python
xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. The per-animal bin width ensures that the spatial grid aligns with the physical partition walls (each 25 cm). The BUFFER prevents boundary positions from exceeding the grid.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both neural and position data undergo the same temporal pooling (3-frame average pooling into 100 ms bins) before being split into trials using the same indices. This ensures frame-by-frame alignment.

ii.
```python
neural, keep = session_neural(trace[day])
bins = session_position_bins(position[day], bin_width)
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. Both streams are pooled with the same `pool_time` function and then sliced with the same trial boundaries. The `min()` handles any small difference in length due to the Gaussian smoothing padding.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms handle data issues:
1. Neurons not registered on a given day (all-NaN trace) are removed
2. Neurons with ≤5 transients are removed (low activity)
3. The incomplete remainder at the end of each session (< 1 minute) is discarded
4. Position values at arena boundaries are clipped to valid bins
5. A tiny BUFFER (1e-15) is added to the max position to prevent boundary edge cases in spatial binning

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
events = np.nansum(trace_day, axis=1)
keep = registered & (events > MIN_EVENTS)
...
BUFFER = 1e-15
bin_width = (np.nanmax(position) + BUFFER) / N_SPATIAL_BINS
...
xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
```

iii. The AI documented that 0.003% of timepoints land in blocked partitions due to tracking jitter at walls, and that these are kept rather than filtered.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading the large joblib files (~6.65 GB total output), (2) Gaussian smoothing of the full neural traces, and (3) the temporal pooling operations on large arrays. The agent's script timed out on the first run attempt (>2 minutes) before completing on a subsequent run.

ii. N/A

iii. The trajectory shows the conversion script timing out initially and requiring multiple execution attempts, indicating the data loading and processing are I/O and memory intensive.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-slicing loop (`for t in range(n_trials)`) could potentially be vectorized using array reshaping instead of a Python loop. However, this loop is not a bottleneck since it just creates views/copies of already-computed arrays.

ii.
```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. The AI chose a straightforward loop for clarity. The bottleneck is in the array operations (smoothing, pooling), not the trial-slicing loop.

## 6-c. What processing does the code repeat multiple times?

i. The `pool_time` function is called separately for neural data and position data for each session. These are independent computations on different arrays, so this is not wasteful duplication.

ii.
```python
# In session_neural:
return pool_time(smoothed).astype(np.float32), keep
# In session_position_bins:
pooled = pool_time(position_day.astype(np.float64))
```

iii. No significant repeated processing is present. Each computation is on different data.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The Gaussian smoothing and temporal pooling are applied to the full session-length data before trial cutting, meaning the remainder frames (those that don't fill a complete trial) are processed but discarded. Additionally, the code processes all neurons before the minimum-events filter could potentially be applied earlier to reduce computation.

ii.
```python
# Full session is smoothed and pooled, then only complete trials are kept
smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
return pool_time(smoothed).astype(np.float32), keep
```

iii. Processing the full session before cutting is simpler and the wasted computation on remainder frames is negligible.
