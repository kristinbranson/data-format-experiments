# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a joblib file in `/app/data/` (e.g., `QLAK-CA1-08`). These are loaded with `joblib.load()`, which returns a dictionary keyed by animal name containing `trace`, `position`, `envs`, and `blocked` arrays. The `.mat` files in the same directory contain equivalent data but were not used.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position = dat["trace"], dat["position"]
envs = [str(e) for e in np.asarray(dat["envs"]).ravel()]
blocked_all = dat["blocked"]
n_days = trace.shape[0]
```

iii. The agent examined both `.mat` and joblib files and chose the joblib format, noting in the docstring that these are "identical content to the .mat files, produced by the paper's own `mat2joblib`." The agent explicitly listed the 7 animal names as constants.

## 1-b. How are the data split into subjects?

i. Each joblib file in `/app/data/` corresponds to one subject (mouse). The 7 animal names are hardcoded in the `ANIMALS` list and iterated over.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for subject, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
```

iii. The agent identified 7 mice from the data directory and hardcoded them, consistent with the paper's report of 7 animals.

## 1-c. How are the data split into sessions?

i. Each joblib file contains multi-day recordings indexed along the first axis of `trace` and `position`. Each day becomes a separate session in the output.

ii.
```python
n_days = trace.shape[0]
days = range(n_days) if max_sessions is None else range(min(n_days, max_sessions))
for day in days:
    ...
```

iii. The `trace` array has shape `(n_days, n_cells, n_frames)`, so iterating over the first axis yields one session per recording day.

## 1-d. How are the data split into trials?

i. Each 40-minute continuous recording session is split into consecutive 1-minute trials. At 100 ms bins (after temporal binning), each trial is 600 time bins. Trailing partial minutes are dropped.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FPS / FRAMES_PER_BIN))  # 600
...
n_trials = n_bins // BINS_PER_TRIAL
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
```

iii. The instructions specify 1-minute trials. After temporal binning (3 frames per bin at 30 Hz = 100 ms bins), 60 seconds / 0.1 s = 600 bins per trial.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All complete 1-minute trials from all sessions are kept. Only trailing partial minutes are dropped.

ii. N/A (no trial filtering code)

iii. The paper does not describe trial-level quality filtering for its decoding analysis, and the instructions do not require it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib files, which contains the binarised rising phase of calcium transients with shape `(n_days, n_cells, n_frames)`.

ii.
```python
trace, position = dat["trace"], dat["position"]
...
tr = trace[day]
```

iii. The agent's docstring states: "trace: (n_days, n_cells, n_frames) binarised rising phase of calcium transients, NaN for a cell that was not registered on that day. The paper treats this binary vector as the firing rate in every analysis."

## 2-b. How is the `neural` data processed?

i. The binary trace data undergoes three processing steps matching the paper's `utils.fit_decoder`: (1) Gaussian smoothing with sigma = 3 frames along the time axis, (2) average pooling over non-overlapping windows of 3 frames (producing 100 ms bins), and (3) multiplication by the frame rate (30 Hz) to express the result as an event rate in Hz.

ii.
```python
SMOOTH_SIGMA_FRAMES = FRAMES_PER_BIN  # 3
...
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
```

iii. The agent explicitly documented this in the code docstring: "gaussian_filter1d along time with sigma = 3 frames, then average pooling over 3 frames... The pooled value is multiplied by the frame rate so that the stored quantity is an event rate in Hz, the same units the paper uses for its rate maps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps are applied: (1) Neurons not registered on a given day (all-NaN) are removed. (2) Neurons with 5 or fewer transients in the session are dropped, matching the paper's `cell_threshold = 5` sparsity criterion.

ii.
```python
CELL_EVENT_THRESHOLD = 5
...
registered = ~np.isnan(tr[:, 0])
tr = tr[registered]
active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
tr = tr[active]
```

iii. The agent noted: "Cells with 5 or fewer transients in the session are then dropped, which is the `cell_threshold = 5` sparsity criterion of the paper's decoder (~1% of cells). No place-cell/spatial-reliability selection is applied, matching the paper's decoding analysis."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous with no trial structure, so trials are simply consecutive 1-minute segments from recording onset. Neural and position data are already frame-aligned from the DAQ (same 30 Hz acquisition).

ii. N/A (no explicit alignment code needed)

iii. The agent documented: "The two streams are acquired by the same DAQ at 30 Hz and are already frame-aligned in the released arrays."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz data is rebinned into 100 ms bins by average-pooling 3 consecutive frames. This matches the paper's `temporal_bin_size` parameter in `utils.fit_decoder`.

ii.
```python
FPS = 30
FRAMES_PER_BIN = 3
TIME_BIN_MS = 1000.0 * FRAMES_PER_BIN / FPS  # 100.0 ms
...
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
```

iii. The agent followed the paper's decoder code which uses `temporal_bin_size = 3` frames, yielding 100 ms bins at 30 Hz.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the joblib data, which contains the indices of occluded partitions for each session.

ii.
```python
blocked_all = dat["blocked"]
...
blocked = blocked_indices(blocked_all[day])
```

iii. The agent noted: "`blocked` is used rather than `utils.get_env_mat(env_name)` because several geometries were run in a vertically mirrored version for some animals, and `blocked` records the configuration actually used."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-dimensional binary vector (one per partition), with 1 for blocked and 0 for open. A value of -1 (no partitions blocked) yields an all-zeros vector. The input is static per session/trial.

ii.
```python
def blocked_indices(blocked_entry):
    vals = np.atleast_1d(np.asarray(blocked_entry[0], dtype=float)).ravel()
    vals = vals[vals >= 0]
    return vals.astype(int)
...
geom = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
geom[blocked] = 1.0
...
input_trials.append(geom.copy())
```

iii. One-hot encoding allows the decoder to treat each blocked position independently.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the joblib files, which contains 2D head coordinates `(2, n_frames)` in cm (0-75 cm range).

ii.
```python
position = dat["position"]
...
pos = pool_mean(position[day], FRAMES_PER_BIN)  # (2, n_time_bins)
```

iii. The agent noted: "head position from DeepLabCut, in cm, 0-75 cm in both x and y, already in a common frame across days."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first average-pooled over 3-frame windows (same windows as neural data), then discretized into a 3x3 grid using floor division by `bin_size`, where `bin_size = (max_position + buffer) / 3`, computed from the maximum position across all days of the animal (matching `utils.decode_position_within`).

ii.
```python
bin_size = (np.nanmax(position) + POSITION_BUFFER) / N_SPATIAL_BINS
...
pos = pool_mean(position[day], FRAMES_PER_BIN)
xb = np.clip((pos[0] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
part = N_SPATIAL_BINS * yb + xb
```

iii. The agent followed the paper's position binning method from `decode_position_within`, using the per-animal maximum position to set the bin size, yielding ~25 cm bins (one partition of the 75 cm arena).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). The partition index is `3 * y_bin + x_bin`. Additionally, positions that fall inside a blocked partition (tracking noise) are snapped to the nearest open partition, mirroring the paper's cleaning step in `decode_position_within`.

ii.
```python
part = N_SPATIAL_BINS * yb + xb

stray = np.isin(part, blocked)
if stray.any():
    d = np.linalg.norm(_PART_CENTRES[part[stray]][:, None, :]
                       - _PART_CENTRES[open_parts][None, :, :], axis=2)
    part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. The agent noted: "The residual handful of samples that fall inside a blocked partition are tracking noise at partition walls; they are reassigned to the nearest open partition, mirroring the 'cleaning' of actual and predicted bins to the nearest valid bin in `decode_position_within`."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are acquired by the same DAQ at 30 Hz and have identical frame counts. Both are average-pooled over the same 3-frame windows and split into trials with the same indices, ensuring alignment.

ii.
```python
pos = pool_mean(position[day], FRAMES_PER_BIN)
...
n_bins = min(rates.shape[1], part.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. Both streams are pooled identically and sliced with the same indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of missing/erroneous data are handled: (1) Neurons not registered on a day (NaN values) are removed. (2) Near-silent cells (≤5 transients) are dropped. (3) Position samples that fall inside blocked partitions (tracking noise) are snapped to the nearest open partition. Trailing frames that don't fill a complete 1-minute trial are discarded.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered]
active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
tr = tr[active]
...
stray = np.isin(part, blocked)
if stray.any():
    ...
    part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. These handling strategies match the paper's processing pipeline and the `decode_position_within` function.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading the large joblib files (I/O bound), (2) Gaussian smoothing of all neural traces with `gaussian_filter1d`, which operates on the full `(n_cells, n_frames)` array for each session.

ii. N/A

iii. The joblib files contain large multi-day arrays and loading them dominates wall time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-slicing loop could potentially be vectorized by reshaping the full session array into `(n_cells, n_trials, BINS_PER_TRIAL)` and splitting along the trial axis, rather than iterating with a Python loop.

ii.
```python
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
```

iii. This loop is not a major bottleneck since `n_trials` is small (~40 per session), but it could be done with a single reshape+split.

## 6-c. What processing does the code repeat multiple times?

i. The `geom.copy()` is called once per trial within a session, creating identical copies of the same blocked vector. This is minor but technically repeated.

ii.
```python
input_trials.append(geom.copy())
```

iii. Since the geometry is static per session, this is repeated `n_trials` times per session unnecessarily (though the overhead is negligible for a 9-element array).

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed `session_info` metadata (environment names, per-session neuron counts, etc.) which is not used by the decoder but is useful for documentation. The snapping of stray positions to nearest open partitions affects a tiny fraction of samples (~0.003%).

ii.
```python
session_info.append({
    "subject": animal,
    "day": int(day),
    "environment": envs[day],
    "blocked_partitions": [int(b) for b in blocked],
    "n_neurons": int(rates.shape[0]),
    "n_neurons_registered": n_registered,
    "n_trials": int(n_trials),
})
```

iii. The session_info metadata is useful for provenance but not consumed by the decoder itself.
