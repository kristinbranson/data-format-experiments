# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from preprocessed joblib files (one per animal) in the `data/` directory using `joblib.load()`. Each file contains a dictionary keyed by the animal ID, with sub-keys `trace`, `position`, `envs`, `blocked`, etc. The data is already preprocessed (3D arrays for trace and position, indexed by day).

ii.
```python
def load_animal_data(animal):
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]
```

iii. The AI identified that the data directory contained both `.mat` files and preprocessed joblib files. The AI chose to use the joblib files because they were already in a convenient Python format with preprocessed arrays (e.g., `trace` as a 3D array of shape `(n_days, n_cells, n_frames)`). The CONVERSION_NOTES.md documents this as the primary data source found during Step 2.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (animal). The AI iterates over a hardcoded list of 7 animal IDs (`ANIMALS`), loading each file separately.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, ...)
```

iii. The AI documented in CONVERSION_NOTES.md that 7 subjects were found in the data directory, matching the paper's description of 7 mice.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days. The AI iterates over the days dimension of the trace array. Each day becomes a separate session.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. The AI found that each animal had 21-31 days of recording, totaling 207 sessions, consistent with the reference paper.

## 1-d. How are the data split into trials?

i. Each session (~40 min continuous recording) is split into 1-minute (1800-frame) non-overlapping segments. The AI additionally keeps partial trials at the end of a session if they are at least 30 seconds (900 frames) long, resulting in variable-length trials.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames
MIN_TRIAL_FRAMES = FPS * 30  # Minimum 30s for a partial trial at end
...
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The AI reasoned that 60-second trials are specified in the instructions. However, it also decided to keep partial trials (>= 30s) at the end of sessions rather than discarding them.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out partial trials shorter than 30 seconds and skips sessions that would produce fewer than 2 trials.

ii.
```python
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
if n_trials < 2:
    print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
    continue
```

iii. The minimum of 2 trials per session is required by the decoder for train/test splitting. The 30-second minimum for partial trials avoids very short segments. In practice, no sessions were skipped since all have ~40 minutes of data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib data, which contains binary calcium event traces (0/1 values from rising-phase extraction with z-score > 2.5 threshold). Shape: `(n_days, n_cells, n_frames)`.

ii.
```python
trace = d['trace'][day]  # (n_cells, n_frames) for this day
```

iii. The AI documented in CONVERSION_NOTES.md that the trace data is already binarized rising-phase calcium events, not raw fluorescence, so no delta F/F computation was needed.

## 2-b. How is the `neural` data processed?

i. The only processing is filtering out unregistered cells (NaN values) and casting to float32. The binary trace data is used as-is without further transformation.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
trace_registered = trace[registered_mask]  # (n_registered, n_frames)
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI noted that the data was already binarized and treated it as the firing rate representation. No smoothing, rate map computation, or other transformations were applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by checking if their trace value at the first timepoint is NaN. Only neurons with a non-NaN value at frame 0 are considered "registered" for that day.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
```

iii. The AI assumed that NaN at the first frame indicates the neuron was not recorded on that day. The CONVERSION_NOTES.md notes this as filtering by "registered cells" since cells tracked via CellReg may not be present in all sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are simply consecutive 1-minute segments starting from the beginning of the session.

ii.
```python
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. There is no stimulus onset or specific event to align to. The session is a continuous free exploration recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI noted that the recording was acquired at 30 Hz and chose to preserve this native resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the input from the `envs` variable in the joblib data, which contains the environment name string for each day (e.g., 'square', 'o', 't', 'u', etc.). This string is then converted to a 3x3 binary matrix using the `get_env_mat()` function from the reference code.

ii.
```python
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI chose to use the environment geometry encoding from the reference code's `get_env_mat` function, which represents which spatial partitions of the arena are accessible (1) or blocked (0).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via `get_env_mat()`, then flattened to a 9-element vector. Values of 1 indicate accessible partitions and 0 indicates blocked partitions.

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    # ... etc for all 10 environments
```

iii. The AI copied `get_env_mat` directly from the reference code (`utils.py`), ensuring consistency with the original analysis.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per session (constant across all trials within a session). It is stored as a per-trial constant vector of shape `(9,)`.

ii.
```python
input_trial = env_mat.astype(np.float32)  # (9,) - same for all trials in session
```

iii. Since the environment doesn't change within a session, no temporal alignment is needed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable in the joblib data, which contains 2D coordinates (x, y) of the animal in the 75x75 cm arena at each frame. Shape: `(2, n_frames)` per day.

ii.
```python
position = d['position'][day]  # (2, n_frames)
```

iii. Position was tracked with DeepLabCut as documented in the paper.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is discretized into a 3x3 grid (9 bins). Each axis is divided into 3 equal bins of ~25 cm each. The bin index is computed as `x_bin * 3 + y_bin` (row-major with x as the first dimension).

ii.
```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx
```

iii. The AI chose a 3x3 grid as specified in the instructions and used `np.floor` with a small buffer to avoid edge effects at 75 cm.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D position is binned into 9 spatial categories (0-8) using a 3x3 grid. The bin size is approximately 25 cm per bin (75 cm / 3). The bin index formula is `x_bin * 3 + y_bin`.

ii.
```python
bin_size = (env_size + buffer) / N_POS_BINS  # ~25.0 cm
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin
```

iii. The AI chose floor-based binning with clipping for boundary handling. Output values are named `x{r}y{c}` where r and c index the row and column of the grid.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and are aligned frame-for-frame. Both are split into trials using the same start/end indices.

ii.
```python
pos_bins = bin_position_3x3(position)  # (n_frames,)
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. Since both arrays share the same time axis, slicing with the same indices ensures alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~33.33 ms (native 30 Hz). No rebinning is applied.

ii.
```python
FPS = 30
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. Same as 2-e.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data share the same 30 Hz frame rate and are sliced identically by trial boundaries. Input (environment geometry) is static per session and doesn't require temporal alignment.

ii.
```python
neural_trial = trace_registered[:, start:end].astype(np.float32)
input_trial = env_mat.astype(np.float32)  # static per trial
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The identical slicing ensures neural and position data remain synchronized.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Unregistered neurons (NaN traces) are filtered out based on the first frame. Partial trials shorter than 30 seconds are dropped. Sessions with fewer than 2 trials are skipped (though this never occurs in practice).

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
...
if trial_len < MIN_TRIAL_FRAMES:
    continue
if n_trials < 2:
    continue
```

iii. The AI documented these handling strategies in CONVERSION_NOTES.md and verified that all sessions produced at least 2 trials.

## 7-a. What are the most time-consuming steps of the code?

i. Loading data from joblib files is the most time-consuming step, as each animal's data file is large (containing 3D arrays for trace, position, etc.). The AI reported ~160 seconds total for all 7 animals.

ii. N/A (timing is from runtime output)

iii. The AI's conversion output shows individual animal processing times of 11-29 seconds each, dominated by I/O.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trial start indices and slices arrays. This could potentially be vectorized using `np.reshape` for sessions where all trials are the same length (1800 frames).

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The AI did not identify this as a bottleneck since the trial splitting is fast compared to I/O.

## 7-c. What processing does the code repeat multiple times?

i. The `astype(np.float32)` conversion is done per-trial for neural data, when it could be done once for the entire session before splitting. Similarly, position binning is done once per session (efficient), but the trial-level slicing creates copies.

ii.
```python
neural_trial = trace_registered[:, start:end].astype(np.float32)  # repeated per trial
```

iii. Not explicitly discussed in CONVERSION_NOTES.md.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores partial trials (between 30-60 seconds) with variable lengths, which may cause issues in some downstream analyses expecting uniform trial lengths. The plotting function (`plot_processing`) does processing that is only used for visualization.

ii.
```python
if trial_len < MIN_TRIAL_FRAMES:
    continue
# Partial trials with 900 <= trial_len < 1800 are kept
```

iii. Not explicitly discussed in CONVERSION_NOTES.md.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. Unregistered neurons are filtered based on first-frame NaN check. Partial trials < 30s are dropped. Sessions with < 2 trials are skipped.

ii. See question 6 code snippets.

iii. See question 6 justification.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. File I/O (loading joblib files) dominates at ~160 seconds for the full dataset.

ii. N/A

iii. The AI printed timing per animal showing 11-29 seconds each.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The per-trial slicing loop could potentially use reshape for uniform-length portions.

ii. See 7-b code snippet.

iii. N/A

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. The `astype(np.float32)` conversion is repeated per trial.

ii. See 7-c code snippet.

iii. N/A

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Partial trials and visualization-only processing.

ii. See 7-d code snippet.

iii. N/A
