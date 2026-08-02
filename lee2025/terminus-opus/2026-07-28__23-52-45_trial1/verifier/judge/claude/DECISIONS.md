# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (one per animal) rather than the `.mat` files. It uses `joblib.load()` to read each animal's data dictionary, which contains trace, position, environment, and blocked fields. The reference code's `load_dat()` function supports both formats, and the AI chose joblib as it is the format used by the reference code's primary loading path.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
n_days = d['trace'].shape[0]
n_cells_total = d['trace'].shape[1]
n_timepoints = d['trace'].shape[2]
```

iii. The AI noted in CONVERSION_NOTES.md Step 1 that `load_dat(animal, p, format="joblib")` is the reference code's loading function. The joblib files contain the same preprocessed data as the `.mat` files in a Python-native format.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (animal). The animal ID string (e.g., "QLAK-CA1-08") serves as the subject identifier. A hardcoded list of all 7 animal IDs is used.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subject_id = animal
subj_idx = all_subjects.index(subject_id)
```

iii. Each joblib file is keyed by the animal ID. The hardcoded list ensures consistent ordering.

## 1-c. How are the data split into sessions?

i. Each day of recording for a given animal becomes a separate session. The trace array has shape `(n_days, n_cells, n_timepoints)`, so iterating over the first dimension yields individual sessions.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. The AI confirmed that each "day" corresponds to one recording session, yielding 207 total sessions across 7 animals, matching the paper's stated count.

## 1-d. How are the data split into trials?

i. Each ~40-minute continuous recording session is split into non-overlapping 60-second trials. After temporal binning to 1-second bins, each trial contains 60 time bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)  # 60 time bins per trial
...
def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_timebins = len(data)
        n_trials = n_timebins // trial_length
        trials = []
        for t in range(n_trials):
            start = t * trial_length
            end = start + trial_length
            trials.append(data[start:end])
        return trials
    else:
        n_timebins = data.shape[-1]
        n_trials = n_timebins // trial_length
        ...
```

iii. Per instructions, trials are defined as 1-minute segments. Due to temporal rebinning to 1-second bins, each trial is 60 bins long instead of the native 1800 frames.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped entirely. No individual trial quality filtering is applied.

ii.
```python
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." In practice, all sessions are ~40 minutes, so no sessions are actually skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib data, which contains binary calcium event traces (0/1 values representing the rising phase of calcium transients) with shape `(n_days, n_cells, n_timepoints)`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. CONVERSION_NOTES.md Step 1 notes: "The trace data is BINARY (0/1) - rising phase of calcium transients" and "No additional dF/F computation needed - data is already preprocessed."

## 2-b. How is the `neural` data processed?

i. The AI applies temporal binning: the binary trace data is averaged within 1-second windows (30 frames per bin) to produce firing rates. Only neurons registered (non-NaN) in that session are included. The result is cast to float32.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
...
def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. CONVERSION_NOTES.md Step 5: "Use 1 second (30 frames) bins. This provides reasonable temporal resolution while reducing data size. The reference code uses temporal_bin_size=3 (100ms) for decoding, but for our decoder format, 1-second bins are more practical and still capture spatial behavior well."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons not registered on a given day are filtered out by checking if the first timepoint is NaN. Only neurons with a valid (non-NaN) first timepoint are included.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]  # (n_valid_cells, n_timepoints)
n_valid = valid_mask.sum()
```

iii. CONVERSION_NOTES.md Step 2: "Cells not registered on a given day have NaN traces." Checking only the first timepoint is an efficiency optimization since unregistered cells have NaN for all timepoints.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The continuous recording is simply split into fixed-length 60-second segments from the start of the recording. Alignment is to the start of the recording session.

ii. N/A (no alignment code - trials are sequential segments)

iii. CONVERSION_NOTES.md Step 5: "Temporal alignment: Start of recording." There is no stimulus event to align to in this free-exploration paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has a temporal resolution of 1 second (1000 ms). Temporal rebinning IS applied: the original 30 Hz data (33.33 ms bins) is averaged within 1-second windows.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
```

iii. The AI chose 1-second bins as a balance between temporal resolution and data size. The reference code's decoder uses 3-frame (100 ms) bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field in the joblib data, which contains the environment name string (e.g., "square", "o", "t") for each recording day. The environment name is then mapped to a 3x3 binary geometry matrix using the `get_env_mat()` function copied from the reference code.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. CONVERSION_NOTES.md Step 4: "Use get_env_mat(env_name) for input, not blocked field directly." The AI explicitly chose the `envs` + `get_env_mat()` approach over the `blocked` indices approach.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is looked up in a dictionary mapping names to 3x3 binary matrices (from the reference code's `get_env_mat()` function). The matrix is flattened to a 9-element vector. Values of 1 indicate accessible positions, 0 indicate blocked positions.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square': [[1,1,1],[1,1,1],[1,1,1]],
        'o': [[1,1,1],[1,0,1],[1,1,1]],
        't': [[0,1,0],[0,1,0],[1,1,1]],
        'u': [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+': [[0,1,0],[1,1,1],[0,1,0]],
        'i': [[1,1,1],[0,1,0],[1,1,1]],
        'l': [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn': [[1,1,0],[1,1,1],[0,1,1]],
    }
    return np.array(env_mats[env], dtype=float)
...
session_input.append(env_mat.astype(np.float32))
```

iii. The AI copied `get_env_mat()` directly from the reference code (`utils.py:215`). This provides the canonical environment geometry representation.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is a static per-trial input (constant across all timepoints and trials within a session). No temporal alignment is needed.

ii.
```python
session_input.append(env_mat.astype(np.float32))  # same for all trials in session
```

iii. Environment geometry doesn't change within a session, so it's replicated identically for each trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position output is derived from the `position` field in the joblib data, which contains 2D (x, y) coordinates of the animal in the arena with shape `(n_days, 2, n_timepoints)` at 30 Hz.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The position data was tracked with DeepLabCut in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is discretized into a 3x3 spatial grid (9 classes). Bin edges are computed dynamically from the maximum position value. The bin index is computed as `x_bin * 3 + y_bin`. Additionally, position is temporally binned to match the 1-second neural bins by taking the mode (most frequent position bin) within each 1-second window.

ii.
```python
def bin_position_to_grid(position, n_spatial_bins=3):
    buffer = 1e-5
    pos_max = np.nanmax(position) + buffer
    bin_size = pos_max / n_spatial_bins
    pos_binned = np.floor(position / bin_size).astype(int)
    pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
    bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
    return bin_idx

def bin_position_temporal(position, time_bin_frames, n_spatial_bins=3):
    bin_idx = bin_position_to_grid(position, n_spatial_bins)
    n_timepoints = len(bin_idx)
    n_bins = n_timepoints // time_bin_frames
    bin_idx_truncated = bin_idx[:n_bins * time_bin_frames]
    bin_idx_reshaped = bin_idx_truncated.reshape(n_bins, time_bin_frames)
    result = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result
```

iii. CONVERSION_NOTES.md Step 5: "Bin position into 3x3 grid (9 bins). Use the same binning approach as the reference code: pos_binned = floor(pos / bin_size)." The mode is used for temporal binning to capture the predominant position within each time bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is divided into 3 equal bins per axis based on the max position value. `np.floor(pos / bin_size)` assigns each position to a bin. Values are clipped to [0, 2] to handle boundary cases. The combined 2D index gives 9 categories (0-8).

ii.
```python
pos_max = np.nanmax(position) + buffer
bin_size = pos_max / n_spatial_bins
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. The 3x3 grid is specified by the instructions ("Mouse position discretized into 3 x 3 = 9 spatial bins").

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both neural and position data undergo the same temporal binning (30 frames averaged/moded to 1-second bins), then are split into trials using the same indices. This ensures frame-for-frame alignment.

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. Both data streams are binned and split identically, ensuring temporal alignment is preserved.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 1 second (1000 ms). Temporal rebinning IS applied: the original 30 Hz data is averaged (neural) or mode-computed (position) within 1-second windows.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30
```

iii. Same as 2-e. The AI chose 1-second bins, deviating from the reference code's 100ms bins and the reference solution's native 30 Hz.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural data is averaged within 1-second bins. Position output is mode-binned within the same 1-second windows. Both produce the same number of time bins, and both are split into 60-bin trials identically. Input is static per trial (no temporal dimension).

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)  # average
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)  # mode
# Both have same n_timebins, split the same way
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. Alignment is ensured by processing both streams with the same temporal bin structure.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Neurons with NaN traces (not registered on a given day) are filtered out by checking the first timepoint. Remainder time bins that don't fill a complete 60-second trial are discarded. Sessions with fewer than 2 trials are skipped.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
...
n_trials = n_timebins // trial_length  # remainder dropped
...
if n_trials < 2:
    continue
```

iii. NaN filtering ensures only recorded neurons are included. Dropping remainders is a minor data loss (~55 seconds per session).

## 7-a. What are the most time-consuming steps of the code?

i. Loading the joblib files via `joblib.load()` is the most I/O-intensive step. Processing (temporal binning, position discretization, trial splitting) is fast by comparison. Total conversion takes ~3 minutes for all 7 animals.

ii. N/A (timing is measured but not a code decision)

iii. CONVERSION_NOTES.md Step 7 reports ~23s per animal, dominated by loading.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The mode computation in `bin_position_temporal` uses a Python loop over time bins, which could be vectorized using `scipy.stats.mode`.

ii.
```python
result = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
```

iii. CONVERSION_NOTES.md Step 6 identifies this: "Mode computation in bin_position_temporal uses a loop (could vectorize with scipy.stats.mode)."

## 7-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified.

ii. N/A

iii. N/A

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporal rebinning to 1-second bins is arguably unnecessary since the decoder could work with native 30 Hz data. Additionally, position temporal binning (mode computation) is a consequence of the rebinning decision and wouldn't be needed at native resolution.

ii. N/A

iii. The AI justified rebinning as reducing data size, but the reference solution keeps native resolution.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. NaN neurons are filtered out, remainder frames are dropped, and sessions with <2 trials are skipped.

ii. See question 6 code snippets.

iii. See question 6 justification.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Data loading is the bottleneck.

ii. N/A

iii. N/A

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The mode computation loop.

ii. See 7-b code snippet.

iii. See 7-b justification.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. No significant repeated processing.

ii. N/A

iii. N/A

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. The temporal rebinning and associated mode computation.

ii. N/A

iii. N/A
