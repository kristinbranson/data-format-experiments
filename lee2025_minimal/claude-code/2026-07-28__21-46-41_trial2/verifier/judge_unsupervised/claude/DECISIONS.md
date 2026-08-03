# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-saved joblib files, one per animal. Each file contains a dictionary keyed by animal ID, with fields including `trace` (neural data), `position` (behavioral tracking), and `envs` (environment labels). The AI iterates through a hardcoded list of 7 animal IDs, loading each file with `joblib.load`. This matches the reference code's `load_dat` function which supports both MATLAB and joblib formats.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. The AI identified the data format by examining the data files and reading the reference `load_dat` function in `utils.py`. The joblib files are compressed versions of the original MATLAB datasets, pre-converted by the reference code's `mat2joblib` function. The AI verified that 7 animals, 207 sessions, and 5,413 total neurons matched the paper's statistics.

## 1-b. How are the data split into subjects?

i. Each animal ID in the ANIMALS list corresponds to one subject (mouse). The `subjects` list is a copy of the ANIMALS list, and `subject_idx` maps each session to its animal's index in this list.

ii.
```python
subjects = animals.copy()
# ...
for a_idx, animal in enumerate(animals):
    # ...
    subject_idx_all.append(a_idx)
```

iii. The AI identified 7 unique mice from the data files and paper, matching the reference. Each animal's data is stored in a separate file, making subject identification straightforward.

## 1-c. How are the data split into sessions?

i. Each day of recording for each animal constitutes one session. The number of days per animal is determined by the first dimension of `d['envs']` (shape `(n_days, 1)`). All days are included without filtering. This yields 207 total sessions matching the paper.

ii.
```python
n_days = d['envs'].shape[0]
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]  # (n_cells, n_frames)
    position_day = d['position'][day]  # (2, n_frames)
```

iii. The AI verified the session count (207) against the paper. Each day uses a specific environment geometry, and days are ordered as sequences starting and ending with the square environment.

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 1-minute trials. At 30 Hz, each trial is 1800 frames. The number of trials per session is `n_frames // 1800`. Remainder frames at the end of each session are discarded. This yields 39 trials for animals with 71,866 frames and 40 trials for animals with ~72,000+ frames.

ii.
```python
TRIAL_FRAMES = FPS * TRIAL_DURATION_S  # 1800 frames per trial
n_trials = n_frames // TRIAL_FRAMES
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
```

iii. The AI followed the instruction to split sessions into 1-minute trials. The CONVERSION_NOTES document which animals have 39 vs 40 trials and how many remainder frames are discarded.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped (to allow decoder evaluation). Sessions with 0 registered cells are skipped. No individual trial-level quality filtering is applied (e.g., no speed filtering, no minimum occupancy filtering). No speed threshold is used to exclude low-velocity time points.

ii.
```python
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The AI applied minimal filtering. The reference code's `decode_position_within` function includes speed filtering (v_filt_size, speed_threshold parameters), but the AI did not apply this. The AI's rationale was to include all registered cells matching the paper's general approach.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in each animal's data dictionary. This contains binary calcium transient traces (rise-extracted, binarized at z > 2.5 as described in the paper's methods).

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_frames)
```

iii. The AI correctly identified `trace` as the source of neural data by reading the reference code and paper. The trace data contains binary {0, 1} values representing calcium transient events.

## 2-b. How is the `neural` data processed?

i. The binary trace data is summed into 1-second time bins (30 frames per bin), yielding event counts per neuron per time bin. No smoothing, normalization, or rate conversion (e.g., dividing by time) is applied. The result is integer event counts (converted to float).

ii.
```python
def bin_neural_data(trace, time_bin_frames):
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    truncated = trace[:, :n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)
```

iii. The AI chose 1-second time bins based on the 30 Hz recording rate. This is a reasonable choice that reduces data dimensionality while preserving temporal information. The reference code uses raw 30 Hz data for rate maps but does not specify a particular time binning for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells registered on each day (non-NaN across all timepoints) are included per session. Registered cells are determined by checking if all timepoints for a cell are NaN. Any remaining NaN values in registered cells are replaced with 0. No additional quality filtering (e.g., place cell selection, minimum firing rate, split-half reliability) is applied.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. The AI followed the CellReg registration approach from the reference paper: cells not registered on a given day have all-NaN traces. The total of 69,744 neuron-session entries (rate maps) matches the paper exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Trials are sequential 1-minute windows from the beginning of each session. The metadata specifies `temporal_alignment_event: 'Start of recording session'`, `off_start: 0.0`, `off_end: 60.0`.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
```

iii. Since this is a free exploration task with continuous recording, there is no natural trial-onset event. The AI chose to align to recording start, which is the only meaningful reference point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 1000 ms (1 second) time bins. This is achieved by summing 30 consecutive frames (at 30 Hz) into each bin. Each 1-minute trial thus has 60 time bins. This is a rebinning from the original 30 Hz (33.3 ms) resolution.

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
# ...
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
```

iii. The AI chose 1-second bins as a compromise between temporal resolution and data manageability. The reference code operates at 30 Hz for rate map computation but doesn't specify a particular temporal resolution for decoding.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry is derived from the `envs` field in each animal's data dictionary, which stores the environment name as a string for each day (e.g., 'square', 'o', 't', etc.).

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
```

iii. The AI correctly identified the `envs` field as the source for environment geometry, consistent with the reference code.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix using the `get_env_mat` function (copied from the reference code). The matrix has 1 for accessible partitions and 0 for blocked partitions. The 3x3 matrix is then flattened to a 9-element vector, which serves as a static (non-time-varying) input per trial.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        # ... etc
    }
    return np.array(envs[env], dtype=float)
```

iii. The AI's `get_env_mat` function exactly matches the reference code's implementation. The 10 environment geometries are all correctly defined.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in each animal's data dictionary, which contains x-y coordinates tracked at 30 Hz. Shape is `(2, n_frames)` per day.

ii.
```python
position_day = d['position'][day]  # (2, n_frames)
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)
```

iii. The AI correctly identified the position field as the source for mouse position output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position (x, y) coordinates are discretized into 3x3 spatial bins. Each dimension is independently binned using `floor(coord / (max_coord + epsilon) * n_bins)`, where max_coord is the per-session maximum for that dimension. A small buffer (1e-5) is added to prevent the maximum value from falling outside the last bin. The bin index is computed as `x_bin * 3 + y_bin`, giving values 0-8.

ii.
```python
def discretize_position(position, n_bins=3):
    x = position[0]
    y = position[1]
    x_max = np.nanmax(x) + POSITION_BUFFER
    y_max = np.nanmax(y) + POSITION_BUFFER
    x_bin = np.floor(x / (x_max / n_bins)).astype(int)
    y_bin = np.floor(y / (y_max / n_bins)).astype(int)
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx
```

iii. This approach mirrors the reference code's `get_rate_maps` binning: `position_binned = (position // ((np.nanmax(position, axis=0) + buffer) / n_bins)).astype(int)`. The AI used per-session normalization rather than a fixed 75 cm arena size.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (0-8), representing a 3x3 spatial grid. The combined bin index is `x_bin * 3 + y_bin`. Output values are labeled as `row0_col0` through `row2_col2`.

ii.
```python
bin_idx = x_bin * n_bins + y_bin
# ...
position_labels = []
for row in range(N_SPATIAL_BINS):
    for col in range(N_SPATIAL_BINS):
        position_labels.append(f"row{row}_col{col}")
```

iii. The 9-bin categorization follows the instructions ("Mouse position discretized into 3 x 3 = 9 spatial bins").

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is first discretized at the original 30 Hz frame resolution, then temporally binned into 1-second bins (matching the neural data) using the mode (most common bin index) within each 30-frame window. This ensures the output has the same number of time bins (60 per trial) as the neural data.

ii.
```python
def bin_position(bin_indices, time_bin_frames):
    n_frames = len(bin_indices)
    n_bins = n_frames // time_bin_frames
    truncated = bin_indices[:n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_bins, time_bin_frames)
    from scipy.stats import mode
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
    return binned
```

iii. Using mode for categorical position data is appropriate (vs. mean or median which aren't meaningful for categorical data). The AI correctly synchronized the temporal resolution of position output with neural data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN values in trace data identify unregistered cells (all-NaN rows), which are excluded per session. Any stray NaN values in registered cells are replaced with 0. Position data has no NaN values in this dataset. Sessions with 0 registered cells or fewer than 2 trials are skipped.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
# ...
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The AI's handling is conservative and defensive. The NaN-to-zero replacement for registered cells is a safety measure that shouldn't trigger in practice (registered cells should have valid data).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading large joblib data files for each animal (each containing ~70,000 frames x hundreds of cells). (2) The `scipy.stats.mode` computation for position binning, called once per trial. (3) The neural data reshaping and summing for time binning.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))  # Large file I/O
binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)  # Per-trial mode
```

iii. The I/O-bound data loading dominates runtime. The mode computation is relatively efficient given the small window size (30 frames).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial loop could potentially be vectorized by processing all trials simultaneously for neural data binning and position binning. Instead of looping over trials, the entire session's data could be reshaped at once (e.g., reshape the full trace into `(n_cells, n_trials, trial_frames)` and then bin in one operation).

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. The loop is straightforward and readable. The per-trial overhead is small since each trial processes only 1800 frames.

## 6-c. What processing does the code repeat multiple times?

i. The position discretization is done once per session (efficient), then sliced per trial. The `get_env_mat` lookup is called once per session and reused for all trials. No significant repeated processing is evident. The `from scipy.stats import mode` import inside `bin_position` is re-executed every call but is a minor issue.

ii.
```python
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)  # Once per session
# ...
for trial in range(n_trials):
    trial_pos_bins = pos_bins[t_start:t_end]  # Slice, not recompute
```

iii. The code is reasonably efficient, avoiding major recomputation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code counts `total_neurons_unique` by summing `n_cells_total` (all cells, including unregistered) per animal. This is used only for metadata reporting. The `deepcopy` import is unused. The `from copy import deepcopy` import is present but never called. The remainder frames at the end of each session (after the last complete trial) are loaded but discarded.

ii.
```python
total_neurons_unique += n_cells_total  # Counts all cells, only for metadata
# ...
from copy import deepcopy  # Imported but not used
```

iii. These are minor inefficiencies. The code is lean overall, computing only what's needed for the output format.
