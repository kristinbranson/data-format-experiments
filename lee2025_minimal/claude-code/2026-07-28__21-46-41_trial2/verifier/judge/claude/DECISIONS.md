# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` from pre-converted joblib files (one per subject) in the `data/` directory. Each file is a dictionary keyed by the animal name, containing fields `trace`, `position`, `envs`, and others. This differs from the reference solution, which loads `.mat` files using `h5py`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
n_days = d['envs'].shape[0]
n_cells_total = d['trace'].shape[1]
n_frames = d['trace'].shape[2]
```

iii. The AI discovered the joblib format by reading the reference code (`main.py` which calls `load_dat(animal, p, format="joblib")`), and confirmed it worked by testing `joblib.load('data/QLAK-CA1-08')`. The reference code supports both formats but defaults to joblib.

## 1-b. How are the data split into subjects?

i. The AI hardcodes a list of 7 animal IDs and iterates over them. Each animal's data is loaded from a separate joblib file.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
# ...
for a_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
```

iii. The animal IDs were taken from the data directory contents. The reference solution uses `sorted(glob.glob(f'{args.datadir}/*.mat'))` to discover subjects dynamically from `.mat` files and extracts names from filenames.

## 1-c. How are the data split into sessions?

i. Each day within a subject's data becomes a separate session. The AI iterates over days using `d['envs'].shape[0]` to get the number of days.

ii.
```python
n_days = d['envs'].shape[0]
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_frames)
    position_day = d['position'][day]  # (2, n_frames)
```

iii. The joblib data stores all days for an animal in 3D arrays indexed by day. This is equivalent to the reference solution's approach of iterating over reference arrays in the .mat file.

## 1-d. How are the data split into trials?

i. Each session is split into 1-minute (1800 frames at 30 Hz) non-overlapping trials. Remainder frames are discarded.

ii.
```python
TRIAL_FRAMES = FPS * TRIAL_DURATION_S  # 1800 frames per trial
n_trials = n_frames // TRIAL_FRAMES
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
```

iii. The instructions specify "1-minute trials within each session." Both AI and reference split identically into 1800-frame segments.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out sessions with fewer than 2 trials and sessions with zero registered neurons. No per-trial quality filtering is applied.

ii.
```python
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The instructions state "There needs to be at least two trials within each session." The reference solution does not apply any trial/session filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib data. In the joblib format, `trace` has shape `(n_days, n_cells, n_frames)` containing binary calcium transient events.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_frames)
```

iii. This is equivalent to the reference solution's `trace` variable from the `.mat` file, which has shape `(timepoints, neurons)`. The reference transposes from (timepoints, neurons) to (neurons, timepoints); the AI's joblib format already stores data as (n_cells, n_frames).

## 2-b. How is the `neural` data processed?

i. The AI applies 1-second temporal rebinning by summing 30 binary frames into event counts per bin. This produces (n_cells, 60) per trial instead of (n_cells, 1800). NaNs in registered neurons are replaced with 0.

ii.
```python
def bin_neural_data(trace, time_bin_frames):
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    truncated = trace[:, :n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)

# In convert_data():
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
```

iii. The AI's reasoning was that 30 Hz (1800 frames/trial) is "quite dense" and 1-second bins would "reduce dimensionality while preserving temporal structure." The reference solution keeps data at native 30 Hz with no rebinning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN on a given day (not registered) are excluded. Any remaining NaN values in registered neurons are replaced with 0.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. This matches the reference solution's approach of filtering all-NaN neurons. The additional `nan_to_num` step is a safety measure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. Trials are consecutive 1-minute segments from the start of the recording session.

ii. N/A - no alignment code, trials are sliced sequentially from the beginning.

iii. There is no stimulus event to align to; the recording is continuous free exploration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins from 30 Hz to 1-second bins (1000 ms time bin size), yielding 60 timepoints per 1-minute trial. This is a departure from the reference solution which keeps the native 30 Hz (~33.33 ms bins).

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
# ...
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
```

iii. The AI justified this as reducing dimensionality while preserving temporal structure.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` field in the joblib data, which contains string names of environments (e.g., "square", "o", "t"). These are mapped to 3x3 binary matrices using a `get_env_mat()` function copied from the reference code.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
```

iii. The AI chose this approach because the reference code contains a `get_env_mat()` function with explicit environment-to-matrix mappings. The reference solution instead uses the `blocked` indices from the `.mat` file directly and one-hot encodes which positions are blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix (1=accessible, 0=blocked), then flattened to a 9-element vector. This is static per trial/session.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        # ... 10 environments
    }
    return np.array(envs[env], dtype=float)

env_flat = env_mat.flatten()  # (9,)
session_input.append(env_flat)
```

iii. The reference solution uses a different representation: one-hot encoding of blocked positions (1 = blocked). The AI's representation is effectively the inverse (1 = accessible). Both encode the same information but with opposite polarity.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Environment geometry is static per session/trial -- it does not vary with time. It is replicated as the same vector for all trials in a session.

ii.
```python
session_input.append(env_flat)  # (9,) static per trial
```

iii. No temporal alignment needed since the environment geometry is constant.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in the joblib data, which contains 2D (x, y) coordinates with shape `(2, n_frames)`.

ii.
```python
position_day = d['position'][day]  # (2, n_frames)
```

iii. Same source variable as the reference solution.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). Each axis is binned using `floor(position / (max + epsilon) * 3)`. The combined bin index is `x_bin * n_bins + y_bin`.

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

iii. The reference solution uses `np.linspace(0, arena_size, n_grid + 1)[1:-1]` with `np.digitize` and the convention `y_bin * n_grid + x_bin`. The AI uses a different binning formula and the opposite axis ordering (`x_bin * n_bins + y_bin`).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous 2D position is discretized into 9 categories (0-8) using the 3x3 grid described above. Then, within each 1-second time bin, the mode (most common bin) is taken.

ii.
```python
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)  # (n_frames,)
# ...
def bin_position(bin_indices, time_bin_frames):
    from scipy.stats import mode
    n_bins = n_frames // time_bin_frames
    truncated = bin_indices[:n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_bins, time_bin_frames)
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
    return binned
```

iii. The reference solution keeps position at native 30 Hz and does not apply temporal rebinning to position. The AI's mode-based rebinning is a consequence of its 1-second temporal binning decision.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are from the same recording at 30 Hz, so they are frame-aligned. Both are split into 1-minute trials at the same indices, then both are temporally rebinned to 1-second bins.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. Alignment is maintained by using the same frame indices for slicing and the same bin size for temporal rebinning.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 1-second temporal rebinning (1000 ms bins), reducing from 1800 frames to 60 timepoints per trial. The reference solution keeps native 30 Hz (~33.33 ms bins) with 1800 timepoints per trial.

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
```

iii. The AI justified this as reducing dimensionality for the decoder. The reference solution does not rebin.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position data are originally frame-aligned at 30 Hz. The AI rebins both to 1-second bins using the same frame boundaries: neural data is summed, position is mode-selected. Input (environment geometry) is static per trial.

ii.
```python
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # sum 30 frames
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # mode of 30 frames
session_input.append(env_flat)  # static (9,)
```

iii. Using the same temporal window boundaries for both neural and output ensures they remain aligned after rebinning.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Neurons with all-NaN values (not registered on a given day) are excluded. Remaining NaN values in registered neurons are replaced with 0 via `nan_to_num`. Sessions with zero registered neurons or fewer than 2 trials are skipped.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
if n_registered == 0:
    continue
if n_trials < 2:
    continue
```

iii. The reference solution only filters all-NaN neurons and does not apply `nan_to_num` or session filtering.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the large joblib files, which is I/O bound. The temporal rebinning (reshaping and summing) and `scipy.stats.mode` for position binning are also relatively expensive.

ii. N/A

iii. The data files contain large 3D arrays (days x cells x frames) that must be loaded entirely.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that calls `bin_neural_data` and `bin_position` separately for each trial could be vectorized by processing all trials at once before splitting.

ii.
```python
for trial in range(n_trials):
    trial_trace = trace_registered[:, t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_pos_bins = pos_bins[t_start:t_end]
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. Binning could be done on the full session data first, then split into trials, avoiding repeated function calls per trial.

## 7-c. What processing does the code repeat multiple times?

i. The `scipy.stats.mode` import is done inside the `bin_position` function, which is called once per trial. This is a minor inefficiency.

ii.
```python
def bin_position(bin_indices, time_bin_frames):
    from scipy.stats import mode  # imported every call
```

iii. While the import is cached by Python, it is unconventional to import inside a function called many times.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporal rebinning to 1-second bins is not required by the instructions and adds processing overhead. The position mode computation is also unnecessary if data is kept at native resolution.

ii. N/A

iii. The reference solution avoids this overhead by keeping data at native 30 Hz.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6: all-NaN neurons are excluded, remaining NaNs are replaced with 0, and sessions with no neurons or too few trials are skipped.

ii. See question 6.

iii. See question 6.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading large joblib files and the temporal rebinning operations.

ii. See question 7-a.

iii. See question 7-a.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-trial binning loop.

ii. See question 7-b.

iii. See question 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: the scipy.stats.mode import per call.

ii. See question 7-c.

iii. See question 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: the 1-second temporal rebinning is unnecessary additional processing not required by the task.

ii. See question 7-d.

iii. See question 7-d.
