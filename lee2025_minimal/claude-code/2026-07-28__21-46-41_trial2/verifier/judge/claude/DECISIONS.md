# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` on pre-converted files (without `.mat` extension) in the data directory. It hardcodes the list of 7 animal IDs and loads each file as a dictionary keyed by the animal name. This differs from the reference, which uses `h5py` to read the `.mat` (HDF5) files directly.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
```

iii. The agent found that the reference code's `load_dat` function supports both `"MATLAB"` and `"joblib"` formats, with `"joblib"` as the default. The data directory contains both `.mat` files and pre-converted joblib files. The agent followed the reference code's default by using joblib.

## 1-b. How are the data split into subjects?

i. Each animal ID in the hardcoded `ANIMALS` list corresponds to one subject. The agent iterates over this list and loads each file separately.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = animals.copy()
for a_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. The agent identified 7 subjects from the data directory files, matching the paper's description of 7 mice.

## 1-c. How are the data split into sessions?

i. Each day of recording for each animal becomes a separate session. The number of days is determined from `d['envs'].shape[0]`. Each day has its own environment, trace data, and position data.

ii.
```python
n_days = d['envs'].shape[0]
...
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]  # (n_cells, n_frames)
    position_day = d['position'][day]  # (2, n_frames)
```

iii. The agent recognized that each day corresponds to one recording session in one environment, consistent with the paper's description.

## 1-d. How are the data split into trials?

i. Each session is split into 1-minute (1800-frame) non-overlapping trials. The number of trials is `n_frames // TRIAL_FRAMES`. Remainder frames that don't fill a complete trial are discarded. Sessions with fewer than 2 trials are skipped.

ii.
```python
TRIAL_FRAMES = FPS * TRIAL_DURATION_S  # 1800 frames per trial
...
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
```

iii. The agent followed the instructions specifying 60-second trials. It added a minimum of 2 trials per session check based on the instruction that "there needs to be at least two trials within each session in order to evaluate the decoder performance."

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filtering is skipping sessions with fewer than 2 trials. No individual trial quality filtering is applied. Sessions with 0 registered neurons are also skipped.

ii.
```python
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The agent added these as safety checks. The reference solution does not filter by minimum trial count.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the loaded data, which contains calcium transient traces with shape `(n_days, n_cells, n_frames)`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_frames)
```

iii. The agent identified `trace` as the neural data variable, which contains binary calcium transient events.

## 2-b. How is the `neural` data processed?

i. The AI applies temporal rebinning: it sums every 30 binary frames into 1-second time bins, producing event counts. This reduces each trial from 1800 frames to 60 timepoints. The reference solution keeps data at the native 30 Hz (1800 timepoints per trial) with no rebinning.

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
...
def bin_neural_data(trace, time_bin_frames):
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    truncated = trace[:, :n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)
```

iii. The agent reasoned: "The 30 Hz recording rate gives 1800 frames per minute trial, which is quite dense--I should consider downsampling to a more manageable time bin size like 1-second intervals to reduce dimensionality while preserving temporal structure." It chose 1-second bins to reduce dimensionality.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN for a given day (not registered/recorded that day) are removed. Any remaining NaN values in registered neurons are replaced with 0.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. The agent recognized that the trace array contains NaN for unregistered neurons and filtered them out, consistent with the reference approach. The `nan_to_num` call is an extra safety measure not in the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are simply 60-second segments from the start of the session. The alignment event is described as "Start of recording session."

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. There is no stimulus onset or behavioral event to align to in these free exploration sessions. The agent correctly identified this.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data from 30 Hz (~33.33 ms bins) to 1-second bins (1000 ms), yielding 60 timepoints per trial instead of 1800. The reference solution keeps the native 30 Hz resolution with no rebinning.

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
...
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
```

iii. The agent justified this as reducing dimensionality while preserving temporal structure, noting the binary traces are sparse.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` variable in the loaded data, which contains environment names (e.g., 'square', 'o', 't') for each day. It maps these names to 3x3 binary accessibility matrices using the `get_env_mat()` function copied from the reference code. The reference solution uses the `blocked` variable from the `.mat` file, which stores indices of blocked reward positions.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
```

iii. The agent copied the `get_env_mat()` function directly from the reference code's `utils.py`, which maps environment names to predefined 3x3 binary matrices.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is looked up in a dictionary to get a 3x3 binary matrix (1=accessible, 0=blocked), which is then flattened to a 9-element vector. This is static per trial within a session. The reference solution uses the `blocked` indices to create a one-hot vector (1=blocked, 0=accessible) -- the polarity is inverted.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(envs[env], dtype=float)
...
env_flat = env_mat.flatten()  # (9,)
session_input.append(env_flat)  # static per trial
```

iii. The agent followed the reference code's representation of environments as accessibility matrices.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the loaded data, which contains 2D x-y coordinates of the mouse. Shape is `(n_days, 2, n_frames)`.

ii.
```python
position_day = d['position'][day]  # (2, n_frames)
```

iii. The agent identified the position variable as the source of mouse location data.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 bins). The AI uses per-session max values (`np.nanmax`) to determine bin edges, then floors the position divided by (max/3). This differs from the reference, which uses a fixed arena size of 75 cm with `np.linspace(0, 75, 4)[1:-1]` and `np.digitize`.

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

iii. The agent chose to use data-driven bin edges (based on max position) rather than fixed arena size.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized into 9 bins (0-8) using a 3x3 spatial grid. The combined bin index is `x_bin * n_bins + y_bin`. Note the reference uses `y_bin * n_grid + x_bin` -- the row/column ordering is swapped.

ii.
```python
bin_idx = x_bin * n_bins + y_bin
```

iii. The agent described this as "row * 3 + col" but implemented it as `x_bin * 3 + y_bin`, which effectively swaps the x and y axis ordering compared to the reference.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame indices in the source data. Both are sliced using the same trial boundaries. However, because the AI rebins neural data to 1-second bins, it must also rebin position data. It uses mode (most common position bin) within each 30-frame window.

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

iii. Because the AI chose to rebin neural data to 1-second bins, it needed to correspondingly rebin position data. It chose mode as the aggregation method for categorical data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons (unregistered cells) are filtered out per session. Remaining NaN values are replaced with 0 via `nan_to_num`. Sessions with 0 registered neurons or fewer than 2 trials are skipped. Remainder frames at the end of sessions are discarded.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
...
if n_registered == 0:
    continue
if n_trials < 2:
    continue
```

iii. The agent noted that NaN filtering ensures only recorded neurons are included. The `nan_to_num` is a safety measure.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the data files via `joblib.load()` is I/O bound and likely the most time-consuming step. The `bin_position` function using `scipy.stats.mode` for each trial is also relatively expensive.

ii. N/A

iii. The data files contain large 3D arrays (e.g., trace shape `(31, 515, 71866)`). Processing steps like rebinning are relatively fast.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that calls `bin_neural_data` and `bin_position` separately for each trial could be vectorized by operating on the full session data before splitting into trials. The `scipy.stats.mode` call for position binning is particularly expensive per-trial.

ii.
```python
for trial in range(n_trials):
    trial_trace = trace_registered[:, t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_pos_bins = pos_bins[t_start:t_end]
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. The `from scipy.stats import mode` import is done inside the `bin_position` function, which is called once per trial. This is a minor inefficiency but functionally not a repeated computation.

ii.
```python
def bin_position(bin_indices, time_bin_frames):
    ...
    from scipy.stats import mode
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
```

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporal rebinning (1-second bins) is additional processing not present in the reference solution. Whether it is unnecessary depends on the decoder's needs, but it adds complexity and discards temporal resolution. The `nan_to_num` call is unnecessary if all-NaN filtering is correct.

ii.
```python
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. N/A
