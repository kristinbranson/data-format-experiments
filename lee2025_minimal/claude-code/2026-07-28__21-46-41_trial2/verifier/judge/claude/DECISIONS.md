# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-serialized pickle files (one per subject, e.g. `QLAK-CA1-08` with no extension) using `joblib.load()`. Each file contains a dictionary keyed by animal ID, with sub-keys `trace`, `position`, and `envs`. This differs from the reference, which loads `.mat` (HDF5) files via `h5py`. Both data sources are present in the data directory.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
n_days = d['envs'].shape[0]
n_cells_total = d['trace'].shape[1]
n_frames = d['trace'].shape[2]
```

iii. The AI chose to load from the pre-processed joblib files rather than the raw `.mat` files. The CONVERSION_NOTES.md states the data originates from the Lee et al. (2025) paper. The joblib files appear to be a pre-processed version of the same data, already organized as 3D arrays (days x cells x frames) rather than arrays of HDF5 references.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of 7 animal IDs. Each animal ID corresponds to one joblib file. All 7 subjects are processed.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for a_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
```

iii. The AI hardcoded the animal list rather than discovering subjects from filenames (as the reference does via `glob`). Both approaches yield the same 7 subjects.

## 1-c. How are the data split into sessions?

i. Each day for each animal becomes one session. The number of days per subject is read from `d['envs'].shape[0]`. Each day corresponds to one environment geometry and one continuous recording.

ii.
```python
n_days = d['envs'].shape[0]
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]  # (n_cells, n_frames)
    position_day = d['position'][day]  # (2, n_frames)
```

iii. This matches the reference approach where each recording session (per `.mat` file reference index) becomes a separate session. The AI correctly produces 207 sessions total.

## 1-d. How are the data split into trials?

i. Each session is split into 1-minute (1800 frame) non-overlapping trials. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_FRAMES = FPS * TRIAL_DURATION_S  # 1800 frames per trial
n_trials = n_frames // TRIAL_FRAMES
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
```

iii. This matches the reference's `split_into_trials()` function which also uses 1800-frame (60-second) non-overlapping windows with the remainder discarded.

## 1-e. How are trials filtered based on quality controls?

i. The AI skips entire sessions (days) that have fewer than 2 trials or zero registered cells. No individual trial-level filtering is applied.

ii.
```python
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The reference does not perform any session or trial filtering. The AI's <2 trial check is motivated by the instruction requirement that "There needs to be at least two trials within each session." However, in practice all sessions have 39-40 trials, so this filter never triggers.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib files, which contains binary calcium transient traces (shape: days x cells x frames). These are binarized rise-extracted calcium events.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_frames)
```

iii. The CONVERSION_NOTES state: "Binary calcium transient traces (trace field) - rise-extracted, binarized at z > 2.5 (per paper methods)." This is the same underlying data as the `trace` variable in the .mat files.

## 2-b. How is the `neural` data processed?

i. The AI applies 1-second temporal binning by summing 30 binary frames into event counts per bin. This reduces the temporal resolution from 30 Hz to 1 Hz (60 time bins per 60-second trial instead of 1800).

ii.
```python
def bin_neural_data(trace, time_bin_frames):
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    truncated = trace[:, :n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)

TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
```

iii. The CONVERSION_NOTES state: "Time binning: Summed over 30 frames = 1-second bins, yielding event counts per bin." The reference does NOT apply temporal rebinning and keeps data at native 30 Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN for a given day (not registered on that day) are excluded. Additionally, any remaining NaN values are replaced with 0.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. The reference only filters all-NaN neurons and does not replace remaining NaN with 0. The AI's `nan_to_num` is a safety measure; per the CONVERSION_NOTES: "Replace any remaining NaNs with 0 (shouldn't happen but be safe)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The continuous recording is split into sequential 1-minute trials starting from the beginning of the session. The metadata specifies `temporal_alignment_event: 'Start of recording session'`.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. There is no stimulus onset or behavioral event to align to. This matches the reference approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 1-second temporal rebinning (30 frames per bin), yielding a time bin size of 1000 ms and 60 time bins per trial. The reference keeps native 30 Hz resolution (~33.33 ms bins, 1800 time bins per trial).

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
```

iii. The CONVERSION_NOTES confirm: "Each trial: 1800 frames (60 seconds) -> 60 time bins (1 second each)." This is a significant difference from the reference.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` field in the joblib files, which stores environment names (e.g., 'square', 'o', 't'). These are mapped to 3x3 binary accessibility matrices using a hardcoded `get_env_mat()` function.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
```

iii. The reference instead uses the `blocked` variable from `.mat` files (indices of blocked positions), encoding them as a one-hot vector of blocked positions. The AI's approach uses environment names to reconstruct the geometry, encoding accessible positions as 1 and blocked as 0.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps environment names to hardcoded 3x3 binary matrices (1=accessible, 0=blocked), then flattens to a 9-element vector. This is static per trial/session.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(envs[env], dtype=float)
```

iii. The reference uses one-hot encoding of blocked positions (1=blocked). The AI uses accessibility encoding (1=accessible). These are complementary representations. Both are 9-element vectors, but with inverted semantics.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` field in the joblib files, containing 2D (x, y) coordinates of the mouse in the arena.

ii.
```python
position_day = d['position'][day]  # (2, n_frames)
```

iii. This is the same underlying position data as in the `.mat` files.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is discretized into a 3x3 grid (9 bins). The AI uses `np.floor(x / (x_max / n_bins))` with per-session max values plus a small buffer, rather than fixed arena size edges. The bin index is computed as `x_bin * n_bins + y_bin`.

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

iii. The reference uses `np.digitize` with fixed 75cm arena edges and computes `y_bin * n_grid + x_bin`. The AI's `x_bin * n_bins + y_bin` ordering differs (row-major vs column-major ordering of the grid). The AI also uses per-session max position rather than a fixed arena size.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 bins (3x3 grid) using floor division with clipping. Each bin covers approximately 25x25 cm of the 75x75 cm arena.

ii.
```python
x_bin = np.floor(x / (x_max / n_bins)).astype(int)
y_bin = np.floor(y / (y_max / n_bins)).astype(int)
x_bin = np.clip(x_bin, 0, n_bins - 1)
y_bin = np.clip(y_bin, 0, n_bins - 1)
bin_idx = x_bin * n_bins + y_bin
```

iii. Both AI and reference produce 9 categories (0-8). The bin boundaries differ slightly (AI uses per-session max, reference uses fixed 75cm) and the index ordering differs.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is first discretized at 30 Hz, then binned to 1-second resolution using mode (most common bin in each 30-frame window), matching the neural data's 1-second time bins. The output shape is (1, 60) per trial.

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

trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
trial_output = trial_output.reshape(1, -1)  # (1, 60)
```

iii. The reference keeps position at native 30 Hz (1800 timepoints per trial) and aligns frame-for-frame with neural data. The AI rebins both neural and position data to 1-second bins to maintain alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons (not registered on a given day) are excluded. Remaining NaN values are replaced with 0 via `np.nan_to_num`. Sessions with 0 registered cells or <2 trials are skipped entirely.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)

if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The reference only filters all-NaN neurons. The AI adds nan_to_num as a safety measure and session-level quality checks.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib files from disk, which contain large 3D arrays (days x cells x frames). The temporal binning and mode computation also add processing time compared to the reference.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
```

iii. The conversion output shows all 7 subjects process relatively quickly. The joblib loading is I/O bound.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial loop could be vectorized by reshaping the entire session's data at once rather than looping over individual trials. The `bin_position` function uses `scipy.stats.mode` inside a per-trial loop, which is relatively slow.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. The trial loop recomputes binning per trial when the entire session could be binned at once and then split.

## 6-c. What processing does the code repeat multiple times?

i. The `from scipy.stats import mode` import is inside the `bin_position` function and gets executed every time the function is called (once per trial per session). The discretization and binning could be done once per session then split, rather than per-trial.

ii.
```python
def bin_position(bin_indices, time_bin_frames):
    ...
    from scipy.stats import mode
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
    return binned
```

iii. The repeated import is a minor inefficiency. The per-trial processing duplicates work that could be done session-wide.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `total_neurons_unique` counter sums all cells per animal (including unregistered ones across sessions), which is only used for metadata. The `deepcopy` import is unused. The `nan_to_num` call processes data that should already be NaN-free after the all-NaN filter.

ii.
```python
from copy import deepcopy  # imported but never used
total_neurons_unique += n_cells_total  # counts all cells, not just registered ones
trace_registered = np.nan_to_num(trace_registered, nan=0.0)  # safety measure for data that should be NaN-free
```

iii. These are minor inefficiencies that don't affect the output data.
