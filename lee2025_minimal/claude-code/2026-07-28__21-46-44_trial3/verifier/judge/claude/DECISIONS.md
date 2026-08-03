# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-serialized files (extensionless files like `QLAK-CA1-08`) in the data directory, not from the `.mat` files. Each joblib file is a dict keyed by animal name containing arrays for `trace`, `position`, `envs`, and `blocked`. Data is loaded with `joblib.load()`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]

n_days = d['trace'].shape[0]
n_cells_total = d['trace'].shape[1]
n_timepoints = d['trace'].shape[2]
```

iii. The AI discovered that the data directory contains both `.mat` files and extensionless joblib files. It chose to use the joblib files because they provide a more convenient Python-native format with pre-organized numpy arrays (shape: `(n_days, n_cells, n_timepoints)` for traces), avoiding the complexity of HDF5 reference resolution required by the `.mat` files. The AI also hard-coded the list of animal names rather than discovering them from the filesystem.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hard-coded list of animal names (`ANIMALS`). Each animal corresponds to one joblib file.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
subjects = list(animals)
```

iii. The AI hard-coded the 7 animal names based on the files found in the data directory. This ensures consistent ordering and correct subject identification.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject's data becomes a separate session. The AI iterates over the first dimension of the trace array (`n_days`).

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. Each recording day corresponds to one environment geometry. The AI correctly identifies each day as a separate session.

## 1-d. How are the data split into trials?

i. Each session (recording day) is split into 1-minute non-overlapping trials of 1800 frames (30 Hz * 60 seconds). Remainder frames that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800
n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. Per the instructions, trials are defined as 1-minute segments of the continuous recording.

## 1-e. How are trials filtered based on quality controls?

i. The AI adds two quality filters: (1) sessions with fewer than 5 registered neurons are skipped, and (2) sessions with fewer than 2 possible trials are skipped. Neither filter triggers on the actual data.

ii.
```python
if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue

if n_trials < 2:
    print(f"  Day {day}: skipping, only {n_trials} possible trials")
    continue
```

iii. The AI added these as safety checks. With ~40 minute sessions (39-40 trials) and hundreds of neurons per session, these never triggered in practice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium transient events with shape `(n_days, n_cells, n_timepoints)`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The `trace` variable contains pre-processed calcium imaging data (binary transient events from rise-phase extraction with z-score > 2.5 threshold).

## 2-b. How is the `neural` data processed?

i. Neurons with all-NaN values are filtered out (see 2-c). Any remaining NaN values are replaced with 0. The data is cast to float32.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]  # (n_registered, n_timepoints)
traces = np.nan_to_num(traces, nan=0.0)
# ...
neural_trials.append(trial_traces.astype(np.float32))
```

iii. The AI determined the traces are binary calcium transient events and applied minimal processing beyond filtering unregistered neurons and replacing NaN.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons where the entire trace is NaN (not registered/recorded in that session) are removed.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()
traces = trace_day[registered]
```

iii. The trace array contains NaN columns for neurons not recorded in a given session (different neurons may be active across days due to cross-day registration). Only registered neurons are kept.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous and trials are artificial 1-minute segments starting from the beginning of each session.

ii. N/A (no alignment code; trials are simply sequential segments)

iii. There is no stimulus-triggered event to align to. All data streams are natively synchronized at 30 Hz.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native 30 Hz frame rate (~33.33 ms time bins). No resampling or rebinning is applied.

ii.
```python
FPS = 30  # recording frame rate
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz frame rate. No temporal rebinning is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` variable (environment names like 'square', 'o', 't', etc.) rather than from the `blocked` variable (blocked position indices).

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
```

iii. The AI used the environment names to look up pre-defined 3x3 binary matrices from a `get_env_mat()` function, which it states was "directly from reference code (utils.py)".

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a 3x3 binary matrix (1 = open, 0 = blocked) via a lookup table, then flattened to a 9-element vector. This is static per trial/session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        # ... etc
    }
    return np.array(env_mats[env], dtype=float)

env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()
input_trials.append(env_input.astype(np.float32))
```

iii. The AI chose to represent the environment geometry as open=1, blocked=0. However, there is a spatial orientation mismatch: the `get_env_mat()` definitions have flipped row ordering compared to the blocked indices in the data. For vertically symmetric environments (square, o, u, rectangle, +, i), this produces correct results, but for asymmetric environments (t, l, bit donut, glenn), the geometry is vertically flipped relative to the actual position bins.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D coordinates (x, y) of the animal in the arena, with shape `(n_days, 2, n_timepoints)`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The position variable records the animal's location in a 75x75 cm arena at each timepoint.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal bins (25 cm each). Positions are clipped to valid arena bounds and binned using floor division.

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS, arena_size=ARENA_SIZE):
    x = np.clip(position[0], 0, arena_size - 1e-10)
    y = np.clip(position[1], 0, arena_size - 1e-10)
    bin_size = arena_size / n_bins
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx
```

iii. The AI uses `x_bin * 3 + y_bin` for bin indexing ("row-major" with row = x), while the reference uses `y_bin * 3 + x_bin`. This results in a different spatial ordering of the 9 position bins (effectively transposed).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by dividing the 75 cm arena into 3 equal 25 cm bins per axis using `np.floor(coord / bin_size)`, yielding 9 categories (0-8).

ii.
```python
bin_size = arena_size / n_bins  # 25.0
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
```

iii. The AI uses `np.floor` with bin_size for discretization, while the reference uses `np.digitize` with `np.linspace` edges. Both approaches produce the same bin assignments for the same inputs, as both create equal-width bins spanning [0, 75].

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are sampled at the same 30 Hz rate and split into trials using the same frame indices, ensuring frame-for-frame alignment.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. Both time series have the same number of timepoints per session and are sliced identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (1) Neurons with all-NaN traces are removed (not registered in that session). (2) Any remaining NaN values in registered neuron traces are replaced with 0 via `np.nan_to_num`. (3) Remainder frames that don't fill a complete 60-second trial are discarded. (4) Sessions with fewer than 5 neurons or fewer than 2 trials are skipped (never triggered).

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)
```

iii. The NaN-to-0 replacement is a safety measure. The reference code does not do this, though in practice it may not affect results if no partial-NaN neurons exist.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib files containing the large neural trace arrays (I/O bound), as each file contains arrays of shape `(n_days, n_cells, n_timepoints)`.

ii. N/A

iii. Processing (NaN filtering, discretization, trial splitting) is fast by comparison.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials to create individual arrays, which could be done with `np.array_split` or array reshaping. However, this is a minor optimization.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
```

iii. The loop is straightforward and not a performance bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The `env_input.astype(np.float32)` cast is repeated for every trial within a session, even though the environment geometry is constant per session.

ii.
```python
input_trials.append(env_input.astype(np.float32))  # repeated for each trial
```

iii. This is a minor inefficiency; the cast could be done once per session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does not perform any significant unnecessary processing. All computed data (neural, position bins, environment geometry) is used in the final output.

ii. N/A

iii. The conversion is straightforward with minimal overhead.
