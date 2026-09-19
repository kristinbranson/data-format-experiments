# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a joblib-serialized file (extensionless, e.g., `QLAK-CA1-08`) in the data directory. The AI loads these using `joblib.load()`, which returns a dict keyed by animal name containing `trace`, `position`, `envs`, and `blocked` arrays. There are 7 such files, one per animal.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]

n_days = d['trace'].shape[0]
n_cells_total = d['trace'].shape[1]
n_timepoints = d['trace'].shape[2]
```

iii. The AI explored the data directory and found both `.mat` files and extensionless files. It chose to use `joblib.load` on the extensionless files after discovering they contained the same data in a Python-friendly format. The AI verified the loaded data structure by printing shapes and sample values before writing the conversion script.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to one file in the data directory. The AI iterates over a hardcoded list of 7 animal names (`ANIMALS`), loading one file per animal.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. The AI discovered 7 animal files and hardcoded the list after verifying all names.

## 1-c. How are the data split into sessions?

i. Each animal file contains multi-day recordings. The `trace` array has shape `(n_days, n_cells, n_timepoints)`. Each day becomes a separate session.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. Each day has a distinct environment geometry and corresponds to one recording session. The AI recognized this from the data structure and the paper's description of one session per day.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute (1800 frame at 30 Hz) non-overlapping segments. Remainder frames that don't fill a full trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
...
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
```

iii. The instructions specified "1-minute trials within each session." The AI implemented this as 1800-frame non-overlapping segments.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two quality filters at the session level: (1) sessions with fewer than 5 registered cells are skipped, and (2) sessions with fewer than 2 possible trials are skipped.

ii.
```python
if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue

n_trials = n_timepoints // FRAMES_PER_TRIAL
if n_trials < 2:
    print(f"  Day {day}: skipping, only {n_trials} possible trials")
    continue
```

iii. The AI added the <5 cells filter as a quality control measure and the <2 trials filter to satisfy the instruction requirement of "at least two trials within each session." In practice, neither filter removes any sessions because all sessions have hundreds of cells and ~40 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium transient events with shape `(n_days, n_cells, n_timepoints)`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The methods section describes that the traces are binarized rising-phase vectors (1 when z-scored derivative > 2.5, 0 otherwise). The AI confirmed this by checking sample values.

## 2-b. How is the `neural` data processed?

i. The AI (1) selects only registered cells (non-all-NaN along time axis), (2) replaces any remaining NaN values with 0, and (3) casts to float32.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]  # (n_registered, n_timepoints)
traces = np.nan_to_num(traces, nan=0.0)
...
neural_trials.append(trial_traces.astype(np.float32))
```

iii. The AI identified that NaN columns correspond to unrecorded neurons and filtered them out. It also defensively replaced any remaining NaN values with 0 as a safety measure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN across the entire session (not recorded that day) are removed. Only neurons with at least one non-NaN timepoint are kept. Additionally, any individual NaN values in otherwise-valid neurons are replaced with 0.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)
```

iii. The trace array contains NaN for neurons not recorded on a given day (cells tracked across sessions may not all be present every day). The AI filters these out and defensively handles any remaining NaN.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are simply consecutive 1-minute segments starting from the beginning of the recording.

ii. N/A (alignment is implicit via frame indexing)

iii. There is no stimulus onset or behavioral event to align to. The recording runs continuously while the mouse freely explores.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30  # recording frame rate
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already recorded at a consistent 30 Hz rate. No resampling is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable in the data, which contains the environment name (e.g., 'square', 'o', 't') for each recording day. This is then mapped to a 3x3 binary geometry matrix using the `get_env_mat()` function from the reference code's utils.py.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
```

iii. The AI examined the reference code and found the `get_env_mat` function which maps environment names to 3x3 binary matrices. It chose to use this approach rather than the `blocked` indices, as it directly captures the full environment geometry as defined in the paper's code.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a 3x3 binary matrix (1=open, 0=blocked) using `get_env_mat()`, then flattened to a 9-element vector. This vector is static per trial (same for all trials within a session).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(env_mats[env], dtype=float)
...
env_input = env_mat.flatten()
input_trials.append(env_input.astype(np.float32))
```

iii. The AI adopted the environment matrix representation directly from the paper's reference code (utils.py). This encoding uses 1 for open partitions and 0 for blocked partitions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena at each timepoint, with shape `(2, n_timepoints)`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. Position data was generated from DeepLabCut head tracking, as described in the methods section.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 bins). Each axis is divided into 3 equal bins of 25 cm each (75 cm / 3). The bin index is computed as `x_bin * 3 + y_bin` (row-major with x as the row index).

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

iii. The instructions specified "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI implemented floor-based binning with clipping to handle edge cases.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position coordinates are clipped to `[0, arena_size - 1e-10]`, then divided by bin_size (25 cm) and floored to get integer bin indices 0-2 for each axis. The final category is `x_bin * 3 + y_bin`, yielding values 0-8.

ii.
```python
x = np.clip(position[0], 0, arena_size - 1e-10)
y = np.clip(position[1], 0, arena_size - 1e-10)
bin_size = arena_size / n_bins
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
bin_idx = x_bin * n_bins + y_bin
```

iii. Clipping slightly below arena_size prevents edge values from producing out-of-range bin indices.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and share the same time axis. Both are sliced into trials using identical frame indices, ensuring alignment.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. Both data streams are sampled at the same rate and indexed identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of missing data handling: (1) Neurons with all-NaN traces (not recorded that day) are removed. (2) Any remaining individual NaN values in valid neurons are replaced with 0. (3) Sessions with fewer than 5 registered cells or fewer than 2 possible trials are skipped entirely. (4) Remainder frames not filling a complete 1-minute trial are discarded.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)

if n_registered < 5:
    continue
if n_trials < 2:
    continue
```

iii. The AI added defensive NaN handling and quality control filters to ensure data integrity.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the data files with `joblib.load()`, which is I/O bound. Each animal file contains large 3D arrays (days x cells x timepoints).

ii. N/A

iii. Processing operations (NaN filtering, discretization, trial splitting) are vectorized NumPy operations and are fast by comparison to file I/O.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial loop that slices neural and position data into 1-minute segments uses a Python for-loop. This could potentially be vectorized using `np.reshape` or `np.split`.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. While the loop is straightforward and not a performance bottleneck (only ~40 iterations per session), it could be replaced with array reshaping for slightly cleaner code.

## 6-c. What processing does the code repeat multiple times?

i. The `get_env_mat` function is called once per session, and `discretize_position` is called once per session. No significant repeated processing is evident. The environment matrix lookup involves dictionary access each time rather than caching, but this is negligible.

ii. N/A

iii. The code is relatively efficient with no major redundant computations.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The sanity checks at the end (`sanity_checks` function) perform validation that isn't needed for the final output but helps verify correctness during development. The code also computes and prints summary statistics that aren't part of the saved output.

ii.
```python
def sanity_checks(data):
    """Run sanity checks on converted data."""
    ...
```

iii. The sanity checks are a development aid rather than part of the data conversion pipeline itself.
