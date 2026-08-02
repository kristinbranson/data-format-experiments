# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` on files without `.mat` extension (the pre-converted joblib format). Each animal's data is stored in a dictionary keyed by animal ID, containing `trace`, `position`, `envs`, and `blocked` arrays. This matches the reference code's `load_dat()` function with `format="joblib"`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]

n_days = d['trace'].shape[0]
n_cells_total = d['trace'].shape[1]
n_timepoints = d['trace'].shape[2]
```

iii. The AI recognized that the data directory contains both `.mat` and joblib-format files, and chose joblib loading which matches the reference code's own `load_dat()` utility function. The trajectory shows the AI read the reference code's `utils.py` to determine the loading approach.

## 1-b. How are the data split into subjects?

i. Each animal ID from a hardcoded list (`ANIMALS`) corresponds to one subject. The animal ID is used as the subject name and each animal's data is loaded from a separate file.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
subjects = list(animals)
```

iii. The AI hardcoded the 7 animal IDs matching the paper's reported 7 mice, rather than dynamically discovering files in the data directory.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal is a separate session. The AI iterates over `n_days = d['trace'].shape[0]` for each animal, creating one session per day.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. The AI correctly identified that the first dimension of the trace array corresponds to recording days/sessions. The total of 207 sessions matches the paper.

## 1-d. How are the data split into trials?

i. Each session is split into 1-minute non-overlapping segments (1800 frames at 30 Hz). Remainder frames that don't fill a complete trial are discarded.

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
    trial_pos = pos_bins[start:end]
```

iii. The 1-minute trial duration is specified in the instructions. The AI implemented this as integer division of total timepoints by frames per trial, discarding the remainder.

## 1-e. How are trials filtered based on quality controls?

i. The AI adds two quality filters not present in the reference: (1) sessions with fewer than 5 registered neurons are skipped, and (2) sessions with fewer than 2 possible trials are skipped.

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

iii. The AI added these filters as safety measures. The minimum 2 trials filter is justified by the instructions stating "There needs to be at least two trials within each session." The minimum 5 neurons filter is not in the instructions or reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium transient events with shape `(n_days, n_cells, n_timepoints)`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The AI correctly identified `trace` as the neural data source and noted it contains binary calcium transient events (0s and 1s).

## 2-b. How is the `neural` data processed?

i. Only neurons registered in the current session (non-NaN traces) are kept. Remaining NaN values are replaced with 0. Data is cast to float32.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]  # (n_registered, n_timepoints)
traces = np.nan_to_num(traces, nan=0.0)
...
neural_trials.append(trial_traces.astype(np.float32))
```

iii. The AI noted that the trace data is already in binary transient event form and no further processing (e.g., deconvolution) is needed. The NaN-to-0 replacement is a safety measure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN across the entire session (not registered on that day) are removed. Additionally, sessions with fewer than 5 registered neurons are skipped entirely (see 1-e).

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()
if n_registered < 5:
    continue
traces = trace_day[registered]
```

iii. The AI followed the paper's note that "not all cells are detected/registered on every day" and filtered accordingly. The minimum 5 neuron threshold is an additional filter not in the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are artificial 1-minute segments from session start. Neural and behavioral data are natively aligned at 30 Hz.

ii. N/A (alignment is implicit through shared indexing)

iii. The AI noted that the DAQ simultaneously acquired behavioral and cellular imaging at 30 Hz, so no explicit alignment is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No rebinning is applied.

ii.
```python
FPS = 30
# ...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The original recording is at 30 Hz and the AI preserved this native resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the input from the `envs` variable in the raw data, which contains environment name strings (e.g., "square", "o", "t", "u", etc.). These are mapped to 3x3 binary matrices using the `get_env_mat()` function from the reference code.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
```

iii. The AI chose to use the environment geometry representation from the reference code's `get_env_mat()` function, where 1=open partition and 0=blocked partition.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via `get_env_mat()`, then flattened to a 9-element vector. The matrix uses 1=open, 0=blocked convention.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        # ... etc
    }
    if env in env_mats:
        return np.array(env_mats[env], dtype=float)
```

iii. The AI directly adopted the `get_env_mat()` function from the reference code's `utils.py`, preserving the exact environment geometry encodings used by the paper authors.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Environment geometry is a per-session constant (static per trial). The same 9-element vector is replicated for all trials within a session.

ii.
```python
input_trials.append(env_input.astype(np.float32))  # static per trial, shape (9,)
```

iii. Since the environment geometry doesn't change within a recording session, no temporal alignment is needed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D coordinates (x, y) of the mouse in the arena with shape `(n_days, 2, n_timepoints)`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The position data records the animal's location in the 75x75 cm arena at each timepoint.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal bins of 25 cm. The bin index is computed as `x_bin * 3 + y_bin` (row-major with x as row).

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

iii. The AI uses `np.floor(x / bin_size)` for binning and `x_bin * 3 + y_bin` for combining, which makes x the "row" dimension. This matches the reference code's rate map convention where `rate_maps[:, x, y]`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (0-8) using a 3x3 grid. Each axis is divided into 3 equal 25cm bins using `np.floor(coord / 25)`, clipped to [0, 2].

ii.
```python
bin_size = arena_size / n_bins  # 25.0 cm
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, n_bins - 1)
y_bin = np.clip(y_bin, 0, n_bins - 1)
bin_idx = x_bin * n_bins + y_bin
```

iii. The bin edges are implicitly at [0, 25, 50, 75] cm, yielding 3 bins per axis. Values at exactly 75 cm are clipped to the last bin.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame rate (30 Hz) and are split into trials using the same indices.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. Both arrays have the same number of timepoints per session and are sliced identically, ensuring frame-for-frame alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e. Native 30 Hz (~33.33 ms bins), no rebinning applied.

ii.
```python
FPS = 30
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is kept at native resolution.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural (trace) and output (position) are recorded simultaneously at 30 Hz and share the same timepoint indices. Input (environment geometry) is static per trial/session. All time-varying signals are split into trials using the same frame indices.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
pos_day = d['position'][day]  # (2, n_timepoints)
# Both have same n_timepoints, split with same indices
```

iii. The simultaneous 30 Hz DAQ ensures alignment without interpolation.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Neurons with all-NaN traces (not registered on that day) are filtered out. Any remaining NaN values are replaced with 0. Sessions with fewer than 5 neurons or fewer than 2 trials are skipped. Remainder frames not filling a complete trial are discarded.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = np.nan_to_num(traces, nan=0.0)
if n_registered < 5:
    continue
if n_trials < 2:
    continue
```

iii. The NaN-to-0 replacement and minimum neuron threshold are safety measures not present in the reference code.

## 7-a. What are the most time-consuming steps of the code?

i. Loading the data files via `joblib.load()` is the most time-consuming step, as it involves decompressing large arrays from disk. The per-trial splitting and discretization are fast vectorized operations.

ii. N/A

iii. The joblib files contain large neural trace arrays (n_days x n_cells x n_timepoints) that must be decompressed.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial loop that slices traces and position for each trial could potentially be vectorized using `np.split` or array reshaping, but the current explicit loop is straightforward and not a bottleneck.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. The loop is simple slicing and the overhead is negligible compared to I/O.

## 7-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each animal is loaded once, each session processed once.

ii. N/A

iii. The code has a clean single-pass structure.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes all 207 sessions, but in sample mode only the first 2 animals are used. The `sanity_checks()` function re-iterates over all sessions to verify consistency.

ii.
```python
def sanity_checks(data):
    for s in range(n_sessions):
        n_trials = len(data['neural'][s])
        for t in range(n_trials):
            # assertions...
```

iii. The sanity checks are diagnostic and add processing time but are valuable for verification.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. All-NaN neurons are filtered out, remaining NaN replaced with 0, sessions with too few neurons or trials are skipped, and remainder frames are discarded.

ii. (See question 6 code snippets)

iii. The AI's approach is conservative, preferring to skip low-quality data rather than include potentially problematic sessions.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Data loading (joblib decompression) dominates runtime.

ii. N/A

iii. N/A

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The trial-splitting loop could theoretically be vectorized.

ii. (See 7-b)

iii. N/A

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. No significant repeated processing.

ii. N/A

iii. N/A

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Sanity checks add overhead but are useful for validation.

ii. N/A

iii. N/A
