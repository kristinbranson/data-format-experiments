# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-serialized files (extensionless files in the `data/` directory) using `joblib.load()`. Each file is a dictionary keyed by animal name, containing `trace`, `position`, `envs`, and other arrays. The data is loaded one animal at a time. This contrasts with the reference solution which loads `.mat` files using `h5py`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]

trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()  # (n_days,) string array
```

iii. The AI identified that the data directory contains both `.mat` files and extensionless joblib files with the same data. It chose joblib because the data was already in a convenient numpy-array format with shape `(n_days, n_cells, n_frames)`, avoiding the complexity of HDF5 reference dereferencing required for the `.mat` files.

## 1-b. How are the data split into subjects?

i. The AI hardcodes a list of 7 animal names and iterates over them. Each animal name corresponds to one joblib file in the data directory.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
subjects = animals  # All 7 animals are subjects
```

iii. The AI found 7 animals in the data directory and hardcoded them. The reference solution discovers subjects dynamically via `glob.glob('*.mat')`.

## 1-c. How are the data split into sessions?

i. Within each animal's data, sessions correspond to recording days. The trace array has shape `(n_days, n_cells, n_frames)`, so iterating over the first axis gives one session per day.

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape

for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
```

iii. Each animal has 21-31 recording days. Each day is a separate session with its own environment geometry.

## 1-d. How are the data split into trials?

i. Each session (~40 min at 30 Hz) is split into 1-minute non-overlapping trials of 1800 frames. Remainder frames are discarded.

ii.
```python
n_trials = n_frames_total // trial_duration_frames

for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The instructions specify "1-minute trials within each session." At 30 Hz, 1 minute = 1800 frames. This yields ~39-40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are kept.

ii. N/A (no filtering code)

iii. The original paper does not filter trials. The AI follows this approach. The reference solution also does not filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binarized calcium event traces (0/1 values indicating significant rising-phase transients).

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]   # (n_cells, n_frames)
```

iii. The AI correctly identified that `trace` contains pre-processed binary calcium events at 30 Hz, not raw fluorescence.

## 2-b. How is the `neural` data processed?

i. Active (non-NaN) cells are selected, remaining NaN values are replaced with 0, and data is cast to float32. No additional processing (smoothing, rate computation, etc.) is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI noted that the trace data is already binarized by the original authors, so no delta F/F or deconvolution is needed. The `np.nan_to_num` call is a safety measure for any residual NaN values in otherwise-active cells.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with all-NaN traces on a given day (unregistered cells) are excluded. No further filtering (e.g., by firing rate, place cell criteria, or event count) is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
n_active = active_mask.sum()
```

iii. Cross-day cell registration via CellReg means some cells are not tracked on all days. NaN rows indicate unregistered cells. The AI includes all registered cells, consistent with the paper's statement that "motivated the inclusion of all cells in subsequent analyses."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are sequential 1-minute segments from the start of the session. The temporal alignment event is the start of each trial segment.

ii. N/A (no alignment code beyond trial slicing)

iii. There is no discrete stimulus onset or behavioral event to align to in this free-exploration task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz frame rate is preserved (time bin size ~33.33 ms). No temporal rebinning is applied.

ii.
```python
fps = 30  # recording frame rate
...
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz across all streams (neural, position). No rebinning is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable (environment name strings like 'square', 'o', 't', etc.) rather than the `blocked` variable. The environment name is passed to `get_env_mat()` (adapted from the reference code) to produce a 3x3 binary accessibility matrix.

ii.
```python
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI chose to use the reference code's `get_env_mat()` function, which maps environment names to 3x3 binary matrices where 1=accessible and 0=blocked. This contrasts with the reference solution which uses the `blocked` variable to create a one-hot encoding where 1=blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is looked up in a dictionary mapping to 3x3 binary matrices (1=accessible, 0=blocked). The matrix is flattened to a 9-element vector and used as a static per-trial input.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
trial_input = env_mat.astype(np.float32)
```

iii. The AI reused `get_env_mat()` directly from the reference code. The encoding is the inverse of the reference solution's one-hot blocked encoding: where the reference has 1=blocked, the AI has 1=accessible. Both convey equivalent information about the environment geometry.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position output is derived from the `position` variable, containing 2D (x, y) coordinates of the mouse in the 75x75 cm arena at each frame.

ii.
```python
pos_day = position[day]  # (2, n_frames)
```

iii. Position was tracked using DeepLabCut at 30 Hz, stored as (2, n_frames) per day.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous (x, y) position is discretized into a 3x3 spatial grid (9 bins of 25 cm each). The bin ID is computed as `x_bin * 3 + y_bin`.

ii.
```python
def discretize_position_3x3(position, env_size=75.0):
    bin_size = env_size / 3.0
    x = np.clip(position[0], 0, env_size - 1e-10)
    y = np.clip(position[1], 0, env_size - 1e-10)
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    x_bin = np.clip(x_bin, 0, 2)
    y_bin = np.clip(y_bin, 0, 2)
    bin_ids = x_bin * 3 + y_bin
    return bin_ids
```

iii. The 3x3 grid matches the task requirement of "3 x 3 = 9 spatial bins." The bin ordering differs from the reference (`x_bin * 3 + y_bin` vs `y_bin * 3 + x_bin`) but this is a labeling convention that doesn't affect decoding.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized by dividing each axis into 3 equal bins of 25 cm (edges at 0, 25, 50, 75 cm). Positions at the boundaries are clipped to valid bins using `np.clip`. The AI uses `np.floor(x / bin_size)` for binning.

ii.
```python
bin_size = env_size / 3.0  # 25 cm
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, 2)
y_bin = np.clip(y_bin, 0, 2)
```

iii. The reference solution uses `np.digitize` with `np.linspace` edges, while the AI uses `np.floor` division. Both produce the same bin assignments for positions in [0, 75].

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and stored with matching indices. Both are sliced using the same trial boundaries, ensuring frame-by-frame alignment.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. Since all data streams share the same time base (30 Hz frame rate), simple index-based slicing maintains alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of missing data are handled: (1) Unregistered neurons (all-NaN traces) are excluded per session. (2) Any residual NaN values in otherwise-active neurons are replaced with 0 via `np.nan_to_num`. (3) Remainder frames that don't fill a complete trial are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
```

iii. The `np.nan_to_num` call is a safety measure not present in the reference solution. The reference only filters all-NaN neurons and does not handle residual NaNs.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib data files (~9-23 seconds per animal). Processing each day within an animal takes <1 second. Total conversion for all 7 animals takes ~215 seconds (~3.6 minutes).

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal_name))
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s", flush=True)
```

iii. The conversion output shows loading times of 9-23 seconds per animal, with processing taking a small fraction of total time. The AI included timing instrumentation to verify this.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trial indices and slices arrays, which could be replaced with `np.split` or array reshaping. However, since the processing time is dominated by I/O, this would not significantly improve performance.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI's code is already reasonably efficient. Position discretization is vectorized. The main loop is over days/sessions and trials, which is inherent to the data structure.

## 6-c. What processing does the code repeat multiple times?

i. The `env_mat.astype(np.float32)` conversion is repeated for every trial within a session, even though the environment geometry is constant per session. This is a minor redundancy.

ii.
```python
for trial_idx in range(n_trials):
    trial_input = env_mat.astype(np.float32)
```

iii. This could be computed once per session and reused, but the overhead is negligible.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads the entire joblib data structure for each animal (including `SFPs`, `maps`, `centroids`, `blocked` etc.) even though only `trace`, `position`, and `envs` are used. These extra arrays consume memory during loading.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
# Only trace, position, envs are used; SFPs, maps, centroids, blocked are loaded but unused
```

iii. The joblib format loads the entire dictionary at once. The `del dat, d` at the end frees memory after processing each animal.
