# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from `joblib` files (not `.mat` files) in the `data/` directory. Each file is loaded with `joblib.load()`, and the data is accessed via a nested dictionary keyed by the animal name. The structure contains `trace`, `position`, `envs`, and other fields already organized as numpy arrays with dimensions `(n_sessions, n_neurons, n_timepoints)` for trace and `(n_sessions, 2, n_timepoints)` for position.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)
```

iii. The AI identified from the reference code's `load_dat()` function that `joblib.load()` was the standard loading method. The reference code also supports `.mat` files via `h5py`, but the AI chose the joblib format since both formats are available in the data directory.

## 1-b. How are the data split into subjects?

i. Subjects are identified by a hardcoded list of 7 animal names (`ANIMALS`). Each animal corresponds to a separate joblib file in the data directory.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for a_idx, animal in enumerate(animals_to_process):
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
```

iii. The AI identified 7 animals from the data directory listing and hardcoded them. The reference code instead discovers `.mat` files via `glob`.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal's data is a separate session. The first dimension of the `trace` array indexes sessions. Each session becomes a separate entry in the output lists.

ii.
```python
n_sessions = trace.shape[0]
for day in range(n_sessions):
    tr = trace[day]  # (n_neurons, n_timepoints)
    pos = position[day]  # (2, n_timepoints)
    env_name = envs[day, 0]
```

iii. Each animal has 21-31 recording sessions (days), resulting in 207 total sessions across all animals. This matches the reference paper.

## 1-d. How are the data split into trials?

i. Continuous recording sessions are split into 1-minute (1800 frame) non-overlapping trials. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. Per the task instructions, continuous recordings are split into 1-minute trials. This yields ~39-40 trials per session from ~40-minute recordings.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are included. The only data excluded are remainder frames at the end of sessions that don't fill a complete trial.

ii. N/A (no filtering code)

iii. The AI noted that the reference code does not have explicit trial structure (continuous 40-minute sessions), and since trials are artificially created, no trial-level filtering is needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary rising-phase calcium transient data (0/1 values) with shape `(n_sessions, n_neurons, n_timepoints)`.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
tr = trace[day]  # (n_neurons, n_timepoints)
```

iii. The AI identified from both the reference code and the paper that `trace` contains the binarized rising-phase vector of calcium transients.

## 2-b. How is the `neural` data processed?

i. Only active (non-all-NaN) neurons are kept. Any remaining NaN values in active neurons are replaced with 0. The data is cast to float32. No additional temporal processing (smoothing, rebinning) is applied.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]  # (n_active, n_timepoints)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The AI kept the raw binary trace without additional processing, noting that the traces are already preprocessed by the original authors. The `nan_to_num` call was added as a safety measure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with all-NaN values (not recorded in that session) are removed. No additional quality filtering (e.g., activity thresholds, velocity-based filtering) is applied at the conversion stage.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
```

iii. The AI decided not to apply the reference code's velocity filtering (v_thresh=5) or cell activity filtering (cell_threshold=5) at conversion time, reasoning that these are decoder-specific preprocessing steps that the downstream decoder can handle differently.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The continuous recording is split into sequential 1-minute segments starting from the beginning of the recording.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
```

iii. There is no stimulus onset or behavioral event to align to in this free-exploration task. Alignment is to the start of each recording session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI noted that the reference code applies temporal binning (3 frames via AvgPool1d) during decoding, but chose to keep the native resolution at the conversion stage.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable, which contains environment name strings for each session (e.g., 'square', 'o', 't', 'u', etc.). The AI uses the `get_env_mat()` function from the reference code to convert the name to a 3x3 binary matrix.

ii.
```python
envs = d['envs']         # (n_sessions, 1)
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI copied `get_env_mat()` directly from the reference code's `utils.py`. This function maps environment names to 3x3 binary matrices indicating which parts of the arena are accessible.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via `get_env_mat()`, then flattened to a 9-element vector. This is static per trial (same for all timepoints within a session).

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
trial_input.append(env_mat.astype(np.float32))
```

iii. The AI used the reference code's `get_env_mat()` function, which represents environment geometry as a binary accessibility matrix (1 = accessible, 0 = blocked).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D coordinates (x, y) of the animal in the arena with shape `(n_sessions, 2, n_timepoints)`.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
pos = position[day]  # (2, n_timepoints)
```

iii. The `position` variable records the animal's location in the 75x75 cm arena, tracked with DeepLabCut.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Continuous x,y position is discretized into a 3x3 spatial grid (9 classes). Each axis is divided into 3 equal bins of 25 cm. The bin index is computed as `x_bin * 3 + y_bin`.

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. The AI discretized position into 9 bins as required by the task. The bin size is 25 cm (75 cm / 3).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using `np.floor(pos / bin_size)` with clipping to [0, 2] per axis. The combined bin index uses `x_bin * n_bins + y_bin` ordering.

ii.
```python
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. The AI uses floor-based binning with clip for boundary handling. Note that the bin order `x * 3 + y` differs from the reference which uses `y * 3 + x`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are from the same continuous recording at the same frame rate (30 Hz), so they are aligned frame-by-frame. Both are split into trials using the same indices.

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. Both arrays share the same timepoint dimension and are sliced identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with all-NaN values (not recorded in a session) are removed. Any remaining NaN values in active neurons are replaced with 0 via `np.nan_to_num`. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. The AI added `nan_to_num` as a safety measure for any sparse NaN values in otherwise active neurons.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files is the most time-consuming step (12-23 seconds per animal). Processing sessions is relatively fast (~8-15 seconds per animal). Total estimated time is ~4 minutes for all 7 animals.

ii. N/A (timing is via `time.time()` calls)

iii. The AI included timing instrumentation and estimated ~35s per animal, ~4 minutes total.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over each trial sequentially, appending to lists. This could potentially be vectorized using `np.split` or array reshaping.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI's code uses explicit loops for trial splitting rather than vectorized array operations.

## 6-c. What processing does the code repeat multiple times?

i. The `env_mat.astype(np.float32)` conversion is repeated for each trial within a session, though the environment geometry is the same for all trials in a session.

ii.
```python
for t in range(n_trials):
    trial_input.append(env_mat.astype(np.float32))
```

iii. This is a minor inefficiency; the float32 cast could be done once outside the loop.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores detailed `session_info` metadata (animal name, env_name, n_active, n_trials) for each session, which is not used by the decoder. Additionally, extensive summary statistics are printed that are not strictly necessary.

ii.
```python
session_info.append({
    'animal': sess['animal'],
    'env_name': sess['env_name'],
    'n_active': sess['n_active'],
    'n_trials': len(sess['neural']),
})
```

iii. This metadata is informational and does not affect decoder performance.
