# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from extensionless joblib files in the `data/` directory (one per animal). Each file is loaded with `joblib.load()`, which returns a dictionary keyed by animal ID containing arrays for `trace`, `position`, `envs`, etc. The list of animals is hardcoded as `ANIMALS`.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    envs = rec['envs'].ravel().tolist()
    traces = rec['trace']
    positions = rec['position']
```

iii. The AI identified that the data directory contains both `.mat` files and extensionless joblib-converted files. Since the reference code uses `joblib.load()` (via `load_dat` and `mat2joblib` utilities), the AI chose to use the joblib format for consistency with the reference code.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The subject name is the animal ID from the hardcoded `ANIMALS` list, matching the filename.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
```

iii. Each joblib file contains all recording sessions for one animal. The filename/animal ID serves as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording sessions, indexed by iterating over the `envs` array. Each session has corresponding entries in `trace` and `position`.

ii.
```python
envs = rec['envs'].ravel().tolist()
traces = rec['trace']
positions = rec['position']
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The `envs` array has one entry per session, providing both the session count and the environment geometry name for each session.

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 60-second trials (1800 frames at 30 Hz). Only the valid portion of the recording is used (determined by `infer_valid_frames`). Remainder frames that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS  # 30 * 60 = 1800
...
def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    ...
    for t in range(n_trials):
        sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
        neural_trials.append(trace_sxn[:, sl])
```

iii. Per instruction, sessions are split into 1-minute trials. The `infer_valid_frames` function determines the last frame with valid data before splitting.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 valid trials are excluded entirely. Additionally, sessions where all neurons have non-finite values after filtering are excluded.

ii.
```python
def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    if n_trials < min_trials:
        return [], [], []
    ...
    if trace_sxn.shape[0] == 0:
        return [], [], []
...
if len(nt) < 2:
    continue
```

iii. The instructions require at least two trials per session for decoder evaluation. The `min_trials=2` filter enforces this.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib-loaded data, which contains calcium imaging traces (binarized rising-phase vectors) with shape `(n_neurons, n_timepoints)` per session.

ii.
```python
traces = rec['trace']
...
trace_sxn = traces[s]  # shape: (n_neurons, n_timepoints)
```

iii. The `trace` variable contains pre-processed binary rising-phase vectors, as confirmed by the paper's methods (z-scored derivative thresholded at 2.5) and the data inspection showing binary {0, 1} values with ~1.5% ones.

## 2-b. How is the `neural` data processed?

i. The only processing is: (1) truncate to valid frames, (2) cast to float32, and (3) filter out neurons with non-finite values. No additional preprocessing (dF/F, rebinning, smoothing) is applied because the traces are already processed.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
```

iii. The stored traces are already binarized rising-phase vectors (confirmed by data inspection and paper methods), so no further preprocessing is needed beyond filtering and type conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by requiring all values within the valid frame range to be finite. Any neuron with any NaN or Inf value (within the valid range) is excluded.

ii.
```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
```

iii. This strict filtering ensures only neurons with complete, valid data within the analyzed time range are included. Neurons not recorded in a given session (all-NaN) are naturally excluded, as well as neurons with partial data gaps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are consecutive 1-minute segments starting from the session beginning. The alignment event is session start time.

ii.
```python
'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. There is no stimulus onset or specific event to align to; trials are artificial temporal segments of the continuous session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. Both neural and behavioral data are acquired at 30 Hz, so the native resolution is preserved.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable in the data, which contains the environment name (e.g., 'square', 'o', 't', 'u', etc.) for each session.

ii.
```python
envs = rec['envs'].ravel().tolist()
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
```

iii. The `envs` variable identifies which of the 10 environment geometries was used in each session. The `get_env_mat` function from the reference code maps each name to a 3x3 binary matrix representing which spatial bins are accessible.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a 3x3 binary matrix via `get_env_mat()`, where 1 indicates an accessible spatial bin and 0 indicates a blocked bin. The matrix is flattened to a 9-dimensional vector. This is static per trial.

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    ...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
input_trials.append(env_vec.copy())
```

iii. This approach directly follows the reference code's `get_env_mat` function, which encodes the environment geometry as a binary accessibility matrix over the conceptual 3x3 grid.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable in the data, which contains 2D (x, y) coordinates of the mouse in the arena at each timepoint.

ii.
```python
positions = rec['position']
pos_sxn = positions[s]  # shape: (2, n_timepoints)
```

iii. The `position` variable records the animal's tracked location in the 75x75 cm arena at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes) using `np.floor((coord / arena_size) * 3)`. Each axis is divided into 3 equal bins. The grid label is computed as `y_bin * 3 + x_bin`.

ii.
```python
def discretize_position_3x3(position_xy, arena_size=75.0):
    x = np.clip(position_xy[0], 0, arena_size - 1e-6)
    y = np.clip(position_xy[1], 0, arena_size - 1e-6)
    xb = np.floor((x / arena_size) * 3).astype(int)
    yb = np.floor((y / arena_size) * 3).astype(int)
    xb = np.clip(xb, 0, 2)
    yb = np.clip(yb, 0, 2)
    return (yb * 3 + xb).astype(np.int64)
```

iii. The 3x3 discretization matches the decoder task specification and the paper's conceptual partition of the arena.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position coordinates are clipped to [0, arena_size - 1e-6], then divided into 3 equal bins per axis using floor division. The resulting 2D bin indices are combined into a single label 0-8 via `y_bin * 3 + x_bin`.

ii.
```python
x = np.clip(position_xy[0], 0, arena_size - 1e-6)
y = np.clip(position_xy[1], 0, arena_size - 1e-6)
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
```

iii. Clipping ensures positions at the boundaries are assigned to valid bins. The floor-based binning creates equal-width bins of 25 cm each.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are stored at the same 30 Hz frame rate in the data files, so they are aligned frame-for-frame. Both are truncated to the same valid frames and split using the same trial slicing.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep]
pos_sxn = pos_sxn[:, :n_keep]
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    output_trials.append(pos_bins[sl][None, :])
```

iii. Both arrays have the same number of timepoints and are sliced identically, ensuring temporal alignment. The `infer_valid_frames` function ensures only the jointly valid portion is used.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The `infer_valid_frames` function detects the last frame where both position and neural data are finite, truncating the session to this range. Neurons with any non-finite values within the valid range are removed. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
    valid = pos_valid & neural_valid
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1] + 1)
```

iii. This approach robustly handles data quality issues: trailing invalid data (e.g., from session ending) is excluded, neurons with incomplete recordings are filtered, and sessions too short for meaningful analysis are dropped.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib files via `joblib.load()`, which involves deserializing large numpy arrays. Processing (filtering, discretization, trial splitting) is fast by comparison. The full conversion runs in ~291 seconds.

ii. N/A

iii. The joblib files contain large trace arrays (up to ~950 neurons x ~72000 timepoints per animal). I/O and deserialization dominate the runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial slicing loop within `session_to_trials` creates trial slices in a Python loop. This could potentially be vectorized using `np.split` or array reshaping, though the loop is not a significant bottleneck.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. Since the number of trials per session is small (~39-40), this loop is not a performance bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The `env_vec.copy()` call repeats the same geometry vector for each trial within a session, creating redundant copies. The `infer_valid_frames` function is called once per session, which is appropriate.

ii.
```python
input_trials.append(env_vec.copy())  # repeated for each trial
```

iii. Since the environment geometry is static per session, a single computation followed by copies is the expected approach.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No significant unnecessary processing is identified. The code is lean and focused on the conversion task. The `--show-processing` flag is accepted but not implemented (prints a message instead of generating plots).

ii.
```python
if args.show_processing:
    print('show-processing requested; plotting not yet implemented')
```

iii. The lack of processing plot implementation is a minor gap but does not affect the data conversion quality.
