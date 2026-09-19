# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from preprocessed `joblib` files in the `/app/data/` directory (one per animal, no file extension). Each file is loaded with `joblib.load()` and contains a dictionary keyed by the animal name, with arrays for `trace`, `position`, `envs`, and `blocked`.

ii.
```python
def load_animal_data(animal):
    """Load preprocessed data for one animal from joblib file."""
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]
```

iii. The AI chose to use the preprocessed joblib files rather than the raw `.mat` files. This is justified because the reference code's own `load_dat` function (in `utils.py`) uses `joblib.load` as its primary loading method. The joblib files contain the same data in a more Python-friendly format (3D numpy arrays rather than HDF5 references).

## 1-b. How are the data split into subjects?

i. Each joblib file in the data directory corresponds to one subject (mouse). The AI uses a hardcoded list of 7 animal names (`ANIMALS`) to iterate over subjects.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
# ...
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)
    subject_id = ANIMALS.index(animal)
```

iii. The AI hardcodes the animal list rather than discovering files dynamically. This is acceptable since the set of animals is fixed and known from the paper (7 mice).

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days. The AI iterates over days (axis 0 of the `trace` array) and each day becomes a separate session.

ii.
```python
n_days = d['trace'].shape[0]
# ...
for day in range(n_days):
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. Each day corresponds to one recording session in one environment. The `trace` array has shape `(n_days, n_cells, n_frames)`, so iterating over axis 0 gives each session.

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into non-overlapping segments starting at every 1800 frames (60 seconds at 30 Hz). The last segment is kept if it has at least 900 frames (30 seconds); otherwise it is dropped. This means some trials can be shorter than 60 seconds.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames
MIN_TRIAL_FRAMES = FPS * 30  # Minimum 30s for a partial trial at end
# ...
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The AI chose to retain partial end-of-session trials that are at least 30 seconds long, rather than discarding all partial trials. This preserves slightly more data. In practice, the partial trials are ~1666 frames (~55.5 seconds), as sessions are ~71866-72219 frames.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped (required by the decoder format). Partial trials shorter than 30 seconds are dropped.

ii.
```python
n_trials = len(neural_trials)
if n_trials < 2:
    print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
    continue
```

iii. The minimum of 2 trials per session is required by the instructions for decoder evaluation. In practice, all sessions have ~40 minutes of data, so all produce at least 39 trials and none are skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium event data (0/1 values representing significant calcium transients from rising-phase extraction).

ii.
```python
trace = d['trace'][day]  # shape: (n_cells, n_frames)
```

iii. The `trace` variable stores preprocessed calcium imaging data that has already been binarized (z-score > 2.5 on the derivative of the calcium signal).

## 2-b. How is the `neural` data processed?

i. The only processing is filtering out unregistered neurons (NaN values) and casting to float32. The binary traces are used as-is without additional processing such as smoothing or rate map computation.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
trace_registered = trace[registered_mask]  # (n_registered, n_frames)
# ...
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The paper states that the binary calcium events are "treated as firing rate" for downstream analyses. No delta F/F computation is needed since the data is already preprocessed. The AI noted this explicitly in CONVERSION_NOTES.md.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are not registered (recorded) in a given session are identified by checking if the first frame is NaN, and are excluded from that session's data. No further quality-based filtering (e.g., place cell selection, activity thresholds) is applied.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue
trace_registered = trace[registered_mask]
```

iii. The AI checks only the first frame for NaN rather than checking if all frames are NaN. For this dataset, neurons that are unregistered on a given day have NaN for all frames, so checking the first frame is functionally equivalent. The paper states it was "motivated the inclusion of all cells in subsequent analyses," so no place cell filtering is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are simply non-overlapping segments of the continuous recording starting from the beginning.

ii. N/A (no alignment code; trials are cut from the start of each session)

iii. There is no stimulus onset or behavioral event to align to. Each session is a continuous 40-minute exploration period, so alignment to the start of the recording is the natural choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per time bin). No temporal rebinning is applied.

ii.
```python
FPS = 30  # Recording frame rate (Hz)
# metadata:
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz frame rate across all sessions. The reference paper states data was "acquired at 30 Hz." No resampling is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable (environment name strings like 'square', 'o', 't', etc.) combined with the reference code function `get_env_mat()` which maps environment names to 3x3 binary matrices.

ii.
```python
envs = d['envs'].squeeze()
# ...
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI chose to use the environment name + `get_env_mat()` from the reference code rather than the raw `blocked` variable from the data files. This directly uses the reference code's own function for representing environment geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix using `get_env_mat()` (copied from the reference code's `utils.py`). The matrix is then flattened to a 9-element vector. Values are 1 for accessible positions and 0 for blocked positions. The input is static (same for all trials within a session).

ii.
```python
def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry. From reference code."""
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    # ... (10 environments total)

env_mat = get_env_mat(env_name).flatten()  # (9,)
input_trial = env_mat.astype(np.float32)
```

iii. Using `get_env_mat` from the reference code ensures consistency with how the original authors represented environment geometry. The encoding uses 1=accessible, 0=blocked. This is inverted relative to a "blocked positions" one-hot encoding but carries the same information.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena at each frame.

ii.
```python
position = d['position'][day]  # (2, n_frames)
```

iii. The `position` variable records the animal's tracked location (via DeepLabCut) in a 75x75 cm open field arena at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is discretized into a 3x3 spatial grid (9 bins). Each axis (x and y) is divided into 3 equal-width bins spanning 75 cm. A small buffer (1e-5) is added to the total arena size to avoid edge effects at position=75.

ii.
```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx
```

iii. The 3x3 grid produces 9 position classes as required by the instructions. `np.clip` and the buffer ensure boundary positions are assigned to valid bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is binned using `np.floor(pos / bin_size)` where `bin_size = 75.00001 / 3 = 25.000003`. The bin index is computed as `x_bin * 3 + y_bin` (row-major with x as the first dimension). Bins at edges are clipped to [0, 2].

ii.
```python
bin_size = (env_size + buffer) / N_POS_BINS  # ~25.000003
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin
```

iii. This produces bin edges at approximately [0, 25, 50, 75] for each axis. The bin ordering `x_bin * 3 + y_bin` is consistent with how `get_env_mat` is flattened, ensuring alignment between input environment geometry and output position bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same frame rate (30 Hz) and are stored with matching indices. Both are sliced using the same trial start/end indices.

ii.
```python
pos_bins = bin_position_3x3(position)  # (n_frames,)
# ...
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. Both arrays have the same number of frames per session and are sliced with identical indices, ensuring frame-for-frame alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered neurons (NaN in trace) are excluded per session. Partial trials shorter than 30 seconds at the end of a session are discarded. Sessions with zero registered cells are skipped entirely. Sessions producing fewer than 2 trials are skipped.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0:
    continue
# ...
if trial_len < MIN_TRIAL_FRAMES:
    continue
if n_trials < 2:
    continue
```

iii. NaN filtering ensures only actually recorded neurons are included. The partial trial and minimum session checks are defensive measures, though in practice all sessions have sufficient data.

## 6-a. What are the most time-consuming steps of the code?

i. Loading data from joblib files is the most time-consuming step, as each file contains large arrays (trace shape: n_days x n_cells x n_frames). The total conversion takes ~160 seconds for all 7 animals.

ii. N/A (timing is printed per animal, ranging from 11-29 seconds each)

iii. Processing operations (NaN masking, position binning, trial splitting) are fast vectorized numpy operations. The I/O dominates.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trial start indices to slice arrays. This could potentially be vectorized using `np.split` or array reshaping, but the current approach is already efficient since numpy slicing is fast.

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The inner loop performs simple array slicing without computation, so vectorization would provide minimal benefit.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once. The `get_env_mat` function is called once per session but is trivial (returns a hardcoded array).

ii. N/A

iii. The code processes each session independently in a single pass.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `session_info` metadata dictionaries with environment name, day index, and cell counts for each session. This metadata is stored but not directly used by the decoder.

ii.
```python
session_info.append({
    'animal': animal,
    'day': day,
    'env': env_name,
    'n_registered': int(n_registered),
    'n_trials': n_trials,
})
```

iii. This metadata is useful for debugging and documentation but adds negligible overhead.
