# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` from non-.mat files in the `data/` directory. Each animal has a corresponding joblib file (e.g., `QLAK-CA1-08`). The loaded data is accessed via `dat[animal]`, which contains a dictionary with keys: `trace`, `position`, `envs`, `blocked`, `maps`, `SFPs`, `centroids`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)
```

iii. The AI noted that the reference code's `load_dat()` function supports both MATLAB and joblib formats, and chose joblib since the pre-converted joblib files are available in the data directory. This avoids the overhead of parsing HDF5/MATLAB files.

## 1-b. How are the data split into subjects?

i. Each joblib file in the `data/` directory corresponds to one subject (mouse). The animal names are hardcoded in the `ANIMALS` list. Each animal is processed sequentially and assigned a subject index.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for a_idx, animal in enumerate(animals_to_process):
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
    for sess in sessions:
        all_subject_idx.append(a_idx)
```

iii. The 7 animals match the paper's reported count. Subject names are hardcoded rather than discovered via directory listing.

## 1-c. How are the data split into sessions?

i. Each animal's `trace` array has shape `(n_sessions, n_neurons, n_timepoints)`. The first dimension indexes recording sessions. Each session (day) becomes a separate session in the output.

ii.
```python
n_sessions = trace.shape[0]
for day in range(n_sessions):
    tr = trace[day]  # (n_neurons, n_timepoints)
    pos = position[day]  # (2, n_timepoints)
    env_name = envs[day, 0]
```

iii. Each recording day for an animal is one session. Total 207 sessions across 7 animals.

## 1-d. How are the data split into trials?

i. Each continuous recording session (~40 minutes) is split into non-overlapping 1-minute (1800 frame) segments. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. Per instruction, trials are 1-minute non-overlapping segments. At 30 Hz, each trial is 1800 frames. Sessions of ~72000 frames yield ~39-40 trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are included. Only incomplete remainder frames at the end of a session are discarded.

ii. N/A (no filtering code)

iii. The reference code does not filter trials at the data loading stage. Velocity filtering in the reference is applied during decoding, not during data preparation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary (0/1) rising-phase calcium transient data with shape `(n_sessions, n_neurons, n_timepoints)`.

ii.
```python
trace = d['trace']  # (n_sessions, n_neurons, n_timepoints)
tr = trace[day]     # (n_neurons, n_timepoints)
```

iii. The `trace` variable contains pre-processed binary rising-phase vectors, where the derivative of calcium traces was smoothed, z-scored, and thresholded at z > 2.5.

## 2-b. How is the `neural` data processed?

i. Active neurons (non-all-NaN) are selected. Any remaining NaN values are replaced with 0. Data is cast to float32. No additional temporal processing (smoothing, binning, etc.) is applied.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The binary trace data is already preprocessed by the original authors. The AI decided not to apply velocity filtering or cell activity filtering at the conversion stage, reasoning that the downstream decoder handles this differently.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with all-NaN values across all timepoints in a session are excluded. These represent neurons not tracked on that day. Additionally, remaining NaN values (if any) in otherwise active neurons are replaced with 0.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. NaN neurons are those not detected by CellReg on a given day. The AI did not apply the reference code's cell activity threshold (cell_threshold=5, filtering cells by total activity when moving), reasoning it was part of the decoder rather than data preparation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of the recording. Alignment is to the start of each recording session.

ii. N/A (no alignment code)

iii. There is no stimulus onset or behavioral event to align to. The continuous recording is simply segmented into fixed-duration trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per frame). No temporal rebinning is applied.

ii.
```python
FPS = 30
# time_bin_size = 1000.0 / FPS  # ~33.33 ms
```

iii. The AI decided not to apply the reference code's 3-frame temporal binning (used in `fit_decoder` via AvgPool1d), reasoning that the downstream decoder handles temporal binning differently.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable, which contains environment name strings (e.g., 'square', 'o', 't'). These are converted to binary 3x3 matrices using the `get_env_mat()` function from the reference code.

ii.
```python
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI chose to use the `get_env_mat()` function directly from the reference code, which maps environment names to canonical binary 3x3 matrices indicating accessible (1) vs blocked (0) positions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary to get a 3x3 binary matrix (1=accessible, 0=blocked). This matrix is flattened to a 9-element vector. The input is static per trial (same for all timepoints).

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

iii. The AI copied the `get_env_mat` function from the reference code's `utils.py`. This encodes environment accessibility rather than blocked positions.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Environment geometry is a per-session constant (static per trial). The same 9-element vector is used for all trials in a session. No temporal alignment is needed.

ii.
```python
trial_input.append(env_mat.astype(np.float32))
```

iii. Since the environment geometry doesn't change within a session, it's simply replicated for each trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output position is derived from the `position` variable, which contains 2D (x, y) coordinates of the animal in the arena, with shape `(n_sessions, 2, n_timepoints)`.

ii.
```python
pos = position[day]  # (2, n_timepoints)
pos_bins = position_to_bin(pos)
```

iii. Position is tracked using DeepLabCut at 30 Hz, with values in the range 0-75 cm.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is discretized into a 3x3 grid (9 bins). Each axis is divided into 3 equal 25 cm bins using `floor(position / bin_size)`. The combined bin index is `x_bin * 3 + y_bin`.

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins  # 25.0
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. The 3x3 grid is specified by the task instructions. The bin size of 25 cm divides the 75 cm arena into 3 equal parts per axis.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded by dividing each axis into 3 bins of 25 cm each. `np.floor(pos / 25)` assigns each position to a bin (0, 1, or 2). `np.clip` handles boundary values. The final label is `x_bin * 3 + y_bin`, giving 9 categories (0-8).

ii.
```python
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. The bin boundaries are at 25 cm and 50 cm for both x and y. Positions at the upper boundary (75 cm) are clipped to bin 2.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same frame rate (30 Hz) in the same array structure. Both are sliced identically when splitting into trials, ensuring frame-for-frame alignment.

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. Both arrays share the same timepoint dimension and are split using identical start/end indices.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e: the data is at the native 30 Hz (~33.33 ms per frame). No rebinning is applied.

ii.
```python
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI chose to preserve the original temporal resolution. The reference code's 3-frame binning is a decoder-specific operation.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data share the same timepoint dimension from the original recording and are sliced with identical indices. Input (environment geometry) is static per trial and requires no temporal alignment.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The temporal alignment is inherent in the data structure since neural and position data are co-recorded at 30 Hz.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Neurons with all-NaN values are removed (not tracked on that day). Remaining NaN values in active neurons are replaced with 0. Remainder frames at session end that don't fill a complete trial are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. The AI documented that NaN values indicate neurons not detected by CellReg on a given day. The `nan_to_num` is described as a safety measure ("shouldn't happen but safety").

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the data files via `joblib.load()`, which takes 12-23 seconds per animal. Processing sessions is relatively fast (8-15 seconds per animal). Total estimated time: ~4 minutes for all 7 animals.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s")
```

iii. The AI included timing instrumentation and documented that I/O (loading large joblib files) dominates runtime.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials and appends slices to a list. This could potentially be vectorized using `np.split` or array reshaping, though the benefit would be minor.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The AI did not explicitly document vectorization opportunities. The loop is simple and the overhead is negligible compared to data loading.

## 7-c. What processing does the code repeat multiple times?

i. The `env_mat.astype(np.float32)` is recomputed for each trial within a session, though the value is the same. This is trivial overhead.

ii.
```python
trial_input.append(env_mat.astype(np.float32))
```

iii. The AI did not document this as a concern, and the overhead is negligible.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes all sessions for each animal's data (including loading `maps`, `SFPs`, `centroids` from joblib files) even though only `trace`, `position`, and `envs` are used. The full joblib file is loaded into memory.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
trace = d['trace']
position = d['position']
envs = d['envs']
```

iii. Loading the full joblib file includes unused fields like `maps`, `SFPs`, and `centroids`, consuming extra memory and I/O time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. All-NaN neurons are filtered out. Remaining NaN values are replaced with 0. Incomplete trial remainders are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. The AI treated NaN columns as intentional (neurons not tracked) rather than data corruption, which is consistent with the reference code and paper.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Data loading via `joblib.load()` is the bottleneck, taking 12-23 seconds per animal.

ii. See 7-a.

iii. See 7-a.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The trial splitting loop is the main candidate but provides minimal benefit.

ii. See 7-b.

iii. See 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. Minor repetition of `astype(np.float32)` on per-trial inputs.

ii. See 7-c.

iii. See 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Loading full joblib files brings in unused data fields.

ii. See 7-d.

iii. See 7-d.
