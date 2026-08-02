# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject's data is stored in a separate `.mat` file. The AI loads each file using `mat73.loadmat`, which reads HDF5-format MATLAB files. Each file contains arrays for `trace` (neural), `position`, and `envs` (environment names). The AI iterates over all 7 `.mat` files in the data directory, processing all sessions within each.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
    n_sessions = len(dat['trace'])
```

iii. The AI chose `mat73.loadmat` since the `.mat` files are MATLAB v7.3+ (HDF5 format). The animal names are hardcoded from the paper (7 mice). The AI verified the data matched paper statistics (207 sessions, 5413 neurons).

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject/mouse. The subject names are hardcoded in the `ANIMALS` list. The `subject_idx` maps each session to the index of the corresponding animal.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for animal_idx, animal in enumerate(ANIMALS):
    ...
    subject_idx_list.append(animal_idx)
```

iii. The AI hardcoded the animal list from the paper rather than discovering filenames dynamically. This is functionally equivalent to scanning the data directory.

## 1-c. How are the data split into sessions?

i. Within each `.mat` file, data is organized as a list of recording sessions. The AI iterates over `dat['trace']` (which has one entry per session) and treats each as a separate session.

ii.
```python
n_sessions = len(dat['trace'])
for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
    position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
    env_name = dat['envs'][sess_idx][0]
```

iii. Each session corresponds to one recording in a specific environment on a specific day. The AI verified 207 total sessions matching the paper.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute non-overlapping trials of 1800 frames (30 Hz x 60 s). Remainder frames are discarded. The AI splits by computing `n_full_trials = n_timepoints // FRAMES_PER_TRIAL`.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
...
n_full_trials = n_timepoints // FRAMES_PER_TRIAL
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. Per the task instructions, sessions are split into 1-minute trials. The AI discards any partial trial at the end of each session.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 complete trials are skipped. Sessions with zero valid neurons are also skipped. No per-trial quality filtering is applied beyond these session-level checks.

ii.
```python
if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue

if n_full_trials < 2:
    print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
    continue
```

iii. The AI noted in CONVERSION_NOTES.md that no velocity filtering or place cell filtering was applied, as these were analysis-specific steps in the paper's decoder rather than data preprocessing. Sessions need at least 2 trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in each `.mat` file, which contains binary rising-phase calcium transients (0 or 1) with shape `(n_neurons, n_timepoints)`.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. The AI identified the trace data as binary rising-phase transients from the paper's description of the calcium signal processing pipeline.

## 2-b. How is the `neural` data processed?

i. Valid (non-all-NaN) neurons are selected. Any remaining NaN values in valid neurons are replaced with 0. Data is cast to float32. No additional processing (rebinning, smoothing, normalization) is applied.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The AI noted that neurons are tracked across sessions via CellReg, so NaN indicates a neuron not detected in that session. The `nan_to_num` step handles any sporadic NaN values within otherwise valid neurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are entirely NaN (not recorded in that session) are excluded. No further neuron quality filtering (e.g., minimum firing rate, signal-to-noise) is applied.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()
if n_valid == 0:
    continue
trace_valid = trace[valid_neurons].copy()
```

iii. The AI chose not to apply the paper's velocity filtering or cell activity thresholds, reasoning these were analysis-specific rather than preprocessing steps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of each session. The alignment event is the start of the recording session.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. There is no stimulus onset or behavioral event to align to. The temporal alignment event is the start of the recording, with `off_start=0.0` and `off_end=60.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz frame rate from the calcium imaging. The AI preserved this native resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable in each `.mat` file, which contains the environment name (e.g., 'square', 'o', 't', 'u', etc.) for each session. The environment name is then mapped to a 3x3 binary matrix using the `get_env_mat` function from the reference code.

ii.
```python
env_name = dat['envs'][sess_idx][0]
env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)
```

iii. The AI used the environment name and the `get_env_mat` function (from the paper's reference code) to represent environment geometry as a binary accessibility matrix, where 1=accessible and 0=blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a hardcoded dictionary that maps each of the 10 environment types to a 3x3 binary matrix. The matrix is flattened to a 9-element vector. 1 indicates an accessible partition, 0 indicates a blocked partition.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
env_mat = get_env_mat(env_name).flatten().astype(np.float32)
```

iii. The AI adopted the `get_env_mat` function directly from the reference code. This provides a richer representation than simple blocked indices, encoding the full spatial accessibility pattern.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per trial (same for all trials within a session). It is stored as a 1D vector of shape (9,) rather than being time-varying.

ii.
```python
trial_input = env_mat.copy()  # (9,) static per trial
session_input.append(trial_input)
```

iii. Environment geometry does not change within a session, so no temporal alignment is needed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable in each `.mat` file, which contains 2D coordinates (x, y) of the mouse in the arena at each timepoint with shape `(2, n_timepoints)`.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
```

iii. Position data comes from DeepLabCut video tracking of the mouse in the 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 spatial bins). Each axis is divided into 3 bins based on the per-session maximum position value (not a fixed arena size). The bin index is computed as `x_bin * 3 + y_bin`.

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS):
    pos = position.copy()
    max_vals = np.nanmax(pos, axis=1, keepdims=True)
    bin_size = (max_vals + BUFFER) / n_bins
    binned = np.floor(pos / bin_size).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    bin_idx = binned[0] * n_bins + binned[1]
    return bin_idx
```

iii. The AI uses data-dependent bin edges (`max_vals / n_bins`) rather than fixed edges based on the known 75 cm arena size. A small buffer (`BUFFER = 1e-5`) is added to prevent the maximum value from landing exactly on a bin boundary.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). Each dimension is divided into 3 equal bins based on the per-session maximum value. The final class label is `x_bin * 3 + y_bin`, giving values 0-8.

ii.
```python
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
```

iii. The 3x3 discretization is specified in the task instructions ("3 x 3 = 9 spatial bins").

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and have the same number of timepoints per session. They are split into trials using the same start/end indices, ensuring frame-by-frame alignment.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    ...
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. Both arrays share the same temporal axis and are sliced identically, ensuring alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~33.33 ms (1000/30 Hz). No temporal rebinning is applied; the native 30 Hz frame rate is preserved.

ii.
```python
FPS = 30
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already at a consistent temporal resolution from the imaging setup. No rebinning was needed.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data share the same 30 Hz sampling and are sliced using identical indices for each trial. Input (environment geometry) is static per trial and does not require temporal alignment.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_input = env_mat.copy()
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. All time-varying data (neural, position) are already co-registered at 30 Hz from the recording. The trial splitting uses the same frame indices for all streams.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Neurons that are entirely NaN are excluded. Remaining NaN values in otherwise valid neurons are replaced with 0. Sessions with zero valid neurons or fewer than 2 complete trials are skipped.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)

if n_valid == 0:
    continue
if n_full_trials < 2:
    continue
```

iii. The AI documented that NaN indicates neurons not tracked in a session due to CellReg cross-day registration. Replacing sporadic NaN with 0 treats missing frames as no neural activity.

## 7-a. What are the most time-consuming steps of the code?

i. Loading the `.mat` files using `mat73.loadmat` is the most I/O-intensive step. The trace arrays are large (hundreds of neurons x tens of thousands of timepoints per session). The code loads 207 sessions across 7 files.

ii. N/A (not specific code, but the `loadmat` call and subsequent array conversions dominate runtime).

iii. The AI noted in the trajectory that the full dataset is ~20 GB, indicating substantial I/O and memory requirements.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over each trial index and slices arrays individually. This could be vectorized using `np.reshape` for the full trace and position arrays.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The loop is straightforward but creates many small arrays. A single reshape operation could produce all trials at once, though the list-of-arrays format requirement limits the benefit.

## 7-c. What processing does the code repeat multiple times?

i. The `env_mat.copy()` is called once per trial, creating many copies of the same static 9-element vector. This is minimal overhead but technically repeated.

ii.
```python
trial_input = env_mat.copy()
```

iii. Since the environment geometry is constant per session, the copy is repeated for each trial unnecessarily.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes all neurons and all timepoints, including those that may be in blocked regions of the arena. The paper's Bayesian decoder applies velocity filtering (discarding low-speed frames) and cell activity thresholds, but the AI chose not to apply these, providing all data to the downstream decoder.

ii. N/A (no explicit discard, but all data is preserved regardless of whether it would be filtered in the paper's analysis pipeline).

iii. The AI reasoned that velocity filtering and neuron selection are analysis-specific and should not be part of data preprocessing.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: NaN neurons are excluded, sporadic NaN in valid neurons are replaced with 0, sessions with no valid neurons or too few trials are skipped. Remainder frames at the end of sessions are discarded.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
n_full_trials = n_timepoints // FRAMES_PER_TRIAL  # remainder discarded
```

iii. The AI's approach preserves maximum data while handling known edge cases in the calcium imaging recordings.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: Loading `.mat` files via `mat73.loadmat` dominates runtime due to the large size of the neural trace arrays (full dataset ~20 GB).

ii. N/A

iii. I/O bound on the large HDF5 files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: The per-trial slicing loop could be replaced with a single reshape, and position discretization is already vectorized.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
```

iii. The current loop is simple and readable, and the overhead is small relative to I/O.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: `env_mat.copy()` is called per trial when it could be created once per session. Also, `position` is fully loaded and discretized even if some frames are discarded (remainder frames).

ii.
```python
trial_input = env_mat.copy()  # repeated per trial
pos_bins = discretize_position(position, N_SPATIAL_BINS)  # includes remainder frames
```

iii. Minimal overhead in practice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: All neurons and all time bins are processed without velocity filtering or neuron quality thresholds. Position discretization is computed for remainder frames that are then discarded. The `create_sample` function processes the full dataset before subsetting.

ii.
```python
pos_bins = discretize_position(position, N_SPATIAL_BINS)  # all timepoints
n_full_trials = n_timepoints // FRAMES_PER_TRIAL  # remainder discarded
```

iii. The extra computation is negligible compared to I/O costs.
