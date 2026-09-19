# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a `.mat` file in the data directory. The AI uses `mat73.loadmat` to load each file, which returns a dictionary containing `trace`, `position`, `envs`, and `blocked` fields. A hardcoded list of 7 animal names is used to iterate over the `.mat` files.

ii.
```python
from mat73 import loadmat

ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
    n_sessions = len(dat['trace'])
    for sess_idx in range(n_sessions):
        trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
        position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
        env_name = dat['envs'][sess_idx][0]
```

iii. The agent chose `mat73` because the reference code repository used `from mat73 import loadmat` in its utility functions. The agent followed the reference code's loading approach directly.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject (mouse). Subject names are hardcoded in the `ANIMALS` list, matching the reference code's `main.py`.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
```

iii. The agent extracted the animal list from the reference code's `main.py` file which listed all 7 animals.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording sessions stored as lists. Each element of `dat['trace']`, `dat['position']`, etc. corresponds to one session. Each session becomes a separate entry in the output data structure.

ii.
```python
n_sessions = len(dat['trace'])
for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])
    position = np.array(dat['position'][sess_idx])
    env_name = dat['envs'][sess_idx][0]
```

iii. The agent identified that each `.mat` file contains multiple recording sessions by examining the data structure, noting "207 sessions total" across 7 animals.

## 1-d. How are the data split into trials?

i. Each session is split into 1-minute (60-second) non-overlapping trials of 1800 frames (30 Hz x 60 s). The remainder frames that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial

n_full_trials = n_timepoints // FRAMES_PER_TRIAL
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. The agent followed the task instructions which specify "split into 1-minute trials within each session." The agent noted: "Each 1-min trial = 1800 frames."

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full trials are skipped. Sessions with 0 valid neurons are also skipped. No per-trial filtering is applied.

ii.
```python
if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue

if n_full_trials < 2:
    print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
    continue
```

iii. The agent added the minimum 2-trial check because the instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The 0-neuron check is a basic validity guard.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in each `.mat` file, which contains calcium imaging traces (binary rising-phase transients) with shape `(n_neurons, n_timepoints)` when loaded via mat73.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. The agent identified `trace` as containing the neural data by reading the reference code and paper, which describe it as "binary rising-phase calcium transients."

## 2-b. How is the `neural` data processed?

i. All-NaN neurons (not recorded in that session) are filtered out. Any remaining NaN values are replaced with 0. Data is cast to float32.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The agent found that 330 out of 515 neurons were all-NaN in the examined session, meaning neurons tracked via CellReg that weren't detected in that session. The `nan_to_num` was described as a safety measure, since in practice all NaN neurons were entirely NaN.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are entirely NaN (not recorded in that session) are removed. Only neurons with at least some valid data are kept.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()
trace_valid = trace[valid_neurons].copy()
```

iii. The agent confirmed that NaN columns represent neurons not present in a session due to CellReg cross-session tracking: "The neurons are tracked across sessions using CellReg, so NaN means a neuron wasn't detected/tracked in that session."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are created by splitting the continuous recording into fixed 1-minute segments from the start of the session.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. The agent set `temporal_alignment_event` to "Start of recording session" and `off_start` to 0.0, indicating trials begin at the start of the recording with no event-based alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning or downsampling is applied.

ii.
```python
FPS = 30  # frames per second
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The agent confirmed the 30 Hz native rate from the paper and noted: "At 30 Hz, 1 frame = 33.33 ms. Each 1-min trial = 1800 frames."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable in each `.mat` file, which contains the name of the environment for each session (e.g., 'square', 'o', 't'). This is mapped to a 3x3 binary matrix via the `get_env_mat()` function copied from the reference code.

ii.
```python
env_name = dat['envs'][sess_idx][0]
env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)
```

iii. The agent examined both `envs` and `blocked` fields but chose to use `envs` + `get_env_mat()` because this is how the reference code represented environments. The agent noted the mapping between environment names and blocked positions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary mapping names to 3x3 binary matrices (1 = accessible, 0 = blocked). The matrix is flattened to a 9-element vector.

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

env_mat = get_env_mat(env_name).flatten().astype(np.float32)
trial_input = env_mat.copy()
```

iii. The agent copied `get_env_mat` directly from the reference code repository. The input is static per trial (same for all trials within a session).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable in each `.mat` file, which contains 2D coordinates (x, y) of the mouse in the arena.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
```

iii. The agent confirmed position values range from ~0 to 75, consistent with the paper's 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 bins). Bin edges are determined per-session based on the maximum position value (using `np.nanmax`). The single grid index is computed as `x_bin * 3 + y_bin`.

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

iii. The agent adapted the binning approach from the reference code's `decode_position_within` function, which used a similar `(max + buffer) / n_bins` calculation but with 15 bins. The agent changed to 3 bins per the task specification.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The position is discretized by dividing each axis into 3 equal bins based on the per-session maximum position value. Each bin spans `(max_val + 1e-5) / 3` units. The two dimensions are combined into a single index: `bin_idx = x_bin * 3 + y_bin`, producing 9 categories (0-8).

ii.
```python
BUFFER = 1e-5
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
bin_idx = binned[0] * n_bins + binned[1]
```

iii. The agent used per-session maximum rather than a fixed 75 cm arena size for bin edges. The BUFFER prevents the maximum value from falling outside the grid.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are stored at the same 30 Hz frame rate, so they are aligned frame-for-frame. Both are split into trials using the same start/end indices.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1)
```

iii. The agent assumed frame-level alignment based on the paper's statement that "all recorded frames were timestamped for post-hoc alignment."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons are removed per session. Remaining NaN values in neural data are replaced with 0. Sessions with 0 valid neurons or fewer than 2 trials are skipped. Remainder frames that don't complete a full 1-minute trial are discarded.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)

if n_valid == 0:
    continue
if n_full_trials < 2:
    continue
```

iii. The agent verified that in practice all NaN neurons were entirely NaN (no partial NaN values existed), making the `nan_to_num` step a safety measure rather than a necessary correction.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the `.mat` files via `mat73.loadmat` is the most I/O-intensive step. The neural trace arrays are large (hundreds of neurons x tens of thousands of timepoints per session).

ii. N/A

iii. The agent noted the resulting pickle file was ~20 GB, indicating large data volumes. The agent spent effort optimizing dtype (float64 to float32) to reduce file size from ~39 GB to ~20 GB.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially to create list entries. This could potentially be vectorized using `np.split` or array reshaping, though the current approach is straightforward and not a significant bottleneck.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_input = env_mat.copy()
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. No explicit reasoning about vectorization was provided by the agent.

## 6-c. What processing does the code repeat multiple times?

i. The `env_mat.copy()` is called for every trial within a session, even though the environment geometry is identical for all trials in a session. This is minimal overhead.

ii.
```python
trial_input = env_mat.copy()
```

iii. No explicit reasoning about repeated processing was provided.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code creates a `sample_data.pkl` file with a subset of sessions (2 per animal), which is not required by the instructions and not used downstream. The `nan_to_num` processing is unnecessary since all NaN neurons are already filtered out.

ii.
```python
sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. The agent created the sample dataset for quick testing during development. The `nan_to_num` was described as a "safety measure."
