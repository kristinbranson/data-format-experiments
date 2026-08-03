# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads each animal's data from individual `.mat` files (one per animal) using `mat73.loadmat`. It iterates over the 7 animals defined in the `ANIMALS` list, loading `{animal_name}.mat` from the `data/` directory. Each `.mat` file contains fields including `trace` (neural data), `position` (behavioral position), and `envs` (environment names). The data is structured as lists indexed by session.

ii.
```python
from mat73 import loadmat

ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
```

iii. The agent confirmed from the reference code (`load_dat` in `utils.py`) that `mat73.loadmat` is the correct loading approach. The agent verified that 7 animals and 207 total sessions were loaded, matching the paper's stated statistics.

## 1-b. How are the data split into subjects (mice)?

i. Each `.mat` file corresponds to one mouse. The 7 mice are hardcoded in the `ANIMALS` list. The outer loop iterates over animals, assigning each an index used for `subject_idx`.

ii.
```python
for animal_idx, animal in enumerate(ANIMALS):
    # ... process all sessions for this animal ...
    subject_idx_list.append(animal_idx)
```

iii. The agent verified 7 animals matching the paper, with session counts per animal: 31 for most mice, 21 for QLAK-CA1-51.

## 1-c. How are the data split into sessions?

i. Within each animal's data, sessions are indexed by position in the `trace` and `position` lists. The code iterates over `range(n_sessions)` where `n_sessions = len(dat['trace'])`. Each session corresponds to one recording day in one environment.

ii.
```python
n_sessions = len(dat['trace'])
for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
    position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
    env_name = dat['envs'][sess_idx][0]
```

iii. The agent verified 207 total sessions across all animals, matching the paper.

## 1-d. How are the data split into trials?

i. Each session's continuous recording (~40 minutes at 30 Hz) is split into non-overlapping 1-minute trials of exactly 1800 frames (60s * 30 Hz). Integer division determines the number of complete trials; any remaining frames at the end of a session are discarded. Sessions with fewer than 2 complete trials are skipped.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
n_full_trials = n_timepoints // FRAMES_PER_TRIAL

if n_full_trials < 2:
    continue

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. The task instructions specified "long recording sessions, which will be split into 1-minute trials." The agent confirmed sessions yielded 39 or 40 trials depending on exact duration (71,866 vs ~72,060 frames), totaling 8,187 trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied beyond requiring at least 2 complete trials per session and at least 1 valid (non-NaN) neuron. Sessions with zero valid neurons or fewer than 2 trials are skipped entirely; no individual trials are removed.

ii.
```python
if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue

if n_full_trials < 2:
    print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
    continue
```

iii. The agent noted in CONVERSION_NOTES.md: "No velocity filtering or place cell filtering applied, as these are analysis-specific in the paper." The reference code's `decode_position_within` applies velocity filtering (`v_thresh=5` cm/s) and cell activity thresholds (`cell_threshold=5`), but the agent chose not to apply these.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in the `.mat` files: `dat['trace'][sess_idx]`, which contains binary rising-phase calcium transients (0/1 values) with shape `(n_neurons, n_timepoints)`.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. The agent confirmed from the paper: "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5, and 0 otherwise." The agent verified the data contained only 0/1 values.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: (1) NaN neurons (those not detected in the current session via CellReg tracking) are identified and excluded. (2) Any remaining NaN values in valid neurons are replaced with 0. (3) Data is cast to float32.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The agent explained that NaN rows represent neurons not tracked in that session (CellReg cross-session tracking). No further processing (smoothing, normalization, z-scoring) is applied. The binary transient data is used as-is.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural filtering is removing neurons that are entirely NaN in a given session. No velocity-based filtering of timepoints, no activity threshold filtering (cell_threshold), and no place cell selection is applied.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()
if n_valid == 0:
    continue
```

iii. The CONVERSION_NOTES state: "All valid neurons included for maximum information available to the decoder." The reference code applies `cell_threshold=5` (requiring minimum transient count) and velocity filtering, but the agent did not replicate these.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Trials are consecutive 1-minute segments from the beginning of the session. The first trial starts at frame 0, the second at frame 1800, etc.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. The metadata sets `temporal_alignment_event: 'Start of recording session'`, `off_start: 0.0`, `off_end: 60.0`. Since this is a free exploration task without discrete trial events, alignment to recording start is appropriate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native 30 Hz sampling rate, giving a time bin size of ~33.33 ms (1000/30 ms). No temporal rebinning or downsampling is applied.

ii.
```python
FPS = 30  # frames per second
# In metadata:
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The agent preserved the original recording rate. The reference code's decoding function does support downsampling via `bin_down` parameter, but the agent kept the native resolution. This means each trial has 1800 timepoints.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `envs` field in the `.mat` files, which contains the environment name string for each session (e.g., 'square', 'o', 't'). The name is mapped to a 3x3 binary matrix using the `get_env_mat` function copied from the reference code.

ii.
```python
env_name = dat['envs'][sess_idx][0]
env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)
```

iii. The agent found and directly copied `get_env_mat` from the reference code (`utils.py` line 215). It also inspected the `blocked` field in the data but chose to use the environment name lookup instead.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary mapping to a 3x3 binary matrix (1=accessible, 0=blocked partition). The matrix is then flattened to a 9-element vector. This is static per trial (same for all trials in a session).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        # ... 10 total environments
    }
    return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)

# In convert_data():
trial_input = env_mat.copy()  # shape (9,), static per trial
```

iii. The agent copied the environment matrix definitions verbatim from the reference code's `get_env_mat` function, ensuring consistency with the paper's description of "partitioning an open square (75 x 75 cm) into a 3 x 3 grid space."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in the `.mat` files: `dat['position'][sess_idx]`, which is a (2, n_timepoints) array of x-y coordinates from DeepLabCut tracking.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
```

iii. The agent confirmed position values range from 0 to ~75, consistent with the 75x75 cm arena described in the paper.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is discretized into a 3x3 grid (9 bins). For each dimension, the bin size is computed as `(max_val + epsilon) / 3`, positions are divided by bin size and floored, then the 2D bin indices are converted to a single index via row-major ordering: `bin_idx = x_bin * 3 + y_bin`.

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS):
    pos = position.copy()
    max_vals = np.nanmax(pos, axis=1, keepdims=True)
    bin_size = (max_vals + BUFFER) / n_bins   # BUFFER = 1e-5
    binned = np.floor(pos / bin_size).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    bin_idx = binned[0] * n_bins + binned[1]
    return bin_idx
```

iii. The agent noted this approach is consistent with the reference code's binning pattern: `bin_down = (behav_max.max() + buffer) / n_bins`. The position max is computed per-session (from `np.nanmax` across the whole session).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is binned into 9 categories (0-8) by dividing each dimension into 3 equal-width bins based on the session's maximum position value. The bins are computed per-session. No velocity threshold or occupancy threshold is applied to exclude timepoints.

ii.
```python
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
# Output shape: (1, FRAMES_PER_TRIAL), dtype int64
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The 9 output values are labeled `bin_(0,0)` through `bin_(2,2)`. The verification output shows the position distribution: most time spent in bin_(2,2) (20.2%), least in bin_(1,1) (5.6%).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same native timepoints (both recorded/tracked at 30 Hz). The same frame indices are used to slice both arrays for each trial, so they are inherently aligned.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. Since `trace` and `position` arrays have the same number of timepoints per session, using the same indices ensures alignment. No interpolation or resampling is needed.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of missing data are handled: (1) Neurons that are entirely NaN in a session are excluded. (2) Any remaining isolated NaN values in otherwise-valid neurons are replaced with 0. NaN values in position data are handled implicitly by `np.nanmax` in the binning function.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
# Position binning uses np.nanmax:
max_vals = np.nanmax(pos, axis=1, keepdims=True)
```

iii. The agent documented that NaN neurons come from CellReg cross-session tracking: neurons not detected in a session have NaN rows. The agent's approach of excluding all-NaN neurons and zeroing remaining NaNs is reasonable for binary transient data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading the large `.mat` files with `mat73.loadmat` (each file contains all sessions for one animal). (2) The inner loop over trials which copies array slices. (3) Saving the ~20 GB pickle file.

ii.
```python
dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))  # slow I/O
# Trial loop with array operations:
for trial_idx in range(n_full_trials):
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The conversion output shows the full dataset is 19.98 GB, indicating significant memory and I/O requirements.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop could be vectorized using `np.split` or array reshaping instead of iterating over trial indices and slicing individually. For example, `trace_valid.reshape(n_valid, n_full_trials, FRAMES_PER_TRIAL)` followed by a list comprehension or `np.split`.

ii.
```python
# Current approach (loop):
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]

# Could be vectorized as:
# trial_neurals = np.split(trace_valid[:, :n_full_trials*FRAMES_PER_TRIAL], n_full_trials, axis=1)
```

iii. The loop creates 39-40 array slices per session (8,187 total), each involving array copying. While not a major bottleneck compared to I/O, it could be more efficient.

## 6-c. What processing does the code repeat multiple times?

i. The position discretization is computed for the entire session and then sliced per trial, which is efficient. However, `env_mat.copy()` is called for every trial even though the geometry is the same within a session. The NaN check and neuron filtering are done once per session, which is appropriate.

ii.
```python
# env_mat.copy() called per trial (unnecessary copies):
for trial_idx in range(n_full_trials):
    trial_input = env_mat.copy()  # Same value every trial in this session
```

iii. This is a minor inefficiency - 8,187 copies of a 9-element array. The overall design is reasonably efficient with session-level processing before trial splitting.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The tail-end frames of each session (beyond the last complete trial) are computed but discarded. For 39-trial sessions, 1,266 frames (~42s) are unused. (2) The `create_sample` function generates a sample dataset that may not be needed for the final analysis. (3) The code stores `n_unique_neurons: 5413` in metadata as a hardcoded value rather than computing it.

ii.
```python
# Unused frames at session end:
n_full_trials = n_timepoints // FRAMES_PER_TRIAL  # remainder discarded

# Sample dataset creation:
sample_data = create_sample(data, max_sessions_per_animal=2)
```

iii. The discarded frames are an inherent consequence of the 1-minute trial design and are expected. The sample dataset serves as a quick-test subset but is not used in final evaluation.
