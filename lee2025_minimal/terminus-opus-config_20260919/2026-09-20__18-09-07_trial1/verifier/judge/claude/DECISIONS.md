# Decisions

**Note:** The agent trajectory (`/logs/agent/trajectory.json`) records the agent writing code that closely mirrors the human reference (using `joblib`, Gaussian smoothing, temporal binning, cell activity thresholds, position snapping). However, the actual output file `/app/convert_data.py` is a substantially different, simpler implementation (using `h5py`, no temporal processing, no cell filtering beyond NaN removal). This document evaluates the **actual code** in `/app/convert_data.py`.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a `.mat` file in the data directory. The AI loads these using `h5py`, accessing `trace`, `position`, and `blocked` as HDF5 reference arrays. Each `.mat` file contains multiple recording sessions stored as arrays of references.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
f = h5py.File(filepath, 'r')
trace_refs = f['trace']
pos_refs = f['position']
blk_refs = f['blocked'][:][0]
n_recording_sessions = trace_refs.shape[0]
```

iii. The trajectory shows the agent exploring data files and discovering both `.mat` and joblib formats. The actual code chose `.mat` files via `h5py`. The trajectory's exploration steps (steps 10-16) show the agent examining the data structure of joblib files, yet the final code uses h5py on `.mat` files instead.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject. The subject name is extracted from the filename by stripping the `.mat` extension.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
name = mat_file.split('/')[-1].replace('.mat', '')
subjects.append(name)
```

iii. The agent identified 7 subjects (QLAK-CA1-08 through QLAK-CA1-75) by listing the data directory.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording sessions (days), indexed by iterating over the HDF5 reference arrays. Each recording session becomes a separate session in the output.

ii.
```python
n_recording_sessions = trace_refs.shape[0]
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. Each reference array entry corresponds to one recording day. The agent determined there are 207 total sessions across 7 animals (matching the paper).

## 1-d. How are the data split into trials?

i. Trials are defined as 60-second non-overlapping segments of the continuous recording. At the native 30 Hz frame rate, each trial is 1800 frames. Remainder frames are discarded.

ii.
```python
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC  # 1800
...
def split_into_trials(data, trial_length=TRIAL_LENGTH):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. The instructions specify 1-minute trials. Because the AI does not apply temporal binning, each trial contains 1800 timepoints (at ~33.33 ms per bin) rather than the reference's 600 timepoints (at 100 ms per bin).

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial or session filtering is applied. Sessions with zero active neurons or zero trials would produce empty lists but are not explicitly skipped. There is no check for a minimum of 2 trials per session.

ii. No filtering code present beyond the implicit zero-trial case.

iii. No justification found in trajectory for the lack of filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the `.mat` file, which contains calcium trace data with shape `(timepoints, neurons)`.

ii.
```python
trace = f[trace_refs[i][0]][:]  # (timepoints, neurons)
```

iii. The agent identified `trace` as containing binarized rising-phase calcium transients.

## 2-b. How is the `neural` data processed?

i. The only processing is: (1) removing all-NaN neurons, (2) transposing from `(timepoints, neurons)` to `(neurons, timepoints)`, and (3) casting to float32. **No Gaussian smoothing or temporal binning is applied.**

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T  # (n_active, n_timepoints)
```

iii. The trajectory shows the agent reading the paper's decoder code (step 9), discovering Gaussian smoothing (sigma=3 frames) and 3-frame average pooling. However, the actual code does not implement this processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons that are all-NaN (not registered in that session) are removed. **No activity-based filtering** (e.g., the paper's >5 transient threshold from `decode_position_within`) is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. The trajectory (step 9) notes the paper uses `cell_threshold=5`, but the actual code does not implement it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous free exploration, and trials are artificial 60-second segments starting from the beginning of the session.

ii. N/A - alignment is implicit through identical slicing of all data streams.

iii. The instructions state there is no task event to align to.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native 30 Hz frame rate (~33.33 ms per bin). **No temporal rebinning is applied.**

ii.
```python
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. The paper's decoder applies 3-frame average pooling (100 ms bins), but the AI does not replicate this. The trajectory shows the agent was aware of this processing (step 9: "temporal_bin_size=3") but the code omits it.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` variable in the `.mat` file, which contains indices of blocked reward positions for each recording session.

ii.
```python
blk_refs = f['blocked'][:][0]
...
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The `blocked` variable stores which of the 9 possible positions were blocked during each session.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-dimensional one-hot (binary) encoding. If no positions are blocked (value is `[-1]`), the vector is all zeros. The blocked vector is static per session and replicated across all trials.

ii.
```python
def encode_blocked(blk_indices, n_positions=N_BLOCKED_POSITIONS):
    blocked = np.zeros(n_positions, dtype=np.float32)
    if not (len(blk_indices) == 1 and blk_indices[0] == -1):
        blocked[blk_indices.astype(int)] = 1
    return blocked
...
input_trials = [blocked] * len(neural_trials)
```

iii. One-hot encoding over 9 positions is consistent with the paper's representation.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the `.mat` file, which contains 2D (x, y) coordinates in cm.

ii.
```python
position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
```

iii. Position data was tracked by DeepLabCut at 30 Hz simultaneously with imaging.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). Bin edges are computed with `np.linspace(0, 75, 4)[1:-1]` giving edges at [25, 50]. `np.digitize` assigns bin indices, which are clipped to [0, 2]. The grid label is `y_bin * 3 + x_bin`.

ii.
```python
def discretize_position(position, n_grid=N_GRID, arena_size=ARENA_SIZE):
    edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
    x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
    y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
    return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The 3x3 grid matches the paper's partition structure.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Positions are binned into 9 categories (0-8) based on the 3x3 grid. No handling of positions that fall in blocked partitions is applied; samples in blocked regions are assigned to the blocked partition's bin index rather than being snapped to the nearest open partition.

ii. Same `discretize_position` function as 4-b. No snapping logic.

iii. No justification for omitting blocked-partition snapping.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are at the same frame rate (30 Hz) in the source data, so they are aligned frame-for-frame. Both are split into trials using the same `split_into_trials` function.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. Both arrays have the same number of timepoints and are sliced identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with all-NaN values (not registered in a session) are removed. Remainder frames that don't fill a complete 60-second trial are discarded. No handling of positions in blocked partitions (tracking noise near walls). No handling of sessions with fewer than 2 trials.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
...
n_trials = data.shape[1] // trial_length  # remainder implicitly dropped
```

iii. The NaN filtering ensures only recorded neurons are included.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the `.mat` files via `h5py` is the most I/O intensive step, as the files are hundreds of MB to ~800 MB each. The HDF5 reference dereferencing for each session is also relatively slow.

ii. N/A

iii. The trajectory shows the conversion took several minutes for all 7 animals.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `split_into_trials` function uses a Python list comprehension to slice arrays, which could be replaced with `np.split` or reshape operations. However, the overhead is minimal.

ii.
```python
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each session's data is processed once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does very little processing overall (no smoothing, no pooling, no cell filtering beyond NaN). However, the lack of temporal binning means the output data is 3x larger than necessary (1800 timepoints per trial vs 600 at 100 ms bins), which increases memory usage and decoder training time without corresponding benefit.

ii. N/A

iii. N/A
