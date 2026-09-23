# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a `.mat` file in `/app/data/`. Files are loaded using `h5py` (MATLAB v7.3 HDF5 format). The agent discovers 7 `.mat` files, iterates over them sorted by filename, and dereferences the HDF5 object references for `trace`, `position`, and `blocked` arrays within each file.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
f = h5py.File(filepath, 'r')
trace_refs = f['trace']
pos_refs = f['position']
blk_refs = f['blocked'][:][0]
n_recording_sessions = trace_refs.shape[0]
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The agent explored the HDF5 structure of the `.mat` files, identified they use MATLAB v7.3 format requiring `h5py`, and determined that each file contains arrays of object references where each reference points to a recording session's data.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject (mouse). The subject name is extracted from the filename by removing the `.mat` extension. Files are sorted alphabetically via `glob.glob` + `sorted()`.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
name = mat_file.split('/')[-1].replace('.mat', '')
subjects.append(name)
```

iii. The agent identified that each `.mat` file contains all recording sessions for one animal and used the filename as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording sessions (days), indexed by iterating over the reference arrays. Each recording day becomes a separate session in the output. Across 7 animals, this yields 207 sessions total.

ii.
```python
n_recording_sessions = trace_refs.shape[0]
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The agent verified that each reference array entry corresponds to one day's recording session and confirmed 207 total sessions matching the paper's count.

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into non-overlapping 60-second (1-minute) trials. At the native 30 Hz frame rate, each trial is 1800 timepoints. The trailing fragment (< 60 seconds) is dropped.

ii.
```python
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC  # 1800
def split_into_trials(data, trial_length=TRIAL_LENGTH):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. Per the task instructions, trials are defined as 1-minute segments. The agent uses integer division to compute the number of complete trials, discarding the remainder.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. The only data dropped are trailing frames that don't fill a complete 60-second trial.

ii. N/A (no filtering code beyond the trial splitting logic above)

iii. The agent did not identify any trial-quality criteria in the paper or instructions. The paper's velocity threshold (>5 cm/s) for decoding analysis was explicitly considered and rejected for this task (see 5).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in each `.mat` file, which contains binarised rising phases of calcium transients with shape `(n_timepoints, n_neurons)`.

ii.
```python
trace = f[trace_refs[i][0]][:]  # (timepoints, neurons)
```

iii. The agent verified that `trace` values are binary (0 and 1 only), consistent with the paper's description of binarised calcium transients.

## 2-b. How is the `neural` data processed?

i. The AI applies **no temporal processing** to the neural data. The raw binarised traces are only transposed from `(timepoints, neurons)` to `(neurons, timepoints)`, cast to float32, and filtered for registered neurons. No Gaussian smoothing, no temporal binning/average pooling, and no conversion to firing rates in Hz is applied.

ii.
```python
trace = trace[:, active_mask].astype(np.float32).T  # (n_active, n_timepoints)
```

iii. The agent acknowledged the paper's `fit_decoder` function uses Gaussian smoothing (sigma = temporal_bin_size frames) followed by average pooling, but chose not to apply this processing. The agent's code retains the raw binarised transients at the native 30 Hz frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN (not registered/recorded in a given session) are removed. Only neurons with at least one valid value are kept. No further neuron curation is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. The agent identified that the `trace` array contains NaN columns for neurons not recorded on a given day and verified that no partial NaN exists within registered neurons. The total of 69,744 registered cell-days matches the paper's count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. Position and trace share the same frame index (the DAQ timestamped both streams together), so they are inherently aligned frame-for-frame. Both are split into trials using the same indices.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. The agent verified that position and trace arrays have the same number of timepoints per session, confirming they share a common clock. No additional alignment is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data is kept at the **native 30 Hz frame rate** with a time bin size of ~33.33 ms. **No temporal rebinning** is applied. Each 1-minute trial has 1800 timepoints.

ii.
```python
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC  # 1800 timepoints
```

iii. The agent did not apply the paper's temporal binning (the reference code uses `temporal_bin_size=3` frames = 100 ms, and the reference solution uses 500 ms bins). The data is stored at the raw acquisition rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` variable in each `.mat` file, which contains indices of blocked reward positions for each recording session.

ii.
```python
blk_refs = f['blocked'][:][0]
...
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The agent explored the `blocked` field and discovered it stores flat indices (0-based) of the 3x3 grid partitions that are walled off, with `-1` indicating no positions blocked (the `square` geometry).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-dimensional binary (one-hot) vector. Positions that are blocked get a value of 1; unblocked positions get 0. If no positions are blocked (stored as `[-1]`), the vector is all zeros. The vector is constant across all timepoints and trials within a session.

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

iii. The agent validated the blocked-index convention empirically by checking that <0.01% of position frames fall in blocked bins across all animals, confirming the index convention is correct.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output mouse position is derived from the `position` variable in each `.mat` file, which contains 2D (x, y) head coordinates in centimeters tracked by DeepLabCut.

ii.
```python
position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
```

iii. The agent verified position coordinates are in cm, ranging from 0 to 75 in the fixed 75x75 cm arena coordinate frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D continuous position is discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal 25 cm bins. The flat grid label is computed as `y_bin * 3 + x_bin`. The discretization is applied per-frame at the native 30 Hz rate. No temporal aggregation (e.g., majority vote within time bins) is applied since the AI does not perform temporal rebinning.

ii.
```python
def discretize_position(position, n_grid=N_GRID, arena_size=ARENA_SIZE):
    edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
    x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
    y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
    return (y_bin * n_grid + x_bin).astype(np.int8)
...
output = discretize_position(position)[np.newaxis, :]  # (1, n_timepoints)
```

iii. The agent validated the flat-index formula `3*y + x` with fixed 25 cm bin edges by checking that occupancy in blocked bins is essentially zero across all animals and sessions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by dividing each axis into 3 equal 25 cm bins using `np.digitize` with edges at [25, 50]. Values are clipped to [0, 2] to handle boundary cases. The flat index `y_bin * 3 + x_bin` produces categories 0-8.

ii.
```python
edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]  # [25.0, 50.0]
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The agent chose fixed physical boundaries (25 cm, 50 cm) rather than per-session rescaling, as the arena coordinates are already in an absolute frame.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are stored at the same frame rate and share the same frame index in the `.mat` file. Both are split into trials using the same `split_into_trials` function with the same indices, ensuring frame-for-frame alignment.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. The agent confirmed both arrays have the same number of timepoints per session and are sliced identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of missing/incomplete data are handled: (1) Neurons with all-NaN values (not registered that day) are removed. (2) Trailing frames that don't fill a complete 60-second trial are discarded. No velocity-based filtering is applied. The agent explicitly considered the paper's 5 cm/s velocity threshold but rejected it, arguing it would break the uniform trial structure and is specific to the paper's decoding-during-locomotion analysis, not the current prediction task.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
...
n_trials = data.shape[1] // trial_length  # remainder implicitly dropped
```

iii. The agent verified there are no partial NaN values within registered neurons and no NaN in position data. The velocity threshold decision was made after computing speed statistics showing ~36-64% of frames exceed 5 cm/s, meaning filtering would remove a large fraction of data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the `.mat` files via `h5py` and dereferencing the HDF5 object references, which is I/O bound. The processing steps (NaN filtering, position discretization, trial splitting) are comparatively fast.

ii. N/A

iii. The `.mat` files contain large neural trace arrays (e.g., ~72k timepoints x 515 neurons per session for some animals).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is already largely vectorized. The `split_into_trials` function uses a list comprehension with array slicing rather than a fully vectorized reshape, but this is a minor inefficiency. The loop over sessions and subjects is inherently sequential due to I/O.

ii.
```python
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session's data is loaded and processed once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No significant unnecessary processing was identified. The code does minimal processing (transpose, filter NaN, discretize position, encode blocked, split trials) and all outputs are used in the final data structure.

ii. N/A

iii. N/A
