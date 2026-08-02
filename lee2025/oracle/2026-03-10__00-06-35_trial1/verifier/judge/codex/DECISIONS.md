# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script enumerates all `.mat` files in `--datadir` with `glob`, treats each file as one subject, opens each file with `h5py`, and reads the HDF5 datasets `trace`, `position`, and `blocked`. Within each file, it iterates through the session reference arrays and resolves each referenced dataset into in-memory NumPy arrays.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
f = h5py.File(filepath, 'r')

trace_refs = f['trace']
pos_refs = f['position']
blk_refs = f['blocked'][:][0]
n_recording_sessions = trace_refs.shape[0]
...
trace = f[trace_refs[i][0]][:]
position = f[pos_refs[i][0]][:].T
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. `CONVERSION_NOTES.md` contains no substantive explanation, and `/logs/agent/trajectory.json` was not present at the stated path, so the justification is inferred from the docstring and code comments: the data are MATLAB v7.3/HDF5 files whose top-level arrays store references to per-session arrays.

## 1-b. How are the data split into subjects?

i. Each `.mat` file is treated as one mouse, and the subject identifier is the filename stem.

ii.
```python
for subj_i, mat_file in enumerate(mat_files):
    ...
    name = mat_file.split('/')[-1].replace('.mat', '')
    subjects.append(name)
```

iii. The only available justification is implicit in the iteration structure and the module docstring stating that each recording session is nested within a per-subject file.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the length of the `trace` reference array in each subject file. The code loops over `range(n_recording_sessions)` and loads one `trace`, one `position`, and one `blocked` entry per session.

ii.
```python
n_recording_sessions = trace_refs.shape[0]
...
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The justification is implicit: the `.mat` schema stores one reference per recording session in each top-level array.

## 1-d. How are the data split into trials?

i. Each continuous session is split into non-overlapping 60-second trials. At 30 Hz this is `1800` frames per trial. Any trailing partial segment is dropped.

ii.
```python
SAMPLING_RATE = 30  # Hz
TRIAL_DURATION_SEC = 60  # 1 minute per trial
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC  # 1800 timepoints

def split_into_trials(data, trial_length=TRIAL_LENGTH):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. The justification is explicit in the comments and docstring: the decoder task requires 1-minute trials from continuous recordings.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filtering. The only implicit filtering is that incomplete trailing segments are discarded because `split_into_trials` uses integer division.

ii.
```python
def split_into_trials(data, trial_length=TRIAL_LENGTH):
    ...
    n_trials = data.shape[1] // trial_length
    return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. No separate justification is documented. The code comments only note that the last short chunk is dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from the raw `trace` variable in each `.mat` file.

ii.
```python
trace_refs = f['trace']
...
trace = f[trace_refs[i][0]][:]  # (timepoints, neurons)
```

iii. The code comments describe `trace` as the per-session neural array and assume it is already the usable signal.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: remove all-NaN neuron columns, cast to `float32`, and transpose from `(timepoints, neurons)` to `(neurons, timepoints)`. No smoothing, normalization, deconvolution, or temporal rebinning is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T  # (n_active, n_timepoints)
```

iii. The justification is only implicit in the comment that the script should keep “only neurons actually recorded in each session.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained if its `trace` column is not entirely NaN within that session. No additional cell-quality or activity thresholds are used.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. The comment indicates the goal is to remove NaN padding for neurons not recorded in that session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is not aligned to a biological or behavioral event. Trials are artificial, contiguous 60-second windows cut from a continuous recording.

ii.
```python
neural_trials = split_into_trials(trace)
```

iii. The justification is implicit in the overall design: the instruction-defined event is effectively trial start after fixed session chunking, not an external task event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data remain at the native 30 Hz sampling rate, corresponding to `1000/30 ≈ 33.33 ms` per time bin. No rebinning is applied.

ii.
```python
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. This is stated directly in the constants and reflected in the absence of any resampling logic.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the raw `blocked` variable, which stores blocked-location indices for each session.

ii.
```python
blk_refs = f['blocked'][:][0]
...
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The code comments identify this as the source for session-level blocked positions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted into a 9-element one-hot vector. If the session stores `[-1]`, the vector remains all zeros. The resulting vector is reused for every trial in that session.

ii.
```python
def encode_blocked(blk_indices, n_positions=N_BLOCKED_POSITIONS):
    blocked = np.zeros(n_positions, dtype=np.float32)
    if not (len(blk_indices) == 1 and blk_indices[0] == -1):
        blocked[blk_indices.astype(int)] = 1
    return blocked
...
blocked = encode_blocked(blk_indices)
input_trials = [blocked] * len(neural_trials)
```

iii. The only explicit justification is the function docstring and comment that this produces a session-level one-hot encoding of blocked positions.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. It is treated as static per trial rather than time-varying. The same session-level blocked vector is attached to every trial produced from that session.

ii.
```python
input_trials = [blocked] * len(neural_trials)
```

iii. The justification is implicit: arena geometry does not change within a session, so no time-axis alignment is performed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the raw `position` variable, read as 2D `(x, y)` coordinates over time.

ii.
```python
pos_refs = f['position']
...
position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
```

iii. The code comments directly describe `position` as the source signal used for the decoder target.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D coordinates are discretized onto a `3 x 3` grid over a `75 x 75` arena. Bin edges are equally spaced, `np.digitize` assigns x and y bins, `np.clip` constrains edge cases, and the final class label is `y_bin * 3 + x_bin`. The labels are then expanded to shape `(1, n_timepoints)`.

ii.
```python
ARENA_SIZE = 75.0
N_GRID = 3

def discretize_position(position, n_grid=N_GRID, arena_size=ARENA_SIZE):
    edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
    x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
    y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
    return (y_bin * n_grid + x_bin).astype(np.int8)
...
output = discretize_position(position)[np.newaxis, :]  # (1, n_timepoints)
```

iii. The justification is embedded in the constants and function design: the decoder target must be a 9-class categorical position signal.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each spatial axis is thresholded using two equally spaced boundaries at one-third and two-thirds of the 75 cm arena, producing 3 bins per axis and 9 combined categories total.

ii.
```python
edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. No prose justification is provided beyond the implementation itself.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Alignment is framewise and inherited from the raw arrays. After position is discretized, neural and output arrays are cut with the same `split_into_trials` indexing, so each trial contains the same time span in both streams.

ii.
```python
output = discretize_position(position)[np.newaxis, :]  # (1, n_timepoints)
...
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. The code assumes `trace` and `position` are already synchronized in the source files.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is approximately `33.33 ms` (`30 Hz`), and no rebinning or interpolation is performed anywhere in the conversion.

ii.
```python
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. The justification is explicit in the constants and in the absence of any temporal aggregation code.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output data are aligned sample-by-sample before trialization and then split with the same trial boundaries. Input is static per session and copied to each trial without a time axis.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
input_trials = [blocked] * len(neural_trials)
```

iii. The justification is implicit: the source arrays already share the same recording timeline, while blocked geometry is constant across that timeline.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing neurons represented by all-NaN trace columns are dropped. Short trailing fragments that do not make a complete 60-second trial are silently discarded. The code assumes other entries are well-formed and does not add defensive validation.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
...
n_trials = data.shape[1] // trial_length
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. No extra justification is documented beyond comments describing NaN padding removal and dropping the final short chunk.

## 7-a. What are the most time-consuming steps of the code?

i. The most expensive operations are likely HDF5 I/O for loading per-session arrays and the repeated per-session slicing/copying involved in filtering traces, discretizing position, and materializing trial lists.

ii.
```python
trace = f[trace_refs[i][0]][:]
position = f[pos_refs[i][0]][:].T
...
trace = trace[:, active_mask].astype(np.float32).T
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. There is no explicit performance analysis in the notes, so this is inferred from the structure of the code.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loop over sessions and the list-comprehension-based `split_into_trials` function could be replaced with reshape/view operations when divisibility permits. The loop that appends each session to aggregate lists is also purely Python-level bookkeeping.

ii.
```python
for i in range(n_recording_sessions):
    ...

return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
...
for sess_i, sess in enumerate(sessions):
    all_neural.append(sess['neural'])
    all_input.append(sess['input'])
    all_output.append(sess['output'])
```

iii. No explicit justification is given by the agent; this is a direct reading of the implementation.

## 7-c. What processing does the code repeat multiple times?

i. The same trial-splitting logic is run separately for neural and output data in every session. The script also repeatedly resolves HDF5 references one session at a time and repeatedly constructs identical blocked vectors across trials within a session.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
input_trials = [blocked] * len(neural_trials)
```

iii. This repetition is evident in the code; no explicit agent-side discussion of it is available.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and prints summary diagnostics for each session, including unique output classes and blocked-state displays, but those values are not stored in the final dataset. It also records `n_neurons` in the temporary session dict only to derive brain-region indices during aggregation.

ii.
```python
sessions.append({
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'n_neurons': n_active,
})
...
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
blocked_str = sess['input'][0].astype(int) if n_trials > 0 else []
print(f"    session {sess_i}: {n_neurons} neurons, {n_trials} trials, "
      f"{trial_len} timepoints/trial, "
      f"neural dtype={sess['neural'][0].dtype}, "
      f"output classes={output_classes.astype(int)}, "
      f"blocked={blocked_str}")
```

iii. There is no explicit self-critique from the agent; this follows from what is produced and later discarded.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is the same as in section 6: all-NaN neuron columns are removed, and any leftover frames that do not fill a full trial are dropped. No additional repair or imputation is attempted.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
...
n_trials = data.shape[1] // trial_length
```

iii. No separate justification is available; the notes file contains no detailed narrative.

## 9-a. What are the most time-consuming steps of the code?

i. As in 7-a, the likely dominant costs are loading arrays from the HDF5-backed `.mat` files and copying/splitting large per-session arrays into trial lists.

ii.
```python
trace = f[trace_refs[i][0]][:]
position = f[pos_refs[i][0]][:].T
...
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. This is inferred from the implementation; the agent did not document profiling results.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. As in 7-b, `split_into_trials` could be implemented with reshape-based views, and the session-by-session accumulation remains a Python loop.

ii.
```python
def split_into_trials(data, trial_length=TRIAL_LENGTH):
    ...
    return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
...
for i in range(n_recording_sessions):
    ...
for sess_i, sess in enumerate(sessions):
    ...
```

iii. No explicit justification is available.

## 9-c. What processing does the code repeat multiple times?

i. As in 7-c, the code repeatedly splits arrays into trials separately for each modality and repeatedly reuses the same blocked vector for each trial within a session.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
input_trials = [blocked] * len(neural_trials)
```

iii. This is evident from the implementation rather than documented rationale.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. As in 7-d, per-session diagnostic summaries and temporary bookkeeping fields are computed for printing or intermediate assembly but are not preserved in `converted_data.pkl`.

ii.
```python
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
blocked_str = sess['input'][0].astype(int) if n_trials > 0 else []
...
'n_neurons': n_active,
```

iii. No explicit justification is available from the agent artifacts.
