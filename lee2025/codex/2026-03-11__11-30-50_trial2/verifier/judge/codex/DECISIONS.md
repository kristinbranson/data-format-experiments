# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The implemented code loads all subjects by globbing `*.mat` files in `data/`, opens each file with `h5py`, and reads per-session references from `trace`, `position`, and `blocked`. Trials are not loaded directly; they are derived later by splitting each session into fixed 1-minute chunks.

ii. ```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
f = h5py.File(filepath, 'r')
trace_refs = f['trace']
pos_refs = f['position']
blk_refs = f['blocked'][:][0]
...
trace = f[trace_refs[i][0]][:]
position = f[pos_refs[i][0]][:].T
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The code docstring and inline comments justify this as a direct conversion from MATLAB/HDF5 session arrays. `CONVERSION_NOTES.md` and the trajectory mostly justify a different, more elaborate joblib-based plan, not the shipped code.

## 1-b. How are the data split into subjects?

i. Each `.mat` file is treated as one subject, and the subject name is the filename stem.

ii. ```python
for subj_i, mat_file in enumerate(mat_files):
    name = mat_file.split('/')[-1].replace('.mat', '')
    subjects.append(name)
```

iii. The implemented rationale is that one file corresponds to one mouse. This matches the code comments and `reference_DECISIONS.md`; the notes file does not materially justify this specific final implementation.

## 1-c. How are the data split into sessions?

i. Within each subject file, the code iterates over rows of the `trace` reference array; each row is one recording session, and corresponding `position` and `blocked` entries are pulled with the same index.

ii. ```python
n_recording_sessions = trace_refs.shape[0]

for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The implemented justification is that the HDF5 reference arrays are organized one entry per recording session. The notes/trajectory instead describe a joblib session-first layout, which was not used in the final code.

## 1-d. How are the data split into trials?

i. Sessions are split into non-overlapping 60 s trials at 30 Hz, so each trial is 1800 frames. Any remainder shorter than 1800 frames is dropped.

ii. ```python
TRIAL_DURATION_SEC = 60
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC  # 1800

def split_into_trials(data, trial_length=TRIAL_LENGTH):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. The code comments justify this directly from the task requirement that long sessions be split into 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. They are effectively not filtered beyond dropping incomplete trailing chunks. Zero-trial sessions can remain; there is no trial-quality rejection based on behavior or signal quality.

ii. ```python
n_trials = data.shape[1] // trial_length
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. No explicit justification is given in the final code beyond using full-length chunks only. The notes/trajectory discuss stricter curation, but that logic was not implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` variable for each session.

ii. ```python
trace = f[trace_refs[i][0]][:]  # (timepoints, neurons)
```

iii. The code comments state that `trace` contains session neural activity. The notes additionally say these are already preprocessed calcium-event traces.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: remove all-NaN neuron columns, cast to `float32`, and transpose from `(timepoints, neurons)` to `(neurons, timepoints)`. Then split into trials.

ii. ```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
neural_trials = split_into_trials(trace)
```

iii. The shipped code’s rationale is that the traces are already in a usable processed form and only need reshaping. The notes explicitly claimed a more complex smoothing/pooling pipeline, which is inconsistent with the final implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons that are all NaN within a session are removed. There is no activity threshold, movement filter, or place-cell filter.

ii. ```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. The code comment justifies this as keeping only neurons actually recorded in that session. The notes/trajectory justify additional activity filtering, but that was not implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. Trials are artificial contiguous chunks of a continuous recording.

ii. ```python
neural_trials = split_into_trials(trace)
```

iii. The implicit justification is that the dataset contains continuous sessions rather than event-locked trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data remain at the native 30 Hz sampling rate, so each bin is about 33.33 ms. No temporal rebinning is applied.

ii. ```python
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. The code constants and metadata encode the rationale: keep the original frame rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The implemented input is derived from the raw `blocked` variable, not `envs`.

ii. ```python
blk_refs = f['blocked'][:][0]
...
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The code and `reference_DECISIONS.md` justify this as directly encoding blocked positions per session. The notes also favored `blocked`, though for a different abandoned pipeline.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted into a length-9 one-hot vector with `1` at blocked positions. `[-1]` means no blocked positions, producing all zeros.

ii. ```python
def encode_blocked(blk_indices, n_positions=N_BLOCKED_POSITIONS):
    blocked = np.zeros(n_positions, dtype=np.float32)
    if not (len(blk_indices) == 1 and blk_indices[0] == -1):
        blocked[blk_indices.astype(int)] = 1
    return blocked
```

iii. The final code treats each blocked partition as an independent binary feature. This is explicitly documented in the function docstring.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. It is session-static. The same length-9 geometry vector is attached to every trial in the session without time variation.

ii. ```python
blocked = encode_blocked(blk_indices)
...
input_trials = [blocked] * len(neural_trials)
```

iii. The code assumes geometry does not change within a session, so no frame-by-frame temporal alignment is needed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw `position` variable for each session.

ii. ```python
position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
```

iii. The code comments identify this as the 2D mouse trajectory used to form the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid over a fixed `75 x 75` arena. The categorical label is `y_bin * 3 + x_bin`, then the 1D label stream is wrapped as shape `(1, n_timepoints)`.

ii. ```python
def discretize_position(position, n_grid=N_GRID, arena_size=ARENA_SIZE):
    edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
    x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
    y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
    return (y_bin * n_grid + x_bin).astype(np.int8)
...
output = discretize_position(position)[np.newaxis, :]
```

iii. The function docstring and constants justify this as a simple 9-class spatial discretization consistent with the requested decoder task.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded by equal-width bin edges over `[0, 75]`, creating 3 bins per axis and therefore 9 categories total. Values are clipped to valid edge bins.

ii. ```python
edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The explicit justification is the decoder requirement for categorical outputs on a 3x3 spatial grid.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The code assumes framewise alignment because `trace` and `position` are session time series with matched sample counts. Both are split with the same trial boundaries.

ii. ```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. The final code relies on identical slicing for temporal alignment. No interpolation or resampling is performed.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is still 30 Hz, or about 33.33 ms per sample. No rebinning or pooling is applied.

ii. ```python
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. This is the same decision reflected in metadata and in the absence of any resampling code.

## 5-b. How are the neural, input, and output data temporally aligned?

i. `neural` and `output` are frame-aligned by applying the same session-to-trial slicing. `input` is a static per-trial vector copied across all trials in a session.

ii. ```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
input_trials = [blocked] * len(neural_trials)
```

iii. The code’s alignment model is simple: one shared session clock for neural and position, with session-constant geometry.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Two minor issues are handled explicitly: all-NaN neurons are dropped, and `blocked == [-1]` is treated as no blocked positions. Incomplete trailing frames are discarded when trials are formed.

ii. ```python
active_mask = ~np.all(np.isnan(trace), axis=0)
...
if not (len(blk_indices) == 1 and blk_indices[0] == -1):
    blocked[blk_indices.astype(int)] = 1
...
n_trials = data.shape[1] // trial_length
```

iii. The code’s implicit rationale is to remove clearly invalid session-specific neuron entries and normalize the special no-blocked sentinel.

## 7-a. What are the most time-consuming steps of the code?

i. The likely bottleneck is opening each HDF5 `.mat` file and reading the large `trace` and `position` arrays. The rest of the processing is light NumPy slicing.

ii. ```python
f = h5py.File(filepath, 'r')
...
trace = f[trace_refs[i][0]][:]
position = f[pos_refs[i][0]][:].T
```

iii. There is no explicit performance discussion in the final code. The notes discuss joblib I/O as the main cost in a different pipeline; by analogy, the implemented HDF5 I/O is the dominant cost here.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python list construction in `split_into_trials` could be replaced with reshaping or strided views for full-trial chunks. The subject/session loops are structurally necessary.

ii. ```python
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
...
for subj_i, mat_file in enumerate(mat_files):
...
    for sess_i, sess in enumerate(sessions):
```

iii. No explicit justification is given. This is an implementation-level observation from the final code.

## 7-c. What processing does the code repeat multiple times?

i. It repeatedly reads per-session `trace`, `position`, and `blocked` datasets and repeatedly slices arrays into trials for each session. There is no caching.

ii. ```python
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
...
    neural_trials = split_into_trials(trace)
    output_trials = split_into_trials(output)
```

iii. No explicit justification is documented; this follows directly from a simple per-session conversion strategy.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes verbose per-session summary values only for printing, such as unique output classes and blocked strings; they are not saved in the pickle.

ii. ```python
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
blocked_str = sess['input'][0].astype(int) if n_trials > 0 else []
print(f"    session {sess_i}: {n_neurons} neurons, {n_trials} trials, "
      f"{trial_len} timepoints/trial, "
      f"neural dtype={sess['neural'][0].dtype}, "
      f"output classes={output_classes.astype(int)}, "
      f"blocked={blocked_str}")
```

iii. This is not justified beyond diagnostic logging.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: all-NaN neurons are removed, `[-1]` in `blocked` means no blocked partitions, and incomplete end-of-session fragments are dropped.

ii. ```python
active_mask = ~np.all(np.isnan(trace), axis=0)
...
if not (len(blk_indices) == 1 and blk_indices[0] == -1):
    blocked[blk_indices.astype(int)] = 1
...
n_trials = data.shape[1] // trial_length
```

iii. The shipped code uses only these minimal checks; no broader malformed-data recovery is implemented.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: HDF5 I/O for per-session neural and position arrays is likely the dominant runtime cost.

ii. ```python
f = h5py.File(filepath, 'r')
trace = f[trace_refs[i][0]][:]
position = f[pos_refs[i][0]][:].T
```

iii. No explicit final-code justification is given.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the trial-splitting list comprehensions are the clearest vectorization opportunity.

ii. ```python
return [data[i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. No explicit justification is given.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: repeated per-session dataset reads and repeated trial slicing.

ii. ```python
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
...
    neural_trials = split_into_trials(trace)
    output_trials = split_into_trials(output)
```

iii. No explicit justification is given.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: diagnostic class summaries and blocked-vector prints are computed only for stdout.

ii. ```python
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
blocked_str = sess['input'][0].astype(int) if n_trials > 0 else []
```

iii. This appears to be purely for human-readable logging during conversion.
