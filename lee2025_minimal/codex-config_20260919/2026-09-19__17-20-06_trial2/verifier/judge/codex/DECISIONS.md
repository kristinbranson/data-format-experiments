# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The submitted script discovers every `.mat` file in the selected data directory, sorts the paths, and opens each MATLAB v7.3/HDF5 file with `h5py`. It reads the `trace`, `position`, and `blocked` reference datasets and dereferences each session inside `process_mat_file`. Full mode is the default; `--sample` limits the result to two sessions.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
for subj_i, mat_file in enumerate(mat_files):
    sessions = process_mat_file(mat_file)
...
f = h5py.File(filepath, 'r')
trace_refs = f['trace']
pos_refs = f['position']
blk_refs = f['blocked'][:][0]
```

iii. The trajectory initially inspected the provided joblib files and said it would use their final rise-extracted traces and aligned positions. It does not justify the submitted switch to duplicate `.mat` files/HDF5. The code comments imply that these files contain the same session data and that HDF5 references are the appropriate loading mechanism.

## 1-b. How are the data split into subjects?

i. One sorted `.mat` file is treated as one mouse. The filename stem is the subject ID, and the file's enumeration index is appended to `subject_idx` for each session.

ii.
```python
name = mat_file.split('/')[-1].replace('.mat', '')
subjects.append(name)
...
subject_idx_list.append(subj_i)
```

iii. The submitted code provides no explicit rationale beyond its one-file-per-subject structure. The trajectory identified the same seven named animals in the source data.

## 1-c. How are the data split into sessions?

i. Every entry in a subject file's `trace` reference array is treated as a recording session. The corresponding trace, position, and blocked-geometry entries at the same index are loaded together, and each session becomes one outer-list element.

ii.
```python
n_recording_sessions = trace_refs.shape[0]
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The trajectory reasoned that a recording day/environment is the natural decoder session and emphasized preserving session boundaries. The HDF5 reference-array entries represent those recording sessions.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 1,800-frame chunks (60 seconds at 30 Hz). Only complete chunks are returned; all trailing frames are discarded.

ii.
```python
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC
...
n_trials = data.shape[1] // trial_length
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. The current docstring justifies this as fixed one-minute trials. This conflicts with the trajectory: after observing slightly different recording lengths, the agent explicitly chose exactly 40 nearly equal `np.array_split` windows to retain every frame and avoid either a short 41st trial or discarded data. That rationale is not implemented by the submitted script.

## 1-e. How are trials filtered based on quality controls?

i. Trials are not quality-filtered. Every complete 1,800-frame chunk is retained, while a final incomplete chunk is dropped solely because of its length.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
input_trials = [blocked] * len(neural_trials)
```

iii. Neither the instructions nor the source analysis specifies a trial-level quality screen. The trajectory also planned to keep all contiguous trials and stationary frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from the raw `trace` session dataset referenced by the MATLAB file's `trace` array.

ii.
```python
trace_refs = f['trace']
...
trace = f[trace_refs[i][0]][:]
```

iii. The trajectory established that `trace` already contains the authors' final, manually curated binary rising-phase calcium-event vectors, so it chose not to reconstruct events from raw fluorescence or video.

## 2-b. How is the `neural` data processed?

i. After absent neurons are removed, traces are cast to `float32` and transposed from time-by-neuron to neuron-by-time. They are otherwise left at native frame resolution and then sliced into trials.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
neural_trials = split_into_trials(trace)
```

iii. The trajectory concluded that the source values are already binary rise-extracted events and that further calcium extraction, interpolation, smoothing, or activity screening would depart from the supplied processing. `float32` was selected for decoder compatibility.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron column is removed only if all its values in the session are NaN. No place-cell, event-count, running-speed, or activity filter is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. The trajectory found that NaNs represent cross-day registered cells absent on that day, and that retaining finite registered cells recovered the paper's 69,744 session-specific maps. It explicitly rejected the paper decoder's `>5` running-event feature screen as model-specific rather than source preprocessing. Unlike the trajectory implementation, the submitted code does not reject partially finite traces or verify the total neuron count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental-event alignment. Time zero is implicitly the start of each artificial contiguous one-minute chunk.

ii.
```python
neural_trials = split_into_trials(trace)
```

iii. The trajectory noted that these are continuous recordings with already frame-aligned neural and position streams and no stimulus event to align to. The submitted metadata omits an explicit `temporal_alignment_event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz samples are retained, giving `1000/30 = 33.333...` ms per time bin. No temporal rebinning or resampling is performed.

ii.
```python
SAMPLING_RATE = 30
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE
...
'time_bin_size': TIME_BIN_SIZE,
```

iii. The trajectory identified simultaneous 30 Hz acquisition and chose to preserve exact framewise neural/behavioral alignment despite the large output size.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry comes from each session's raw `blocked` entry, which lists blocked 3×3 arena-cell indices; `-1` denotes the unobstructed square.

ii.
```python
blk_refs = f['blocked'][:][0]
...
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The trajectory compared blocked IDs against observed positions and concluded that they use the same row-major spatial convention as the desired output.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked IDs are converted into a static nine-element `float32` multi-hot vector, with 1 for blocked and 0 for accessible. The `[-1]` sentinel produces all zeros. The same vector object is placed in every trial of a session.

ii.
```python
blocked = np.zeros(n_positions, dtype=np.float32)
if not (len(blk_indices) == 1 and blk_indices[0] == -1):
    blocked[blk_indices.astype(int)] = 1
...
input_trials = [blocked] * len(neural_trials)
```

iii. The trajectory justified one independent binary feature per arena cell and a static per-session/per-trial representation. It also validated the row-major convention empirically. Sharing the immutable array is an implementation shortcut; the trajectory version instead copied it per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from each session's raw `position` dataset, transposed into a `(2, time)` x/y array.

ii.
```python
position = f[pos_refs[i][0]][:].T
```

iii. The trajectory identified these as the source's frame-aligned DeepLabCut coordinates in the 75×75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The x and y coordinates are independently discretized using fixed 25 cm boundaries at 25 and 50 cm. Bins are clipped to 0–2, then flattened row-major as `y_bin * 3 + x_bin`; the result is stored as a one-row `int8` time series.

ii.
```python
edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
return (y_bin * n_grid + x_bin).astype(np.int8)
...
output = discretize_position(position)[np.newaxis, :]
```

iii. The trajectory tested coordinate conventions against blocked cells and found that `y*3+x` minimized impossible wall occupancy and agrees with the blocked IDs.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Values below 25 cm map to axis bin 0, values from 25 up to 50 cm map to bin 1, and values at or above 50 cm map to bin 2; clipping also assigns boundary overshoots to an edge bin. Combining x/y yields classes 0–8.

ii.
```python
edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The requested output is a 3×3 grid in a 75 cm square, so the agent used equal 25 cm spatial bins. The trajectory specifically noted clipping exact 75 cm values and small overshoots into the last bin.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural arrays are assumed to be frame-aligned in the source and are independently sliced using the same 1,800-frame boundaries. Thus retained output trials align sample-for-sample with retained neural trials.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. The trajectory verified matching source frame dimensions and chose not to interpolate either stream. It emphasized retaining exact correspondence; the submitted code preserves alignment but jointly drops each session's remainder.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Entirely NaN neuron columns are treated as absent registered cells and removed. Incomplete trailing time segments are silently discarded. Position values outside the nominal arena are clipped into edge bins. There are no explicit shape, partial-NaN, binary-value, blocked-position, or empty-session checks, and the HDF5 file is closed only after normal completion.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
...
n_trials = data.shape[1] // trial_length
...
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
```

iii. The trajectory justified removing absent-day cells and clipping small tracking overshoots. It also decided to retain and audit rare positions in blocked bins, reject partially finite traces and structural mismatches, and retain all frames. Those safeguards and the retain-all-frames policy are absent from the submitted code.

## 6-a. What are the most time-consuming steps of the code?

i. Reading the large per-session trace arrays from HDF5 and serializing the resulting full pickle dominate conversion time and I/O. Array casting/transposition and all-NaN scans touch every neural value and are the main computational work. Decoder training is downstream and not part of this converter.

ii.
```python
trace = f[trace_refs[i][0]][:]
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory observed that the native-frame dataset contains billions of neural values and produced an approximately 18.8 GiB pickle, making loading, memory movement, and writing the natural bottlenecks.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The subject/session loops are required for variable neuron counts and referenced datasets. The Python trial-slicing comprehensions could be replaced by reshape/split operations for fixed-length retained data, and summary concatenation could be avoided, but these loops are small (about 40 iterations per session) relative to I/O.

ii.
```python
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
...
for subj_i, mat_file in enumerate(mat_files):
    ...
    for sess_i, sess in enumerate(sessions):
```

iii. The trajectory did not identify a loop-vectorization problem. Its performance concern focused on total native-frame volume and the downstream trainer, not the short trial-list construction loops.

## 6-c. What processing does the code repeat multiple times?

i. `split_into_trials` separately computes the same trial count and boundaries for neural and output data. Per-session diagnostic printing also concatenates all output trials and scans unique labels after the full output was already discretized.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
...
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
```

iii. No justification was recorded for this repetition. It is straightforward and low-cost compared with reading and copying neural arrays, but shared slice boundaries could make the alignment invariant more explicit.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes `output_classes`, converts the geometry to integers for `blocked_str`, and derives display-only trial dimensions for console logging; none are saved or used downstream. It also seeds NumPy although the converter uses no randomness. `--show-processing` is parsed but intentionally has no effect.

ii.
```python
np.random.seed(sum(ord(c) for c in _lee2025_seed) % 2**31)
...
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
blocked_str = sess['input'][0].astype(int) if n_trials > 0 else []
...
parser.add_argument('--show-processing', action='store_true',
                    help='Show processing details (no effect, for testing)')
```

iii. The trajectory did not discuss these remnants. They appear intended for reproducibility conventions, diagnostics, and test-interface compatibility, but they do not affect the converted dataset.
