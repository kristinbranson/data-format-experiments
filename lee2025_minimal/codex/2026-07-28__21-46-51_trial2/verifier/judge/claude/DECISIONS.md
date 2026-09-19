# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a `.mat` file in the `/app/data/` directory. The `.mat` files are MATLAB v7.3+ HDF5 format and are loaded using `h5py`. Each file contains reference arrays (`trace`, `position`, `blocked`) that point to per-session data. Files are discovered via `glob.glob` and sorted alphabetically.

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

iii. The agent's trajectory shows it explored the data directory, identified both `.mat` and joblib-format files, and investigated data structures. The agent's trajectory indicates it originally wrote code using `joblib.load()` on the joblib-format files (without `.mat` extension), but the final `convert_data.py` uses `h5py` on `.mat` files. The agent recognized that `.mat` files use HDF5 format requiring `h5py`.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject. The subject name is extracted from the filename by stripping the `.mat` extension.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
name = mat_file.split('/')[-1].replace('.mat', '')
subjects.append(name)
```

iii. The agent identified that each file in the data directory corresponds to one animal (mouse). This is consistent with how both the `.mat` and joblib files are organized.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording sessions, indexed by iterating over the HDF5 reference arrays. The first dimension of the `trace` reference array gives the number of sessions. Each reference is dereferenced to get per-session neural and behavioral data.

ii.
```python
n_recording_sessions = trace_refs.shape[0]
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]     # (timepoints, neurons)
    position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The agent confirmed from the paper and data that each recording day is treated as one session, consistent with the paper's experimental design of one session per day.

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into 60-second non-overlapping segments (1800 frames at 30 Hz). Remainder frames that don't fill a complete trial are discarded.

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

iii. The agent followed the instructions specifying "1-minute trials" and confirmed sessions are ~40 minutes long, yielding 39-40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are kept. The only filtering is implicit: incomplete trailing segments are discarded.

ii. N/A

iii. The agent did not apply trial-level filtering. The trajectory and CONVERSION_NOTES.md confirm no additional quality controls beyond discarding incomplete trailing segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the `.mat` file, which contains calcium imaging traces with shape `(timepoints, neurons)` per session.

ii.
```python
trace = f[trace_refs[i][0]][:]  # (timepoints, neurons)
```

iii. The agent identified `trace` as the neural data source. The trajectory and CONVERSION_NOTES.md describe these as "binary rise-event traces" (binarized rising phase of calcium transients), though the code treats them generically without explicitly noting the binary nature.

## 2-b. How is the `neural` data processed?

i. The trace data is filtered to remove inactive neurons (all-NaN), transposed from `(timepoints, neurons)` to `(neurons, timepoints)`, and cast to float32. No additional smoothing, deconvolution, or normalization is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T  # (n_active, n_timepoints)
```

iii. The agent recognized that the traces are already preprocessed (binarized rising-phase events) by the original authors, so no further signal processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are entirely NaN across all timepoints in a given session are removed. Only neurons with at least one valid (non-NaN) value are retained.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. The agent identified that unregistered neurons for a given day appear as all-NaN columns. No additional place-cell or activity-threshold filtering was applied. The CONVERSION_NOTES state: "No additional place-cell or activity-threshold filtering was applied to the exported neural arrays."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of each session.

ii. N/A (trials are simply contiguous windows)

iii. There is no stimulus onset or behavioral event to align to. The alignment is simply the start of each 60-second window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. The agent confirmed the 30 Hz sampling rate from both the paper methods and the data files. No rebinning was needed since the native rate is consistent.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` variable in the `.mat` file, which contains indices of blocked reward positions for each recording session.

ii.
```python
blk_refs = f['blocked'][:][0]
...
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The agent identified `blocked` as the source for geometry information. The trajectory shows the agent discovered that environment names alone don't fully specify geometry orientation, and that `blocked` provides authoritative per-session geometry information.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a one-hot encoding over 9 possible positions. The encoding uses 1 to mark blocked positions and 0 for open positions. If no positions are blocked (indicated by `[-1]`), the vector is all zeros. The blocked vector is static (constant across all timepoints and trials within a session).

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

iii. The agent chose a one-hot encoding where 1 indicates a blocked position. Note: this is the **opposite polarity** from the agent's trajectory and CONVERSION_NOTES.md, which describe an "open-bin mask" where 1=open and 0=blocked. The code in `convert_data.py` implements 1=blocked, 0=open.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the `.mat` file, which contains 2D coordinates `(x, y)` of the animal tracked via DeepLabCut pose estimation.

ii.
```python
position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
```

iii. The agent identified the `position` variable as the source of spatial data, consistent with the paper's description of DeepLabCut-based head tracking.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). The arena is assumed to be 75cm, and bin edges are computed using `np.linspace(0, 75, 4)[1:-1]` = [25, 50]. Positions are binned using `np.digitize` and clipped to valid range [0, 2].

ii.
```python
def discretize_position(position, n_grid=N_GRID, arena_size=ARENA_SIZE):
    edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
    x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
    y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
    return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The agent assumed a fixed 75cm arena size based on the paper's description ("75 cm x 75 cm"). However, the agent's trajectory and CONVERSION_NOTES.md actually describe using per-axis max normalization (`floor(position / (max/3))`), which is inconsistent with the code's fixed-size approach. The code at `convert_data.py` does not match the described decision.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The grid label is computed as `y_bin * 3 + x_bin`, giving values 0-8. No positions in blocked bins are reassigned.

ii.
```python
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The code uses `y_bin * 3 + x_bin` ordering. Note: the agent's CONVERSION_NOTES.md describes the label definition as `x_bin * 3 + y_bin`, which is the opposite convention. Furthermore, the CONVERSION_NOTES describe a "coarse-bin cleanup" that reassigns positions in blocked bins to the nearest valid bin, but this cleanup is absent from the code.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are sampled at the same 30 Hz frame rate in the `.mat` file, so they are frame-aligned. Both are split into trials using the same `split_into_trials` function with identical indices.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. Both arrays share the same number of timepoints and are sliced identically, ensuring temporal alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with all-NaN values (unregistered in that session) are removed before any further processing. Trailing frames that don't fill a complete 60-second trial are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
...
n_trials = data.shape[1] // trial_length  # remainder implicitly dropped
```

iii. NaN filtering ensures only actually recorded neurons are included. The agent noted this aligns with the paper's approach where neurons are tracked across sessions but may not be present on all days.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the `.mat` files via `h5py` and dereferencing the HDF5 object references, which is I/O-bound due to the large neural trace arrays.

ii. N/A

iii. The agent's trajectory confirms that data loading took substantial time (~2 minutes per full conversion run), while the actual processing (filtering, discretization, trial splitting) was fast by comparison.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `split_into_trials` function uses a Python list comprehension to create individual trial arrays, which could be replaced with `np.split` or array reshape operations. The per-session loop could potentially be parallelized across subjects.

ii.
```python
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. N/A - the agent did not discuss vectorization in the trajectory.

## 6-c. What processing does the code repeat multiple times?

i. The `encode_blocked` function is called once per session and the result is replicated for each trial (`[blocked] * len(neural_trials)`), which is efficient. The code does not significantly repeat processing.

ii.
```python
input_trials = [blocked] * len(neural_trials)
```

iii. N/A - no significant repeated processing noted.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No obviously unnecessary processing. The code is relatively minimal. The `--sample` mode flag and related logic is unused in the full conversion run but is harmless.

ii. N/A

iii. N/A
