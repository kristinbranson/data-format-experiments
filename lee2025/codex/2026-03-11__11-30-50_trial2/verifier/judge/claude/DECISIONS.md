# Decisions

**CRITICAL NOTE**: There is a major discrepancy between the AI's `convert_data.py` file and the actual conversion output. The `convert_data.py` in `/app/` is a simple 222-line script using `h5py` to load `.mat` files with no movement filtering, smoothing, or temporal pooling. However, the `conversion_full_out.txt` shows the data was produced by a sophisticated script using `joblib` files with movement filtering, cell activity filtering, Gaussian smoothing, and 3-frame average pooling -- matching the reference preprocessing. The CONVERSION_NOTES also describe this sophisticated pipeline. The current `convert_data.py` was apparently overwritten after the data was produced. Below, I document what the current `convert_data.py` does, while noting what the CONVERSION_NOTES and output files describe.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The current `convert_data.py` loads data from `.mat` files in the `data/` directory using `h5py`. Each `.mat` file is opened as an HDF5 file, and the `trace`, `position`, and `blocked` arrays are accessed via object references. However, the CONVERSION_NOTES and actual conversion output show the data was loaded from joblib files (not `.mat` files) using `joblib.load()`, matching the reference code's approach.

ii.
```python
# Current convert_data.py (h5py approach)
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
f = h5py.File(filepath, 'r')
trace_refs = f['trace']
pos_refs = f['position']
blk_refs = f['blocked'][:][0]
```

iii. The CONVERSION_NOTES state: "Uses joblib animal files from `data/` to match the reference code storage path." The code that actually produced the output used joblib, as shown in `conversion_full_out.txt` (e.g., "Loading /app/data/QLAK-CA1-08" without `.mat` extension). The current `convert_data.py` uses the `.mat` files instead, which contain the same data but in HDF5/MATLAB format.

## 1-b. How are the data split into subjects?

i. Each file in the data directory corresponds to one subject (mouse). The subject name is derived from the filename. In the current `convert_data.py`, `.mat` filenames are used; in the code that actually ran, joblib filenames (without `.mat` extension) were used.

ii.
```python
# Current convert_data.py
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
name = mat_file.split('/')[-1].replace('.mat', '')
subjects.append(name)
```

iii. Both versions agree on subject identification: 7 mice (QLAK-CA1-08, QLAK-CA1-30, QLAK-CA1-50, QLAK-CA1-51, QLAK-CA1-56, QLAK-CA1-74, QLAK-CA1-75). This is documented in CONVERSION_NOTES Step 2.

## 1-c. How are the data split into sessions?

i. Each subject file contains multiple recording sessions. In the current `convert_data.py`, sessions are indexed by iterating over reference arrays in the HDF5 file. In the code that actually ran, sessions correspond to the first axis of `dat['trace']`, `dat['position']`, etc. from the joblib file. Each recording session becomes a separate session in the output.

ii.
```python
# Current convert_data.py
n_recording_sessions = trace_refs.shape[0]
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. CONVERSION_NOTES document 207 total sessions (31 for 6 mice, 21 for QLAK-CA1-51), matching the reference paper. Both versions handle session splitting identically in concept.

## 1-d. How are the data split into trials?

i. The current `convert_data.py` splits continuous sessions into fixed 60-second non-overlapping segments (1800 frames at 30 Hz). Remainder frames are discarded. The code that actually produced the data also splits into 60-second chunks, but within each chunk only movement-valid frames are kept and then temporally pooled, yielding variable-length trials.

ii.
```python
# Current convert_data.py
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC  # 1800
def split_into_trials(data, trial_length=TRIAL_LENGTH):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. CONVERSION_NOTES Step 5: "Split sessions into full one-minute chunks first, but within each chunk keep only movement-valid frames and pool in groups of 3." The actual output confirms variable trial lengths (mean T=313.25, not fixed 1800).

## 1-e. How are trials filtered based on quality controls?

i. The current `convert_data.py` does NOT filter trials -- all complete 60-second chunks are kept. The code that actually ran filtered trials based on: (a) minimum number of movement-valid frames (must have at least POOL_SIZE=3 movement-valid frames), (b) minimum pooled samples per trial (MIN_POOLED_SAMPLES_PER_TRIAL=10), and (c) excluding all-zero neural trials. This reduced the total from 8187 raw chunks to 8109 exported trials.

ii.
```python
# Current convert_data.py -- no trial filtering
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
input_trials = [blocked] * len(neural_trials)
```

iii. CONVERSION_NOTES Step 10: "Converted total trials are 8109 rather than the raw floor-split count 8187 because four low-movement sessions lose chunks during trial-quality curation." The reference code also filters trials with insufficient movement-valid frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains rise-extracted calcium traces (binary event indicators). Both the current `convert_data.py` and the code that ran agree on this.

ii.
```python
# Current convert_data.py
trace = f[trace_refs[i][0]][:]  # (timepoints, neurons)
```

iii. CONVERSION_NOTES Step 1: "trace is already 'rise-extracted calcium traces' where 1 marks a significant event." The reference code also uses `trace` directly.

## 2-b. How is the `neural` data processed?

i. The current `convert_data.py` only transposes from (timepoints, neurons) to (neurons, timepoints) and casts to float32. No smoothing, no temporal pooling. The code that actually ran applies: (1) velocity-based movement filtering, (2) Gaussian smoothing (sigma=3 frames), and (3) non-overlapping 3-frame average pooling, matching the reference decoder's preprocessing.

ii.
```python
# Current convert_data.py -- minimal processing
trace = trace[:, active_mask].astype(np.float32).T  # (n_active, n_timepoints)
```

iii. CONVERSION_NOTES Step 6: "Applies preprocessing modeled on the reference within-session decoder: movement filtering from smoothed speed (5 cm/s threshold), session-level low-activity cell filtering (>5 events during movement-valid frames), Gaussian smoothing of traces, non-overlapping 3-frame average pooling."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The current `convert_data.py` filters only all-NaN neurons (not recorded in that session). The code that actually ran applies two filters: (1) remove neurons with any NaN values (not just all-NaN), and (2) remove neurons with fewer than 5 calcium events during movement-valid frames (CELL_EVENT_THRESHOLD=5), matching the reference decoder's cell filtering.

ii.
```python
# Current convert_data.py -- only all-NaN filter
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. CONVERSION_NOTES Step 1: "The decoder excludes time points with low running speed and cells with too few events during those valid time points." The verification output shows mean 332.67 active cells/session (vs ~337 registered), confirming additional cell filtering beyond NaN removal.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus event to align to. The recording is continuous, and trials are artificial 60-second segments. Neural data is aligned to the start of each one-minute chunk. Both the current code and the code that ran agree on this.

ii. N/A -- no explicit alignment code beyond trial splitting.

iii. CONVERSION_NOTES Step 5: temporal alignment is described as preserving "the simultaneous neural/behavior streams and applies filtering on the same frame axis." The metadata in the output sets `temporal_alignment_event` to "Start of each consecutive one-minute chunk from a continuous recording session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The current `convert_data.py` keeps the native 30 Hz frame rate (time_bin_size = 33.33 ms) with no rebinning. The code that actually ran applies 3-frame average pooling, resulting in time_bin_size = 100.0 ms (with further reduction from movement filtering, yielding variable trial lengths).

ii.
```python
# Current convert_data.py
SAMPLING_RATE = 30  # Hz
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. The verification output confirms time_bin_size of 100.0 ms and variable trial lengths (mean T=313.25), consistent with 3-frame pooling of movement-valid frames. CONVERSION_NOTES Step 5: "average-pool every 3 frames."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Both versions derive the geometry input from the `blocked` variable in the raw data, which lists which of the 9 possible reward positions in the 3x3 grid are blocked for each session.

ii.
```python
# Current convert_data.py
blk_refs = f['blocked'][:][0]
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. CONVERSION_NOTES Step 5: "Use raw blocked partitions to build decoder inputs."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The current `convert_data.py` converts blocked indices to a one-hot encoding where 1=blocked, 0=open. The code that actually ran and the reference use the OPPOSITE convention: 1=open, 0=blocked. Additionally, the code that ran infers a coordinate transform per animal to align position data with the geometry labels.

ii.
```python
# Current convert_data.py -- INVERTED encoding (1=blocked)
def encode_blocked(blk_indices, n_positions=N_BLOCKED_POSITIONS):
    blocked = np.zeros(n_positions, dtype=np.float32)
    if not (len(blk_indices) == 1 and blk_indices[0] == -1):
        blocked[blk_indices.astype(int)] = 1
    return blocked
```

iii. CONVERSION_NOTES Step 5: "Encode geometry as 9 binary partition features: 1 = open, 0 = blocked." The verification output confirms input_names are `partition_X_open`, indicating the open=1 convention. The current convert_data.py uses the inverted convention.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry input is static per session (same vector for all trials and timepoints within a session). Both versions agree on this. There is no temporal alignment needed since it is a per-trial constant.

ii.
```python
# Current convert_data.py
input_trials = [blocked] * len(neural_trials)
```

iii. CONVERSION_NOTES: "blocked vector is constant across all timepoints and trials within a session."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Both versions derive position from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena at each timepoint.

ii.
```python
# Current convert_data.py
position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
```

iii. CONVERSION_NOTES Step 5: position is derived from `position[session, :, frame]`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The current `convert_data.py` discretizes the 2D position into a 3x3 grid using `np.linspace` edges and `np.digitize`, yielding 9 classes. It does NOT apply any coordinate transform or snap blocked positions. The code that actually ran: (1) computes bin assignments using `np.floor(pos/bin_size)`, (2) applies a per-animal coordinate transform to align bins with geometry labels, (3) snaps positions in blocked bins to the nearest open bin, and (4) applies 3-frame average pooling to position coordinates.

ii.
```python
# Current convert_data.py
def discretize_position(position, n_grid=N_GRID, arena_size=ARENA_SIZE):
    edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
    x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
    y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
    return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. CONVERSION_NOTES Step 5: "Snap blocked-bin position samples to the nearest open partition before export" and "Derive 3x3 position bins from the same arena scale used by the reference decoder logic." The reference code uses `np.floor(pos/bin_size)` for discretization.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The current `convert_data.py` directly assigns the 2D position into one of 9 spatial bins (3x3 grid) using `np.digitize` with bin edges at 1/3 and 2/3 of the 75 cm arena size. The result is a single categorical variable with values 0-8 (y_bin * 3 + x_bin). The code that actually ran uses `np.floor(pos/bin_size)` for discretization and also applies snapping to avoid labels in blocked bins.

ii.
```python
# Current convert_data.py
edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]  # [25.0, 50.0]
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The reference code uses `np.floor(pos / bin_size)` which produces equivalent 3x3 bins. Both approaches produce 9 categories in row-major order.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. In the current `convert_data.py`, position and neural data are at the same 30 Hz frame rate and are split into trials using identical indexing, ensuring alignment. In the code that actually ran, both neural and position data undergo the same movement filtering (keeping only frames where velocity > 5 cm/s) and 3-frame average pooling, maintaining alignment.

ii.
```python
# Current convert_data.py
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. CONVERSION_NOTES: "Both arrays have the same number of timepoints and are sliced identically, ensuring alignment."

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The current `convert_data.py` keeps data at 30 Hz (33.33 ms bins) with no rebinning. The code that actually ran applies 3-frame average pooling, yielding 100 ms bins (10 Hz effective rate), matching the reference decoder's temporal resolution.

ii.
```python
# Current convert_data.py
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE  # ~33.33 ms
```

iii. CONVERSION_NOTES Step 5: "average-pool every 3 frames." The actual metadata in the output confirms `time_bin_size: 100.0`.

## 5-b. How are the neural, input, and output data temporally aligned?

i. In the current `convert_data.py`, neural and position data are at the same 30 Hz rate and split with identical indices. Input (geometry) is static per session. In the code that actually ran, neural and position data undergo the same movement filtering and pooling operations before trial splitting, ensuring frame-by-frame alignment at the pooled level.

ii.
```python
# Current convert_data.py
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
input_trials = [blocked] * len(neural_trials)
```

iii. CONVERSION_NOTES Step 10: "Temporal alignment: preserves the simultaneous neural/behavior streams and applies filtering on the same frame axis."

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The current `convert_data.py` handles NaN neurons by filtering all-NaN columns and discards remainder frames that don't fill a complete trial. The code that actually ran additionally: handles the `blocked=[-1]` sentinel for "no positions blocked," filters low-activity neurons, drops trials with too few movement-valid frames, and snaps positions landing in blocked bins to nearest open bin.

ii.
```python
# Current convert_data.py
active_mask = ~np.all(np.isnan(trace), axis=0)
# Remainder discarded implicitly: n_trials = data.shape[1] // trial_length
```

iii. CONVERSION_NOTES Step 10: documents handling of edge cases including sessions with unusually low movement-valid fractions and the all-zero neural trial warning.

## 7-a. What are the most time-consuming steps of the code?

i. The CONVERSION_NOTES identify data loading as the bottleneck. Loading animal joblib files takes 20-112 seconds per animal. Processing each session within an already-loaded animal is fast (~1.67 s/session).

ii. N/A (timing information from CONVERSION_NOTES, not from current code).

iii. CONVERSION_NOTES Step 7: "Sample conversion (measured): ~1.67 s/session after first animal load; first animal load ~62.46 s." Total conversion took 675 seconds for all 207 sessions.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The current `convert_data.py` uses a list comprehension for trial splitting which could be replaced with array reshaping. The reference code's `snap_to_open_bins` function uses a Python loop over individual frames to snap blocked positions, which could be vectorized.

ii.
```python
# Current convert_data.py -- list comprehension for trial splitting
return [data[:, i * trial_length:(i + 1) * trial_length] for i in range(n_trials)]
```

iii. The CONVERSION_NOTES note that "heavy preprocessing" was moved to vectorized NumPy/SciPy operations. However, the reference code's `snap_to_open_bins` still loops per-frame.

## 7-c. What processing does the code repeat multiple times?

i. The current `convert_data.py` recomputes the blocked encoding for each session but then copies it for all trials (which is redundant but cheap). More significantly, in the code that actually ran, the coordinate transform inference is done once per animal rather than per session, which is a noted optimization.

ii.
```python
# Current convert_data.py
input_trials = [blocked] * len(neural_trials)  # same vector repeated
```

iii. CONVERSION_NOTES Step 6: "Performs one transform inference per animal instead of per session."

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The current `convert_data.py` does minimal processing so there is nothing obviously unnecessary. The code that actually ran computes occupancy matrices for plotting/debugging that are not needed for the final output.

ii. N/A

iii. No significant unnecessary processing is documented in CONVERSION_NOTES.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. NaN neurons are filtered, remainder frames are discarded, and the `blocked=[-1]` sentinel is handled. The code that actually ran also handles edge cases with low-movement sessions and snaps positions in blocked bins.

ii.
```python
# Current convert_data.py
active_mask = ~np.all(np.isnan(trace), axis=0)
if not (len(blk_indices) == 1 and blk_indices[0] == -1):
    blocked[blk_indices.astype(int)] = 1
```

iii. CONVERSION_NOTES Step 10 documents edge case handling.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Data loading (joblib files) is the bottleneck at 20-112 seconds per animal.

ii. N/A

iii. CONVERSION_NOTES Step 7 timing estimates.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The `snap_to_open_bins` function in the reference-style code iterates per frame. Trial splitting uses list comprehensions.

ii. See 7-b.

iii. See 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. Coordinate transform inference is shared across sessions within an animal. Blocked encoding is repeated per trial but is a trivial copy.

ii. See 7-c.

iii. See 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Occupancy matrix computation for debugging/plotting is not needed for final output. The code that actually ran stores extensive session metadata that may not all be used downstream.

ii. N/A

iii. See 7-d.
