# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files in the `data/` directory. Each animal has a separate joblib file (e.g., `data/QLAK-CA1-08`). The file is loaded with `joblib.load(animal_path)[animal]`, which returns a dictionary containing fields `trace`, `position`, `blocked`, `envs`, etc. Animals are discovered by listing the `data/` directory and filtering for non-`.mat`, non-hidden files. The data is loaded one animal at a time to manage memory, with `gc.collect()` called after each animal is processed.

ii.
```python
def get_animal_files(data_dir):
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if (
            os.path.isfile(path)
            and not name.endswith(".mat")
            and name not in {"behav_dict"}
            and not name.startswith(".")
        ):
            animals.append(name)
    return animals

# In main():
dat = joblib.load(animal_path)[animal]
```

iii. The AI chose joblib files as the primary source because they "expose the same field names as the reference code and already match the analysis-ready axis ordering, reducing the risk of silent transpose errors relative to the MATLAB HDF5 layout." This matches the reference code's `load_dat` function which also uses joblib.

## 1-b. How are the data split into subjects (mice)?

i. Each joblib file in `data/` corresponds to one mouse/subject. The list of all animal file names becomes the `subjects` list. The `subject_idx` array maps each session to its animal's index in this list.

ii.
```python
# In main():
selected_animals = animals  # list of animal file names
# ...
for animal in selected_animals:
    # ...
    subject_idx.append(animals.index(animal))

data = {
    "subjects": animals,
    "subject_idx": np.array(subject_idx, dtype=np.int64),
}
```

iii. The AI noted that `main.py` in the reference code lists 7 animals, and the data directory contains 7 joblib files. This is consistent with the paper stating "5,413 unique neurons across 207 sessions."

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple sessions indexed along the first axis of arrays like `trace` (shape: `(n_sessions, n_cells, n_frames)`) and `position` (shape: `(n_sessions, 2, n_frames)`). The code iterates over all sessions within each animal using `for session_idx in range(dat["position"].shape[0])`.

ii.
```python
for session_idx in range(dat["position"].shape[0]):
    session_id = f"{animal}_s{session_idx:02d}"
    env_name = str(dat["envs"][session_idx, 0])
    session_neural, session_input, session_output, plot_payload, trial_lengths, n_finite, n_active, velocity_mask = preprocess_session(
        trace_session=dat["trace"][session_idx],
        position_session=dat["position"][session_idx],
        blocked_entry=dat["blocked"][session_idx],
        # ...
    )
```

iii. The AI documented that 6 animals have 31 sessions and 1 animal (QLAK-CA1-51) has 21 sessions, totaling 207 sessions. This matches the paper's "207 sessions."

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into consecutive 1-minute (1800-frame) chunks. The number of full trials per session is computed as `trace.shape[1] // RAW_TRIAL_FRAMES` where `RAW_TRIAL_FRAMES = 30 * 60 = 1800`. Remaining frames at the end of a session are discarded.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800

n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
```

iii. The AI noted "There are no native trial objects in the raw dataset. For the requested decoder format, one-minute trials will need to be derived by splitting each session time series into consecutive 1800-frame chunks." This follows the instruction to split into "1-minute trials within each session."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on three criteria: (1) the number of movement-valid frames within the chunk must be at least `POOL_SIZE` (3), (2) the number of pooled time bins must be at least `MIN_POOLED_SAMPLES_PER_TRIAL` (10), and (3) the pooled neural trace must not be all zeros. Sessions with fewer than 2 valid trials after filtering raise an error.

ii.
```python
chunk_mask = velocity_mask[start:end]
if int(chunk_mask.sum()) < POOL_SIZE:
    continue

# ...
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue

if not np.any(pooled_trace):
    continue

# After all trials:
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")
```

iii. The AI documented that these filters reduced the total trial count from 8187 (raw floor-split count) to 8109 exported trials. Four sessions with low movement-valid fractions (0.217 to 0.351) had the most trial loss.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field of each animal's joblib file. The `trace` array has shape `(n_sessions, n_cells, n_frames)` and contains binary rising-phase calcium event traces (0s and 1s).

ii.
```python
trace_session=dat["trace"][session_idx]  # shape: (n_cells, n_frames)
```

iii. The AI noted: "Neural data are not raw fluorescence and no delta-F/F computation appears in the reference code. The README states `trace` is already 'rise-extracted calcium traces' where `1` marks a significant event." This matches the paper's description of the binarized rising-phase vector.

## 2-b. How is the `neural` data processed?

i. Processing follows several steps matching the reference decoder code:
1. Remove cells with any NaN values (non-registered cells for that session).
2. Filter to only cells with >5 events during movement-valid frames.
3. Apply Gaussian smoothing (sigma=3 frames) to the traces within each trial chunk.
4. Apply non-overlapping 3-frame average pooling.

ii.
```python
TRACE_SMOOTH_SIGMA = 3
POOL_SIZE = 3

# NaN filtering
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)

# Activity filtering
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]

# Per-trial: extract movement-valid frames, smooth, pool
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. The AI justified this by referencing the `fit_decoder` and `decode_position_within` functions in the reference code, which apply Gaussian smoothing with `sigma=temporal_bin_size` (default 3) and average pooling with kernel size 3.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage cell filtering is applied per session:
1. NaN filtering: cells with any NaN values in the session are removed.
2. Activity threshold: cells must have >5 total events during movement-valid frames.

ii.
```python
CELL_EVENT_THRESHOLD = 5

finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)

activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
```

iii. The AI noted this matches the reference `decode_position_within` code: `cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold` with `cell_threshold=5`. The paper also states high reliability "motivated the inclusion of all cells in subsequent analyses," confirming no place-cell filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 1-minute chunk. The neural and position data share the same frame axis (both recorded at 30 Hz), so alignment is maintained by indexing both streams with the same frame indices. Within each trial, only movement-valid frames are kept, and both neural and position data use the same velocity mask.

ii.
```python
# Same frame indices used for neural and position:
start = trial_idx * RAW_TRIAL_FRAMES
end = start + RAW_TRIAL_FRAMES
chunk_mask = velocity_mask[start:end]

chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
```

iii. The AI set `temporal_alignment_event` to "Start of each consecutive one-minute chunk from a continuous recording session" with `off_start=0.0` and `off_end=60.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native frame rate is 30 Hz (33.33 ms per frame). After movement filtering (which removes stationary frames), the remaining frames are pooled in non-overlapping groups of 3, resulting in an effective time bin of 100 ms (3 frames x 33.33 ms). The `time_bin_size` metadata is set to 100.0 ms.

ii.
```python
FPS = 30
POOL_SIZE = 3

def trial_average_pool(values, pool_size=POOL_SIZE):
    n_full = values.shape[-1] // pool_size
    trimmed = values[..., : n_full * pool_size]
    new_shape = values.shape[:-1] + (n_full, pool_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

# metadata:
"time_bin_size": 100.0,
```

iii. The AI noted this matches the reference code's `fit_decoder` which uses `temporal_bin_size=3` with `AvgPool1d(kernel_size=3, stride=3)`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the `blocked` field in each animal's data. Each session has a `blocked` entry that lists the indices (in a 3x3 grid) of blocked/inaccessible partitions.

ii.
```python
blocked_entry=dat["blocked"][session_idx]

def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
```

iii. The AI decided to use the raw `blocked` field rather than the `envs` labels with `get_env_mat()` because "it directly represents blocked partitions in the raw dataset" and avoids ambiguities from coordinate orientation conventions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The `blocked` indices are parsed into a sorted integer array. A length-9 binary vector is created (1=open, 0=blocked). Additionally, the AI infers a per-animal coordinate transform by testing 8 candidate transforms (identity, rotations, flips, transposes) and selecting the one that minimizes occupancy in blocked partitions. This transform is applied to align position data with the blocked-partition definitions.

ii.
```python
def parse_blocked_indices(blocked_entry):
    if isinstance(blocked_entry, list):
        if len(blocked_entry) == 0:
            return np.array([], dtype=np.int64)
        blocked_entry = blocked_entry[0]
    arr = np.array(blocked_entry, dtype=np.float64).reshape(-1)
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
    return np.sort(arr.astype(np.int64))

def infer_transform(position_all, blocked_all, scale_cm):
    # Tests 8 transforms, selects one minimizing blocked-partition occupancy
    # ...

# Input stored as static per-trial vector:
input_trials.append(geometry_open.copy())
```

iii. The AI found that the raw `blocked` indices and the canonical `get_env_mat` matrices use different coordinate conventions. Rather than hardcoding a single transform, the AI empirically infers the correct transform per animal. The input is stored as a static (non-time-varying) 9-element vector per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field, which has shape `(n_sessions, 2, n_frames)` containing x-y coordinates of the mouse's head position tracked via DeepLabCut.

ii.
```python
position_session=dat["position"][session_idx]  # shape: (2, n_frames)
```

iii. The AI confirmed this matches the reference code which uses the same `position` field for computing rate maps and decoding.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position processing involves:
1. Compute spatial scale as per-animal maximum position value: `scale_cm = float(np.nanmax(dat["position"]))`.
2. Bin x-y coordinates into a 3x3 grid: `bin_size_cm = (scale_cm + 1e-6) / 3`.
3. Apply the inferred coordinate transform to map raw grid indices to the geometry-aligned coordinate system.
4. Snap positions that fall in blocked bins to the nearest open bin.
5. Filter to movement-valid frames using velocity mask.
6. Average-pool position indices in groups of 3 (matching neural pooling).
7. Collapse to a single categorical output (0-8) representing the 3x3 bin index.

ii.
```python
def raw_position_to_rc(position_xy, bin_size_cm):
    x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
    y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
    x = np.clip(x, 0, GEOMETRY_SIZE - 1)
    y = np.clip(y, 0, GEOMETRY_SIZE - 1)
    return y, x

# Transform and snap:
raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
raw_ids_all = raw_rows_all * GEOMETRY_SIZE + raw_cols_all
mapped_ids_all = coord_map[raw_ids_all]
mapped_rows_all = mapped_ids_all // GEOMETRY_SIZE
mapped_cols_all = mapped_ids_all % GEOMETRY_SIZE
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
```

iii. The AI noted that the spatial scale uses `np.nanmax(dat["position"])` across all sessions for an animal, matching the reference code's `behav_max = behav.max(axis=0).max(axis=1)` which computes max across all days.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (0-8) by binning x-y coordinates into a 3x3 grid. The bin index is computed as `row * 3 + col` in row-major order. After pooling, positions are clipped to [0, 2] per dimension and any positions in blocked bins are snapped to the nearest open bin.

ii.
```python
# After pooling:
pooled_rows = np.clip(pooled_rows, 0, GEOMETRY_SIZE - 1)
pooled_cols = np.clip(pooled_cols, 0, GEOMETRY_SIZE - 1)
invalid = geometry_mat[pooled_rows, pooled_cols] == 0
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. The instructions specified "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI implemented this directly, with output values labeled `bin_0_r0c0` through `bin_8_r2c2`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by using the same frame indices. Both streams are originally at 30 Hz on the same frame axis. Within each trial chunk, the same velocity mask is applied to both, and the same 3-frame average pooling is applied. The position is pooled using `np.floor` of the averaged coordinates cast to int.

ii.
```python
# Same chunk_mask applied to both:
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]

# Same pooling applied:
pooled_trace = trial_average_pool(chunk_trace)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. The AI ensured temporal alignment by processing neural and position data in lockstep within each trial chunk. This mirrors the reference code's `fit_decoder` which applies the same `AvgPool1d` to both behavior and traces.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
1. **NaN cells**: Cells with any NaN values in a session (non-registered cells) are removed before further processing.
2. **Blocked partition encoding**: The `blocked` field has variable formats (empty list, list-of-list, negative sentinel values). `parse_blocked_indices` handles all these cases.
3. **Positions in blocked bins**: Snapped to the nearest open bin using Euclidean distance.
4. **Short trials**: Trials with <3 movement-valid frames or <10 pooled samples are excluded.
5. **All-zero neural trials**: Excluded.
6. **Sessions with <2 valid trials**: Raise an error (but none occurred in practice).

ii.
```python
def parse_blocked_indices(blocked_entry):
    if isinstance(blocked_entry, list):
        if len(blocked_entry) == 0:
            return np.array([], dtype=np.int64)
        blocked_entry = blocked_entry[0]
    arr = np.array(blocked_entry, dtype=np.float64).reshape(-1)
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
    return np.sort(arr.astype(np.int64))

# NaN filtering:
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
```

iii. The AI documented investigating edge cases in Step 10 (Critical Review 1), finding that only 4 sessions had unusually low trial counts due to low movement, and confirmed these were not indexing bugs.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the animal joblib files, which take 20-112 seconds each. Total conversion time was ~675 seconds for all 207 sessions. Per-session processing is relatively fast (~1.67 seconds per session after loading).

ii.
```python
# From conversion_full_out.txt:
# Loading QLAK-CA1-08: 61.94s
# Loading QLAK-CA1-30: 107.05s
# Loading QLAK-CA1-50: 112.79s
# Total elapsed: 675.01s
```

iii. The AI documented: "Animal joblib loads are relatively slow (roughly tens of seconds per animal), so the script processes animals sequentially and frees memory between animals."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `snap_to_open_bins` function uses a Python loop over each frame to check if it falls in a blocked bin and find the nearest open bin. This could be vectorized using broadcasting/distance matrix computation.

ii.
```python
def snap_to_open_bins(row_idx, col_idx, geometry_open):
    # ...
    for i in range(row_idx.shape[0]):
        if geometry_mat[row_idx[i], col_idx[i]] == 1:
            continue
        distances = np.sum((open_coords - np.array([row_idx[i], col_idx[i]])) ** 2, axis=1)
        nearest = open_coords[np.argmin(distances)]
        row_idx[i] = nearest[0]
        col_idx[i] = nearest[1]
    return row_idx, col_idx
```

iii. The AI noted it "moved heavy preprocessing to vectorized NumPy/Scipy operations" but this particular function still uses a per-frame loop.

## 6-c. What processing does the code repeat multiple times?

i. The velocity mask is computed once per session and reused across all trials within that session, which is efficient. The coordinate transform is inferred once per animal and reused. Position-to-grid-bin conversion is also done once per session. The snapping to open bins is done both session-wide and again after pooling within trials (a redundancy since pooled averages can re-introduce blocked bin values).

ii.
```python
# Session-wide snap:
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)

# Per-trial post-pooling snap (repeated):
invalid = geometry_mat[pooled_rows, pooled_cols] == 0
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
```

iii. The double snapping is necessary because average-pooling integer bin indices can produce fractional values that, when floored, may land in blocked bins again. The AI did not explicitly discuss this redundancy but it is a deliberate choice.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are done for visualization/metadata purposes that are not used in the final pickle:
1. Occupancy matrices (raw, aligned, snapped) are computed for each session but only used in processing plots.
2. The `SessionPlotPayload` dataclass stores copies of raw trial data, velocity masks, etc. for plotting.
3. Transform scores for all 8 candidate transforms are computed and stored but only the best is used.
4. Session-level statistics (movement-valid fraction, raw frame counts) are computed and stored in metadata but not used by the decoder.

ii.
```python
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
occupancy_aligned = np.zeros_like(occupancy_raw)
np.add.at(occupancy_aligned, (mapped_rows_all, mapped_cols_all), 1)
occupancy_snapped = np.zeros_like(occupancy_raw)
np.add.at(occupancy_snapped, (snapped_rows_all, snapped_cols_all), 1)

# SessionPlotPayload stores copies of data only needed for plots
```

iii. The occupancy matrices and plot payloads are only generated for validation/visualization. The transform scores serve as a useful diagnostic but only the best transform is applied.
