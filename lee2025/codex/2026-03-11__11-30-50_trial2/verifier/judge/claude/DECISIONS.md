# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-format animal files in the `data/` directory (e.g., `data/QLAK-CA1-08`), using `joblib.load()`. Each file is a dictionary keyed by animal name, containing fields `trace`, `position`, `blocked`, `envs`, and others. The AI chose joblib over the .mat files because the joblib files match the axis ordering used by the reference code and avoid HDF5 transpose issues.

ii.
```python
animal_path = os.path.join(data_dir, animal)
dat = joblib.load(animal_path)[animal]
```

iii. From CONVERSION_NOTES Step 5: "Use joblib files as the primary raw source: They expose the same field names as the reference code and already match the analysis-ready axis ordering, reducing the risk of silent transpose errors relative to the MATLAB HDF5 layout."

## 1-b. How are the data split into subjects?

i. Each joblib file in the `data/` directory corresponds to one subject (mouse). The AI identifies animal files by listing the directory and excluding `.mat` files, the `behav_dict` file, and hidden files. The subject name is the filename itself.

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
```

iii. The animal files are named by mouse ID (e.g., `QLAK-CA1-08`). The AI lists all non-.mat files to find them, producing 7 subjects.

## 1-c. How are the data split into sessions?

i. Each animal's data contains arrays indexed by session (e.g., `dat["trace"][session_idx]` for session `session_idx`). The AI iterates over the session dimension of the position array to determine the number of sessions per animal.

ii.
```python
for session_idx in range(dat["position"].shape[0]):
    ...
    trace_session=dat["trace"][session_idx],
    position_session=dat["position"][session_idx],
    blocked_entry=dat["blocked"][session_idx],
```

iii. Sessions are the first dimension of the per-animal arrays, with 31 sessions for 6 animals and 21 for one animal (207 total).

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into 60-second non-overlapping segments (1800 frames at 30 Hz). However, within each trial, only movement-valid frames (above a velocity threshold) are retained, and the remaining frames are then pooled in groups of 3. This means trials have variable length after processing.

ii.
```python
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
    chunk_mask = velocity_mask[start:end]
    ...
    chunk_trace = trace_active[:, start:end][:, chunk_mask]
    ...
    pooled_trace = trial_average_pool(chunk_trace)
```

iii. From CONVERSION_NOTES Step 5: "Split sessions into full one-minute chunks first, but within each chunk keep only movement-valid frames and pool in groups of 3."

## 1-e. How are trials filtered based on quality controls?

i. Trials are discarded if they have fewer than `POOL_SIZE` (3) movement-valid frames, fewer than `MIN_POOLED_SAMPLES_PER_TRIAL` (10) pooled timepoints, or no non-zero neural activity after processing. Sessions with fewer than 2 valid trials are also rejected.

ii.
```python
if int(chunk_mask.sum()) < POOL_SIZE:
    continue
...
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue
if not np.any(pooled_trace):
    continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")
```

iii. From CONVERSION_NOTES Step 10: "All-zero sample trial warning: Found during Step 7 verification. Resolved by excluding trials with fewer than 10 pooled samples or no processed neural signal." This led to 8109 exported trials from a possible 8187 raw segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib files, which contains rise-extracted calcium event traces with shape `(n_sessions, n_cells, n_frames)`.

ii.
```python
trace_session=dat["trace"][session_idx],
```

iii. From CONVERSION_NOTES Step 1: "The README states `trace` is already 'rise-extracted calcium traces' where `1` marks a significant event."

## 2-b. How is the `neural` data processed?

i. The AI applies a multi-step processing pipeline modeled on the reference code's within-session decoder:
1. Remove neurons with any NaN values (non-finite cells)
2. Filter to movement-valid frames using a velocity threshold (5 cm/s with Gaussian smoothing, sigma=5)
3. Remove low-activity neurons (those with <=5 events during movement-valid frames)
4. Apply Gaussian temporal smoothing (sigma=3 frames) to the traces
5. Average-pool every 3 consecutive frames

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)

activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]

# Within each trial:
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES Step 5: "Apply the reference decoder's session-level preprocessing before trial export: The paper/code decode position after filtering to movement-valid samples, dropping very-low-activity cells, smoothing traces, and average-pooling every 3 frames."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied at the session level:
1. Neurons with any NaN values across the session are removed (these are cells not registered in that session)
2. Neurons with 5 or fewer calcium events during movement-valid frames are removed

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)

activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
```

iii. From CONVERSION_NOTES Step 1: "The decoder excludes time points with low running speed and cells with too few events during those valid time points."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The recording is continuous free exploration, and trials are artificial 60-second segments starting from the beginning of each session. The alignment event is simply "Start of each consecutive one-minute chunk from a continuous recording session."

ii. N/A (alignment is implicit in the trial-splitting logic)

iii. From CONVERSION_NOTES metadata: `temporal_alignment_event: "Start of each consecutive one-minute chunk from a continuous recording session"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 3-frame average pooling to the 30 Hz data, resulting in an effective time bin size of 100 ms (3 frames x 33.33 ms/frame). The metadata reports `time_bin_size: 100.0`.

ii.
```python
POOL_SIZE = 3
...
def trial_average_pool(values, pool_size=POOL_SIZE):
    n_full = values.shape[-1] // pool_size
    trimmed = values[..., : n_full * pool_size]
    new_shape = values.shape[:-1] + (n_full, pool_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. From CONVERSION_NOTES Step 1: "Temporal binning is by average pooling over `3` frames in both traces and position." This directly follows the reference code's `fit_decoder` function.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the raw data, which stores which of the 9 possible 3x3 partitions are blocked for each session.

ii.
```python
blocked_entry=dat["blocked"][session_idx],
...
geometry_open, blocked_idx = blocked_to_open_vector(blocked_entry)
```

iii. From CONVERSION_NOTES Step 5: "Use raw `blocked` partitions to build decoder inputs: This satisfies the requested input specification directly."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted to a 9-element binary vector where `1 = open` and `0 = blocked`. If no positions are blocked (indicated by `-1`), all entries are 1. The vector is static per trial (constant within a session). Notably, the AI's encoding is the **inverse** of the reference solution: `1=open, 0=blocked` vs the reference's `1=blocked, 0=unblocked`.

ii.
```python
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
```

iii. From CONVERSION_NOTES Step 5: "Encode geometry as 9 binary partition features: One feature per 3x3 partition provides the decoder with the full static environmental context. The planned feature semantics are `1 = open`, `0 = blocked`, consistent with the reference code's `get_env_mat` convention."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse at each frame, with shape `(n_sessions, 2, n_frames)`.

ii.
```python
position_session=dat["position"][session_idx],
```

iii. Position tracks the mouse location in the arena at each timepoint at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI applies several processing steps:
1. Compute a per-animal spatial scale from the maximum tracked coordinate
2. Infer a coordinate transform per animal by testing 8 possible rotations/reflections and choosing the one that minimizes mouse occupancy in blocked regions
3. Discretize position into 3x3 grid bins using floor division
4. Apply the inferred coordinate transform to the bin indices
5. Snap any positions that fall in blocked bins to the nearest open bin
6. Filter to movement-valid frames (velocity > 5 cm/s)
7. Average-pool position bins in groups of 3 frames

ii.
```python
scale_cm = float(np.nanmax(dat["position"]))
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
coord_map = build_coord_map(transform_name)
...
raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
raw_ids_all = raw_rows_all * GEOMETRY_SIZE + raw_cols_all
mapped_ids_all = coord_map[raw_ids_all]
...
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
```

iii. From CONVERSION_NOTES Step 5: "Derive 3x3 position bins from the same arena scale used by the reference decoder logic... Snap blocked-bin position samples to the nearest open partition before export."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using floor division: `floor(pos / bin_size)` where `bin_size = scale_cm / 3`. With a 75 cm arena, this gives 25 cm bins. The resulting (row, col) is clipped to [0, 2] and encoded as a single integer `row * 3 + col` for 9 categories. Additionally, positions landing in blocked bins are snapped to the nearest open bin.

ii.
```python
def raw_position_to_rc(position_xy, bin_size_cm):
    x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
    y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
    x = np.clip(x, 0, GEOMETRY_SIZE - 1)
    y = np.clip(y, 0, GEOMETRY_SIZE - 1)
    return y, x
...
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. The 3x3 grid with 9 categories matches the decoder task specification.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are processed from the same raw frames. Within each 60-second trial, the same velocity mask selects the movement-valid frames for both streams, and the same 3-frame pooling is applied. This ensures frame-for-frame alignment.

ii.
```python
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
...
pooled_trace = trial_average_pool(chunk_trace)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. Both neural and position data use the same trial boundaries, velocity mask, and pooling window, guaranteeing temporal alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several forms of missing/problematic data are handled:
- Neurons with any NaN values (not registered in that session) are removed
- Low-activity neurons (<=5 events during movement) are removed
- Trials with very few movement-valid frames (<3 raw or <10 pooled) are discarded
- Trials with all-zero neural activity after processing are discarded
- Positions in blocked bins are snapped to nearest open bin
- Sessions with fewer than 2 valid trials raise an error

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
...
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
...
if int(chunk_mask.sum()) < POOL_SIZE:
    continue
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue
if not np.any(pooled_trace):
    continue
```

iii. From CONVERSION_NOTES Step 10: "All-zero sample trial warning: Found during Step 7 verification. Resolved by excluding trials with fewer than 10 pooled samples or no processed neural signal."

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib animal files is the most time-consuming step, taking roughly 60 seconds per animal. Per-session processing is relatively fast (~1.67 s/session).

ii. N/A

iii. From CONVERSION_NOTES Step 7: "Sample conversion (measured): ~1.67 s/session after first animal load; first animal load ~62.46 s."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `snap_to_open_bins` function iterates over every timepoint with a Python for loop to check if each position falls in a blocked bin and, if so, find the nearest open bin. This could be vectorized using array operations.

ii.
```python
def snap_to_open_bins(row_idx, col_idx, geometry_open):
    ...
    for i in range(row_idx.shape[0]):
        if geometry_mat[row_idx[i], col_idx[i]] == 1:
            continue
        distances = np.sum((open_coords - np.array([row_idx[i], col_idx[i]])) ** 2, axis=1)
        nearest = open_coords[np.argmin(distances)]
        row_idx[i] = nearest[0]
        col_idx[i] = nearest[1]
    return row_idx, col_idx
```

iii. The AI's CONVERSION_NOTES does not identify this loop as a vectorization candidate, though it processes thousands of timepoints per session.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is obviously repeated. The coordinate transform is inferred once per animal and reused across sessions. Occupancy matrices are computed once per session.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several quantities that are not included in the final pickle output:
- `occupancy_raw`, `occupancy_aligned`, `occupancy_snapped` matrices for every session (only used in optional processing plots)
- `transform_scores` dictionary for every animal (only used for logging)
- Detailed `session_info` metadata entries including movement fractions and trial lengths

ii.
```python
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
occupancy_aligned = np.zeros_like(occupancy_raw)
np.add.at(occupancy_aligned, (mapped_rows_all, mapped_cols_all), 1)
occupancy_snapped = np.zeros_like(occupancy_raw)
np.add.at(occupancy_snapped, (snapped_rows_all, snapped_cols_all), 1)
```

iii. The AI's CONVERSION_NOTES does not explicitly identify these as unnecessary. The occupancy matrices serve a validation/debugging purpose during development but are computed for all sessions even in non-plotting mode.
