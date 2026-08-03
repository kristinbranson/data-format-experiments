# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (not .mat files) in the `data/` directory using `joblib.load()`. Each joblib file contains all sessions for one animal, with fields including `trace`, `position`, `blocked`, and `envs`. The AI chose joblib over .mat/HDF5 because the joblib files match the reference code's axis ordering and avoid transpose errors.

ii.
```python
animal_path = os.path.join(data_dir, animal)
dat = joblib.load(animal_path)[animal]
```
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

iii. From CONVERSION_NOTES.md Step 5: "Use joblib files as the primary raw source: They expose the same field names as the reference code and already match the analysis-ready axis ordering, reducing the risk of silent transpose errors relative to the MATLAB HDF5 layout."

## 1-b. How are the data split into subjects?

i. Each joblib file in the `data/` directory corresponds to one subject (mouse). The animal name is the filename itself. The AI lists all non-.mat, non-hidden files in `data/` (excluding `behav_dict`) and sorts them alphabetically.

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

iii. The AI noted that each joblib file contains all recording sessions for one animal, and the filename serves as the subject identifier (CONVERSION_NOTES.md Step 2).

## 1-c. How are the data split into sessions?

i. Within each animal's data, sessions are indexed by the first dimension of the `trace`, `position`, `blocked`, and `envs` arrays. Each session index becomes a separate session in the output.

ii.
```python
for session_idx in range(dat["position"].shape[0]):
    ...
    session_neural, session_input, session_output, ... = preprocess_session(
        trace_session=dat["trace"][session_idx],
        position_session=dat["position"][session_idx],
        blocked_entry=dat["blocked"][session_idx],
        ...
    )
```

iii. From CONVERSION_NOTES.md Step 2: "Six animals have 31 sessions; one animal has 21 sessions" and "Session lengths are nearly fixed at about 40 minutes."

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into non-overlapping 1-minute (1800-frame at 30 Hz) chunks. However, within each chunk, only movement-valid frames are retained (velocity > 5 cm/s after Gaussian smoothing), and these are then average-pooled in groups of 3 frames. Remainder frames not filling a complete 1-minute chunk are discarded. Trials with fewer than 10 pooled samples or no neural activity are excluded.

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800
...
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
    chunk_mask = velocity_mask[start:end]
    if int(chunk_mask.sum()) < POOL_SIZE:
        continue
    chunk_trace = trace_active[:, start:end][:, chunk_mask]
    ...
    pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
    if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
        continue
```

iii. From CONVERSION_NOTES.md Step 5: "Split sessions into full one-minute chunks first, but within each chunk keep only movement-valid frames and pool in groups of 3."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on three criteria: (1) the trial must have at least `POOL_SIZE` (3) movement-valid frames, (2) the pooled trial must have at least `MIN_POOLED_SAMPLES_PER_TRIAL` (10) samples, and (3) the pooled neural trace must not be all zeros.

ii.
```python
if int(chunk_mask.sum()) < POOL_SIZE:
    continue
...
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue
if not np.any(pooled_trace):
    continue
```

iii. From CONVERSION_NOTES.md Step 7: "Fixed by excluding trials with fewer than 10 pooled samples or no processed neural activity."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib files, which contains rise-extracted calcium traces (binary events where 1 marks a significant calcium transient).

ii.
```python
trace_session=dat["trace"][session_idx]
```

iii. From CONVERSION_NOTES.md Step 1: "Neural data are not raw fluorescence and no delta-F/F computation appears in the reference code. The README states `trace` is already 'rise-extracted calcium traces' where `1` marks a significant event."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes several processing steps modeled on the reference paper's within-session decoder: (1) NaN filtering to remove unrecorded cells, (2) movement velocity filtering to keep only frames where the mouse is moving > 5 cm/s, (3) session-level active cell filtering (cells must have > 5 events during movement-valid frames), (4) Gaussian smoothing with sigma=3 frames, and (5) non-overlapping 3-frame average pooling.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
...
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES.md Step 5: "Apply the reference decoder's session-level preprocessing before trial export: The paper/code decode position after filtering to movement-valid samples, dropping very-low-activity cells, smoothing traces, and average-pooling every 3 frames."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two levels of filtering: (1) Cells with any NaN values in a session are removed (non-finite cells), and (2) remaining cells are filtered by activity — only cells with more than 5 events during movement-valid frames are kept.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
```

iii. From CONVERSION_NOTES.md Step 5: "Retain all session-valid registered cells initially, then apply the reference decoder's low-activity cell filter per session: This matches the code path in `decode_position_within` more closely than exporting every registered cell regardless of activity."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The recording is continuous, and trials are 1-minute non-overlapping segments starting from the beginning of each session. Within each trial, only movement-valid frames are kept and pooled, so the temporal axis represents movement-filtered, pooled time.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
```

iii. From CONVERSION_NOTES.md metadata: `"temporal_alignment_event": "Start of each consecutive one-minute chunk from a continuous recording session"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has a time bin size of 100 ms (3 native frames at 30 Hz pooled together). The AI applies non-overlapping 3-frame average pooling to match the reference decoder's preprocessing, changing the effective resolution from the native 33.33 ms to 100 ms. Additionally, only movement-valid frames are included before pooling, so the pooled time axis is not uniformly spaced in wall-clock time.

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
```python
"time_bin_size": 100.0,  # in metadata
```

iii. From CONVERSION_NOTES.md Step 1: "Temporal binning is by average pooling over 3 frames in both traces and position." The AI adopted the reference decoder's pooling as part of the conversion pipeline.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The geometry input is derived from the `blocked` variable in the joblib files, which contains a list of blocked partition indices for each session in the 3x3 arena layout.

ii.
```python
blocked_entry=dat["blocked"][session_idx]
...
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
```

iii. From CONVERSION_NOTES.md Step 5: "Use raw `blocked` partitions to build decoder inputs: This satisfies the requested input specification directly and avoids ambiguities from canonical geometry labels and plotting/masking orientation conventions."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted into a 9-element binary vector where 1=open and 0=blocked. A value of -1 (no positions blocked) results in an all-ones vector. The input is static per trial (same for all trials in a session). Additionally, the AI infers a coordinate transform per animal by minimizing occupancy in blocked partitions, and this transform is applied to align position bins with the geometry encoding.

ii.
```python
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
...
input_trials.append(geometry_open.copy())
```

iii. From CONVERSION_NOTES.md Step 5: "Encode geometry as 9 binary partition features: One feature per 3x3 partition provides the decoder with the full static environmental context. The planned feature semantics are `1 = open`, `0 = blocked`."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` variable in the joblib files, which contains 2D (x, y) coordinates of the animal tracked at 30 Hz.

ii.
```python
position_session=dat["position"][session_idx]
```

iii. From CONVERSION_NOTES.md Step 3: "Position was derived from DeepLabCut head tracking."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The raw x-y positions are: (1) discretized into a 3x3 grid using the per-animal maximum coordinate as the arena scale, (2) transformed using the inferred coordinate mapping to align with geometry encoding, (3) snapped to the nearest open bin if the position falls in a blocked partition, and (4) average-pooled in groups of 3 movement-valid frames (same pooling as neural data).

ii.
```python
raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
raw_ids_all = raw_rows_all * GEOMETRY_SIZE + raw_cols_all
mapped_ids_all = coord_map[raw_ids_all]
mapped_rows_all = mapped_ids_all // GEOMETRY_SIZE
mapped_cols_all = mapped_ids_all % GEOMETRY_SIZE
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
```

iii. From CONVERSION_NOTES.md Step 5: "Snap blocked-bin position samples to the nearest open partition before export: The reference decoder cleans positions/predictions to the nearest valid bin when evaluating decoding."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Continuous x-y position is discretized into a 3x3 grid (9 categories). The arena is divided into 3 equal bins along each axis using the per-animal maximum position coordinate as the scale. The bin label is computed as `row * 3 + col` in row-major order. Positions in blocked bins are snapped to the nearest open bin.

ii.
```python
def raw_position_to_rc(position_xy, bin_size_cm):
    x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
    y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
    x = np.clip(x, 0, GEOMETRY_SIZE - 1)
    y = np.clip(y, 0, GEOMETRY_SIZE - 1)
    return y, x
```

iii. From CONVERSION_NOTES.md Step 5: "Derive 3x3 position bins from the same arena scale used by the reference decoder logic."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same native frame rate (30 Hz) and are aligned frame-by-frame. Both are subjected to the same movement filtering (keeping only frames where velocity > 5 cm/s), and then both are average-pooled in non-overlapping groups of 3 valid frames. This ensures temporal alignment is maintained throughout processing.

ii.
```python
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. From CONVERSION_NOTES.md Step 5: "Output is time-varying and uses the same temporally pooled samples as the neural data."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of data issues are handled: (1) Cells with any NaN values in a session are removed, (2) remainder frames not filling a complete 1-minute trial are discarded, (3) trials with insufficient movement-valid frames (< 3) are dropped, (4) trials with too few pooled samples (< 10) are dropped, (5) trials with all-zero neural activity are dropped, (6) sessions with fewer than 2 valid trials raise an error. Additionally, positions in blocked bins are snapped to nearest open bins rather than being discarded.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
...
if int(chunk_mask.sum()) < POOL_SIZE:
    continue
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue
if not np.any(pooled_trace):
    continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")
```

iii. From CONVERSION_NOTES.md Step 10: "All-zero sample trial warning: Found during Step 7 verification. Resolved by excluding trials with fewer than 10 pooled samples or no processed neural signal."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the animal joblib files, which takes approximately 62 seconds per animal. The total estimated conversion time is ~13 minutes for all 207 sessions.

ii. N/A (timing is printed at runtime)

iii. From CONVERSION_NOTES.md Step 7: "Sample conversion (measured): ~1.67 s/session after first animal load; first animal load ~62.46 s."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `snap_to_open_bins` function contains a per-frame Python loop that iterates over every position sample to check if it falls in a blocked bin and snap it to the nearest open bin. This could be vectorized using boolean indexing and broadcast distance computation.

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

iii. The AI did not explicitly note this inefficiency in CONVERSION_NOTES.md.

## 6-c. What processing does the code repeat multiple times?

i. The coordinate transform inference (`infer_transform`) computes occupancy matrices for all sessions of an animal, which involves iterating over all sessions and computing position binning. This occupancy computation is then partially repeated during individual session preprocessing. The `snap_to_open_bins` function is also called twice — once at the session level for all frames, and again after pooling if any pooled positions land in blocked bins.

ii.
```python
# First call at session level:
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
# Second call after pooling:
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
```

iii. Not explicitly documented in CONVERSION_NOTES.md.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several items used only for visualization/debugging that are discarded: occupancy matrices (raw, aligned, snapped), processing plot payloads including raw trial position data, and detailed session_info metadata. The coordinate transform inference involves testing 8 different geometric transforms across all sessions, when only one is selected. The `SessionPlotPayload` dataclass stores copies of raw data for visualization that are only used in `--show-processing` mode but are computed unconditionally.

ii.
```python
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
occupancy_aligned = np.zeros_like(occupancy_raw)
np.add.at(occupancy_aligned, (mapped_rows_all, mapped_cols_all), 1)
occupancy_snapped = np.zeros_like(occupancy_raw)
np.add.at(occupancy_snapped, (snapped_rows_all, snapped_cols_all), 1)
```

iii. Not explicitly documented as unnecessary in CONVERSION_NOTES.md. The occupancy computations and plot payload are part of the validation/visualization pipeline.
