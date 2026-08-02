# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files in the `data/` directory. Each file is named by animal ID (e.g., `QLAK-CA1-08`) and contains a dictionary keyed by animal name. The AI iterates over all animal files found in sorted order, loading each with `joblib.load()`. Within each animal file, the data includes `trace` (neural), `position` (behavior), `blocked` (geometry), and `envs` (environment labels). Sessions are accessed as indices into the first dimension of these arrays. Trials do not exist natively -- they are derived by splitting each session's time series into consecutive 1-minute (1800-frame) chunks.

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
data_dir = os.path.join(os.getcwd(), "data")
animals = get_animal_files(data_dir)
for animal in selected_animals:
    animal_path = os.path.join(data_dir, animal)
    dat = joblib.load(animal_path)[animal]
```

iii. The AI justified using joblib files (rather than .mat files) because they "expose the same field names as the reference code and already match the analysis-ready axis ordering, reducing the risk of silent transpose errors relative to the MATLAB HDF5 layout" (CONVERSION_NOTES.md Step 5, Key Decision 1).

## 1-b. How are the data split into subjects?

i. Subjects (mice) correspond to individual animal files in the `data/` directory. The AI creates a list of all animal filenames sorted alphabetically. Each animal file is processed sequentially, and the animal's index in this list is stored in `subject_idx` for each session.

ii.
```python
animals = get_animal_files(data_dir)
# ...
for animal in selected_animals:
    # ...
    subject_idx.append(animals.index(animal))
# ...
data = {
    "subjects": animals,
    "subject_idx": np.array(subject_idx, dtype=np.int64),
}
```

iii. The AI noted that there are 7 mice in the dataset, matching the reference paper and code. The sorting ensures deterministic ordering across runs.

## 1-c. How are the data split into sessions?

i. Sessions correspond to the first dimension index of arrays within each animal's data (e.g., `dat["trace"][session_idx]`). The AI iterates over all session indices within each animal, treating each as an independent session.

ii.
```python
for session_idx in range(dat["position"].shape[0]):
    session_id = f"{animal}_s{session_idx:02d}"
    env_name = str(dat["envs"][session_idx, 0])
    # process session...
```

iii. The CONVERSION_NOTES document that 6 animals have 31 sessions and 1 animal has 21 sessions, totaling 207 sessions, consistent with the paper.

## 1-d. How are the data split into trials?

i. There are no native trials in the raw data. The AI derives trials by splitting each session's time series into consecutive non-overlapping 1-minute chunks of 1800 frames (30 fps * 60 seconds). The number of full trials per session is `floor(n_frames / 1800)`.

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 30 * 60 = 1800
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
```

iii. The AI justified this by noting "No native trials reported; conversion will derive one-minute trials solely to satisfy the target decoder format" (CONVERSION_NOTES.md Step 3). The instructions specify "1-minute trials within each session."

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three quality filters to trials: (1) a minimum of `POOL_SIZE` (3) movement-valid frames within the 1-minute chunk, (2) a minimum of `MIN_POOLED_SAMPLES_PER_TRIAL` (10) pooled time bins after filtering and pooling, and (3) at least some non-zero neural activity in the processed trace. This reduced total trials from 8187 (raw floor-division count) to 8109 exported trials.

ii.
```python
chunk_mask = velocity_mask[start:end]
if int(chunk_mask.sum()) < POOL_SIZE:
    continue

# ... processing ...

if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue

if not np.any(pooled_trace):
    continue
```

iii. The AI documented that sessions with unusually low trial counts (e.g., `QLAK-CA1-75_s01` with 24 trials) had the lowest movement-valid fractions (0.217-0.351), explaining the reduced counts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field of each animal's data file. The `trace` array has shape `(n_sessions, n_cells, n_frames)` and contains binary rise-extracted calcium event traces where `1` marks a significant calcium transient event.

ii.
```python
trace_session = dat["trace"][session_idx]
# shape: (n_cells, n_frames)
```

iii. The AI noted: "Neural data are not raw fluorescence and no delta-F/F computation appears in the reference code. The README states `trace` is already 'rise-extracted calcium traces' where `1` marks a significant event" (CONVERSION_NOTES.md Step 1).

## 2-b. How is the `neural` data processed?

i. The processing pipeline for neural data follows the reference decoder's approach:
1. Filter out cells with any NaN values within the session.
2. Filter out cells with fewer than `CELL_EVENT_THRESHOLD` (5) events during movement-valid frames.
3. For each 1-minute trial chunk, extract only movement-valid frames.
4. Apply Gaussian smoothing with `sigma=3` frames along the time axis.
5. Average-pool every 3 consecutive frames into one time bin.

ii.
```python
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

iii. The AI justified this by mapping to the reference decoder function `decode_position_within`: "filtering to moving periods, filtering low-activity cells, uses 5-fold CV, and reports position decoding error" and `fit_decoder`: "temporally smooths traces, averages in non-overlapping 3-frame bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two levels of filtering are applied:
1. **NaN filtering**: Cells with any NaN values in a session are removed (these represent unregistered cells for that session).
2. **Activity filtering**: Cells with 5 or fewer calcium events during movement-valid periods across the entire session are removed. This is done at the session level, not per-trial.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)

activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
```

iii. The AI noted this matches the reference decoder's approach. The CONVERSION_NOTES state the paper says high reliability "motivated the inclusion of all cells in subsequent analyses" -- but the within-session decoder specifically filters by activity, and the AI followed this.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the start of a consecutive 1-minute chunk from the continuous recording session. The `off_start` is 0.0 seconds and `off_end` is 60.0 seconds. Within each chunk, only movement-valid frames are kept, so the effective temporal samples are irregularly spaced within the 1-minute window. After movement filtering, the remaining frames are smoothed and pooled in groups of 3.

ii.
```python
# Alignment: consecutive 1-minute chunks
start = trial_idx * RAW_TRIAL_FRAMES  # trial_idx * 1800
end = start + RAW_TRIAL_FRAMES
chunk_mask = velocity_mask[start:end]

# metadata
"temporal_alignment_event": "Start of each consecutive one-minute chunk from a continuous recording session",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The AI explained there is no specific stimulus or behavioral event to align to, as the experiment is free exploration. Alignment is simply to the start of each derived 1-minute trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native data is at 30 Hz (33.33 ms per frame). After movement-speed filtering removes stationary frames, the remaining frames are average-pooled in non-overlapping groups of 3. The reported `time_bin_size` in metadata is 100.0 ms (3 frames * 33.33 ms/frame). This rebinning matches the reference decoder's `fit_decoder` function.

ii.
```python
POOL_SIZE = 3
def trial_average_pool(values, pool_size=POOL_SIZE):
    n_full = values.shape[-1] // pool_size
    trimmed = values[..., : n_full * pool_size]
    new_shape = values.shape[:-1] + (n_full, pool_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

# metadata
"time_bin_size": 100.0,
```

iii. The AI justified the 100 ms bin size as matching the reference decoder's 3-frame average pooling at 30 Hz.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field of each animal's data. This field contains, for each session, a list of blocked partition indices in the 3x3 layout (or -1 for no blocked partitions).

ii.
```python
blocked_entry = dat["blocked"][session_idx]
```

iii. The AI chose to use `blocked` rather than the `envs` labels because "it directly represents blocked partitions in the raw dataset" (CONVERSION_NOTES.md Step 4).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The `blocked` partition indices are converted to a 9-element binary vector where `1 = open` and `0 = blocked`. A value of -1 or an empty list means all partitions are open (the square environment).

ii.
```python
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx

def parse_blocked_indices(blocked_entry):
    if isinstance(blocked_entry, list):
        if len(blocked_entry) == 0:
            return np.array([], dtype=np.int64)
        blocked_entry = blocked_entry[0]
    arr = np.array(blocked_entry, dtype=np.float64).reshape(-1)
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
    return np.sort(arr.astype(np.int64))
```

iii. The AI documented that this encoding is "consistent with the reference code's `get_env_mat` convention" (CONVERSION_NOTES.md Step 5, Key Decision 7).

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry input is static per trial -- the same 9-element vector is used for all time points within a trial. It does not vary across time within a session. The input shape is `(9,)` per trial (not `(9, n_timepoints)`).

ii.
```python
input_trials.append(geometry_open.copy())
# geometry_open has shape (9,)
```

iii. The instructions state "Static per-trial" for environment geometry as decoder input. The AI matches this requirement.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field of each animal's data, which has shape `(n_sessions, 2, n_frames)` containing x-y coordinates of the mouse's tracked position at 30 Hz.

ii.
```python
position_session = dat["position"][session_idx]
# shape: (2, n_frames)
```

iii. The AI noted that "Position was derived from DeepLabCut head tracking" as stated in the paper.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position processing involves multiple steps:
1. Compute a per-animal spatial scale from `max(position)` across all sessions.
2. Discretize x-y positions into a 3x3 grid using `floor(position / bin_size)` where `bin_size = scale / 3`.
3. Infer and apply a coordinate transform to align the raw spatial bins with the `blocked` geometry definition (minimizing occupancy in blocked partitions).
4. Snap any positions that land in blocked bins to the nearest open bin.
5. Filter to movement-valid frames only.
6. Average-pool the discrete position indices in groups of 3 frames (same as neural data).
7. Collapse 2D row/col indices to a single categorical index: `row * 3 + col`.

ii.
```python
scale_cm = float(np.nanmax(dat["position"]))
bin_size_cm = (scale_cm + 1e-6) / GEOMETRY_SIZE
raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
# apply coordinate transform
mapped_ids_all = coord_map[raw_ids_all]
# snap to open bins
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
# per-trial: filter, pool
chunk_rows = snapped_rows_all[start:end][chunk_mask]
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. The AI justified the transform inference step by noting that the `blocked` field indices and the spatial coordinate system may not be in the same orientation, requiring an alignment step.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous x-y position is discretized into a 3x3 grid (9 categories). The arena is divided into 3 equal-width columns (x) and 3 equal-height rows (y) using `floor(coord / bin_size)` where `bin_size = arena_scale / 3`. The 2D bin is then collapsed to a single categorical label: `row * 3 + col`, yielding values 0-8. Positions in blocked bins are snapped to the nearest open bin.

ii.
```python
def raw_position_to_rc(position_xy, bin_size_cm):
    x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
    y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
    x = np.clip(x, 0, GEOMETRY_SIZE - 1)
    y = np.clip(y, 0, GEOMETRY_SIZE - 1)
    return y, x  # row=y, col=x

pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. The instructions specify "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying." The AI matches this requirement.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The output is aligned with the neural data because both use the same movement-valid frame selection, the same 1-minute chunk boundaries, and the same 3-frame average pooling. Position and neural traces are sliced from the same frame indices, so they are inherently temporally synchronized.

ii.
```python
# Same chunk and mask applied to both neural and position
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
# Same pooling applied to both
pooled_trace = trial_average_pool(chunk_trace)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. The AI notes that neural and behavioral streams were acquired simultaneously at 30 Hz, so using the same frame indices ensures alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final temporal resolution is 100 ms (3 frames at 30 Hz pooled together). Temporal rebinning is applied via non-overlapping 3-frame average pooling, matching the reference decoder's approach.

ii.
```python
POOL_SIZE = 3
FPS = 30
# time_bin_size = POOL_SIZE / FPS * 1000 = 100.0 ms
"time_bin_size": 100.0,
```

iii. Same as 2-e. The AI consistently reports 100 ms bins matching the reference decoder's 3-frame pooling.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data are temporally aligned by construction: both are derived from the same session-level frame indices, filtered by the same movement-validity mask, and pooled with the same 3-frame window. The input (geometry) is static per trial and does not require temporal alignment.

ii.
```python
# All use the same chunk boundaries and velocity mask
chunk_mask = velocity_mask[start:end]
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
# Both pooled with same function
pooled_trace = trial_average_pool(chunk_trace)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0])
```

iii. The AI documented that neural and behavioral streams are simultaneously sampled at 30 Hz and the same frame-level operations are applied to both.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI handles several data quality issues:
- **NaN cells**: Cells with any NaN values in a session are removed before processing (these represent cells not registered for that session).
- **Malformed `blocked` entries**: The `parse_blocked_indices` function handles various formats: empty lists, nested single-element lists, and -1 as "no blocked bins."
- **Low-activity cells**: Cells with fewer than 5 events during movement-valid frames are excluded.
- **Short trials**: Trials with fewer than 3 movement-valid frames or fewer than 10 pooled samples are skipped.
- **All-zero neural activity**: Trials where the processed neural trace is entirely zero are skipped.
- **Positions in blocked bins**: Positions that land in blocked partitions are snapped to the nearest open bin.

ii.
```python
# NaN handling
finite_cells = np.isfinite(trace_session).all(axis=1)

# Blocked entry parsing
def parse_blocked_indices(blocked_entry):
    if isinstance(blocked_entry, list):
        if len(blocked_entry) == 0:
            return np.array([], dtype=np.int64)
        blocked_entry = blocked_entry[0]
    arr = np.array(blocked_entry, dtype=np.float64).reshape(-1)
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
    return np.sort(arr.astype(np.int64))

# Minimum trial quality
if int(chunk_mask.sum()) < POOL_SIZE:
    continue
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue
if not np.any(pooled_trace):
    continue
```

iii. The AI documented investigating edge cases in CONVERSION_NOTES Step 10, noting only 4 sessions had fewer than 35 trials due to low movement fractions.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the animal joblib files. According to the conversion output, loading times range from ~20 seconds (QLAK-CA1-51, smallest animal) to ~113 seconds (QLAK-CA1-50). Total elapsed time for all 207 sessions was 675 seconds (~11.25 minutes), with the majority spent on I/O. Per-session processing after loading is fast (~1-2 seconds).

ii.
```python
animal_load_start = time.time()
dat = joblib.load(animal_path)[animal]
load_seconds = time.time() - animal_load_start
print(f"  loaded in {load_seconds:.2f}s")
```

iii. The AI estimated full conversion at ~13 minutes (well under the 15-minute target) and noted that "Animal joblib loads are relatively slow."

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the `snap_to_open_bins` function, which iterates element-by-element over position indices to snap blocked-bin positions to the nearest open bin. This per-element loop could be replaced with vectorized distance computation. The trial loop itself is largely vectorized within each trial.

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

iii. The CONVERSION_NOTES note that the AI "moved heavy preprocessing to vectorized NumPy/Scipy operations" but the snap_to_open_bins function remains an explicit Python loop.

## 7-c. What processing does the code repeat multiple times?

i. The transform inference (`infer_transform`) computes occupancy matrices for all sessions of an animal, but this is done once per animal, not repeated. The `snap_to_open_bins` function is called twice -- once for the full session and potentially once more after pooling if any pooled positions fall in blocked bins. The `blocked_to_open_vector` function is called once in `preprocess_session` and once more per session in the `infer_transform` loop (for all sessions). The `parse_blocked_indices` function is called both within `blocked_to_open_vector` and separately when building session info metadata.

ii.
```python
# snap_to_open_bins called at session level:
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)

# And again per-trial if pooled positions land in blocked bins:
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
```

iii. The AI did not explicitly document repeated processing but addressed efficiency in CONVERSION_NOTES Step 6.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce results not used in the final converted data:
1. **Transform inference scores**: The `infer_transform` function computes blocked-fraction scores for all 8 transforms, but only the best transform is used.
2. **Occupancy matrices**: Raw, aligned, and snapped occupancy matrices are computed for visualization but not stored in the output pickle.
3. **Plot payloads**: `SessionPlotPayload` objects store extensive per-session visualization data that is only used when `--show-processing` is enabled.
4. **Session-level position processing** for frames in the tail portion of the session (beyond the last full 1-minute trial) is computed but never used.

ii.
```python
# Transform scores computed for all 8 transforms
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)

# Occupancy matrices
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
occupancy_aligned = np.zeros_like(occupancy_raw)
occupancy_snapped = np.zeros_like(occupancy_raw)
```

iii. The AI documented that occupancy and transform scores are used for visualization/debugging but not in the final output.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. The AI handles: NaN-valued cells (removed per session), malformed `blocked` entries (parsed robustly), low-activity cells (filtered out), short/empty trials (skipped), and positions in blocked bins (snapped to nearest open bin). Memory is managed by deleting animal data after processing each animal and calling `gc.collect()`.

ii.
```python
# Memory management
del dat
gc.collect()
```

iii. The AI verified in CONVERSION_NOTES Step 10 that edge cases were handled correctly, documenting specific sessions with low trial counts due to low movement fractions.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. The dominant cost is loading joblib files (20-113 seconds per animal). Per-session processing is fast (~1-2 seconds). Total conversion time is ~675 seconds for all 207 sessions.

ii. See 7-a code snippet.

iii. The AI estimated and validated timing, noting full conversion completed in ~11 minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The `snap_to_open_bins` function uses a Python for-loop over individual position samples. This could be vectorized using broadcasting to compute distances to all open bins simultaneously. However, for most samples the early `continue` branch applies (position already in an open bin), so the practical impact may be small.

ii. See 7-b code snippet.

iii. The AI noted it "moved heavy preprocessing to vectorized NumPy/Scipy operations" but this particular function was not vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. The `snap_to_open_bins` is potentially called twice (at session level and again per-trial after pooling). The `parse_blocked_indices` / `blocked_to_open_vector` are called in both `infer_transform` and `preprocess_session`. The overall architecture processes each animal sequentially with loading/unloading, avoiding cross-animal data in memory simultaneously.

ii. See 7-c code snippets.

iii. The AI chose this design for memory efficiency rather than avoiding redundant computation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. The main unnecessary processing includes:
1. Computing transform scores for all 8 geometric transforms (only the best is used).
2. Computing occupancy matrices (raw, aligned, snapped) that are only used for visualization.
3. Processing position data for the tail frames of each session beyond the last full trial.
4. Building `SessionPlotPayload` objects for visualization even when `--show-processing` is not enabled (though the payload is only saved for up to 2 sessions when the flag is set).

ii. See 7-d code snippets.

iii. The AI designed these for debugging and validation rather than efficiency, which is reasonable for a data conversion pipeline.
