# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-subject `.mat` files (HDF5 format) located in a data directory. Each file contains fields: `trace` (neural activity), `position` (mouse position), `blocked` (blocked bin indices), `envs` (environment names), `SFPs` (spatial footprints), and `centroids`. Seven animals are hardcoded in the `ANIMALS` list. For each animal, the corresponding `.mat` file is opened with `h5py`, and all sessions within are iterated over. HDF5 object references are dereferenced to extract numeric arrays and strings.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]

for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
    with h5py.File(path, "r") as h5:
        n_sessions = h5["envs"].shape[1]
        for session_index in session_indices:
            # dereference each field
            position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
            trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The AI examined the `.mat` file structure, identified the keys and their shapes, and iterated over all sessions within each animal file. This matches the reference approach of iterating `sorted(glob(datadir + '/*.mat'))` and calling `process_mat_file` on each.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by a hardcoded list of 7 animal IDs. Each `.mat` file corresponds to one subject. The subject index is tracked by `enumerate(ANIMALS)` and stored in `subject_idx`.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
dataset["subjects"] = ANIMALS.copy()
for subject_idx, animal in enumerate(ANIMALS):
    ...
    dataset["subject_idx"].append(subject_idx)
```

iii. The AI verified there are 7 subjects matching the paper. The reference code similarly derives subject names from `.mat` filenames. Both produce the same subject list.

## 1-c. How are the data split into sessions?

i. Each recording session within a `.mat` file becomes a separate session in the output. Session count is determined by `h5["envs"].shape[1]`. Sessions are processed in their stored order within each animal file.

ii.
```python
n_sessions = h5["envs"].shape[1]
session_indices = list(range(n_sessions))
for session_index in session_indices:
    neural_trials, input_trials, output_trials, region_idx, summary = convert_session(
        h5=h5, animal=animal, session_index=session_index,
    )
    dataset["neural"].append(neural_trials)
```

iii. The AI verified 207 total sessions across all animals, matching the paper. The reference code uses the same approach (iterating sessions within each subject's `.mat` file).

## 1-d. How are the data split into trials?

i. Each session's data is split into contiguous 1-minute windows (1800 frames at 30 fps) from the session start. The last window can be shorter than 60 seconds if the total frame count isn't a multiple of 1800.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800

for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
```

iii. The AI chose 1-minute trials as specified in the instructions. However, the reference code drops the last trial if it is shorter than the full trial length (`n_trials = len(data) // trial_length`), while the agent keeps partial last trials. This is a minor difference.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple trial-level quality filters:
1. Trials with **no moving frames** (speed > 5 cm/s) are dropped.
2. Trials with **fewer than 10 pooled time bins** after movement filtering and temporal pooling are dropped.
3. Trials where **all neural data is zero** are dropped.
4. Sessions with **fewer than 2 surviving trials** raise an error.

ii.
```python
if not np.any(trial_movement):
    dropped_stationary_trials += 1
    continue
# ...
if trace_pooled.shape[0] < MIN_POOLED_TIMEPOINTS_PER_TRIAL:
    dropped_short_trials += 1
    continue
# ...
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
# ...
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. The AI applied these filters to ensure data quality for decoder training. The reference code does NOT apply any trial-level quality filtering beyond dropping the final incomplete trial. There is no movement filtering, no minimum timepoints check, and no zero-neural check in the reference. This is a significant difference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in the `.mat` files. This field contains binary rise-event traces (0s and 1s) for calcium imaging events, stored as a (n_timepoints, n_neurons) matrix per session.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The AI identified that the `trace` field contains the paper's "rise-extracted binary calcium-event vectors." This matches the reference code, which also uses the `trace` field.

## 2-b. How is the `neural` data processed?

i. The AI applies several processing steps to the raw trace data:
1. **NaN-based cell filtering**: Cells not registered in a session (NaN in first row) are dropped.
2. **NaN-to-zero conversion**: Remaining NaN values are replaced with 0.
3. **Movement frame restriction**: Only frames where the mouse is moving (speed > 5 cm/s) are kept.
4. **Gaussian smoothing**: Traces are smoothed with `sigma=3` frames along the time axis.
5. **Temporal pooling**: Traces are average-pooled in non-overlapping 3-frame windows.
6. **Transposition**: Final neural data is stored as (n_neurons, n_timepoints).

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
# ... movement filtering applied ...
trace_trial = trace_trial[trial_movement]
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The AI justified smoothing and pooling as producing decoder-ready time series. However, the reference code does NOT apply Gaussian smoothing, does NOT apply movement filtering, and does NOT apply temporal pooling. The reference simply filters NaN columns, casts to float32, transposes, and splits into trials. The smoothing and pooling are significant additions not in the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data filtering consists of:
1. **Cell-level**: Only cells registered for a given session (non-NaN in first row of trace) are included.
2. **Frame-level**: Only frames where the mouse is moving are retained.
3. **Trial-level**: Trials with all-zero neural activity after processing are dropped.

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], ...)
# movement filtering
trace_trial = trace_trial[trial_movement]
# all-zero check
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
```

iii. The cell-level NaN filtering matches the reference code, which also filters neurons using `active_mask = ~np.all(np.isnan(trace), axis=0)`. However, the reference checks for columns that are ALL NaN (using `np.all(np.isnan(...), axis=0)`), while the agent checks only the first row (`np.isnan(trace[0])`). The movement filtering and all-zero trial checks are not in the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. Trials are contiguous 1-minute windows starting from the beginning of each session. The metadata explicitly states: "No event alignment; contiguous 1-minute windows from session start."

ii.
```python
dataset["metadata"] = {
    "temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
    "off_start": None,
    "off_end": None,
}
```

iii. This is consistent with the instructions and the reference code. The recording sessions are continuous, with no specific stimulus events to align to.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 3 original frames (at 30 fps) are average-pooled into each time bin, yielding a bin size of 100 ms. This reduces 1800 raw frames per trial to at most 600 pooled time bins.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100.0 ms
# ...
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
```

iii. The AI chose 100 ms bins as a processing step. However, the reference code does NOT apply temporal rebinning. The reference `TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE = 33.33 ms` (one original frame at 30 Hz), and `TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC = 1800` frames per trial. Each time point in the reference corresponds to one raw frame. This is a significant difference: the agent's data has ~3x fewer time points per trial than the reference.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the `.mat` files, which stores indices of blocked spatial bins for each session.

ii.
```python
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
env_open_flat = env_open.reshape(-1).astype(np.float32)
input_trials.append(env_open_flat.copy())
```

iii. The AI used the `blocked` field from the source data, which matches the reference code. Both derive the environment geometry from the same raw variable.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI creates a 3x3 binary open mask: blocked indices get 0, open positions get 1. The mask is flattened to a 9-element vector. The encoding is: **1 = open, 0 = blocked**. The input is static per trial (same for all time points within a session).

ii.
```python
def open_mask_from_blocked(blocked_indices: list[int]) -> np.ndarray:
    open_mask = np.ones(POSITION_BINS_PER_AXIS * POSITION_BINS_PER_AXIS, dtype=np.float32)
    for idx in blocked_indices:
        open_mask[idx] = 0.0
    return open_mask.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS)
```

iii. The reference code uses the OPPOSITE encoding: **1 = blocked, 0 = not blocked** (`encode_blocked` sets `blocked[blk_indices.astype(int)] = 1`). The input variable names also differ: the agent uses `env_open_r{r}c{c}` while the reference uses `blocked_{i}`. The information content is the same (just inverted), but the encoding direction is opposite.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field in the `.mat` files, which contains 2D (x, y) coordinates of the mouse in centimeters, recorded at 30 fps.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. Both the agent and the reference use the `position` field. This matches.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI processes position in several steps:
1. Remove NaN position rows (and corresponding trace rows).
2. Restrict to moving frames only (speed > 5 cm/s).
3. Scale position using a session-specific maximum coordinate value: `bin_down = max(position) / 3`.
4. Average-pool in non-overlapping 3-frame windows (same as neural data).
5. Discretize into 3x3 bins by flooring the scaled position and clipping to [0, 2].
6. Remap any positions landing in blocked bins to nearest valid bin.

ii.
```python
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
bin_down = arena_extent / POSITION_BINS_PER_AXIS
pos_scaled = pos_trial / bin_down
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The reference code uses a FIXED `ARENA_SIZE = 75.0` for discretization with `np.linspace(0, 75, 4)` edges and `np.digitize`. The agent uses a session-specific maximum coordinate, which can vary between sessions. The reference does not apply movement filtering, temporal pooling, or blocked-bin remapping. These are significant differences.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D position is discretized into a 3x3 grid (9 bins total). The position is scaled so that `[0, max_coordinate)` maps to `[0, 3)`, then floored to get integer bin indices `{0, 1, 2}` for each axis. The 2D bin indices are flattened to a single categorical variable `position_bin` with values 0-8 using `row * 3 + col`.

ii.
```python
pos_scaled = pos_trial / bin_down
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins = bin_xy[:, 0] * POSITION_BINS_PER_AXIS + bin_xy[:, 1]
```

iii. The reference uses `np.digitize` with `np.linspace(0, 75, 4)` edges (equally-spaced at 0, 25, 50, 75) and clips to 0..2, then computes `y_bin * N_GRID + x_bin`. The agent's approach is similar in spirit (3x3 grid, flattened index) but differs in: (a) using session-specific scaling vs. fixed 75cm arena, (b) using `row * 3 + col` vs `y * 3 + x` ordering, and (c) applying temporal pooling before discretization.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by sharing the same frame indices. Both go through the same movement filtering and temporal pooling pipeline: movement mask is applied to both, then both are average-pooled in the same 3-frame windows. Lengths are synchronized after pooling.

ii.
```python
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
if pos_pooled.shape[0] != trace_pooled.shape[0]:
    min_len = min(pos_pooled.shape[0], trace_pooled.shape[0])
    pos_pooled = pos_pooled[:min_len]
    trace_pooled = trace_pooled[:min_len]
```

iii. This ensures temporal alignment between neural and position data. The reference code achieves alignment more simply: position and trace share the same time axis from the raw data, and `split_into_trials` splits both with the same trial length, so they are inherently aligned without any additional processing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of missing/problematic data:
1. **NaN positions**: Rows with NaN position values are removed from both position and trace.
2. **Unregistered neurons**: Neurons with NaN in the first trace row are excluded for that session.
3. **Remaining NaN in traces**: Replaced with 0 using `np.nan_to_num`.
4. **Positions in blocked bins**: After temporal pooling, positions landing in blocked bins are remapped to the nearest valid open bin.
5. **Infinite/NaN speed values**: Set to 0 in the movement mask computation.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]

registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)

speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The reference code handles NaN neurons with `active_mask = ~np.all(np.isnan(trace), axis=0)` and casts to float32 (which preserves NaN but the reference trace field is binary so NaN only appears for unregistered cells). The reference does not handle NaN positions, blocked-bin remapping, or speed NaN values since it doesn't compute speed. The agent's more defensive handling is reasonable but goes beyond the reference.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading .mat files**: Each file is ~100-500MB HDF5, and dereferencing object references is slow.
2. **Gaussian smoothing**: Applied to every trial's trace data with `gaussian_filter1d`.
3. **Movement mask computation**: Computing speed, smoothing it, and thresholding for every session.
4. **Blocked-bin remapping**: Iterating over invalid samples with nearest-neighbor search.
5. **Building the full dataset twice**: The code builds both full and sample datasets by calling `build_dataset` twice, re-loading all .mat files for the sample.

ii.
```python
# Full dataset is built first, then sample is built by re-reading from disk
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
# ... save full ...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The most expensive operations involve I/O (reading large HDF5 files) and the per-frame Gaussian smoothing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `remap_invalid_bins` function uses a Python loop to remap each invalid bin index individually, computing nearest-neighbor distances one sample at a time. This could be vectorized using broadcasting.

ii.
```python
def remap_invalid_bins(bin_xy, open_mask_flat):
    valid_bins = np.argwhere(open_mask_flat.reshape(3,3) > 0)
    for idx in np.where(invalid)[0]:
        diffs = valid_bins - bin_xy[idx]
        nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
        flat[idx] = nearest[0] * 3 + nearest[1]
```

iii. With 441,410 remapped samples across the full dataset, this loop is substantial. A vectorized approach could compute all remappings at once.

## 6-c. What processing does the code repeat multiple times?

i. The code rebuilds the entire dataset from scratch twice: once for the full dataset and once for the sample. This means the first animal's .mat file is loaded and processed twice.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
# ...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The sample could be extracted as a subset of the already-built full dataset instead of reprocessing from disk.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that is not needed downstream:
1. **Gaussian smoothing of traces**: Applied before temporal pooling, but the reference doesn't do this and the decoder operates on the pooled values.
2. **Movement filtering**: Frames below the speed threshold are removed, significantly reducing data. The reference keeps all frames.
3. **Blocked-bin remapping**: Positions in blocked bins are remapped to valid bins. The reference doesn't need this since it doesn't do temporal pooling that could create cross-bin artifacts.
4. **Environment name lookup** (`get_env_mat`): A function is defined to map environment names to 3x3 matrices, but this function is never called in the conversion code. The actual env geometry comes from the `blocked` field.
5. **Detailed per-session summary statistics**: Extensive summary dicts are computed but only stored in metadata.

ii.
```python
# get_env_mat is defined but never called
def get_env_mat(env_name: str) -> np.ndarray:
    if env_name == "square":
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    # ... more env types ...
```

iii. The movement filtering and Gaussian smoothing represent significant processing that the reference does not perform. These change the data characteristics and could affect decoder performance.
