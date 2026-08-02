# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `.mat` files from a data directory using `h5py`. It has a hardcoded list of 7 animal names (`ANIMALS`). For each animal, it opens the corresponding `.mat` file and reads `trace`, `position`, `blocked`, and `envs` variables. Each file contains multiple sessions accessed via HDF5 references.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
path = data_dir / f"{animal}.mat"
with h5py.File(path, "r") as h5:
    n_sessions = h5["envs"].shape[1]
```

iii. The AI identified from the paper that there are 7 subjects with 207 total sessions and verified these counts. The hardcoded animal list ensures deterministic ordering.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject. The subject names are hardcoded in the `ANIMALS` list rather than discovered from filenames.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
```

iii. The AI hardcoded the animal names based on the paper and data files, ensuring consistent ordering.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording sessions. The number of sessions is determined from `h5["envs"].shape[1]`. Each session index is processed separately via `convert_session()`.

ii.
```python
n_sessions = h5["envs"].shape[1]
session_indices = list(range(n_sessions))
for session_index in session_indices:
    neural_trials, input_trials, output_trials, region_idx, summary = convert_session(
        h5=h5, animal=animal, session_index=session_index,
    )
```

iii. Sessions are stored as indexed references within each `.mat` file. The AI verified the total count matches the paper's 207 sessions.

## 1-d. How are the data split into trials?

i. Sessions are split into contiguous 1-minute windows (1800 frames at 30 Hz) from session start. However, unlike the reference, within each trial the AI first applies velocity filtering (removing stationary frames), then temporally rebins the remaining moving frames into 3-frame average-pooled bins. This means trial lengths vary depending on how many moving frames exist in each 1-minute window.

ii.
```python
TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
    trial_movement = movement_mask[start:end]
    # Filter to moving frames only
    pos_trial = pos_trial[trial_movement]
    trace_trial = trace_trial[trial_movement]
    # Average pool in 3-frame bins
    trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. The AI followed the paper's decoder methodology which filters by velocity. The 1-minute trial duration matches the instructions.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple quality filters: (1) trials with no moving frames are dropped, (2) trials with fewer than 10 pooled time bins after movement filtering are dropped, (3) trials where all neural activity is zero after processing are dropped. Sessions must have at least 2 remaining trials.

ii.
```python
if not np.any(trial_movement):
    dropped_stationary_trials += 1
    continue
if trace_pooled.shape[0] < MIN_POOLED_TIMEPOINTS_PER_TRIAL:
    dropped_short_trials += 1
    continue
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. These filters ensure only trials with sufficient moving data and neural activity are included, following the paper's approach.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the `.mat` file, which contains binary rising-phase calcium event vectors (shape: timepoints x neurons).

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The AI correctly identified these as the binary rise-event traces described in the paper.

## 2-b. How is the `neural` data processed?

i. The AI applies substantial processing: (1) NaN rows in position are removed along with corresponding trace rows, (2) unregistered neurons (NaN in first timepoint) are removed, (3) remaining NaN values are set to 0, (4) only moving frames are kept (speed > 5 cm/s), (5) Gaussian smoothing with sigma=3 frames is applied, (6) average pooling in non-overlapping 3-frame bins. The result is transposed to (neurons, timepoints).

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
# After movement filtering:
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The AI followed the paper's decoder pipeline, applying velocity filtering, smoothing, and temporal binning as described in the reference code's `decode_position_within` function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons not registered in a session are removed by checking if the first timepoint is NaN. After all processing, trials where all neural values are zero are dropped.

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
# Later:
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
```

iii. The first-timepoint NaN check identifies neurons not registered in that session. The zero-activity check removes trials with no usable neural signal.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials are contiguous 1-minute windows from session start. There is no stimulus event to align to.

ii. N/A

iii. The recording is continuous free exploration with no discrete trial events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning from the native 30 Hz (33.33 ms) to 100 ms bins by average-pooling every 3 frames. The metadata reports `time_bin_size: 100.0` ms.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100.0
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. The AI chose 3-frame binning to reduce noise and match the temporal scale used in the paper's decoding analyses.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the `.mat` file, which lists which of the 9 grid positions are blocked in each session. The AI also reads `envs` for environment name but uses `blocked` for the actual input encoding.

ii.
```python
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
```

iii. The AI uses the `blocked` indices to construct an open/blocked mask for the 3x3 grid.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 3x3 binary mask where 1 = open and 0 = blocked, then flattened to a 9-element vector. This is the inverse convention of the reference (which uses 1 = blocked).

ii.
```python
def open_mask_from_blocked(blocked_indices: list[int]) -> np.ndarray:
    open_mask = np.ones(POSITION_BINS_PER_AXIS * POSITION_BINS_PER_AXIS, dtype=np.float32)
    for idx in blocked_indices:
        open_mask[idx] = 0.0
    return open_mask.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS)
...
input_trials.append(env_open_flat.copy())
```

iii. The AI chose an "open mask" representation (1=open, 0=blocked) rather than a "blocked mask" (1=blocked, 0=open) used in the reference. Both encode the same information.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is a static per-trial input (constant across timepoints). The same 9-element vector is used for all trials within a session.

ii.
```python
input_trials.append(env_open_flat.copy())
```

iii. Blocked positions don't change within a session, so no temporal alignment is needed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the `.mat` file, which contains 2D (x, y) coordinates of the animal.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. Position data comes from DeepLabCut tracking as described in the paper.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI applies: (1) NaN position rows are removed, (2) only moving frames are kept, (3) position is scaled by session-wise maximum coordinate / 3, (4) average pooled in 3-frame bins, (5) floored and clipped to get 3x3 grid bin indices, (6) positions landing in blocked bins are remapped to the nearest valid open bin.

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

iii. The session-wise scaling and blocked-bin remapping follow the paper's decoder approach. The AI noted that 441,410 samples were remapped from blocked to open bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into a 3x3 grid (9 categories). The AI uses session-wise scaling (max position / 3) rather than the fixed 75 cm arena size used by the reference. The bin index is `row * 3 + col`.

ii.
```python
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
bin_down = arena_extent / POSITION_BINS_PER_AXIS
pos_scaled = pos_trial / bin_down
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The AI used session-wise scaling to match the reference code's `decode_position_within` function, which also uses `behav.max()` for scaling.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are sampled at the same 30 Hz rate and processed identically: both are filtered to moving frames and average-pooled in the same 3-frame bins, ensuring frame-by-frame alignment.

ii.
```python
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. Identical filtering and pooling operations on both streams maintain temporal alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms (3 native frames at 30 Hz averaged together). This is a rebinning from the native 33.33 ms resolution.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100.0
```

iii. The AI chose 3-frame bins to reduce noise, resulting in 100 ms temporal resolution.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data are aligned by applying the same movement mask and average pooling operations. Input (environment geometry) is static per trial and does not require temporal alignment.

ii.
```python
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. The same frame indices are used for both neural and position data throughout the pipeline.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. (1) NaN position rows are removed along with corresponding trace rows. (2) Unregistered neurons (NaN at first timepoint) are removed. (3) Remaining NaN values in traces are set to 0 via `nan_to_num`. (4) Trials with no moving frames, too few pooled bins, or all-zero neural data are dropped. (5) Position bins landing in blocked regions are remapped to nearest valid bin.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The AI took a comprehensive approach to handling data quality issues, documented in CONVERSION_NOTES.md.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading large `.mat` files via h5py (I/O bound), (2) computing the movement mask with Gaussian-smoothed velocity for each session, (3) Gaussian smoothing of traces for each trial.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
movement_mask = compute_movement_mask(position)
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
```

iii. File I/O dominates due to large trace arrays. The Gaussian filtering adds computational overhead per trial.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop over 1-minute windows could potentially be vectorized by reshaping the entire session data into a 3D array and processing all trials at once, though the variable-length output after movement filtering makes this difficult.

ii.
```python
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    # Per-trial processing
```

iii. The movement filtering creates variable-length trials, making full vectorization impractical.

## 7-c. What processing does the code repeat multiple times?

i. When not using `--sample-only`, the code builds both the full and sample datasets, which means all sessions for the sample animal are processed twice.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
# ... save full ...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The full dataset includes the sample animal's data, so building the sample dataset separately duplicates that processing.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `remap_invalid_bins` function remaps position samples that land in blocked bins to the nearest valid bin. This adds complexity but handles an edge case from temporal pooling. The environment name lookups via `deref_string` and `envs` are used only for summary statistics, not for the actual data.

ii.
```python
env_name = deref_string(h5, h5["envs"][0, session_index])
# Only used for summary statistics
```

iii. The environment name lookups are overhead for validation rather than data conversion.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6: NaN positions removed, unregistered neurons removed, NaN traces zeroed, empty/short trials dropped, blocked-bin positions remapped.

ii. See question 6 code snippets.

iii. See question 6 justification.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: file I/O for large `.mat` files, movement mask computation, and Gaussian smoothing.

ii. See 7-a code snippets.

iii. See 7-a justification.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-trial loop, and also the `remap_invalid_bins` function which loops over invalid samples.

ii.
```python
for idx in np.where(invalid)[0]:
    diffs = valid_bins - bin_xy[idx]
    nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
```

iii. The nearest-valid-bin computation loops over invalid samples individually.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: the full run processes the sample animal's sessions twice.

ii. See 7-c code snippets.

iii. See 7-c justification.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: environment name lookups for summaries, and the blocked-bin remapping adds complexity for edge cases. Additionally, the extensive summary statistics computation is not needed for the actual converted data.

ii. See 7-d code snippets.

iii. See 7-d justification.
