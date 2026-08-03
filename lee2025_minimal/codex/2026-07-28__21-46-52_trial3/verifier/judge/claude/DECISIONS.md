# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a `.mat` file in the data directory. Files are loaded using `h5py` (HDF5/MATLAB v7.3+ format). Within each file, `trace`, `position`, `blocked`, and `envs` reference arrays are used to access per-session data. The AI hardcodes the list of 7 animal names (ANIMALS) and constructs paths from them.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
    with h5py.File(path, "r") as h5:
        n_sessions = h5["envs"].shape[1]
```

iii. The AI explicitly hardcodes the 7 animal names based on the paper. It reads the `envs` field to determine session count, which also enables reading the environment name per session. The CONVERSION_NOTES.md states these counts were verified against the paper (207 sessions, 5413 unique neurons, 69744 registered cell-session instances).

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject. The subject names are hardcoded in the `ANIMALS` list. The `subject_idx` array maps each session to its subject index in that list.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for subject_idx, animal in enumerate(ANIMALS):
    ...
    dataset["subject_idx"].append(subject_idx)
```

iii. The AI verified that the 7 animals match the paper's reported subjects.

## 1-c. How are the data split into sessions?

i. Within each `.mat` file, sessions are indexed by iterating over the columns of the `envs` array (and corresponding `trace`, `position`, `blocked` references). Each recording session becomes a separate session in the output.

ii.
```python
n_sessions = h5["envs"].shape[1]
...
for session_index in session_indices:
    env_name = deref_string(h5, h5["envs"][0, session_index])
    ...
    neural_trials, input_trials, output_trials, region_idx, summary = convert_session(
        h5=h5, animal=animal, session_index=session_index,
    )
```

iii. The AI verified that the total session count across all animals is 207, matching the paper.

## 1-d. How are the data split into trials?

i. Each session's continuous recording is split into contiguous 1-minute (1800-frame at 30 Hz) windows from the session start. However, unlike the reference, the AI first filters for movement frames and then splits into trial windows. After movement filtering, Gaussian smoothing, and temporal pooling (3-frame bins), each trial has variable length. Trials with no movement or fewer than 10 pooled time bins are dropped.

ii.
```python
TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800

for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
    trial_movement = movement_mask[start:end]

    if not np.any(trial_movement):
        dropped_stationary_trials += 1
        continue

    pos_trial = pos_trial[trial_movement]
    trace_trial = trace_trial[trial_movement]
```

iii. The AI's CONVERSION_NOTES.md explains: "Sessions are split into contiguous 1-minute windows from session start. ... Trials with no movement frames or fewer than 10 pooled time bins are dropped." This results in 8,266 total trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on three criteria: (1) trials with no movement frames are dropped, (2) trials with fewer than `TEMPORAL_BIN_FRAMES` (3) moving frames are dropped, (3) trials with fewer than `MIN_POOLED_TIMEPOINTS_PER_TRIAL` (10) pooled time bins after processing are dropped, and (4) trials where all neural activity is zero are dropped. Sessions must have at least 2 remaining trials.

ii.
```python
if not np.any(trial_movement):
    dropped_stationary_trials += 1
    continue
...
if pos_trial.shape[0] < TEMPORAL_BIN_FRAMES:
    dropped_short_trials += 1
    continue
...
if trace_pooled.shape[0] < MIN_POOLED_TIMEPOINTS_PER_TRIAL:
    dropped_short_trials += 1
    continue
...
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
...
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. The CONVERSION_NOTES.md states: "Trials with no movement frames or fewer than 10 pooled time bins are dropped. This still leaves at least 24 trials in every session, and usually about 40."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the `.mat` file, which contains the paper's rise-extracted binary calcium-event vectors (deconvolved spike traces).

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The CONVERSION_NOTES.md confirms: "Source traces are the paper's rise-extracted binary calcium-event vectors."

## 2-b. How is the `neural` data processed?

i. The AI applies substantial processing beyond the reference:
1. Rows with NaN positions are removed from both trace and position
2. Unregistered neurons (NaN in first row) are removed
3. Remaining NaN values are replaced with 0
4. Only frames where the mouse is moving (speed > 5 cm/s) are kept
5. Gaussian smoothing with sigma=3 frames is applied
6. Average pooling in non-overlapping 3-frame bins is applied

ii.
```python
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
movement_mask = compute_movement_mask(position)
...
trace_trial = trace_trial[trial_movement]
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The CONVERSION_NOTES.md justifies: "For decoder-ready time series, traces are: restricted to movement frames, Gaussian-smoothed with sigma=3 frames, average pooled in non-overlapping 3-frame windows." The movement filtering is described as matching "the same rule used by the paper's within-session position decoder."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons not registered in a session are removed using the first row's NaN pattern. Trials where all neural activity is zero after processing are dropped.

ii.
```python
registered_mask = ~np.isnan(trace[0])
if not np.any(registered_mask):
    raise ValueError(f"{animal} session {session_index}: no registered cells")
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
```

iii. The CONVERSION_NOTES.md states: "Cells not registered on a given session are dropped from that session. I kept all registered cells so the converted data preserves the published 69,744 session-level cell count."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous and trials are contiguous 1-minute windows from session start. However, within each trial, only movement frames are kept and then temporally pooled, so there is no fixed alignment event.

ii.
```python
"temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
"off_start": None,
"off_end": None,
```

iii. The AI's metadata explicitly states no event alignment. This matches the nature of the free-exploration task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning. Raw data is at 30 Hz (~33.33 ms per frame). After movement filtering, data is average-pooled in non-overlapping 3-frame bins, resulting in an effective time bin size of 100 ms.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100.0

trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. The CONVERSION_NOTES.md states: "Time bin size: 100 ms." The metadata records `time_bin_size: 100.0`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the `.mat` file, which contains indices of blocked reward locations for each session. This is converted into a 3x3 open mask (1=open, 0=blocked).

ii.
```python
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
...
def open_mask_from_blocked(blocked_indices: list[int]) -> np.ndarray:
    open_mask = np.ones(POSITION_BINS_PER_AXIS * POSITION_BINS_PER_AXIS, dtype=np.float32)
    for idx in blocked_indices:
        open_mask[idx] = 0.0
    return open_mask.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS)
```

iii. The CONVERSION_NOTES.md explains: "The decoder input is a static 3x3 environment-open mask. I used the source blocked field rather than a geometry-name template, because asymmetric environments can appear in session-specific orientations. Input values are 1=open, 0=blocked."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices from the raw data are converted to a flattened 9-element open mask. The mask has 1 for open positions and 0 for blocked positions. The `[-1]` sentinel (meaning no blocked positions) is handled by returning an empty list. The mask is stored as a flat (9,) float32 array, constant across all trials in a session.

ii.
```python
def normalize_blocked(raw_blocked: np.ndarray) -> list[int]:
    if raw_blocked.size == 0:
        return []
    if raw_blocked.size == 1 and raw_blocked[0] == -1:
        return []
    return sorted(int(x) for x in raw_blocked.tolist())

env_open_flat = env_open.reshape(-1).astype(np.float32)
input_trials.append(env_open_flat.copy())
```

iii. The AI chose to represent the environment as an "open mask" (1=open, 0=blocked) rather than a "blocked mask" (1=blocked, 0=open) as used in the reference. This is an inverted encoding. The input_names are `env_open_r0c0` through `env_open_r2c2`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable in the `.mat` file, which contains 2D (x, y) coordinates of the mouse in the arena.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. The position data records the animal's location in the open field arena at each timepoint.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI applies several processing steps:
1. NaN position rows are removed
2. Only movement frames (speed > 5 cm/s) are kept
3. Position is scaled by the session's maximum coordinate extent divided by 3
4. Positions are average-pooled in 3-frame bins
5. Pooled positions are floored to integer bin indices and clipped to [0, 2]
6. Positions falling in blocked bins are remapped to the nearest valid open bin

ii.
```python
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
bin_down = arena_extent / POSITION_BINS_PER_AXIS

pos_trial = pos_trial[trial_movement]
pos_scaled = pos_trial / bin_down
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)

bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The CONVERSION_NOTES.md explains the session-wise common spatial scale and the blocked-bin remapping logic.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into a 3x3 grid (9 categories). The arena is divided using a session-specific scale factor: `max(position) / 3`. The scaled position is floored to integer bin indices, then clipped to [0, 2]. The flat bin index is `row * 3 + col`. Positions in blocked bins are remapped to the nearest valid open bin.

ii.
```python
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
bin_down = arena_extent / POSITION_BINS_PER_AXIS
pos_scaled = pos_trial / bin_down
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The AI uses session-specific spatial scaling rather than a fixed 75 cm arena size. The reference uses `np.linspace(0, 75, 4)[1:-1]` (fixed 25 cm bin edges). The AI also remaps blocked-bin positions, which the reference does not do.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are processed in lockstep: both are restricted to the same movement frames, then both are average-pooled with the same 3-frame bins, ensuring frame-for-frame alignment.

ii.
```python
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
...
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. By applying identical movement filtering and temporal pooling to both streams, alignment is preserved throughout processing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/invalid data are handled:
1. NaN position rows: removed from both position and trace arrays
2. Unregistered neurons (NaN in first row): removed from the session
3. Remaining NaN/inf values in traces: replaced with 0 via `np.nan_to_num`
4. Speed NaN/inf: replaced with 0
5. Positions in blocked bins after temporal pooling: remapped to nearest valid bin
6. Short/incomplete trials: dropped

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

iii. The AI handles multiple edge cases robustly, including blocked-bin remapping which is not present in the reference.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading the large `.mat` files via `h5py`, (2) computing Gaussian smoothing on the neural traces for each trial, and (3) the blocked-bin remapping loop which iterates over individual invalid samples.

ii.
```python
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
...
for idx in np.where(invalid)[0]:
    diffs = valid_bins - bin_xy[idx]
    nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
```

iii. The Gaussian smoothing is applied per-trial (inside a loop over trials), and the blocked-bin remapping iterates sample-by-sample.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `remap_invalid_bins` function contains a Python loop over individual invalid samples that could be vectorized using scipy's `cdist` or broadcasting. The trial-level processing loop could potentially be partially vectorized.

ii.
```python
for idx in np.where(invalid)[0]:
    diffs = valid_bins - bin_xy[idx]
    nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
    flat[idx] = nearest[0] * POSITION_BINS_PER_AXIS + nearest[1]
    remapped += 1
```

iii. The loop processes 441,410 remapped samples across the full dataset.

## 6-c. What processing does the code repeat multiple times?

i. The code builds the full dataset and then rebuilds the sample dataset from scratch (re-reading and re-processing the `.mat` files), rather than extracting the sample as a subset of the already-built full dataset. The `open_mask_from_blocked` computation is also repeated for every session even when multiple sessions share the same environment.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The sample is a subset of the first animal's sessions, so it could have been extracted from the full dataset without reprocessing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs movement filtering, Gaussian smoothing, and temporal rebinning that go beyond what the reference solution does. While these are defensible processing choices inspired by the paper's decoder methodology, they add complexity and change the data substantially from the raw format. The blocked-bin remapping is also extra processing not present in the reference. Additionally, the code computes and stores extensive per-session summary statistics in `session_info` metadata.

ii.
```python
movement_mask = compute_movement_mask(position)
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The reference solution keeps data at native 30 Hz with no movement filtering, no smoothing, and no temporal rebinning. The AI added these based on the paper's decoder methodology, but the instructions asked for the same processing as described in the reference code.
