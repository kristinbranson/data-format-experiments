# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, opens one `.mat` file per animal with `h5py`, and then dereferences per-session `position`, `trace`, `blocked`, and `envs` entries from each HDF5 file. It does not discover subjects by scanning the directory.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]
...
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
    with h5py.File(path, "r") as h5:
        n_sessions = h5["envs"].shape[1]
...
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
```

iii. The recorded justification is that this subject order matches the repository and reproduces the published totals of 7 subjects, 207 sessions, 5,413 tracked neurons, and 69,744 session-level registered-cell instances.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. `dataset["subjects"]` is copied directly from that list, and each appended session gets the enumerated subject index.

ii.
```python
dataset = {
    ...
    "subjects": ANIMALS.copy(),
    "subject_idx": [],
    ...
}
...
for subject_idx, animal in enumerate(ANIMALS):
    ...
    dataset["subject_idx"].append(subject_idx)
```

iii. The AI justified this as preserving the repository order of animals: `QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`, `QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75`.

## 1-c. How are the data split into sessions?

i. Within each animal file, sessions are split by iterating over the second dimension of `h5["envs"]`. Each `session_index` becomes one output session.

ii.
```python
with h5py.File(path, "r") as h5:
    n_sessions = h5["envs"].shape[1]
    ...
    session_indices = list(range(n_sessions))
    for session_index in session_indices:
        neural_trials, input_trials, output_trials, region_idx, summary = convert_session(
            h5=h5,
            animal=animal,
            session_index=session_index,
        )
        dataset["neural"].append(neural_trials)
        dataset["input"].append(input_trials)
        dataset["output"].append(output_trials)
```

iii. The notes say session order is preserved exactly as stored in each source file, and the published 207-session total is used as a sanity check.

## 1-d. How are the data split into trials?

i. Each session is split into contiguous 60 s windows of `1800` source frames, but the AI keeps the final partial raw window if it survives later filtering and pooling. Trials are not fixed-length after preprocessing because the code drops stationary frames and rebins time.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS
...
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
```

iii. The notes say sessions are split into contiguous 1-minute windows from session start, and justify allowing a short final raw window by saying source sessions can be slightly shorter than exactly 72,000 frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with no moving frames, with fewer than 3 moving frames before pooling, with fewer than 10 pooled bins after smoothing/pooling, or whose pooled neural activity is all zeros. It also raises an error if a session has fewer than two surviving trials.

ii.
```python
if not np.any(trial_movement):
    dropped_stationary_trials += 1
    continue

pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
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
    raise ValueError(
        f"{animal} session {session_index}: only {len(neural_trials)} non-empty trials after preprocessing"
    )
```

iii. The notes explicitly justify dropping trials with no movement or fewer than 10 pooled bins, and say this still leaves at least 24 trials in every session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` variable for each session.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The notes say the source traces are the paper's rise-extracted binary calcium-event vectors.

## 2-b. How is the `neural` data processed?

i. After loading `trace`, the AI removes unregistered cells, converts remaining NaNs/Infs to zero, restricts to moving frames, Gaussian-smooths each cell's time series with `sigma=3` frames, average-pools in non-overlapping 3-frame windows, and transposes to `(neurons, time)`.

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
movement_mask = compute_movement_mask(position)
...
trace_trial = trace_trial[trial_movement]
...
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The notes justify this as making decoder-ready time series and claim the movement selection matches the paper's within-session decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells whose first frame is not NaN, removes timepoints where position is missing, converts remaining invalid neural values to zero, and drops trials whose pooled neural activity is entirely zero.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]

registered_mask = ~np.isnan(trace[0])
if not np.any(registered_mask):
    raise ValueError(f"{animal} session {session_index}: no registered cells")

trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
```

iii. The notes explicitly justify dropping cells not registered in a given session. No separate explicit justification is recorded for the first-row NaN mask, zero-filling, or dropping all-zero pooled trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI states there is no event alignment. Instead, it uses contiguous 1-minute windows from session start as the temporal reference.

ii.
```python
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
```

```python
"temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
"off_start": None,
"off_end": None,
```

iii. The notes explicitly describe trialization as contiguous 1-minute windows from session start and do not claim any stimulus- or event-locked alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data from 30 Hz frames to 3-frame bins, yielding a `100 ms` time bin size. It also smooths before pooling.

ii.
```python
FPS = 30
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
TRACE_SMOOTH_SIGMA_FRAMES = 3
...
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
```

iii. The notes justify this as producing decoder-ready time series and explicitly report a final time bin size of 100 ms.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the raw `blocked` field for each session. The code also reads `envs`, but the final input mask is built from `blocked`, not from the environment-name template.

ii.
```python
env_name = deref_string(h5, h5["envs"][0, session_index])
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
```

iii. The notes explicitly say the AI used the source `blocked` field rather than a geometry-name template because asymmetric environments can appear in session-specific orientations.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts `blocked` indices into a length-9 flattened 3x3 open-mask where `1` means open and `0` means blocked, and copies that same static vector into every trial of the session.

ii.
```python
def open_mask_from_blocked(blocked_indices: list[int]) -> np.ndarray:
    open_mask = np.ones(POSITION_BINS_PER_AXIS * POSITION_BINS_PER_AXIS, dtype=np.float32)
    for idx in blocked_indices:
        open_mask[idx] = 0.0
    return open_mask.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS)
...
env_open_flat = env_open.reshape(-1).astype(np.float32)
...
input_trials.append(env_open_flat.copy())
```

iii. The recorded justification is that the decoder input should be a static 3x3 environment-open mask and that `blocked` is safer than geometry-name templates for orientation-specific sessions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position is derived from the raw per-session `position` variable.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. The notes say this uses the stored session-aligned DeepLabCut trajectories.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI removes timepoints with NaN positions, computes a movement mask from smoothed speed, keeps only moving frames, rescales positions by a session-specific maximum extent divided by 3, average-pools the rescaled positions in 3-frame bins, then converts them to categorical grid bins.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]
...
movement_mask = compute_movement_mask(position)
...
pos_trial = pos_trial[trial_movement]
...
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
bin_down = arena_extent / POSITION_BINS_PER_AXIS
...
pos_scaled = pos_trial / bin_down
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
```

iii. The notes justify movement selection with the paper's within-session decoder rule and justify position scaling as a session-wise common 3x3 spatial scale based on the session's maximum coordinate extent.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. After pooling, the AI floors the scaled coordinates, clips each axis to `0..2`, converts the 3x3 `(row, col)` bin to a flat label `0..8`, and remaps any blocked-bin labels to the nearest open bin.

ii.
```python
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
...
output_trials.append(flat_bins[np.newaxis, :].astype(np.int64))
```

iii. The notes justify the 3x3 categorization as a coarse position binning and justify the blocked-bin remapping by saying pooled samples can cross blocked regions and should be snapped to the nearest valid open bin.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are first sliced by the same 1-minute raw windows, then filtered by the same movement mask, then pooled with the same 3-frame kernel. If the pooled lengths differ, both streams are truncated to the shared minimum length.

ii.
```python
pos_trial = position[start:end]
trace_trial = trace[start:end]
trial_movement = movement_mask[start:end]
...
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
...
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
if pos_pooled.shape[0] != trace_pooled.shape[0]:
    min_len = min(pos_pooled.shape[0], trace_pooled.shape[0])
    pos_pooled = pos_pooled[:min_len]
    trace_pooled = trace_pooled[:min_len]
```

iii. The recorded justification is that both streams are processed with the same movement-selection rule and temporal pooling before decoding.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed values are handled by: converting empty `blocked` or `[-1]` to "no blocked bins"; removing rows where position contains NaNs; removing unregistered cells using a NaN mask on the first trace row; zero-filling remaining invalid neural values; dropping short/stationary/all-zero trials; and raising errors on bad shapes or sessions with too few surviving trials.

ii.
```python
def normalize_blocked(raw_blocked: np.ndarray) -> list[int]:
    if raw_blocked.size == 0:
        return []
    if raw_blocked.size == 1 and raw_blocked[0] == -1:
        return []
...
if position.ndim != 2 or position.shape[1] != 2:
    raise ValueError(...)
if trace.ndim != 2 or trace.shape[0] != position.shape[0]:
    raise ValueError(...)
...
valid_position_rows = ~np.isnan(position).any(axis=1)
...
registered_mask = ~np.isnan(trace[0])
...
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes explicitly justify handling `[-1]` as no blocked bins and say cells not registered on a session are dropped. The remaining missing-data handling choices are not separately justified beyond keeping a clean decoder-ready dataset.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming work in this AI code is likely repeated HDF5 loading over all sessions, Gaussian smoothing of speed and neural traces, temporal pooling for every trial, and blocked-bin remapping. It also rebuilds the dataset twice on a full run: once for the full set and once for the sample subset.

ii.
```python
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
    with h5py.File(path, "r") as h5:
        ...
        neural_trials, input_trials, output_trials, region_idx, summary = convert_session(...)
```

```python
speed = gaussian_filter1d(speed, sigma=VELOCITY_SMOOTH_SIGMA_FRAMES)
...
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. No explicit efficiency justification is recorded. The notes focus on matching paper-level counts and decoder performance rather than runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loop over raw trials and the inner loop over invalid pooled bins are the clearest vectorization targets. The subject/session iteration is structurally necessary, but the per-bin nearest-open-bin remapping is especially scalar.

ii.
```python
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    ...
```

```python
for idx in np.where(invalid)[0]:
    diffs = valid_bins - bin_xy[idx]
    nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
    flat[idx] = nearest[0] * POSITION_BINS_PER_AXIS + nearest[1]
    remapped += 1
```

iii. No explicit justification is recorded for leaving these loops in Python.

## 6-c. What processing does the code repeat multiple times?

i. The full run repeats the entire data-loading and preprocessing pipeline a second time to generate the sample dataset. It also dereferences `env_name` both in `build_dataset` for counting and again inside `convert_session`, and it recomputes summary statistics after conversion.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

```python
env_name = deref_string(h5, h5["envs"][0, session_index])
source_stats["environment_counts"][env_name] += 1
...
def convert_session(...):
    env_name = deref_string(h5, h5["envs"][0, session_index])
```

iii. No explicit justification is recorded for the repeated processing. The notes only state that both full and sample outputs were required.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes substantial bookkeeping that is not needed for the decoder inputs/outputs themselves: environment histograms, output-bin counts, remapped-bin counts, detailed `session_info`, and consistency `checks`. It also defines helper functions such as `get_env_mat` and `blocked_indices_from_open_mask` that are unused in the final script.

ii.
```python
dataset_stats = {
    "n_sessions": len(dataset["neural"]),
    "n_trials_total": int(sum(len(session) for session in dataset["neural"])),
    ...
    "total_remapped_invalid_position_bins": int(
        sum(info["n_remapped_invalid_position_bins"] for info in session_info)
    ),
}
...
checks = {
    "matches_paper_session_count": source_stats["raw_session_count"] == 207 if not sample_only else True,
    "matches_paper_unique_neurons": source_stats["raw_unique_neurons"] == 5413 if not sample_only else True,
    "matches_paper_rate_maps": source_stats["raw_registered_cell_session_count"] == 69744 if not sample_only else True,
    "matches_environment_histogram": (
        dict(source_stats["environment_counts"]) == EXPECTED_ENV_COUNTS if not sample_only else True
    ),
}
```

```python
def get_env_mat(env_name: str) -> np.ndarray:
    ...

def blocked_indices_from_open_mask(open_mask: np.ndarray) -> list[int]:
    ...
```

iii. The notes justify these as sanity checks against the paper and as documentation of validation results, not as part of the downstream decoder representation itself.
