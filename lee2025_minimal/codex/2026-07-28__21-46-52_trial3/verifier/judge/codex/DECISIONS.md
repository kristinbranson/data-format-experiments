# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a fixed list of seven subject `.mat` files with `h5py`, opens each file, and then dereferences session-level HDF5 references for `envs`, `blocked`, `position`, and `trace`. It builds the dataset by iterating over every subject and every session inside each file.

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

for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
    with h5py.File(path, "r") as h5:
        n_sessions = h5["envs"].shape[1]
```

```python
env_name = deref_string(h5, h5["envs"][0, session_index])
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The notes say the agent wanted to preserve the paper’s 7 subjects, 207 sessions, and session ordering. In the trajectory it explicitly says it will “read the `.mat` files session-by-session” and preserve “the paper’s session order and geometry labels.”

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hard-coded `ANIMALS` list, and `subject_idx` is assigned by enumerating that list. The subject names in the output are exactly the strings from `ANIMALS`.

ii.
```python
dataset = {
    "subjects": ANIMALS.copy(),
    "subject_idx": [],
    ...
}

for subject_idx, animal in enumerate(ANIMALS):
    ...
    dataset["subject_idx"].append(subject_idx)
```

iii. The notes say sessions are ordered “by animal in the repository order,” and then list the same seven animal IDs.

## 1-c. How are the data split into sessions?

i. Each entry along the session axis of each animal’s `.mat` file becomes one output session. The AI gets the number of sessions from `h5["envs"].shape[1]` and iterates `session_index` over that range.

ii.
```python
with h5py.File(path, "r") as h5:
    n_sessions = h5["envs"].shape[1]
    ...
    session_indices = list(range(n_sessions))

    for session_index in session_indices:
        ...
        dataset["neural"].append(neural_trials)
        dataset["input"].append(input_trials)
        dataset["output"].append(output_trials)
```

iii. The notes say “Within each animal, session order is preserved exactly as stored in the source files.”

## 1-d. How are the data split into trials?

i. Each session is cut into contiguous 60-second windows (`1800` frames at `30` Hz) starting from session start. Unlike the human reference, the AI initially keeps a final shorter raw window by using `end = min(start + TRIAL_FRAMES, position.shape[0])`; later preprocessing may drop that trial.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS

for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
```

iii. In the trajectory the agent says the stored sessions are “slightly short” of exactly `72,000` frames, so it is “likely to keep a shorter final chunk per session rather than silently drop data.” The notes repeat that choice.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they contain no movement frames, if fewer than 3 movement frames remain before pooling, if pooling produces empty arrays, if the pooled trial has fewer than 10 time bins, or if the pooled neural activity is entirely zero. Sessions with fewer than 2 kept trials are rejected.

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

if pos_pooled.shape[0] == 0 or trace_pooled.shape[0] == 0:
    dropped_short_trials += 1
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

iii. The notes say trials with “no movement frames or fewer than 10 pooled time bins are dropped.” In the trajectory the agent says it tightened trial curation after seeing “low-information edge cases.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data is derived from the raw `trace` field in each `.mat` file.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The notes describe the source traces as the paper’s “rise-extracted binary calcium-event vectors.”

## 2-b. How is the `neural` data processed?

i. The AI removes timepoints with NaN position values, keeps only registered cells, replaces remaining NaNs/Infs with zero, restricts to movement frames, Gaussian-smooths traces with `sigma=3` frames, average-pools them in non-overlapping 3-frame windows, and then transposes to `(neurons, time)`.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]

registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
movement_mask = compute_movement_mask(position)
...
trace_trial = trace_trial[trial_movement]
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The notes justify this by saying the decoder-ready representation should match the paper’s within-session decoder preprocessing: movement selection, Gaussian smoothing, and 3-frame temporal pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered with `registered_mask = ~np.isnan(trace[0])`, so only cells that are present in the session are kept. Timepoints with NaN position are removed before that. Trials are also filtered if they become too short or all-zero after preprocessing.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
...
registered_mask = ~np.isnan(trace[0])
if not np.any(registered_mask):
    raise ValueError(...)

trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
if trace_pooled.shape[0] < MIN_POOLED_TIMEPOINTS_PER_TRIAL:
    dropped_short_trials += 1
    continue
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
```

iii. The notes say “Cells not registered on a given session are dropped,” and the trajectory says the agent wanted to “keep the published per-session registered-cell counts by only dropping cells that are actually absent on that day.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align trials to an experimental event. Trials are contiguous 1-minute windows from session start, and within each trial only movement frames are kept before pooling.

ii.
```python
"temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
...
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    ...
    trial_movement = movement_mask[start:end]
    ...
    pos_trial = pos_trial[trial_movement]
    trace_trial = trace_trial[trial_movement]
```

iii. The notes explicitly say “No event alignment,” and the trajectory says the representation uses “movement selection and 3-frame temporal binning for position decoding.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data is saved at `100 ms` resolution by average-pooling every 3 frames at 30 Hz. Temporal rebinning is applied.

ii.
```python
FPS = 30
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
...
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. The trajectory says the agent chose to “bake that 10 Hz representation into the saved dataset,” and the notes say traces are “average pooled in non-overlapping 3-frame windows.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The final input geometry is derived from the raw `blocked` field for each session. The `envs` string is read for metadata and summary counts, but the actual input vector comes from `blocked`.

ii.
```python
env_name = deref_string(h5, h5["envs"][0, session_index])
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
```

iii. The notes say the agent used the source `blocked` field rather than a geometry-name template because asymmetric environments can appear in session-specific orientations.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI normalizes empty or `[-1]` blocked entries to “nothing blocked,” converts blocked indices into a flattened 3x3 open-mask with `1=open` and `0=blocked`, and uses that 9-element vector as the static per-trial input.

ii.
```python
def normalize_blocked(raw_blocked: np.ndarray) -> list[int]:
    if raw_blocked.size == 0:
        return []
    if raw_blocked.size == 1 and raw_blocked[0] == -1:
        return []
    return sorted(int(x) for x in raw_blocked.tolist())

def open_mask_from_blocked(blocked_indices: list[int]) -> np.ndarray:
    open_mask = np.ones(POSITION_BINS_PER_AXIS * POSITION_BINS_PER_AXIS, dtype=np.float32)
    for idx in blocked_indices:
        open_mask[idx] = 0.0
    return open_mask.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS)
```

```python
env_open_flat = env_open.reshape(-1).astype(np.float32)
input_trials.append(env_open_flat.copy())
```

iii. The trajectory says the sample run showed `blocked` was the “authoritative orientation” for asymmetric environments, so the agent switched to source `blocked` indices directly.

## 3-c. How is `input` *Environment geometry* aligned with the neural data?

i. The geometry input is static rather than time-varying. After trial filtering, the same 9-element vector is copied once for every kept neural trial in the session.

ii.
```python
input_trials: list[np.ndarray] = []
...
input_trials.append(env_open_flat.copy())
```

iii. The notes describe the decoder input as “a static 3x3 environment-open mask” for each trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The final output position labels are derived from the raw `position` field in each `.mat` file.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. The notes say position uses the stored session-aligned DLC trajectories.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI removes NaN position rows, keeps only movement frames, rescales position by a session-specific extent, average-pools position in 3-frame windows, floors and clips the pooled coordinates into a 3x3 grid, then remaps pooled samples that land in blocked bins to the nearest open bin.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
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
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The notes say movement selection is meant to match the paper’s within-session decoder, and that remapping blocked-bin samples “mirrors the spirit of the paper’s decoder cleanup.”

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. After pooling and session-wise scaling, each sample is assigned to one of 9 row-major bins (`r0c0` to `r2c2`). Samples in blocked bins are reassigned to the nearest valid open bin.

ii.
```python
def position_bin_names() -> list[str]:
    names = []
    for r in range(POSITION_BINS_PER_AXIS):
        for c in range(POSITION_BINS_PER_AXIS):
            names.append(f"r{r}c{c}")
    return names

flat = bin_xy[:, 0] * POSITION_BINS_PER_AXIS + bin_xy[:, 1]
invalid = open_mask_flat[flat] == 0
...
nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
flat[idx] = nearest[0] * POSITION_BINS_PER_AXIS + nearest[1]
```

iii. The notes say the output is stored as one categorical variable, `position_bin`, with 9 values `r0c0 ... r2c2`, and blocked-bin remapping is applied after temporal pooling.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by applying the same row filtering, the same 1-minute trial windows, the same movement mask, and the same 3-frame pooling. If pooled lengths disagree, both arrays are trimmed to the shorter length.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]
...
trial_movement = movement_mask[start:end]
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
...
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
if pos_pooled.shape[0] != trace_pooled.shape[0]:
    min_len = min(pos_pooled.shape[0], trace_pooled.shape[0])
    pos_pooled = pos_pooled[:min_len]
    trace_pooled = trace_pooled[:min_len]
```

iii. The metadata says output processing uses the same movement restriction and 3-frame pooling as neural processing.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final dataset uses `100 ms` time bins, not the raw `33.3 ms` frames. Both neural traces and time-varying position labels are rebinned by 3-frame average pooling.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
...
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
"time_bin_size": float(TEMPORAL_BIN_MS),
```

iii. The notes and metadata both state the saved representation is pooled to `100 ms`.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position start from the same raw frame index, have the same NaN-position rows removed, are split with the same 1-minute boundaries, are restricted by the same movement mask, and are pooled with the same 3-frame windows. The input geometry is static and copied once per kept trial.

ii.
```python
position = position[valid_position_rows]
trace = trace[valid_position_rows]
...
trial_movement = movement_mask[start:end]
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
...
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
input_trials.append(env_open_flat.copy())
output_trials.append(flat_bins[np.newaxis, :].astype(np.int64))
neural_trials.append(neural_trial)
```

iii. The notes describe a shared movement-filtered, temporally pooled representation, with static geometry input per trial.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI drops timepoints with NaN positions, converts `blocked=[]` or `blocked=[-1]` to “no blocked bins,” errors out if a session has no registered cells, replaces trace NaNs/Infs with zeros after cell filtering, and drops trials that become too short or empty after preprocessing.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
...
if raw_blocked.size == 0:
    return []
if raw_blocked.size == 1 and raw_blocked[0] == -1:
    return []
...
if not np.any(registered_mask):
    raise ValueError(...)
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
if pos_pooled.shape[0] == 0 or trace_pooled.shape[0] == 0:
    dropped_short_trials += 1
    continue
```

iii. The notes frame these as practical cleanup steps needed to keep the decoder-ready dataset valid after movement filtering and pooling.

## 7-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the large HDF5 arrays for every subject/session, Gaussian smoothing traces for every kept trial, 3-frame pooling of both position and neural data, and per-sample blocked-bin remapping.

ii.
```python
with h5py.File(path, "r") as h5:
    ...
    position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
    trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

```python
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. In the trajectory the agent repeatedly notes the source files are large and slow to load, and its own code adds per-trial smoothing, pooling, and remapping work.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest avoidable loop is the invalid-bin remapping loop, which remaps one sample at a time. The repeated Python trial loop also drives per-trial smoothing/pooling and could only partly be batched.

ii.
```python
for idx in np.where(invalid)[0]:
    diffs = valid_bins - bin_xy[idx]
    nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
    flat[idx] = nearest[0] * POSITION_BINS_PER_AXIS + nearest[1]
    remapped += 1
```

iii. The code uses NumPy for pooling, but nearest-open-bin assignment stays scalar and runs once per invalid pooled sample.

## 7-c. What processing does the code repeat multiple times?

i. It rebuilds the dataset twice in normal execution, once for the full output and again for the sample output. It also dereferences `env_name` once in `build_dataset` and again inside `convert_session`, and it recomputes summary statistics by flattening all outputs after the dataset is already built.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

```python
env_name = deref_string(h5, h5["envs"][0, session_index])
...
neural_trials, input_trials, output_trials, region_idx, summary = convert_session(...)
```

```python
flat_outputs = np.concatenate(
    [trial.reshape(-1) for session in dataset["output"] for trial in session]
)
```

iii. The agent prioritized validation and summary checks, so it re-reads and re-processes source sessions for both full and sample outputs.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The unused helper functions `get_env_mat` and `blocked_indices_from_open_mask` are dead code. The code also computes extensive summary/check metadata and output histograms that are useful for notes and validation but not consumed by downstream decoder analyses.

ii.
```python
def get_env_mat(env_name: str) -> np.ndarray:
    ...

def blocked_indices_from_open_mask(open_mask: np.ndarray) -> list[int]:
    return [idx for idx, value in enumerate(open_mask.reshape(-1)) if value == 0]
```

```python
dataset_stats = {
    "n_sessions": len(dataset["neural"]),
    ...
    "output_bin_counts": {
        position_bin_names()[idx]: int(output_counts.get(idx, 0)) for idx in range(9)
    },
}

checks = {
    "matches_paper_session_count": ...,
    "matches_paper_unique_neurons": ...,
    "matches_paper_rate_maps": ...,
    "matches_environment_histogram": ...,
}
```

iii. These computations support documentation and sanity checks rather than the saved trial tensors themselves.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed entries by removing NaN position rows, normalizing empty/`[-1]` blocked entries, replacing remaining trace NaNs/Infs with zero after cell filtering, and dropping sessions/trials that become invalid after preprocessing.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
...
if raw_blocked.size == 0:
    return []
if raw_blocked.size == 1 and raw_blocked[0] == -1:
    return []
...
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. This is the same cleanup policy described in the notes under trial curation and input normalization.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are subject/session HDF5 reads plus per-trial smoothing, pooling, and blocked-bin remapping.

ii.
```python
with h5py.File(path, "r") as h5:
    ...
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The trajectory explicitly mentions the source files are large and the full run is expensive; the added preprocessing steps account for the rest of the cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The nearest-open-bin remapping loop is the main obvious vectorization target. The code otherwise already uses vectorized NumPy for pooling and masking.

ii.
```python
for idx in np.where(invalid)[0]:
    diffs = valid_bins - bin_xy[idx]
    nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
    flat[idx] = nearest[0] * POSITION_BINS_PER_AXIS + nearest[1]
    remapped += 1
```

iii. This loop runs once per invalid pooled sample and is not vectorized.

## 9-c. What processing does the code repeat multiple times?

i. It rebuilds the data separately for full and sample outputs, repeats some per-session metadata dereferencing, and recomputes aggregate histograms after building the dataset.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

```python
env_name = deref_string(h5, h5["envs"][0, session_index])
...
flat_outputs = np.concatenate(
    [trial.reshape(-1) for session in dataset["output"] for trial in session]
)
```

iii. This repeated work supports both output files and validation summaries.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Dead helper functions and validation-only summary/check computations are the clearest unnecessary work for downstream analyses.

ii.
```python
def get_env_mat(env_name: str) -> np.ndarray:
    ...

def blocked_indices_from_open_mask(open_mask: np.ndarray) -> list[int]:
    ...
```

```python
summary = {
    "source_stats": {...},
    "dataset_stats": dataset_stats,
    "checks": checks,
}
```

iii. These parts are useful for diagnostics and notes, but not for the saved decoder inputs/outputs.
