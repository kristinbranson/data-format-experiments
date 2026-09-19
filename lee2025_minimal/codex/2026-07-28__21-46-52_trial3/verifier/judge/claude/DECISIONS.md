# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a `.mat` file in the data directory. The AI loads these using `h5py` (HDF5 format). It reads `trace`, `position`, `blocked`, and `envs` arrays from each file via HDF5 object reference dereferencing. The list of animals is hardcoded as `ANIMALS`.

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

iii. The AI read the repository structure and determined the `.mat` files use MATLAB v7.3 HDF5 format requiring `h5py`. It hardcoded the animal list after verifying the 7 subjects from the paper.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject. The subject name is the animal ID (e.g., "QLAK-CA1-08"). The list is hardcoded rather than discovered via glob.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
```

iii. The AI determined the subject list from inspecting the data directory and paper, then hardcoded it for reproducibility.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording sessions indexed by the second dimension of `envs`. The AI iterates over `session_indices` within each file, where each becomes a separate session in the output.

ii.
```python
n_sessions = h5["envs"].shape[1]
session_indices = list(range(n_sessions))
for session_index in session_indices:
    neural_trials, input_trials, output_trials, region_idx, summary = convert_session(
        h5=h5, animal=animal, session_index=session_index,
    )
```

iii. The AI verified the total session count (207) matches the paper and validated per-animal session counts.

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into 1-minute (1800 frame) non-overlapping windows. However, within each window, stationary frames are removed (movement filtering), and the remaining frames are temporally pooled into 3-frame bins. This means trials have variable numbers of timepoints.

ii.
```python
TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800
...
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
    trial_movement = movement_mask[start:end]
    ...
    pos_trial = pos_trial[trial_movement]
    trace_trial = trace_trial[trial_movement]
    ...
    trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. The AI stated: "use the paper's movement selection and 3-frame temporal binning for position decoding." It followed the paper's `decode_position_within` function which filters by movement before decoding.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple trial-level quality filters: (1) trials where no frame passes the movement mask are dropped, (2) trials with fewer than 10 pooled time bins are dropped, (3) trials with all-zero neural activity after pooling are dropped. Sessions with fewer than 2 valid trials are rejected entirely.

ii.
```python
if not np.any(trial_movement):
    dropped_stationary_trials += 1
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

iii. The AI noted: "There's at least one trial with almost no retained movement and completely flat neural activity after pooling, so I'm tightening the trial curation slightly to drop those low-information edge cases."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the `.mat` file, which contains calcium imaging traces (binary rise-event vectors) with shape `(timepoints, neurons)`.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The AI verified: "I've verified the source traces are already the binary rise-event vectors described in the paper."

## 2-b. How is the `neural` data processed?

i. The AI applies several processing steps following the paper's decoder code: (1) identify registered cells via first-row NaN check, (2) replace remaining NaN/Inf with zero, (3) restrict to moving frames, (4) Gaussian smooth with sigma=3 frames, (5) average pool in non-overlapping 3-frame bins, (6) transpose to (neurons, time).

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
trace_trial = trace_trial[trial_movement]
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The AI stated: "The paper decoder also applies a 3-frame temporal bin after smoothing traces, which matters because it changes both the neural representation and the effective position labels. I'm checking whether the conversion should bake that 10 Hz representation into the saved dataset or leave it at raw 30 Hz." It decided to bake it in to match the paper's decoder pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons not registered in a session are identified by NaN in the first row of the trace matrix. These are excluded. Remaining NaN/Inf values are replaced with zero. Additionally, trials with all-zero neural activity after pooling are dropped.

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
```

iii. The AI identified registered cells using the first row's NaN pattern, following the structure of the `.mat` files where absent neurons have NaN columns.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials are contiguous 1-minute windows from session start. The metadata explicitly states this.

ii.
```python
"temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
"off_start": None,
"off_end": None,
```

iii. There is no stimulus event to align to in this continuous free-exploration task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 3-frame average pooling, changing the temporal resolution from 30 Hz (~33.33 ms) to 10 Hz (100 ms). This rebinning is baked into the saved data.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100.0 ms
...
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
```

iii. The AI followed the paper's `fit_decoder` function which uses `AvgPool1d(kernel_size=3, stride=3)` for temporal binning, and decided to bake this into the converted data rather than leaving it at the native 30 Hz.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the `.mat` file, which contains indices of blocked reward locations. It is validated against the `envs` variable containing environment names.

ii.
```python
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
```

iii. The AI initially tried to use environment names to look up geometry templates but found that the `blocked` field carries the authoritative orientation for asymmetric geometries: "The first sample run exposed a real data detail: the `blocked` field carries the authoritative orientation of some asymmetric geometries, and it can disagree with a simple name-to-template lookup."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary "open mask" where 1 = open/accessible and 0 = blocked. If no positions are blocked (`[-1]`), all values are 1. This is a static vector per session, the same for all trials.

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

iii. The AI used the open mask representation (1=open, 0=blocked), which is the inverse of the reference approach (1=blocked, 0=not-blocked). Both encode the same information.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the `.mat` file, which contains 2D coordinates (x, y) of the animal in the arena.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. The `position` variable records the animal's location at each timepoint.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is restricted to moving frames, scaled by the session-wise maximum coordinate divided by 3, average-pooled in 3-frame bins, then floored and clipped to yield 3x3 grid labels (0-8). Positions landing in blocked bins are remapped to the nearest open bin.

ii.
```python
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
bin_down = arena_extent / POSITION_BINS_PER_AXIS
...
pos_trial = pos_trial[trial_movement]
pos_scaled = pos_trial / bin_down
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The AI followed the paper's `get_rate_maps` function which uses `np.nanmax(position)` for normalization. It also added blocked-bin remapping because temporal averaging of position coordinates could place samples in blocked regions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into 3 equal bins based on the session-wise maximum coordinate. The bin index is computed as `row * 3 + col`. Positions at boundaries are clipped to valid range [0, 2].

ii.
```python
bin_down = arena_extent / POSITION_BINS_PER_AXIS
pos_scaled = pos_trial / bin_down
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins = bin_xy[:, 0] * POSITION_BINS_PER_AXIS + bin_xy[:, 1]
```

iii. This follows the paper's binning approach using session-wise maximum for normalization rather than a fixed 75 cm arena size.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data undergo the same processing pipeline: both are restricted to moving frames, then both are temporally pooled using the same 3-frame average pooling. They are split into trials using the same 1800-frame windows.

ii.
```python
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
...
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. Both streams go through identical frame selection and temporal pooling, ensuring frame-for-frame alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (1) NaN position rows are removed before processing. (2) Neurons not registered in a session (NaN in first row) are excluded. (3) Remaining NaN/Inf values in traces are replaced with zero. (4) Trials with entirely stationary animals, too few timepoints, or zero neural activity are dropped.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]

registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The AI checked for edge cases including NaN positions and unregistered neurons, and added trial-level quality controls after encountering issues during validation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading the large `.mat` files via h5py (I/O bound), (2) computing movement masks with Gaussian-smoothed velocity, (3) Gaussian smoothing of neural traces, and (4) the full dataset is built twice (once for full, once for sample).

ii. N/A

iii. The `.mat` files contain large neural trace arrays. The additional processing (smoothing, pooling) adds computation beyond simple loading.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial processing loop within `convert_session` iterates over each 1-minute window sequentially. The movement filtering, smoothing, pooling, and binning within each trial could potentially be vectorized across trials.

ii.
```python
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    # ... per-trial processing
```

iii. The loop structure was chosen for clarity and to handle variable-length outputs after movement filtering.

## 6-c. What processing does the code repeat multiple times?

i. The full `build_dataset` function is called twice in the main function: once for the full dataset and once for the sample dataset. This re-reads and re-processes all data files for the sample, rather than extracting the sample from the already-built full dataset.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The AI chose to build both datasets independently for simplicity and correctness, at the cost of redundant I/O and processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Movement velocity filtering removes a substantial fraction of frames that would otherwise be used. (2) Gaussian smoothing and temporal rebinning are decoder-internal preprocessing steps that the reference solution does not bake into the converted data. (3) Blocked-bin remapping of position labels adds complexity. (4) The `get_env_mat` function is defined but not used in the final code (the agent switched to using blocked indices directly).

ii.
```python
def get_env_mat(env_name: str) -> np.ndarray:  # defined but unused
    ...
```

iii. The movement filtering and smoothing follow the paper's decoder code, but the reference solution treats these as decoder-internal steps rather than data conversion steps.
