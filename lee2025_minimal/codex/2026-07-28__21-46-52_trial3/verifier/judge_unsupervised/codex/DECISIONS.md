# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven mouse IDs, opens each corresponding MATLAB HDF5 file with `h5py`, and iterates over every session index in `h5["envs"]`. For each session it dereferences the session-level `envs`, `blocked`, `position`, and `trace` entries and converts that session with `convert_session(...)`. Trials are not loaded as separate source objects; they are created later by splitting the session time series.

ii. ```python
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
        ...
        for session_index in session_indices:
            env_name = deref_string(h5, h5["envs"][0, session_index])
            ...
            neural_trials, input_trials, output_trials, region_idx, summary = convert_session(...)
```

iii. In `CONVERSION_NOTES.md`, the agent says it matched the paper’s headline counts by reading the source `.mat` files directly and preserving the repository’s subject/session order. In the trajectory it explicitly says it chose to read the `.mat` files session-by-session and preserve the source ordering rather than guess from downstream products.

## 1-b. How are the data split into subjects?

i. Subjects are split by filename and by the fixed `ANIMALS` list. Each animal contributes one block of sessions, and `subject_idx` stores the integer index of the animal for each appended session.

ii. ```python
dataset = {
    ...
    "subjects": ANIMALS.copy(),
    "subject_idx": [],
    ...
}

for subject_idx, animal in enumerate(ANIMALS):
    ...
    for session_index in session_indices:
        ...
        dataset["subject_idx"].append(subject_idx)
```

iii. The agent documents that session ordering is “by animal in the repository order” and that within each animal the stored session order is preserved. That is also stated in the trajectory plan and final summary.

## 1-c. How are the data split into sessions?

i. Sessions are split by per-animal session index. The code uses the second dimension of `h5["envs"]` as the authoritative session count and processes each `session_index` independently. Each converted session contributes one element to `dataset["neural"]`, `dataset["input"]`, `dataset["output"]`, and `dataset["brain_region_idx"]`.

ii. ```python
with h5py.File(path, "r") as h5:
    n_sessions = h5["envs"].shape[1]
    ...
    for session_index in session_indices:
        ...
        dataset["neural"].append(neural_trials)
        dataset["input"].append(input_trials)
        dataset["output"].append(output_trials)
        dataset["brain_region_idx"].append(region_idx)
```

iii. The justification in `CONVERSION_NOTES.md` is that the session order should match the source files exactly. The trajectory also notes that the agent verified the apparent session-count mismatch for `QLAK-CA1-51` and retained the true total of 207 sessions.

## 1-d. How are the data split into trials?

i. Trials are not present in the raw source; the agent creates them by chopping each continuous session into contiguous 1-minute windows from session start. At 30 Hz, each raw trial window is `1800` frames. The final window can be shorter because many sessions have slightly fewer than exactly `72,000` frames.

ii. ```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS

for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
```

iii. The agent’s notes say “Sessions are split into contiguous 1-minute windows from session start” and justify keeping a shorter last chunk because the stored frame counts are slightly short of exactly 40 minutes. The trajectory says this was chosen because the task instructions explicitly requested 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. After making each 1-minute chunk, the agent drops trials with no movement frames, trials with fewer than `3` movement frames, trials that pool down to zero bins, trials with fewer than `10` pooled time bins, and trials whose pooled neural matrix is entirely zero. It also raises an error if a session ends up with fewer than two kept trials.

ii. ```python
trial_movement = movement_mask[start:end]

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

if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. The agent explains in `CONVERSION_NOTES.md` and the trajectory that these extra exclusions were added to avoid low-information edge cases, warnings in `train_decoder.py`, and sessions with too few valid trials for evaluation. The notes say this “still leaves at least 24 trials in every session.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` field, session by session. The code does not use the precomputed rate maps in `maps`; it uses the time-series traces directly.

ii. ```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
...
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The agent’s notes explicitly say the source traces are the paper’s “rise-extracted binary calcium-event vectors.” In the trajectory it says it verified that the source traces were already the binary rise-event representation described in the paper.

## 2-b. How is the `neural` data processed?

i. The code removes session-absent cells, replaces remaining NaN/inf values with zero, restricts the trace to movement frames within each trial, Gaussian-smooths the time axis with `sigma=3` frames, average-pools in non-overlapping 3-frame windows, and finally transposes to `(n_neurons, n_timepoints)` for the saved trial array.

ii. ```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
trace_trial = trace_trial[trial_movement]
...
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The justification is in both `CONVERSION_NOTES.md` and the trajectory: the agent wanted the saved neural representation to match the paper’s within-session decoder preprocessing, which smooths traces and temporally bins them at 3 frames, while still preserving all registered cells.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code filters neural data in two main ways: it drops cells that are unregistered in a session (`NaN` in the session trace), and it discards whole trials if the pooled neural activity is all zeros. It does not apply the paper decoder’s additional per-session active-cell threshold (`> 5` events on movement frames).

ii. ```python
registered_mask = ~np.isnan(trace[0])
if not np.any(registered_mask):
    raise ValueError(...)
...
if np.all(neural_trial == 0):
    dropped_zero_neural_trials += 1
    continue
```

iii. The agent explicitly justifies keeping all registered cells in `CONVERSION_NOTES.md`: it says this preserves the published `69,744` session-level cell count. In the trajectory it says it wanted to “keep the published per-session registered-cell counts by only dropping cells that are actually absent on that day.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not align to a discrete behavioral event. Instead, neural trials are aligned to contiguous 1-minute windows from session start, with movement-frame selection and temporal pooling performed inside each window.

ii. ```python
"temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
"off_start": None,
"off_end": None,
```

iii. The agent justifies this by the task setup: the experiment is continuous exploration, and the instructions themselves define the decoder trials as 1-minute windows within long sessions. The same wording appears in the metadata and notes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved data are at 100 ms resolution. The code rebins from the native 30 Hz recording to non-overlapping 3-frame bins, so `3 / 30 s = 0.1 s = 100 ms`.

ii. ```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
...
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
```

iii. The agent’s notes say it chose 100 ms because the paper’s within-session decoder applies a 3-frame temporal bin after smoothing. The trajectory explicitly mentions deciding whether to “bake that 10 Hz representation into the saved dataset” and then doing so.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The saved decoder input comes primarily from the raw `blocked` field, which stores omitted 3x3 partition indices per session. `envs` is also read, but only for bookkeeping and summaries; it is not the direct source of the saved mask.

ii. ```python
env_name = deref_string(h5, h5["envs"][0, session_index])
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
...
input_trials.append(env_open_flat.copy())
```

iii. The justification is explicit in `CONVERSION_NOTES.md`: the agent says it used `blocked` “rather than a geometry-name template, because asymmetric environments can appear in session-specific orientations.” In the trajectory it says a sample run showed that `blocked` was the “authoritative orientation” for some asymmetric geometries.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code converts the list of blocked coarse-bin indices into a 3x3 open-mask with `1=open` and `0=blocked`, flattens that to length 9, and copies the same static vector into every trial from the session.

ii. ```python
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

iii. The agent’s notes say the decoder input is “a static 3x3 environment-open mask” and that `1=open`, `0=blocked`. The trajectory says this was chosen to match “which part of the arena is blocked” from the task instructions while respecting session-specific geometry orientation.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output variable is derived from the raw `position` time series for each session. No other raw variable contributes directly to the saved mouse-position labels.

ii. ```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
...
pos_trial = position[start:end]
```

iii. The notes say “Position uses the stored session-aligned DLC trajectories.” The trajectory also says the agent was checking the raw position structure and the reference decoder’s behavior so it could base the output labels on the same stored trajectories.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code removes any NaN position rows, keeps only movement frames, rescales position to a session-wise 3x3 grid using the session’s maximum coordinate extent, average-pools the 2D position in 3-frame bins, floors the pooled coordinates to integer bins, clips them into `[0, 2]`, and converts them to a flattened 0-8 category.

ii. ```python
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
...
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The notes justify these steps as a coarse adaptation of the paper’s within-session position decoder: movement-frame selection with a 5 cm/s threshold, session-wise common spatial scaling, and 3-frame temporal pooling. The trajectory repeatedly says the agent was trying to preserve the paper’s normalization and temporal-binning logic while converting to the required 3x3 decoder output.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. After pooling, each 2D position sample is converted to one of 9 categorical bins by flooring the scaled x/y coordinates into a 3x3 grid, clipping to valid indices, flattening with `row * 3 + col`, and then remapping any sample that falls into a blocked coarse bin to the nearest open bin.

ii. ```python
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
...
output_trials.append(flat_bins[np.newaxis, :].astype(np.int64))
```

iii. The agent justifies the remapping in `CONVERSION_NOTES.md` by saying temporal pooling can push a sample into a blocked coarse bin and that remapping “mirrors the spirit of the paper’s decoder cleanup.” The trajectory likewise says pooling produced labels inside blocked bins and that it therefore added an explicit remap.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The output labels are aligned to neural data by applying the same per-trial movement mask and the same 3-frame temporal pooling to both position and traces. If pooled lengths differ, both arrays are truncated to the same minimum length before being saved.

ii. ```python
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

iii. The agent’s notes state that both neural and position streams are restricted to movement frames and pooled in the same non-overlapping 3-frame windows. The trajectory says it was deliberately checking alignment between different streams before finalizing the saved representation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several small data issues defensively: it removes rows with NaN position values, drops session-absent cells identified by NaNs in the trace, converts remaining NaN/inf trace values to zero, treats `blocked == [-1]` as “no blocked bins,” truncates pooled position and neural arrays to the same minimum length if they differ, and errors out if a session has no registered cells or too few valid trials.

ii. ```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]

registered_mask = ~np.isnan(trace[0])
if not np.any(registered_mask):
    raise ValueError(...)

trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)

def normalize_blocked(raw_blocked: np.ndarray) -> list[int]:
    if raw_blocked.size == 0:
        return []
    if raw_blocked.size == 1 and raw_blocked[0] == -1:
        return []
```

iii. The notes frame these choices as practical cleanup needed to pass verification and preserve the published counts. The trajectory also says the agent tightened trial curation after seeing flat-neural edge cases in the sample run.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work in `convert_data.py` is reading every large `.mat` file, dereferencing session objects, processing every session/trial time series, and running `gaussian_filter1d` plus temporal pooling on large trial matrices. The code also rebuilds a sample dataset after the full dataset, repeating part of the conversion work.

ii. ```python
for subject_idx, animal in enumerate(ANIMALS):
    with h5py.File(path, "r") as h5:
        ...
        for session_index in session_indices:
            ...
            neural_trials, input_trials, output_trials, region_idx, summary = convert_session(...)

trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The trajectory repeatedly mentions that the “source files are bigger than they looked,” that full loading takes real time, and that the final full pickle is about 3.3 GB. Those comments explain where the agent saw the runtime cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization target is `remap_invalid_bins`, which loops over every invalid pooled sample and computes nearest valid bins one by one. The per-trial Python loop over `range(0, position.shape[0], TRIAL_FRAMES)` is also coarse-grained and could be reorganized, but the nearest-neighbor remap loop is the clearest avoidable hotspot.

ii. ```python
def remap_invalid_bins(bin_xy: np.ndarray, open_mask_flat: np.ndarray) -> tuple[np.ndarray, int]:
    ...
    valid_bins = np.argwhere(open_mask_flat.reshape(POSITION_BINS_PER_AXIS, POSITION_BINS_PER_AXIS) > 0)
    for idx in np.where(invalid)[0]:
        diffs = valid_bins - bin_xy[idx]
        nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
        flat[idx] = nearest[0] * POSITION_BINS_PER_AXIS + nearest[1]
        remapped += 1
```

iii. The agent did not explicitly justify this in notes, but the structure of the code makes the inefficiency visible: the remap is pure Python over many pooled samples, and the notes report `441,410` remapped samples on the full dataset.

## 6-c. What processing does the code repeat multiple times?

i. The conversion work is repeated for the sample subset after the full dataset is already built, so sessions from `QLAK-CA1-08` are processed twice. Within each session, the code also repeats environment/session dereferencing and summary bookkeeping even though only the converted arrays are needed downstream.

ii. ```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
...
env_name = deref_string(h5, h5["envs"][0, session_index])
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
...
session_summary = {
    "animal": animal,
    "session_index_within_animal": session_index,
    ...
}
```

iii. The trajectory shows the agent intentionally ran both full and sample conversions and then both full and sample verifier/training passes. The code mirrors that by recomputing the sample subset instead of slicing it from the already-built full conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive summaries, counters, and metadata that are useful for validation but not used by downstream decoder training: source-statistic counters, output histograms, trial-length summaries, remap counters, and per-session `session_info`. It also computes `env_name` for summaries even though the saved decoder input itself only uses the blocked-mask representation.

ii. ```python
session_summary = {
    "animal": animal,
    "session_index_within_animal": session_index,
    "environment": env_name,
    "blocked_bins": raw_blocked,
    ...
    "n_remapped_invalid_position_bins": int(remapped_samples),
    "trial_timepoints_summary": summarize_trial_lengths(neural_trials),
}

dataset_stats["output_bin_counts"] = {
    position_bin_names()[idx]: int(output_counts.get(idx, 0)) for idx in range(9)
}

summary = {
    "source_stats": {...},
    "dataset_stats": dataset_stats,
    "checks": checks,
}
```

iii. The agent’s notes make clear that these were produced for sanity checks and documentation rather than model input. The trajectory says it added explicit sanity checks against paper counts and then wrote those validation details into `CONVERSION_NOTES.md`.
