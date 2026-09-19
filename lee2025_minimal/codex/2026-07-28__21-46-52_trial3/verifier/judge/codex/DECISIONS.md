# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, opens one `.mat` HDF5 file per animal with `h5py`, and iterates through every session in each file. Within each session it dereferences the `envs`, `blocked`, `position`, and `trace` entries and then converts that session into trial lists with `convert_session`.

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
        for session_index in session_indices:
            env_name = deref_string(h5, h5["envs"][0, session_index])
            ...
            neural_trials, input_trials, output_trials, region_idx, summary = convert_session(
                h5=h5,
                animal=animal,
                session_index=session_index,
            )
```

iii. In the trajectory, the AI first planned to "load the reference joblib datasets" but then switched to direct HDF5 access because it could "avoid a lot of slow full-file loads by reading the `.mat` HDF5 structure directly" and because that was more reliable for curation checks. It also said the conversion would "read the `.mat` files session-by-session" and preserve source order.

## 1-b. How are the data split into subjects?

i. Subjects are split by file. Each animal in the hard-coded `ANIMALS` list is treated as one subject, and the subject index used for every converted session is the index of that animal in the list.

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

iii. The AI justified subject handling by matching the paper’s reported seven animals and preserving "session order from the source files." Its final summary says it matched "7 subjects" and used the repository-order animal list.

## 1-c. How are the data split into sessions?

i. Each per-animal `.mat` file is split into sessions by iterating over the second dimension of `h5["envs"]`. Every `session_index` becomes one output session, and the other raw arrays are dereferenced using that same index.

ii.
```python
with h5py.File(path, "r") as h5:
    n_sessions = h5["envs"].shape[1]
    ...
    for session_index in session_indices:
        env_name = deref_string(h5, h5["envs"][0, session_index])
        raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
        position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
        trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The trajectory says the AI wanted to "preserve the paper’s session order" and explicitly checked the session count discrepancy, concluding that the raw dataset already contains the paper’s 207 sessions because `QLAK-CA1-51` has 21 sessions.

## 1-d. How are the data split into trials?

i. The AI splits each session into contiguous 60 s windows using `TRIAL_FRAMES = 30 * 60 = 1800` frames, but it does not discard the final incomplete raw window. Instead it processes the last shorter chunk too, then later drops only chunks that fail movement/length checks.

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

iii. The AI explicitly justified this in the trajectory: "sessions are nominally 40 minutes but the stored frame counts are slightly short, so I’m likely to keep a shorter final chunk per session rather than silently drop data." Its notes repeat that "the final raw window can be shorter than 60 s."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered aggressively after splitting. The AI drops trials with no movement frames, trials with fewer than 3 movement frames before pooling, trials that become empty after pooling, trials with fewer than 10 pooled bins, and trials whose pooled neural activity is entirely zero. It also errors out if fewer than 2 trials remain in a session.

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
if pos_pooled.shape[0] == 0 or trace_pooled.shape[0] == 0:
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

iii. The trajectory says the AI was "tightening the trial curation slightly to drop those low-information edge cases" after seeing "at least one trial with almost no retained movement and completely flat neural activity after pooling." It also said it wanted the sample/full verifier to pass with no warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data is derived from the per-session `trace` array in the `.mat` file.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The trajectory says the AI "verified the source traces are already the binary rise-event vectors described in the paper" and described them as "the paper’s rise-extracted binary calcium-event vectors."

## 2-b. How is the `neural` data processed?

i. The AI treats the raw traces as binary rise-event vectors, removes unregistered cells, replaces remaining NaNs/Infs with zero, restricts each trial to movement frames only, smooths the trace along time with a Gaussian filter (`sigma=3` frames), average-pools in non-overlapping 3-frame bins, transposes to `(neurons, time)`, and stores float32.

ii.
```python
registered_mask = ~np.isnan(trace[0])
...
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
movement_mask = compute_movement_mask(position)
...
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
...
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
...
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The trajectory states that the AI was trying to match "the paper’s movement selection and 3-frame temporal binning for position decoding." It specifically noted that "the paper decoder also applies a 3-frame temporal bin after smoothing traces" and that it chose to "bake that 10 Hz representation into the saved dataset."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells whose first sample is not NaN, effectively treating those as registered cells for that session. After that, it drops entire trials if the pooled neural matrix is all zeros. Position rows with NaNs are also removed before neural processing, which co-filters the neural timepoints.

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

iii. The AI justified this as keeping "all registered cells so the converted data preserves the published `69,744` session-level cell count." It also said later that it dropped flat pooled trials because they were "low-information edge cases."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to any experimental event. It defines trials as contiguous 1-minute windows from the start of each session and states explicitly that there is "No event alignment."

ii.
```python
"temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start.",
"off_start": None,
"off_end": None,
```

iii. The trajectory says the conversion would "split sessions into contiguous 1-minute windows" and preserve source order, not align around a stimulus or action event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data is stored at 100 ms resolution. The AI rebins from the native 30 Hz stream to non-overlapping 3-frame bins after movement filtering, so trial lengths vary depending on how many moving frames survived.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
...
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
...
"time_bin_size": float(TEMPORAL_BIN_MS),
```

iii. The trajectory says the AI debated whether to keep the raw 30 Hz data but concluded that "the paper decoder also applies a 3-frame temporal bin after smoothing traces" and proceeded with "the current 100 ms, movement-filtered representation."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the per-session `blocked` field, while also reading `envs` only for reporting and checks. After an initial attempt to infer geometry from `env_name`, the final code treats `blocked` as authoritative.

ii.
```python
env_name = deref_string(h5, h5["envs"][0, session_index])
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
```

iii. The trajectory says the AI discovered that "`blocked` field carries the authoritative orientation of some asymmetric geometries" and switched the encoder so "the geometry representation matches each recorded session exactly."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the blocked indices into a length-9 open-mask vector with `1=open` and `0=blocked`, flattens the 3x3 geometry, and copies that same static vector into every trial of the session.

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
...
"input_names": flatten_env_names(),
```

iii. The AI justified this by saying "the decoder input is a static 3x3 environment-open mask" and that using `blocked` directly is safer than a name-template lookup because asymmetric layouts can appear in different orientations.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the per-session `position` array in the `.mat` file.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. The trajectory says "Position uses the stored session-aligned DLC trajectories" and repeatedly describes the task as decoding mouse location from those recorded positions.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI removes any timepoints with NaNs in position, restricts to movement-only frames within each 1-minute chunk, rescales coordinates by a session-specific arena extent (`max(position)/3`), average-pools the scaled positions in 3-frame bins, floors and clips the pooled coordinates into a 3x3 grid, remaps pooled samples that land in blocked bins to the nearest open bin, and stores the flattened 0-8 category sequence.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
if not np.all(valid_position_rows):
    position = position[valid_position_rows]
    trace = trace[valid_position_rows]
...
trial_movement = movement_mask[start:end]
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
...
output_trials.append(flat_bins[np.newaxis, :].astype(np.int64))
```

iii. The trajectory says the AI was trying to match the "paper’s binning logic" and the "within-session position decoder." It specifically highlighted "session-level 3x3 binning based on the same per-session position normalization the paper code uses."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI thresholds position categories by dividing the session-specific coordinate extent into three equal bins per axis, then using `floor` and flattening row/column indices into labels `0..8`. Because it pools before thresholding, the categories are based on pooled mean positions, not raw framewise coordinates.

ii.
```python
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
bin_down = arena_extent / POSITION_BINS_PER_AXIS
...
pos_scaled = pos_trial / bin_down
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
...
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat = bin_xy[:, 0] * POSITION_BINS_PER_AXIS + bin_xy[:, 1]
```

iii. The trajectory justification is the same as for 4-b: the AI said it wanted "the same per-session position normalization the paper code uses" and a "coarse 3x3 grid using a session-wise common spatial scale based on the session's maximum coordinate extent."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output and neural data by slicing them with the same session windows, applying the same movement mask to both, temporally pooling both with the same 3-frame factor, and trimming to the shorter pooled length if the two arrays differ.

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

iii. The trajectory says the AI wanted the converted trials to "match the paper’s binning logic" and chose to "bake that 10 Hz representation into the saved dataset," implying that both neural and position streams should be transformed together.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI applies several defensive fixes: it drops position rows containing NaNs, removes cells deemed unregistered by a NaN first sample, converts remaining NaN/Inf neural values to zero, interprets empty or `[-1]` blocked lists as "no blocked bins," remaps pooled position labels that fall into blocked bins to the nearest open bin, and drops trials that are stationary, too short, or all-zero after pooling.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
...
registered_mask = ~np.isnan(trace[0])
...
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
...
if raw_blocked.size == 0:
    return []
if raw_blocked.size == 1 and raw_blocked[0] == -1:
    return []
...
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
...
if not np.any(trial_movement):
    ...
if trace_pooled.shape[0] < MIN_POOLED_TIMEPOINTS_PER_TRIAL:
    ...
if np.all(neural_trial == 0):
    ...
```

iii. The trajectory justifies these as sanity-preserving cleanups: it says the AI wanted to "avoid warnings," identified "low-information edge cases," and believed remapping invalid pooled bins "mirrors the spirit of the paper's decoder cleanup."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are reading the large `.mat` files across all 207 sessions, converting each session trial-by-trial, and writing the very large full pickle. The AI’s trajectory emphasizes I/O scale, but its code also adds per-trial smoothing/pooling/remapping work on top of that.

ii.
```python
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
    with h5py.File(path, "r") as h5:
        ...
        for session_index in session_indices:
            neural_trials, input_trials, output_trials, region_idx, summary = convert_session(...)
...
with args.output.open("wb") as f:
    pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory says "loading the full set takes real time," that it could avoid some cost by reading HDF5 metadata directly, and later notes that the "full pass has to walk all 207 sessions and write both pickles" and that the full pickle is about `3.3G`.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the explicit trial loop over every 1-minute chunk and the per-invalid-sample loop in `remap_invalid_bins`. The session loop itself is necessary, but the nearest-open-bin remapping and some list construction are done with Python loops rather than array operations.

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

iii. The AI did not explicitly discuss vectorization in the trajectory. Its comments instead focus on getting the paper-like decoder preprocessing correct and on reducing slow full-file loads.

## 6-c. What processing does the code repeat multiple times?

i. The code rebuilds datasets more than once. In the default path it builds the full dataset once and then separately rebuilds a sample dataset from the raw files. During the recorded run, the agent also reran conversion and verification multiple times for sample/full outputs after earlier exploratory runs. Within each kept trial it separately pools position and trace after parallel preprocessing.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The trajectory shows repeated conversion/verification passes: sample conversion, sample verification, sample training, full conversion, full verification, full training, and then extra reruns to capture output logs. The AI justified this as stepwise validation so it could catch structural issues early and save the required artifacts.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are not used by downstream decoder training: `get_env_mat` and `blocked_indices_from_open_mask` are effectively dead code in the final script, `env_name` is used mainly for summaries/checks, extensive per-session summaries and dataset-level sanity checks are only metadata/logging, and the code does extra sample-dataset generation plus log-capture work that is not needed for the final `converted_data.pkl`. The blocked-bin remap counter is also bookkeeping rather than a decoder input/output.

ii.
```python
def get_env_mat(env_name: str) -> np.ndarray:
    ...

def blocked_indices_from_open_mask(open_mask: np.ndarray) -> list[int]:
    return [idx for idx, value in enumerate(open_mask.reshape(-1)) if value == 0]
...
session_summary = {
    ...
    "n_remapped_invalid_position_bins": int(remapped_samples),
    "trial_timepoints_summary": summarize_trial_lengths(neural_trials),
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

iii. The trajectory shows the AI intentionally added these as "sanity checks" against the paper and to document conversion choices. It explicitly said it wanted the conversion to "verify itself against the paper before I write anything out."
