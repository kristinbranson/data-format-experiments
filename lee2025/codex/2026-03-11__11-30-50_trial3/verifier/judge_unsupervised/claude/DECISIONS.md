# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded from primary joblib-format animal files in the `data/` directory. Each file (e.g., `data/QLAK-CA1-08`) is loaded using `joblib.load()` and indexed by the animal ID string. The code discovers animal IDs by listing the `data/` directory for filenames starting with `QLAK-CA1-` that contain no dot (excluding `.mat` files). Each animal file contains all sessions (days) for that animal, with fields including `trace`, `position`, `envs`, `blocked`, and `maps`. Sessions are iterated by indexing the first axis of arrays like `dat["envs"]` and `dat["trace"]`. Trials do not exist natively in the raw data; they are created by splitting each continuous ~40-minute session into non-overlapping 1-minute windows (1800 frames at 30 Hz).

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )

def load_animal_dataset(data_dir: str, animal: str) -> dict:
    return joblib.load(os.path.join(data_dir, animal))[animal]

def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
            ...
```

iii. The agent documented in CONVERSION_NOTES.md Step 1 that the reference code uses `load_dat(..., format="joblib")` to load the primary joblib animal files. In Step 5, the agent explicitly decided to "use primary joblib animal files, not cached analysis results" to match the reference loading path and avoid inheriting downstream analysis assumptions.

## 1-b. How are the data split into subjects?

i. Subjects (mice) are identified by their animal ID strings (e.g., `QLAK-CA1-08`). The code discovers all unique animal IDs from the data directory filenames, sorts them alphabetically, and assigns each a numeric index. Each session is associated with a subject through the `SessionRef.animal` field, and the `subject_idx` array maps each session to its subject's index in the sorted subject list.

ii.
```python
animals = get_animal_ids(data_dir)
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}
# ...
session_data = {
    ...
    "subject_idx": subject_lookup[session_ref.animal],
    ...
}
# Final: converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)
```

iii. The agent stated in CONVERSION_NOTES.md Step 5: "Unique sorted subject IDs; one target session per original recording day." The agent confirmed 7 subjects across the dataset, matching the reference paper's count.

## 1-c. How are the data split into sessions?

i. Each recording day for each animal constitutes one session. The code iterates over all animals and all days within each animal's dataset (indexed by `dat["envs"].shape[0]`), creating one session per day. Sessions are ordered by animal (alphabetical) then by day index within each animal.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. The agent noted in CONVERSION_NOTES.md Step 5: "Keep one target session per original recording day: The raw data are organized by day/session, and the target format supports multiple trials within each session." This yields 207 total sessions matching the paper's count.

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into non-overlapping 1-minute windows of exactly 1800 frames (at 30 Hz). Any trailing frames that don't fill a complete 1-minute window are discarded. This yields 39-40 trials per session and 8187 total trials.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. The agent justified this in CONVERSION_NOTES.md Step 5: "Create trials by splitting each continuous 40-minute session into non-overlapping 1-minute windows: This satisfies the decoder task while preserving within-session context. Trial length will be exactly 1800 frames at 30 Hz; any trailing partial minute will be discarded." The instructions specified "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session."

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 complete 1-minute trials are rejected (raising a `ValueError`). Within the trials that are kept, no further quality filtering is applied -- all complete 1-minute windows are retained. There is no velocity filtering, no minimum-activity filtering, and no removal of trials based on behavioral criteria.

ii.
```python
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The agent noted that the reference experiment has no native trial structure, and the 1-minute trialization is a task-specific transformation. The minimum-2-trials requirement comes from the instruction that "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from the `trace` field of each animal's dataset: `dat[animal]["trace"]`, which has shape `(n_sessions, n_registered_cells, n_frames)`. These are pre-processed binary rising-phase calcium event traces where `1` indicates a significant calcium event and `0` indicates no event.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
# ...
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The agent documented in CONVERSION_NOTES.md Step 1: "The neural signal is not raw fluorescence and does not require delta F/F computation in the reference code. The README explicitly describes `trace` as rise-extracted calcium traces where `1` indicates a significant event." In Step 5: "No delta-F/F. Use released binary rising-phase event traces directly."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes minimal processing: (1) cells absent on a given day (indicated by NaN values) are removed; (2) the remaining traces are sliced into 1-minute trial windows; (3) the data is cast to `float16` for storage efficiency. No additional smoothing, normalization, z-scoring, or temporal rebinning is applied. The raw 30 Hz binary event traces are used directly.

ii.
```python
present_mask = session_present_cell_mask(trace_day)  # ~np.isnan(trace_day[:, 0])
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The agent stated in CONVERSION_NOTES.md Step 6: "reference-consistent use of released binary calcium-event traces" and Step 5: "Do not compute new calcium features: The released `trace` is already the binary rising-phase representation used in the paper/code."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural filtering applied is the removal of cells not present (registered) on a given session day. These are identified by NaN values in the first time point of the trace for that day. No place-cell filtering is applied. No velocity filtering or cell-activity-threshold filtering (as used in the reference `decode_position_within` function) is applied.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. The agent justified this in CONVERSION_NOTES.md Step 5: "Use all session-present cells: This matches the paper's statement that all cells were included in subsequent analyses." And: "Do not pre-filter to place cells: The paper's position decoder is not place-cell-restricted, and place-cell status is an analysis label rather than a required curation step for decoding."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the start of its respective 1-minute window within the continuous session recording. The first trial starts at frame 0 of the session, the second at frame 1800, etc. Since the behavioral and neural imaging streams were simultaneously acquired at 30 Hz and are already temporally aligned in the raw data, no additional alignment is needed.

ii.
```python
slices = trial_slices(n_frames)  # [slice(0,1800), slice(1800,3600), ...]
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
```

iii. The agent documented in the metadata: `"temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment"` with `"off_start": 0.0` and `"off_end": 60.0`. In CONVERSION_NOTES.md Step 10: "converter preserves the raw 30 Hz synchronized time base and uses the released aligned `position` and `trace` streams directly."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 1/30 seconds (~33.33 ms), the native frame rate of the imaging system. No temporal rebinning is applied. The reference code's `decode_position_within` uses a 3-frame temporal pooling (`temporal_bin_size=3` in `fit_decoder`), but the agent did not apply this rebinning in the conversion, preserving the raw 30 Hz resolution.

ii.
```python
FPS = 30.0
# ...
"time_bin_size": 1000.0 / FPS,  # = 33.33 ms
```

iii. The agent documented in CONVERSION_NOTES.md Step 3: "behavioral and cellular imaging streams at 30 Hz" and set the metadata time_bin_size to `1000.0 / 30 = 33.33` ms.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field of each animal's dataset: `dat[animal]["blocked"]`. This is a Python list of length `n_sessions`, where each entry stores the indices of blocked partitions in a 3x3 arena numbering scheme, with `-1` meaning no partitions are blocked (full square environment).

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)

def blocked_to_geometry_vector(blocked_list: list, day_index: int) -> tuple[np.ndarray, np.ndarray]:
    blocked = extract_day_blocked_entry(blocked_list, day_index)
    geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
    if not (blocked.size == 1 and blocked[0] == -1):
        geometry[blocked] = 0.0
    geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
    return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)
```

iii. The agent documented in CONVERSION_NOTES.md Step 4: "Raw data also contain an explicit `blocked` list with partition IDs in the stated 3x3 indexing scheme." The agent cross-checked this against the valid mask from `maps['smoothed']` to verify correctness.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The processing involves: (1) extracting the blocked partition indices for the session day, handling nested list structures; (2) creating a 9-element vector of ones (all open); (3) setting blocked partition indices to 0; (4) reshaping to 3x3 and transposing to align with the position/map coordinate frame; (5) flattening back to a 9-element float32 vector. The transpose is critical for asymmetric environments to match the spatial coordinate system used by position data.

ii.
```python
geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
if not (blocked.size == 1 and blocked[0] == -1):
    geometry[blocked] = 0.0
geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)
```

iii. The agent verified this in CONVERSION_NOTES.md Step 4: "Cross-checking against the valid-mask of `maps['smoothed']` shows that `blocked` reshaped to 3x3 and then transposed matches the actual map/position coordinate frame for all 207 sessions."

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per session (and therefore per trial). The same 9-element geometry vector is copied identically for every trial within a session. No temporal alignment is needed since it does not vary over time.

ii.
```python
for trial_slice in slices:
    ...
    input_trials.append(geometry_vector.copy())
```

iii. The agent stated in CONVERSION_NOTES.md Step 5: "Static per trial; same value for every 1-minute trial within a session."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output mouse position is derived from the `position` field of each animal's dataset: `dat[animal]["position"]`, which has shape `(n_sessions, 2, n_frames)` -- continuous x-y position coordinates obtained from DeepLabCut head tracking at 30 Hz.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
# ...
position_bins, output_class = compute_position_bins(position_day)
```

iii. The agent documented in CONVERSION_NOTES.md Step 2: "`position`: continuous x-y position, shape `(n_sessions, 2, n_frames)`" and Step 3: "Position was obtained from DeepLabCut head tracking."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x-y position is discretized into a 3x3 spatial grid using session-wide floor-division binning: (1) position is transposed to (n_frames, 2); (2) a scale factor is computed as `(session_max + buffer) / 3` for each dimension; (3) bin indices are computed via `floor(position / scale)` and clipped to [0, 2]; (4) the 2D bin indices are flattened to a single class index: `x_bin * 3 + y_bin`, yielding values 0-8.

ii.
```python
POSITION_BINS = 3
POSITION_BUFFER = 1e-5

def compute_position_bins(position_day: np.ndarray, n_bins: int = POSITION_BINS) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(position_day, dtype=np.float64).T
    scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
    binned = np.floor(coords / scale).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
    return binned, class_idx
```

iii. The agent justified this in CONVERSION_NOTES.md Step 5: "Compute 3x3 spatial bins using the reference floor-division rule with session-wide position maxima and buffer." And: "Using session-wide maxima preserves a single spatial partition per original session across all derived trials."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous position is thresholded into 9 categories (3x3 grid). The binning uses `floor(position / scale)` where `scale = (max_position + 1e-5) / 3`. The buffer (1e-5) prevents the maximum position value from being binned into index 3 (out of range). Results are clipped to [0, 2] per dimension. The final class index combines x and y bins: `class = x_bin * 3 + y_bin`, giving 9 categories labeled 0-8.

ii.
```python
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The instructions specified "Mouse position discretized into 3 x 3 = 9 spatial bins." The agent chose session-wide normalization matching the reference code's approach in `get_rate_maps`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The output position is aligned frame-by-frame with the neural data because both come from the same 30 Hz synchronized recording stream. The position is binned over the full session first, then the same trial slices used for neural data are applied to the binned position array. This guarantees exact temporal alignment between neural and output data within each trial.

ii.
```python
position_bins, output_class = compute_position_bins(position_day)
# ...
slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The agent documented in CONVERSION_NOTES.md Step 10: "converter preserves the raw 30 Hz synchronized time base and uses the released aligned `position` and `trace` streams directly."

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms (1/30 Hz), the native acquisition frame rate. No temporal rebinning is applied. The reference code's `fit_decoder` applies a 3-frame temporal pooling (`temporal_bin_size=3`), but this was not replicated in the conversion -- the data is stored at native resolution.

ii.
```python
FPS = 30.0
# ...
"time_bin_size": 1000.0 / FPS,  # 33.33 ms
```

iii. The agent documented the time bin as `1000.0/FPS` ms in the metadata. No discussion of rebinning was found in the agent's notes; the raw 30 Hz resolution was preserved.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output data are temporally aligned by using identical trial slice indices on the frame-synchronized 30 Hz streams. Both `trace` and `position` share the same time axis in the raw data. The input (geometry) is static per trial and requires no temporal alignment. All three data streams use exactly 1800 frames per trial.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    input_trials.append(geometry_vector.copy())
    output_trials.append(output_trial)
```

iii. The agent noted in CONVERSION_NOTES.md Step 3 that "behavioral and cellular imaging streams were simultaneously acquired at 30 Hz and timestamped for post-hoc alignment."

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Several edge cases are handled: (1) Cells absent on a given day (NaN rows in trace) are excluded via `session_present_cell_mask`. (2) The `blocked` field has inconsistent nesting (some entries are nested lists like `[[indices]]`); the code unwraps single-element nested lists via `extract_day_blocked_entry`. (3) Sessions with no present cells raise an error. (4) The geometry derived from `blocked` is cross-validated against the valid spatial support of `maps['smoothed']`; mismatches raise an error. (5) Trailing frames that don't complete a full 1-minute trial are discarded. (6) Sessions with fewer than 2 full trials raise an error.

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)

# NaN cell handling
present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(...)

# Geometry validation
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
```

iii. The agent documented in CONVERSION_NOTES.md Step 10: "no NaN/Inf values remained in neural, input, or output arrays" and verified edge cases like discarded tail frames.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the large joblib animal files from disk. Each animal file is 68-145 MB and contains all sessions and all registered cells. The code mitigates this by caching one animal at a time and processing all its sessions before loading the next. The agent measured ~6.5 seconds per session for the sample run and estimated 7-10 minutes for the full 207-session conversion.

ii.
```python
animal_cache: dict[str, dict] = {}
# ...
if session_ref.animal not in animal_cache:
    animal_cache.clear()
    gc.collect()
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. The agent documented in CONVERSION_NOTES.md Step 6: "Loading the full animal files is the main runtime cost" and Step 7: "Sample conversion observed 6.51 s/session over 2 sessions."

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could benefit from vectorization is the trial-slicing loop, which iterates over slices to extract neural and output trials. However, this loop is lightweight since it uses NumPy array slicing rather than element-wise operations. The occupancy computation in the plotting function uses a Python for-loop over position bins that could be replaced with `np.add.at` or similar. Overall, the code is already reasonably vectorized for its workload.

ii.
```python
# Trial slicing loop (already uses NumPy slicing)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)

# Plotting occupancy loop (could be vectorized)
for xbin, ybin in position_bins:
    occupancy[xbin, ybin] += 1.0
```

iii. The agent noted in CONVERSION_NOTES.md Step 6: "Slice full-session binned outputs and present-cell traces directly without redundant recomputation inside trials."

## 7-c. What processing does the code repeat multiple times?

i. The code loads each animal's dataset once and reuses it for all of that animal's sessions, so no data loading is repeated. Within each session, position binning is done once for the entire session then sliced into trials. The geometry computation is done once per session. There is no significant repeated processing. The `aggregate_valid_map` function computes a valid mask from `maps['smoothed']` for each session as a cross-check; this is only used for validation and could be skipped if the check were removed.

ii.
```python
# Animal caching prevents repeated loads
if session_ref.animal not in animal_cache:
    animal_cache.clear()
    gc.collect()
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. The agent documented in CONVERSION_NOTES.md Step 6: "Reuse one loaded animal dataset across all of its sessions before releasing it."

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `aggregate_valid_map` from `maps['smoothed']` for every session as a geometry cross-check, which is only used for validation (the assertion that geometry matches the valid map). This result is not stored in the converted data. Additionally, the `position_bins` (2D x-y bin indices) are computed but only the flattened `output_class` (1D class index) is used in the output; the 2D bins are only used in the optional processing plots.

ii.
```python
# Computed for validation only, not stored in output
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)

# 2D position_bins computed but only class_idx used in output
position_bins, output_class = compute_position_bins(position_day)
# position_bins only used in optional plot_processing_figure
```

iii. The agent considered this a worthwhile trade-off for validation, documenting in CONVERSION_NOTES.md Step 10 that the geometry-vs-map check passed for all 207 sessions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. This is the same as question 6. Missing cells (NaN rows) are excluded per session. Malformed `blocked` entries (nested lists) are unwrapped. Incomplete trailing frames are discarded. The code raises errors for sessions with no present cells, geometry mismatches, or fewer than 2 trials. The `POSITION_BUFFER` (1e-5) prevents edge-case binning errors at position maxima.

ii. (Same code as question 6)

iii. The agent verified in CONVERSION_NOTES.md Step 10: "no NaN/Inf values remained in neural, input, or output arrays" and confirmed all 207 sessions passed validation.

## 9-a. What are the most time-consuming steps of the code?

i. (Same as 7-a) Loading large joblib animal files from disk is the dominant cost. The agent cached one animal at a time and measured ~2-3 s/session effective rate for the full conversion (approximately 7-10 minutes total for 207 sessions).

ii. (Same code as 7-a)

iii. The agent documented timing in CONVERSION_NOTES.md Step 7 and estimated the full conversion at 7-10 minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. (Same as 7-b) The main trial-slicing loop is lightweight and already uses NumPy slicing. The occupancy computation in the plotting function uses a Python loop. No significant vectorization opportunities remain that would meaningfully improve performance, as the bottleneck is I/O rather than computation.

ii. (Same code as 7-b)

iii. The agent documented speedups in CONVERSION_NOTES.md Step 7.

## 9-c. What processing does the code repeat multiple times?

i. (Same as 7-c) No significant processing is repeated. Animal data is cached and reused. Position binning and geometry computation are done once per session. The `maps['smoothed']` valid-mask computation is done per session for validation only.

ii. (Same code as 7-c)

iii. Documented in CONVERSION_NOTES.md Step 6.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (Same as 7-d) The `aggregate_valid_map` validation check and the 2D `position_bins` (used only for plots) are computed but not stored in the final output. The `smoothed_day` maps array is loaded and processed for the geometry cross-check but not included in the converted data.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
# ...
valid_grid = aggregate_valid_map(smoothed_day)
```

iii. The agent considered this validation overhead acceptable for ensuring correctness of the conversion.
