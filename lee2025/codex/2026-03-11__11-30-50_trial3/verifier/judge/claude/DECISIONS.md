# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files in the `data/` directory. Each animal has a joblib file (e.g., `data/QLAK-CA1-08`) which is loaded using `joblib.load()`. The loaded dictionary is keyed by the animal name and contains fields: `trace`, `position`, `blocked`, `envs`, `maps`, `SFPs`, `centroids`. Animal IDs are discovered by listing directory entries matching the `QLAK-CA1-*` pattern (without file extensions).

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
```

iii. The AI chose joblib over h5py/.mat because the reference code's `load_dat()` function defaults to joblib format. From CONVERSION_NOTES Step 5: "Use primary joblib animal files, not cached analysis results: This matches the reference loading path in `load_dat(..., format='joblib')`."

## 1-b. How are the data split into subjects?

i. Each joblib file in the data directory corresponds to one subject. Subject names are the filenames matching `QLAK-CA1-*` without extensions, sorted alphabetically. All 7 animals are discovered automatically.

ii.
```python
animals = get_animal_ids(data_dir)
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}
```

iii. Each joblib file contains all recording sessions for one animal. The filename serves as the subject identifier, consistent with the reference code's animal enumeration.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days. The number of sessions is determined from the `envs` array shape (`dat["envs"].shape[0]`). Each day index becomes a separate session in the output. Sessions are enumerated across all animals in sorted animal order, then by day index.

ii.
```python
def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. Each entry in the data arrays (trace, position, blocked) corresponds to one recording day/session. This matches the reference code's organization.

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into non-overlapping 1-minute (1800-frame at 30 Hz) segments. Remainder frames that do not fill a complete 60-second trial are discarded.

ii.
```python
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. Per instruction, the experiment consists of long recording sessions split into 1-minute trials. The AI documented this yields 39-40 trials per session given the ~71866-72219 frame session lengths.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full 1-minute trials are rejected (raises ValueError). No other trial-level filtering is applied.

ii.
```python
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." No further trial filtering was needed as all sessions had 39+ trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib data, which contains binary rise-extracted calcium event traces with shape `(n_sessions, n_registered_cells, n_frames)`.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The CONVERSION_NOTES document that `trace` contains pre-processed binary calcium event traces (1=significant event, 0=otherwise), matching the paper's description: "the rising phase of each calcium transient was extracted... thresholding at 2.5."

## 2-b. How is the `neural` data processed?

i. Present (non-NaN) neurons are selected, then the trace is cast to float16. No additional processing (smoothing, delta F/F, rebinning) is applied. The trace is used directly as the neural signal.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. From CONVERSION_NOTES Step 5: "No delta-F/F. Use released binary rising-phase event traces directly." The paper states these traces are already the final processed signal used for all analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are NaN at the first timepoint of a session are removed. Only neurons present (non-NaN) in that session are kept.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The NaN columns indicate neurons not registered/recorded in a given session. The paper states "all cells were included in subsequent analyses" so no further quality filtering (e.g., place-cell filtering) is applied. From CONVERSION_NOTES Step 4: "For the conversion I should keep all session-present cells and avoid pre-filtering to place cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of the session. Neural and behavioral data are already synchronized at 30 Hz.

ii.
```python
# Neural and position are sliced with the same trial_slices
neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. From CONVERSION_NOTES Step 4: "The 1-minute trialization is a task-specific transformation required by the target format, not part of the original acquisition." The metadata records `temporal_alignment_event` as "start of each non-overlapping 1-minute within-session segment".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
# ...
"time_bin_size": 1000.0 / FPS,  # ~33.33 ms
```

iii. Both behavioral and neural imaging streams were acquired at 30 Hz. The AI preserves this native rate without rebinning.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the `blocked` field in the joblib data, which contains a list of blocked partition indices for each session. Additionally, the `maps["smoothed"]` field is used for cross-validation of the geometry.

ii.
```python
geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
```

iii. From CONVERSION_NOTES Step 5: "Convert blocked partition IDs to 3x3 binary open/blocked matrix, transpose to align with map/position axes, flatten to 9-dim float vector."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted to a 9-element geometry vector where 1=open and 0=blocked. The vector is reshaped to 3x3, transposed to align with the position/map coordinate frame, then flattened back to a 9-element vector. If blocked is `[-1]` (no blocked positions), all entries are 1 (all open).

ii.
```python
def blocked_to_geometry_vector(blocked_list: list, day_index: int) -> tuple[np.ndarray, np.ndarray]:
    blocked = extract_day_blocked_entry(blocked_list, day_index)
    geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
    if not (blocked.size == 1 and blocked[0] == -1):
        geometry[blocked] = 0.0
    geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
    return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)
```

iii. The AI discovered the need for transposing through systematic validation against `maps['smoothed']` valid masks. From CONVERSION_NOTES Step 10: "Environment orientation ambiguity: resolved before full conversion by proving that `blocked.reshape(3,3).T` exactly matches the valid spatial support of `maps['smoothed']` for all 207 sessions."

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry is a per-session constant (static per trial), not a time-varying signal. The same 9-element geometry vector is assigned to every trial within a session.

ii.
```python
input_trials.append(geometry_vector.copy())
```

iii. Blocked positions do not change within a session, so the geometry input is constant across all trials in a session. This is consistent with the instructions specifying the input is "Static per-trial."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output position is derived from the `position` field in the joblib data, which contains 2D coordinates of the animal in the arena with shape `(n_sessions, 2, n_frames)`.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
```

iii. The `position` variable records the animal's location at each timepoint in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). Binning uses session-wide position maxima with a small buffer (1e-5) to normalize, then floor-division to assign bin indices. The class label is computed as `x_bin * 3 + y_bin`.

ii.
```python
def compute_position_bins(position_day: np.ndarray, n_bins: int = POSITION_BINS) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(position_day, dtype=np.float64).T
    scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
    binned = np.floor(coords / scale).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
    return binned, class_idx
```

iii. From CONVERSION_NOTES Step 5: "Use session-wide position normalization when binning to 3x3 outputs: The reference code bins position using maxima from the full session/day, not from smaller windows."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). The bin edges are determined per session by dividing the session-wide maximum coordinate (plus a tiny buffer) by 3. Each coordinate axis gets 3 equal bins. The final class is `x_bin * 3 + y_bin`, producing classes 0-8.

ii.
```python
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The session-wide normalization ensures consistent spatial bins across all trials within a session, matching the reference code's approach in `get_rate_maps` and `fit_decoder`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are sampled at the same 30 Hz rate and stored frame-aligned in the original data. Both are sliced using the same trial boundaries.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    output_trials.append(output_trial)
```

iii. Both arrays share the same timepoint axis and are sliced identically, preserving temporal alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~33.33 ms (native 30 Hz). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
"time_bin_size": 1000.0 / FPS,  # ~33.33 ms
```

iii. The paper states "behavioral and cellular imaging streams at 30 Hz." The AI preserves this native rate.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural (trace) and output (position) are both frame-aligned at 30 Hz in the source data. Both are sliced at the same trial boundaries. Input (geometry) is static per session and does not require temporal alignment.

ii.
```python
slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice]
    output_trial = output_class[trial_slice][np.newaxis, :]
    input_trials.append(geometry_vector.copy())
```

iii. The AI's CONVERSION_NOTES Step 10 confirms: "converter preserves the raw 30 Hz synchronized time base and uses the released aligned `position` and `trace` streams directly."

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. NaN neurons (not present in a session) are filtered out by checking the first timepoint. The `blocked` field has variable formats (sometimes nested lists) which are handled by `extract_day_blocked_entry`. Remainder frames that don't fill a complete trial are discarded. Sessions with no present cells or fewer than 2 trials raise errors.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)
```

iii. From CONVERSION_NOTES Step 10: "no NaN/Inf values remained in neural, input, or output arrays" and "discarded_tail_frames values were exactly the raw remainders."

## 7-a. What are the most time-consuming steps of the code?

i. Loading the joblib animal files is the most time-consuming step. Each file is 68-145 MB. The first session of each animal shows a large time spike (10-40 seconds) due to file loading, while subsequent sessions of the same animal take <0.2 seconds.

ii. N/A (observable from conversion_full_out.txt timing data)

iii. From CONVERSION_NOTES Step 6: "Loading the full animal files is the main runtime cost because each file contains all sessions and registered cells for one subject."

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial slicing loop iterates over each trial to extract neural and output data. This could potentially be replaced with a single reshape operation for contiguous data, though the remainder-frame handling makes this slightly more complex.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The AI acknowledged this but the loop is simple and fast compared to I/O. From CONVERSION_NOTES: "Slice full-session binned outputs and present-cell traces directly without redundant recomputation inside trials."

## 7-c. What processing does the code repeat multiple times?

i. The `aggregate_valid_map` function recomputes the valid mask from `maps['smoothed']` for every session, even though it's only used for validation (assertion check). The geometry vector is copied for each trial within a session.

ii.
```python
valid_grid = aggregate_valid_map(smoothed_day)
# ...
input_trials.append(geometry_vector.copy())
```

iii. The validation check is intentional (verifying geometry consistency) and not performance-critical.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `aggregate_valid_map` computation from `maps['smoothed']` is used solely for validation and not included in the output. The `session_meta` dictionary stores per-session metadata that is included in the output but may not be used by the decoder. Processing plots are generated when `--show-processing` is used.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
valid_grid = aggregate_valid_map(smoothed_day)
```

iii. The smoothed maps validation serves as a sanity check during development and is not needed for the final conversion.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN neurons are filtered out per session. The `blocked` field format inconsistencies (nested lists, scalar vs array) are handled by `extract_day_blocked_entry`. Sessions with no present cells raise errors. Remainder frames are discarded. The geometry is cross-validated against map valid masks to catch any data inconsistencies.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. From CONVERSION_NOTES Step 10: "Edge-case checks: all converted trials had length 1800, all sessions had at least 2 trials, no NaN/Inf values remained."

## 9-a. What are the most time-consuming steps of the code?

i. Loading the joblib animal files dominates runtime. The full conversion takes ~259 seconds for 207 sessions (~1.25 s/session average), with the first session of each animal taking 7-40 seconds due to file I/O.

ii. N/A (timing visible in conversion_full_out.txt)

iii. From CONVERSION_NOTES: "Loading the full animal files is the main runtime cost."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop could be replaced with array reshaping for the portion of data that fits complete trials. However, the loop is simple and the overhead is negligible compared to I/O.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
```

iii. The AI chose to keep the loop for clarity, noting that I/O dominates runtime.

## 9-c. What processing does the code repeat multiple times?

i. The `aggregate_valid_map` computes a valid mask from smoothed maps for each session as a validation step. Each animal's full data is loaded once and reused across all its sessions, avoiding redundant file I/O.

ii.
```python
if session_ref.animal not in animal_cache:
    animal_cache.clear()
    gc.collect()
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. From CONVERSION_NOTES Step 6: "Reuse one loaded animal dataset across all of its sessions before releasing it."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `maps['smoothed']` data is loaded and processed (via `aggregate_valid_map`) purely for validation purposes. This data is not included in the final output. The per-session metadata dictionary contains fields that may not be used by the decoder.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
valid_grid = aggregate_valid_map(smoothed_day)
```

iii. The AI included this as a deliberate sanity check but it adds unnecessary computation for the final production run.
