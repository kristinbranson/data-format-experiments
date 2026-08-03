# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-format animal files (not the `.mat` files) in the `data/` directory using `joblib.load()`. Each file is a dictionary keyed by the animal ID containing fields `trace`, `position`, `blocked`, `envs`, `maps`, etc. This matches the default `format="joblib"` path in the reference code's `load_dat` function.

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

iii. The AI chose joblib files because the reference code's `load_dat` function uses joblib as the default format. The AI verified this matches the reference loading path and avoids inheriting downstream analysis assumptions from precomputed results.

## 1-b. How are the data split into subjects?

i. Each joblib file in the data directory corresponds to one subject. Subject names are derived from the filenames (e.g., `QLAK-CA1-08`). All 7 animal IDs are always included in the `subjects` list.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )
# ...
converted = {
    "subjects": animals,
    # ...
}
```

iii. The AI identified that each joblib file contains all recording sessions for one animal, and filenames serve as subject identifiers.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject's dataset becomes a separate session. The number of sessions per animal is determined by `dat["envs"].shape[0]`. Sessions are iterated via `SessionRef` objects pairing animal ID and day index.

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

iii. The raw data are organized by day/session within each animal file, and the AI preserves this structure with one target session per original recording day.

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute recording session is split into non-overlapping 60-second (1800-frame at 30 Hz) segments. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. The instructions specify 1-minute trials. At 30 Hz, this gives 1800 frames per trial. Sessions yield 39-40 trials each.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full 1-minute trials are rejected (raises ValueError). Additionally, the AI validates that the geometry derived from the `blocked` field matches the valid spatial support of `maps['smoothed']`, raising an error on mismatch.

ii.
```python
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")

if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )
```

iii. The minimum-2-trials check satisfies the instruction requirement that each session have at least two trials for decoder evaluation. The geometry validation is a cross-check against the spatial maps.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib dataset, which contains rise-extracted calcium event traces. The shape is `(n_sessions, n_registered_cells, n_frames)`, with NaN for cells not recorded on a given session.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The AI identified that `trace` contains binary calcium event traces (1 = significant rising-phase event, 0 otherwise), already preprocessed by the original authors. No delta F/F computation is needed.

## 2-b. How is the `neural` data processed?

i. Processing consists of: (1) filtering out neurons not present in the session (NaN at first timepoint), (2) casting to float16, and (3) splitting into 1800-frame trial segments.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
# ...
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The AI chose to use the released binary traces directly, consistent with the reference code. Float16 was used to reduce disk footprint.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with NaN at the first timepoint (indicating they were not recorded in that session) are excluded. No place-cell filtering or additional quality criteria are applied.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The AI followed the paper's statement that all cells were included in subsequent analyses. Checking only the first timepoint is sufficient because absent neurons have NaN across all timepoints.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous and trials are artificial 60-second segments starting from the beginning of the session. Alignment is to the start of each non-overlapping 1-minute segment.

ii.
```python
"temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The original experiment has no stimulus events to align to; sessions are continuous free exploration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
# ...
"time_bin_size": 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz frame rate. The reference code's decoder does use temporal binning (3-frame pooling), but that is a decoder-specific processing step, not part of the data conversion.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the `blocked` field in the joblib dataset, which contains a list of blocked partition indices for each session. The AI also cross-validates against the `maps['smoothed']` field.

ii.
```python
geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
valid_grid = aggregate_valid_map(smoothed_day)
```

iii. The `blocked` field stores which of the 9 possible positions in a 3x3 grid are blocked. A value of [-1] indicates no positions are blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary vector where **1 = open (accessible)** and **0 = blocked**. The 3x3 grid is also **transposed** to align with the position/map coordinate frame. The geometry vector is static per trial (same for all trials within a session).

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

iii. The AI chose 1 = open, 0 = blocked to match the reference code's `get_env_mat` function which uses the same convention. The transpose was verified by cross-checking against the valid spatial support of `maps['smoothed']` for all 207 sessions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in the joblib dataset, which contains continuous 2D coordinates with shape `(n_sessions, 2, n_frames)`.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
```

iii. The position tracks the animal's location in the arena at each frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). Each axis is binned using the session-wide maximum position + a small buffer, divided by 3, with floor division. The class label is computed as `dim0_bin * 3 + dim1_bin`.

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

iii. The AI explicitly followed the reference code's `get_rate_maps` position binning approach: `position // ((np.nanmax(position, axis=0) + buffer) / n_bins)`. Session-wide maxima are used to maintain a consistent spatial partition across all derived trials within a session.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into 9 categories (0-8) using floor division of position coordinates by `(session_max + buffer) / 3`. Values are clipped to [0, 2] per axis, then combined as `dim0 * 3 + dim1`.

ii.
```python
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. This produces 9 categories matching the instruction's "3 x 3 = 9 spatial bins" requirement. The session-wise normalization ensures consistent bin boundaries within each session.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and are already aligned in the raw data. Both are split into trials using the same slice indices.

ii.
```python
slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. Using identical slices guarantees frame-for-frame alignment between neural and position data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality checks are implemented: (1) Neurons absent in a session (NaN) are excluded. (2) Sessions with fewer than 2 trials raise an error. (3) The blocked-derived geometry is validated against the spatial maps for every session. (4) Remainder frames that don't fill a complete trial are discarded and counted in metadata as `discarded_tail_frames`. (5) No NaN/Inf values remain in converted arrays (verified in Step 10).

ii.
```python
present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")

if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)

if len(slices) < 2:
    raise ValueError(...)
```

iii. The AI documented all edge cases, including the exact frame counts discarded per session as tail remainders.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib animal files (68-145 MB each). Processing per session is fast by comparison (~2-3 seconds/session effective).

ii. N/A (timing is measured via `time.time()` around the processing loop)

iii. The AI estimated 7-10 minutes for all 207 sessions with per-animal caching. Loading is I/O bound.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial slicing loop iterates over slices to create per-trial arrays. This could potentially be done with a single `np.split` or reshape operation instead of a list comprehension. However, the loop is not a significant bottleneck.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
```

iii. The AI acknowledged that loading is the main cost and the trial slicing loop is fast.

## 6-c. What processing does the code repeat multiple times?

i. The code loads each animal file once and reuses it across all sessions for that animal, avoiding redundant I/O. Position binning and present-cell masking are computed once per session and reused across trials. No significant redundant processing was identified.

ii.
```python
if session_ref.animal not in animal_cache:
    animal_cache.clear()
    gc.collect()
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. The caching strategy avoids reloading the same large file for every session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and validates `aggregate_valid_map` from `maps['smoothed']` for every session as a geometry cross-check. This validation data is not included in the output but serves as a sanity check during conversion. The `session_meta` dictionary also stores extra metadata (environment name, raw frame counts, discarded tail frames) that are informational only.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
```

iii. This is validation processing, not wasted computation. It ensures the geometry encoding is correct.
