# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from primary joblib files in the `data/` directory. Each file is named by animal ID (e.g., `QLAK-CA1-08`) and contains a dictionary keyed by the animal ID. The AI uses `joblib.load()` to read each file, then iterates over all days (sessions) within each animal to enumerate session references. An animal-level cache keeps one animal dataset in memory at a time to avoid redundant I/O.

ii.
```python
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

iii. The AI chose joblib files because the reference code's `load_dat` function defaults to `format="joblib"`. This matches the reference loading path. The AI documented this in CONVERSION_NOTES Step 1 and Step 5.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by listing all directories in `data/` that start with `QLAK-CA1-` and have no file extension. Each such directory name is treated as a unique subject. The AI sorts them alphabetically and creates a lookup dictionary mapping animal ID to integer index.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )
```

iii. The AI noted there are 7 subjects matching the paper's description. This is consistent with the reference code's `main.py` which iterates over the same animal IDs.

## 1-c. How are the data split into sessions?

i. Each day within each animal's dataset is treated as one session. The number of days is determined by `dat["envs"].shape[0]` (the number of rows in the environment labels array). Sessions are enumerated in order: all days for the first animal, then all days for the second, etc.

ii.
```python
for day_index in range(dat["envs"].shape[0]):
    session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. The AI verified that this yields 207 total sessions across 7 animals (31 days each for 6 animals, 21 for one), matching the paper's stated "207 sessions."

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into non-overlapping 1-minute (1800-frame) windows. Trials are created by computing floor division of total frames by 1800 and taking sequential slices. Any trailing frames that do not fill a complete 1-minute window are discarded.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. The AI justified this by noting the original data has no native trial structure (continuous 40-min sessions), and the instructions require splitting into 1-minute trials. This yields 39-40 trials per session and 8187 total trials.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full 1-minute trials are rejected (raising a ValueError). Sessions where no cells are present (all NaN in trace) are also rejected. Additionally, the geometry derived from the `blocked` field must match the valid spatial support from `maps['smoothed']`; mismatches raise an error. No individual trial-level quality filtering is applied.

ii.
```python
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")

if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)

if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The AI noted that the reference paper does not define trial-level quality filtering since sessions are continuous recordings. The geometry consistency check is the AI's own sanity check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dat[animal]['trace']`, which is the rise-extracted calcium event trace. Shape is `(n_sessions, n_registered_cells, n_frames)`. Values are binary (0 or 1), where 1 indicates a significant calcium transient rising phase.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The AI documented that the trace field is already preprocessed (rising-phase extracted, z-scored, thresholded at 2.5) and does not require delta F/F computation. This matches the reference code's README and the paper's methods.

## 2-b. How is the `neural` data processed?

i. For each session, only cells present on that day are retained (NaN filtering). The selected rows of the trace matrix are then sliced into 1-minute trial windows and cast to float16 for storage efficiency.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
# ...
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The AI justified keeping all present cells without additional filtering (e.g., no place-cell filtering, no activity threshold) based on the paper's statement that all cells were included in subsequent analyses. The reference decoder (`decode_position_within`) applies velocity and cell-activity thresholds online during decoding, but these are not pre-applied to the stored data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural data quality filter is removing cells that are absent on a given day, identified by NaN values in the first frame of the trace matrix. No place-cell filtering, no minimum activity threshold, and no velocity-based filtering are applied at the data conversion stage.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The AI explicitly noted that `decode_position_within` applies velocity filtering (v_thresh=5) and cell activity thresholding (cell_threshold=5) but that these are decoder-internal operations, not data curation steps. The AI chose not to replicate them in the conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the start of its non-overlapping 1-minute window within the session. The temporal alignment event is "start of each non-overlapping 1-minute within-session segment." The offset from alignment to trial start is 0.0 seconds, and offset to trial end is 60.0 seconds.

ii.
```python
"temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The AI noted that position and trace data are already synchronized at 30 Hz from the original acquisition, so no additional alignment is needed beyond slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the raw 30 Hz frame rate, giving a time bin size of 1000/30 = 33.33 ms. No temporal rebinning or smoothing is applied.

ii.
```python
FPS = 30.0
# ...
"time_bin_size": 1000.0 / FPS,  # 33.33 ms
```

iii. The AI kept the native acquisition rate. The reference code's `fit_decoder` applies 3-frame temporal binning and Gaussian smoothing internally during decoding, but this is decoder-specific processing, not a data format requirement.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from `dat[animal]['blocked']`, a Python list where each entry specifies which partitions of the 3x3 arena grid are blocked for that session/day. A value of -1 means no partitions are blocked.

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

iii. The AI chose to use the raw `blocked` field rather than the `envs` + `get_env_mat()` approach from the reference code. The AI verified that the `blocked`-derived 3x3 geometry (after transposing) exactly matches the valid spatial support of `maps['smoothed']` for all 207 sessions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked partition indices are used to create a 9-element binary vector: start with all 1s (open), set blocked positions to 0. The resulting 3x3 grid is transposed to align with the position/map coordinate frame, then flattened to a 9-dimensional float32 vector. This vector is static per trial (same for all trials within a session).

ii.
```python
geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
if not (blocked.size == 1 and blocked[0] == -1):
    geometry[blocked] = 0.0
geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)
```

iii. The transpose step was validated against the smoothed maps' non-NaN spatial support for all sessions. The AI documented this as a key decision in Step 5 of CONVERSION_NOTES.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output mouse position is derived from `dat[animal]['position']`, which has shape `(n_sessions, 2, n_frames)` and contains continuous x-y position tracked via DeepLabCut.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
```

iii. The AI identified position as the primary behavioral variable for the decoder output, consistent with the paper's position decoding analysis.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x-y position is discretized into a 3x3 spatial grid using floor division. For each dimension, the bin size is computed as `(session_max + buffer) / 3`. The position is divided by this scale and floored, then clipped to [0, 2]. The two bin indices are combined into a single class index: `x_bin * 3 + y_bin`, yielding 9 classes.

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

iii. The AI noted that session-wide position normalization keeps spatial bins consistent across all trials within a session, matching the reference approach in `get_rate_maps`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into 9 categories (3x3 grid) by floor-dividing each coordinate by the per-dimension bin scale. Each bin scale is `(max_position + 1e-5) / 3`. The small buffer prevents the maximum position from exceeding bin 2. Values are clipped to [0, 2] as a safety measure.

ii.
```python
POSITION_BUFFER = 1e-5
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The AI used per-dimension maxima for normalization (matching `get_rate_maps`), rather than the global max across both dimensions used in `decode_position_within`. Since the arena is 75cm x 75cm (square), the practical difference is negligible.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are inherently aligned because they share the same 30 Hz time base from simultaneous acquisition. Both are sliced using the same trial window indices (1800-frame non-overlapping segments), ensuring frame-by-frame correspondence.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    output_trials.append(output_trial)
```

iii. The AI noted that the reference paper states behavioral and cellular imaging streams were simultaneously acquired at 30 Hz and timestamped for post-hoc alignment, so no additional alignment processing is necessary.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of missing/irregular data:
- **Cells absent on a session day**: Identified by NaN in `trace[:, 0]`; these cells are excluded from that session's neural data.
- **Nested blocked entries**: The `extract_day_blocked_entry` function handles cases where `blocked[day]` is a nested list `[[value]]` by unwrapping it.
- **Trailing frames**: Frames that don't fill a complete 1-minute window are discarded (66-1666 frames depending on session).
- **Geometry consistency**: If the blocked-derived geometry doesn't match the maps' valid mask, a ValueError is raised.

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)

def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The AI documented edge cases in CONVERSION_NOTES Step 10, noting that discarded tail frames ranged from 60 to 1666 and that all sessions had at least 2 full trials.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib animal files from disk. Each file is 68-145 MB. The first session of each animal takes ~10 seconds (dominated by I/O), while subsequent sessions for the same animal take ~0.05-4 seconds depending on neuron count.

ii. From conversion output:
```
Processed QLAK-CA1-08_day00: 185 neurons, 39 trials, 71866 raw frames in 10.27s
Processed QLAK-CA1-08_day01: 153 neurons, 39 trials, 71866 raw frames in 0.05s
```

iii. The AI identified this in CONVERSION_NOTES Step 6 and implemented an animal-level cache to avoid reloading the same file for every session. Total conversion time was ~259 seconds for 207 sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial slicing loop iterates over each trial window sequentially, appending to lists. This could potentially be vectorized by reshaping the full session arrays into `(n_neurons, n_trials, trial_frames)` in one operation. The position occupancy computation in plotting also uses a Python loop over bins.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    input_trials.append(geometry_vector.copy())
    output_trials.append(output_trial)
```

iii. The loop is simple (39-40 iterations) and each iteration does a slice operation, so the overhead is minimal. The AI focused optimization efforts on I/O (the dominant bottleneck) rather than this loop.

## 6-c. What processing does the code repeat multiple times?

i. The code loads each animal file twice during the full conversion: once in `iter_session_refs` to enumerate all sessions (getting day counts), and again in the main processing loop. This redundant loading doubles I/O for the first pass.

ii.
```python
def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)  # First load
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(...)

# Then in convert_dataset:
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)  # Second load
```

iii. The AI did not explicitly document this redundancy. The first pass is used to enumerate all sessions before processing begins.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and validates the `aggregate_valid_map` from `maps['smoothed']` for every session, loading the smoothed maps array even though it is not used in the final converted data. This validation check (geometry vs. map consistency) is useful for correctness but is computationally wasteful since it loads a large array (`15 x 15 x n_cells x n_sessions`) just to check spatial support.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
# ...
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
```

iii. This is a sanity check that the AI implemented to verify geometry correctness. The smoothed maps data is not used in the final output. Additionally, the `geometry_vector.copy()` call in the trial loop creates redundant copies of the same static vector for each trial, though the memory cost is trivial (9 floats per trial).
