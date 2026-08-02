# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files in the `data/` directory. It scans for files matching the pattern `QLAK-CA1-\d+`, loads each with `joblib.load()`, and iterates over all days (sessions) within each animal. The data dictionary is keyed by animal ID and contains `trace`, `position`, `envs`, `blocked`, and other fields. Sessions are processed sequentially, one animal at a time.

ii.
```python
def get_animals(data_dir: str) -> list[str]:
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and re.fullmatch(r"QLAK-CA1-\d+", name):
            animals.append(name)
    return animals

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

# In build_dataset:
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    ndays = animal_data["trace"].shape[0]
    for day_idx in range(ndays):
        # ... process each session
```

iii. The AI identified that the reference code uses `load_dat(..., format='joblib')` to load per-animal joblib files (CONVERSION_NOTES Step 1). The AI followed the same loading path, loading each animal's dictionary and iterating over days. This matches the reference code's data loading approach.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the per-animal files in the data directory. Each file (e.g., `QLAK-CA1-08`) corresponds to one mouse. The AI discovers subjects by scanning the directory for files matching the naming pattern and sorting them alphabetically. A subject index maps each session to its animal.

ii.
```python
animals = get_animals(data_dir)  # sorted list of animal IDs
subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
# ...
converted["subject_idx"].append(subject_to_idx[animal])
```

iii. The AI documented 7 mice in the dataset (CONVERSION_NOTES Step 2) matching the paper's count of 7 mice. The sorted directory listing produces a deterministic subject ordering.

## 1-c. How are the data split into sessions?

i. Sessions correspond to recording days within each animal. The AI iterates over `day_idx in range(ndays)` for each animal, where `ndays` is determined from `animal_data["trace"].shape[0]`. Each day is one session. The total is 207 sessions (6 mice with 31 sessions + 1 mouse with 21 sessions).

ii.
```python
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    # ... convert this session
```

iii. The AI documented that sessions correspond to recording days per animal (CONVERSION_NOTES Step 2), matching the paper's description of one session per day. The total of 207 sessions matches the paper's report of "5,413 unique neurons across 207 sessions."

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into 40 consecutive 1-minute (1800-frame) chunks. The first 40 minutes (72,000 frames) are used; any frames beyond are discarded. If the recording is slightly shorter than 40 minutes, the final trial is shorter.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800
NOMINAL_SESSION_SECONDS = 40 * 60
NOMINAL_SESSION_FRAMES = NOMINAL_SESSION_SECONDS * FPS  # 72000

def get_trial_slices(n_frames_session: int) -> list[tuple[int, int]]:
    usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
    slices = []
    for trial_idx in range(NOMINAL_SESSION_SECONDS // TRIAL_SECONDS):  # 40 trials
        start = trial_idx * TRIAL_FRAMES
        end = min(start + TRIAL_FRAMES, usable_frames)
        if end - start >= TEMPORAL_BIN_FRAMES:
            slices.append((start, end))
    return slices
```

iii. The AI justified this by noting the paper's 40-minute session duration and the instructions requiring "1-minute trials within each session" (CONVERSION_NOTES Step 5, Key Decision 3). Sessions slightly shorter than 40 minutes result in a shorter final trial rather than being discarded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are not explicitly filtered for quality. The only condition is that a trial must have at least `TEMPORAL_BIN_FRAMES` (3) frames to be included, which effectively only filters out zero-length segments. No velocity filtering or other quality control is applied at the trial level during data conversion.

ii.
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The AI noted that the reference code's velocity and activity filtering (`v_thresh=5`, `cell_threshold=5`) are applied internally within the decoder function `decode_position_within`, not during data preparation (CONVERSION_NOTES Step 1). Since the conversion prepares data for a different downstream decoder, these filters were not applied during conversion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field of each animal's dataset. This field contains binarized rising-phase calcium event traces, stored as a matrix of shape `(n_days, n_registered_cells, n_frames)`. Values are binary (0/1) event indicators at 30 Hz.

ii.
```python
trace_day = animal_data["trace"][day]
# ...
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The AI documented that the `trace` field contains pre-extracted binary rising-phase calcium events and that no dF/F computation is needed (CONVERSION_NOTES Steps 1, 5). This matches the reference code which treats these binary traces as the neural signal directly.

## 2-b. How is the `neural` data processed?

i. The neural traces are (1) filtered to keep only valid (non-NaN) cells for that session, (2) clipped to the usable session duration (max 40 minutes), (3) split into 1-minute trial segments, and (4) temporally binned by averaging non-overlapping 3-frame windows to produce 100ms time bins. No Gaussian smoothing is applied.

ii.
```python
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
# ...
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The AI followed the reference code's temporal binning approach (3-frame averaging, matching `decode_position_within`'s `temporal_bin_size=3`). However, the reference code's `fit_decoder` function applies Gaussian smoothing (`gaussian_filter1d` with `sigma=temporal_bin_size`) to traces before temporal binning. The AI did not apply this smoothing step (CONVERSION_NOTES Step 5, Key Decision 4).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered only by removing cells that are unregistered on a given day (marked by NaN in the first frame). No further quality filtering (place-cell classification, activity thresholds, etc.) is applied.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

# In convert_session:
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The AI justified keeping all valid registered cells by citing the paper's statement that "observed reliability motivated inclusion of all cells in subsequent analyses" and noting that `decode_position_within` in `main.py` uses all cells (CONVERSION_NOTES Step 5, Key Decision 2). The reference decoder applies cell activity thresholds (`cell_threshold=5`) internally during decoding, not during data preparation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the start of its corresponding 1-minute chunk within the session. Trial 0 starts at frame 0, trial 1 at frame 1800, etc. The temporal alignment event is "start of each consecutive 1-minute chunk within a session" with `off_start=0.0` and `off_end=60.0` seconds.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    # ...

# In metadata:
"temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),  # 60.0
```

iii. The AI explained that since the source data has no native trial structure, trials are defined by consecutive 1-minute chunks starting from session onset (CONVERSION_NOTES Step 5).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms (3 frames at 30 Hz). Temporal rebinning is applied: non-overlapping 3-frame averages reduce the raw 30 Hz data to 10 Hz. Each 60-second trial produces 600 time bins (or 555 for slightly short recordings).

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
# ...
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI chose 100 ms bins to match the reference decoder's `temporal_bin_size=3` parameter used in `fit_decoder` and `test_decoder` (CONVERSION_NOTES Step 5, Key Decision 4).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field of each animal's dataset. This field is a list of length `n_days`, where each element contains an array of blocked partition indices in the 3x3 environment layout. A value of `-1` indicates no blocked partitions (fully open environment).

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. The AI chose `blocked` over the `envs` field + `get_env_mat()` helper because the raw `blocked` field "directly encodes blocked partitions per session" and avoids "helper-template simplifications" that may not exactly match raw data for some shapes like `glenn`, `t`, `l`, and `bit donut` (CONVERSION_NOTES Steps 4-5, Key Decision 5).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked partition indices are converted into a 9-element binary open-mask vector. Partitions listed as blocked get value 0, all others get value 1. A transpose of the 3x3 mask is applied to convert from the raw y-major (row-major) index ordering to the x-major ordering used by the output position classes (`x_bin * 3 + y_bin`).

ii.
```python
def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    # Raw blocked indices are stored in a y-major 3x3 layout, while output classes use x_bin * 3 + y_bin.
    return open_mask.reshape(3, 3).T.reshape(-1)
```

iii. The AI initially had a geometry orientation bug causing ~17% of output bins to fall in blocked partitions. This was discovered in Step 10 (Critical Review 1) and fixed by adding the `.T` transpose. After the fix, blocked-bin occupancy dropped to ~4e-5 (CONVERSION_NOTES Step 10).

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per trial -- the same 9-element vector is assigned to all trials within a session. It is stored as a 1D array of shape `(9,)` per trial, not tiled across time.

ii.
```python
input_trials.append(open_mask.copy())
```

iii. The AI noted that geometry is constant within each session and that the downstream decoder/validator will automatically tile 1D inputs across time (CONVERSION_NOTES Step 5, Key Decision 8).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output mouse position is derived from the `position` field of each animal's dataset, which has shape `(n_days, 2, n_frames)` containing x-y coordinates from DeepLabCut head tracking at 30 Hz.

ii.
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
# ...
position_raw = position_valid[:, start:end]
```

iii. The AI documented that position is derived from DeepLabCut head tracking, synchronized with neural imaging at 30 Hz (CONVERSION_NOTES Steps 2-3).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position data is (1) split into trial segments matching neural data, (2) temporally binned by averaging non-overlapping 3-frame windows (same as neural), (3) discretized into a 3x3 grid using session-wise maximum-based binning, and (4) encoded as a single categorical variable (0-8) using `x_bin * 3 + y_bin`.

ii.
```python
position_raw = position_valid[:, start:end]
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64)

def discretize_position_3x3(position_xy_by_time, session_max_xy):
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    denom = np.where(denom <= 0, 1.0, denom)
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```

iii. The AI stated it matched "the reference code's flooring rule" adapted from 15x15 bins to 3x3 bins (CONVERSION_NOTES Step 5, Key Decision 6). However, the reference code's `decode_position_within` computes the maximum across ALL days for each animal (`behav.max(axis=0).max(axis=1)`), while the AI uses per-session maxima (`session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)`).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Continuous x-y position is discretized into 9 categories (3x3 spatial grid). Each coordinate is divided by `(session_max + buffer) / 3`, floored to get a bin index (0, 1, or 2), and the two indices are combined as `x_bin * 3 + y_bin` to produce a single class label 0-8.

ii.
```python
POSITION_BINS = 3
BUFFER = 1e-5

def discretize_position_3x3(position_xy_by_time, session_max_xy):
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```

iii. The AI adapted the reference code's position binning logic from 15x15 to 3x3 as required by the task instructions (CONVERSION_NOTES Step 5, Key Decision 6). The floor-and-clip approach and buffer parameter match the reference code's integer conversion pattern.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The output position data uses the exact same trial segmentation and temporal binning as the neural data. Both are split at the same frame boundaries and averaged in the same 3-frame windows, ensuring frame-by-frame alignment.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI ensured alignment by applying the same `temporal_bin_mean` function with the same bin size to both neural and position data within the same trial boundaries (CONVERSION_NOTES Step 5).

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms. Temporal rebinning is applied: raw 30 Hz data (33.3 ms per frame) is averaged in non-overlapping windows of 3 frames to produce 100 ms bins.

ii.
```python
TEMPORAL_BIN_FRAMES = 3  # 3 frames * 33.3 ms = 100 ms
TIME_BIN_MS = 100.0
```

iii. Same as 2-e. The 3-frame binning matches the reference decoder's `temporal_bin_size=3` parameter.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data are aligned by applying identical trial segmentation (same start/end frames) and identical temporal binning (3-frame averaging). The input (geometry) is static per trial and does not require temporal alignment.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy)
    neural_trials.append(neural_binned)
    input_trials.append(open_mask.copy())
    output_trials.append(output_binned)
```

iii. The AI verified temporal alignment through processing plots and sanity checks, confirming matching dimensions and no visible misalignment (CONVERSION_NOTES Steps 7, 10).

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing/unregistered neurons (NaN traces) are excluded on a per-session basis. Sessions slightly shorter than 40 minutes produce a shorter final trial (555 bins instead of 600). The `blocked` field value of `-1` (no blocked partitions) is handled as a special case. Negative or zero denominators in position discretization are replaced with 1.0 to avoid division errors.

ii.
```python
# NaN neurons excluded
valid_cells = get_valid_cell_mask(trace_day)  # ~np.isnan(trace_day[:, 0])

# Short sessions produce shorter final trial
end = min(start + TRIAL_FRAMES, usable_frames)

# -1 means no blocked partitions
if not (blocked_values.size == 1 and blocked_values[0] == -1):
    open_mask[blocked_values.astype(int)] = 0.0

# Safe denominator
denom = np.where(denom <= 0, 1.0, denom)
```

iii. The AI documented these edge cases in CONVERSION_NOTES Steps 5 and 10, including verification that short trials (from 3 recordings at ~39.93 min) produce 555-bin final trials.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading/decompressing the 7 large per-animal joblib files from disk. The actual per-session conversion is fast (~0.73 s/session) since it uses vectorized operations.

ii.
```python
animal_data = load_animal(data_dir, animal)  # joblib.load - the bottleneck
```

iii. The AI documented timing information showing that full conversion takes approximately 3-4 minutes, with most time spent on file I/O rather than computation (CONVERSION_NOTES Steps 6-7).

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loops are the outer iteration over animals and sessions, which cannot easily be vectorized since they involve sequential file I/O and session-specific processing. The inner temporal binning is already vectorized using reshape+mean. The trial slicing loop is lightweight.

ii.
```python
# Already vectorized temporal binning:
def temporal_bin_mean(arr, bin_size):
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The AI implemented vectorized temporal binning from the start (CONVERSION_NOTES Step 6) and noted no significant additional vectorization opportunities.

## 7-c. What processing does the code repeat multiple times?

i. The `temporal_bin_mean` function is called twice per trial (once for neural data, once for position data), but these operate on different arrays so both calls are necessary. The `open_mask.copy()` creates a new copy per trial, but the mask is identical across trials within a session -- this could theoretically be optimized to share references but the overhead is negligible.

ii.
```python
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
# ...
input_trials.append(open_mask.copy())  # same mask copied 40 times per session
```

iii. Not explicitly discussed in CONVERSION_NOTES.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores extensive metadata (session IDs, geometry labels, frame counts, trial counts, neuron counts) that is not used by the downstream decoder but serves documentation purposes. The `debug_info` dictionary is constructed for each session but only used for plotting. The `session_max_xy` is computed over all usable frames but only used for position discretization.

ii.
```python
debug_info = {
    "env_label": str(env_label),
    "open_mask": open_mask,
    "usable_frames": usable_frames,
    "original_frames": int(trace_valid.shape[1]),
    "trial_slices": trial_slices,
    "session_max_xy": session_max_xy,
}
# Stored in metadata but not used by decoder:
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
```

iii. The AI did not discuss unnecessary processing in CONVERSION_NOTES.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. Missing/unregistered neurons are excluded per session via NaN detection. Short recordings produce shorter final trials. The `-1` sentinel value in the `blocked` field is handled explicitly. Position discretization includes safe division guards.

ii. (Same code snippets as question 6)

iii. The AI documented these handling approaches in CONVERSION_NOTES Steps 5 and 10.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Loading per-animal joblib files dominates the runtime. Per-session conversion is fast due to vectorized operations.

ii. (Same code as 7-a)

iii. (Same justification as 7-a)

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The trial-level loop could theoretically be eliminated by processing all trials at once as a single large array, but the overhead is minimal and the current approach is clear.

ii. (Same code as 7-b)

iii. (Same justification as 7-b)

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. The main repetition is the geometry mask being copied identically for each trial within a session.

ii. (Same code as 7-c)

iii. (Same justification as 7-c)

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Extensive metadata and debug info are computed but not used by the downstream decoder.

ii. (Same code as 7-d)

iii. (Same justification as 7-d)
