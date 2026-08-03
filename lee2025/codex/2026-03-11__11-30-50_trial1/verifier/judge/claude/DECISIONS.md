# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files in the `data/` directory. It identifies animal files by matching the regex `QLAK-CA1-\d+` (files without the `.mat` extension). Each file is loaded using `joblib.load()`, returning a dictionary keyed by the animal ID containing `trace`, `position`, `blocked`, and `envs` arrays. Animals are processed one at a time to limit memory usage.

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
```

iii. The AI noted in CONVERSION_NOTES.md Step 1 that "the joblib data files are the reference loading path used throughout `main.py`" and chose joblib over the `.mat` files as the primary data source. This is consistent with the reference code's `load_dat(..., format='joblib')` function.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to a separate joblib file in `data/`. The animal name is the filename itself (e.g., `QLAK-CA1-08`). Animals are processed in sorted filename order.

ii.
```python
animals = get_animals(data_dir)
subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
```

iii. The AI identified 7 subjects matching the paper's reported count. Subject splitting follows the data organization directly.

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple recording days (sessions). The AI iterates over the day index (`day_idx`) within each animal's `trace` array, where each day becomes a separate session.

ii.
```python
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    ...
```

iii. The AI confirmed 207 total sessions (6 mice with 31 sessions + 1 mouse with 21 sessions), matching the paper.

## 1-d. How are the data split into trials?

i. The AI splits each continuous session into exactly 40 one-minute trials (1800 frames at 30 Hz). Sessions are capped at `NOMINAL_SESSION_FRAMES = 72000` (40 minutes). Any frames beyond 40 minutes are discarded. The last trial may be shorter if the recording is slightly shorter than 40 minutes.

ii.
```python
NOMINAL_SESSION_SECONDS = 40 * 60
NOMINAL_SESSION_FRAMES = NOMINAL_SESSION_SECONDS * FPS

def get_trial_slices(n_frames_session: int) -> list[tuple[int, int]]:
    usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
    slices = []
    for trial_idx in range(NOMINAL_SESSION_SECONDS // TRIAL_SECONDS):
        start = trial_idx * TRIAL_FRAMES
        end = min(start + TRIAL_FRAMES, usable_frames)
        if end - start >= TEMPORAL_BIN_FRAMES:
            slices.append((start, end))
    return slices
```

iii. The AI justified this in CONVERSION_NOTES.md: "paper sessions are nominally 40 min. For recordings shorter than 72,000 frames, the 40th trial is shorter; for recordings longer than 72,000 frames, discard the small tail beyond 40 min."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied beyond the session duration cap. All 40 trials per session are retained, including potentially shorter final trials (555 bins instead of 600 bins after temporal binning).

ii.
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The AI noted that the original data has no native trial structure, so there is no basis for trial quality filtering. The only filter is that a trial must have at least 3 frames (one temporal bin).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the per-animal joblib files. This contains binarized rising-phase calcium event traces, stored as `(n_days, n_registered_cells, n_frames)`.

ii.
```python
trace_day = animal_data["trace"][day]
```

iii. The AI noted in CONVERSION_NOTES.md Step 1: "The `trace` field is already a rise-extracted event matrix where `1` marks significant calcium events. No code computes `dF/F`."

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps: (1) filtering out unregistered neurons (NaN at first timepoint), and (2) temporal binning by averaging 3 consecutive frames into 100ms bins. Data is cast to float32.

ii.
```python
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The AI justified temporal binning by noting "the reference decoder temporally bins data in 3-frame windows. Using 100 ms bins matches this scale."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered based on whether they are registered (non-NaN) in the current session. The check is performed by testing whether the first timepoint is NaN.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The AI noted: "NaN traces/maps for cells absent on a given day. Several functions treat NaN entries as unregistered cells rather than applying an additional cell-quality filter."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials are consecutive 1-minute segments starting from the session onset. The alignment event is the start of each chunk.

ii.
```python
"temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The AI noted there is no stimulus event to align to in this free-exploration task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 3 native frames (at 30 Hz) are averaged into one 100ms bin. This results in 600 time bins per full 1-minute trial instead of 1800 frames.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0

neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI justified this by referencing the decoder code: "the reference decoder temporally bins data in 3-frame windows (`temporal_bin_size=3`)." The `fit_decoder` function in `utils.py` uses 3-frame average pooling.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the per-animal data, which contains blocked partition indices for each session day.

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. The AI chose raw `blocked` over the `envs -> get_env_mat()` helper because "Step 4 showed helper templates are an abstract geometry representation and do not exactly match raw blocked partitions for all shapes."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary "open mask" vector where 1 = open and 0 = blocked. The mask is transposed from a y-major to x-major layout to match the output position class convention. The vector is static per trial.

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

iii. The AI initially had a geometry orientation bug where blocked indices didn't match the output bin convention. This was fixed by transposing the 3x3 mask before flattening, as documented in CONVERSION_NOTES.md Step 10.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` field in the per-animal data, which contains 2D (x, y) coordinates at each timepoint, shape `(n_days, 2, n_frames)`.

ii.
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
```

iii. The position is the tracked location of the mouse in the arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first temporally binned (3-frame averaging, same as neural data), then discretized into a 3x3 grid using session-wise maximum-based flooring.

ii.
```python
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy)
```

iii. The AI adapted the reference code's spatial binning logic from 15x15 to 3x3 bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using session-wise maximum values and floor-based binning. Each x/y coordinate is divided by `(session_max + buffer) / n_bins` and floored. The combined class is `x_bin * 3 + y_bin`, giving 9 classes (0-8).

ii.
```python
def discretize_position_3x3(position_xy_by_time: np.ndarray, session_max_xy: np.ndarray) -> np.ndarray:
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    denom = np.where(denom <= 0, 1.0, denom)
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```

iii. The AI justified session-wise max binning: "the paper's code bins position by dividing by `(session_max + buffer) / n_bins`. Reusing the same logic preserves coordinate handling."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned frame-for-frame in the source data (both at 30 Hz). Both are split into the same trial slices and undergo the same 3-frame temporal binning, preserving alignment.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI noted that neural and behavioral streams are "aligned frame-for-frame" and applies "a common 3-frame binning to both."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered neurons (NaN at first timepoint) are excluded per session. Sessions slightly shorter than 40 minutes produce a shorter final trial. Frames beyond 40 minutes are discarded. The blocked value of `-1` is treated as "no positions blocked."

ii.
```python
valid_cells = get_valid_cell_mask(trace_day)  # ~np.isnan(trace_day[:, 0])
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The AI documented handling of edge cases in CONVERSION_NOTES.md Steps 7 and 10, including verification that short final trials have 555 bins and that geometry-output alignment is correct after the transpose fix.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and decompressing the large joblib animal files. The full conversion took ~467 seconds, with most time spent on I/O.

ii. N/A (runtime is I/O-dominated)

iii. The AI noted: "Full-data runtime will be dominated by decompressing the 7 large joblib animal files."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop over `trial_slices` processes each trial individually with separate `temporal_bin_mean` calls. This could potentially be vectorized by processing the entire session's neural and position data at once, then slicing into trials.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    ...
```

iii. The AI did not identify this as an inefficiency. The temporal binning itself is vectorized within each trial via reshape/mean.

## 6-c. What processing does the code repeat multiple times?

i. The `temporal_bin_mean` function is called separately for neural and position data for every trial within every session. The session-max computation (`np.nanmax`) is performed once per session but could be cached.

ii.
```python
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI did not explicitly note repeated processing as a concern.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores extensive metadata (session IDs, geometry labels, frame counts, trial counts per session) that is not used by the downstream decoder. It also computes and stores `session_max_xy` in debug info.

ii.
```python
converted["metadata"]["session_ids"].append(session.session_id)
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
converted["metadata"]["session_usable_frames"].append(debug_info["usable_frames"])
```

iii. The AI considered this metadata useful for debugging and documentation rather than unnecessary overhead.
