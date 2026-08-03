# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files stored in the `data/` directory. It iterates over sorted filenames matching the pattern `QLAK-CA1-\d+`, loading each file with `joblib.load()`. Each file contains a nested dictionary keyed by animal ID with fields `trace`, `position`, `envs`, `blocked`, `maps`, etc. The AI then iterates over days within each animal to process sessions sequentially, streaming one animal at a time to limit memory usage.

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

iii. The AI documented (CONVERSION_NOTES Step 1) that "the joblib data files are the reference loading path used throughout main.py" and that `load_dat` loads per-animal preprocessed datasets. This matches the reference code's `load_dat(..., format='joblib')` function.

## 1-b. How are the data split into subjects?

i. Subjects are identified by animal ID from the sorted filenames in the `data/` directory. Each filename like `QLAK-CA1-08` corresponds to one subject. The AI maintains a `subjects` list and a `subject_to_idx` mapping to assign each session to its subject.

ii.
```python
animals = get_animals(data_dir)
subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
# ...
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    # ...
    converted["subject_idx"].append(subject_to_idx[animal])
```

iii. The AI noted (CONVERSION_NOTES Step 2) 7 mice: `QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`, `QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75`. This matches the paper's report of 7 mice.

## 1-c. How are the data split into sessions?

i. Sessions correspond to individual recording days within each animal. The AI iterates over the day index dimension of each animal's data arrays (e.g., `trace[day]`, `position[day]`). Each day is one session. Six mice contribute 31 sessions each; one mouse (QLAK-CA1-51) contributes 21 sessions, for 207 total.

ii.
```python
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    # ...process this session...
```

iii. The AI confirmed (CONVERSION_NOTES Step 4) that "207 sessions" matches the paper ("up to three total repetitions (31 days)") and that one mouse has only 21 sessions.

## 1-d. How are the data split into trials?

i. Since the raw data has no native trial structure (continuous ~40 min recordings), the AI splits each session into consecutive 1-minute chunks. Each session is capped at 40 minutes (72,000 frames at 30 Hz), yielding up to 40 trials per session. Each trial spans `TRIAL_FRAMES = 1800` frames (60 seconds * 30 fps).

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS
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

iii. The AI justified (CONVERSION_NOTES Step 5): "paper sessions are nominally 40 min. For recordings shorter than 72,000 frames, the 40th trial is shorter; for recordings longer than 72,000 frames, discard the small tail beyond 40 min." This follows the instruction that "The experiment consists of long recording sessions, which will be split into 1-minute trials."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All 40 trials per session are kept. The only restriction is that a trial must have at least `TEMPORAL_BIN_FRAMES = 3` frames to be included (line 102). The AI does not apply the reference code's velocity filtering (`v_thresh=5`) or cell activity threshold (`cell_threshold=5`) that are used in `decode_position_within`. Some sessions have a slightly shorter final trial (555 bins instead of 600) because the recording is under 40 minutes.

ii.
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The AI noted (CONVERSION_NOTES Step 1) that velocity and activity filtering exist "inside `decode_position_within`" and are part of the decoder rather than data preprocessing. The AI chose not to replicate these filters in the conversion, leaving them for the downstream decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in the per-animal data. This contains binarized rising-phase calcium event traces, where `1` marks significant calcium events. The trace has shape `(n_days, n_registered_cells, n_frames)`.

ii.
```python
trace_day = animal_data["trace"][day]
# ...
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The AI documented (CONVERSION_NOTES Step 1): "The `trace` field is already a rise-extracted event matrix where `1` marks significant calcium events. No code computes `dF/F`; for this conversion the reference neural signal is the provided event trace."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes two processing steps: (1) removal of unregistered cells (NaN values), and (2) temporal binning into 100ms bins via non-overlapping 3-frame averaging. No Gaussian smoothing or additional filtering is applied to the neural traces.

ii.
```python
TEMPORAL_BIN_FRAMES = 3

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

# In convert_session:
neural_raw = trace_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The AI justified (CONVERSION_NOTES Step 5): "the reference decoder temporally bins data in 3-frame windows. Using 100 ms bins matches this scale." The reference code's `fit_decoder` uses `AvgPool1d(kernel_size=3, stride=3)` for temporal binning, which the AI's reshape+mean approach replicates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only unregistered cells (those with NaN in the first frame of a session) are removed. No velocity-based frame filtering, no cell activity threshold, and no place-cell filtering is applied.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

# In convert_session:
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The AI noted (CONVERSION_NOTES Step 1): "Place-cell filtering exists in the repository for specific analyses, but the position decoder in `main.py` does not restrict to place cells before decoding." And: "Registration quality is encoded by `NaN` traces/maps for cells absent on a given day." The paper says all cells were retained for main analyses after demonstrating high spatial reliability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the start of its corresponding 1-minute chunk within the session. There is no discrete stimulus or event to align to (free exploration). The alignment event is described as "start of each consecutive 1-minute chunk within a session" with `off_start=0.0` and `off_end=60.0` seconds.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    # start = trial_idx * TRIAL_FRAMES
```
```python
"temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),  # 60.0
```

iii. The AI noted this is a continuous free exploration task with no discrete trials or events, so alignment to chunk boundaries is the natural choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100ms (stored as `time_bin_size = 100.0` ms). Temporal rebinning is applied: the raw 30 Hz data (33.3ms frames) is averaged in non-overlapping 3-frame bins, yielding ~10 Hz / 100ms resolution. Each full 60-second trial yields 600 time bins.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The AI justified: "the reference decoder temporally bins data in 3-frame windows" matching `temporal_bin_size=3` in the reference `fit_decoder` function.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field in the per-animal data. This field contains lists of blocked partition indices in the 3x3 environment layout, where `-1` denotes no blocked partitions. The AI chose NOT to use the reference code's `get_env_mat` function which maps `envs` string labels to binary 3x3 matrices.

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. The AI justified (CONVERSION_NOTES Step 4): "For the decoder input, use raw `blocked` as canonical geometry because it directly encodes blocked partitions per session. Keep `envs` as metadata / human-readable labels. This preserves actual session geometry and avoids helper-template simplifications." The AI noted that `get_env_mat` templates "do not exactly match raw blocked partitions for all shapes."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The raw blocked partition indices are converted to a 9-element binary vector (1=open, 0=blocked). If the blocked value is `-1` (no blocked partitions), all positions are marked open. The resulting 3x3 mask is transposed before flattening to align with the output class convention (`x_bin * 3 + y_bin`). The geometry is static per trial.

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

iii. The AI initially had a geometry orientation bug where ~17.15% of output bins appeared in blocked partitions. After discovering this in Step 10, the AI added the `.T` transpose to fix alignment between blocked indices and output class indices. After the fix, blocked-bin occupancy dropped to 4.15e-05.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field, which contains x-y coordinates of shape `(n_days, 2, n_frames)` sampled at 30 Hz. Position is derived from DeepLabCut head tracking.

ii.
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
# ...
position_raw = position_valid[:, start:end]
```

iii. The AI noted (CONVERSION_NOTES Step 1): "position is stored per day as 2 x time" which matches the reference code usage.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position undergoes: (1) temporal binning via 3-frame averaging (same as neural), (2) discretization into a 3x3 grid using session-wise maximum-based flooring, (3) encoding as a single categorical variable (classes 0-8) with shape `(1, T)`.

ii.
```python
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The AI justified (CONVERSION_NOTES Step 5): "Reusing the same [session-wise max-based flooring] logic preserves coordinate handling while adapting to the requested 3 x 3 output grid."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using session-wise maximum-based integer flooring. For each coordinate, `bin = floor(position / ((session_max + buffer) / n_bins))`, clipped to [0, 2]. The final class is `x_bin * 3 + y_bin`, yielding 9 classes (0-8). The buffer (1e-5) prevents the maximum value from falling into an extra bin.

ii.
```python
POSITION_BINS = 3
BUFFER = 1e-5

def discretize_position_3x3(position_xy_by_time: np.ndarray, session_max_xy: np.ndarray) -> np.ndarray:
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    denom = np.where(denom <= 0, 1.0, denom)
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```

iii. The AI documented this matches the reference code's spatial binning: "the paper's code bins position by dividing by `(session_max + buffer) / n_bins`" but adapted from 15 to 3 bins per instruction.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame indices within each trial. Both undergo identical 3-frame temporal binning, applied to the same frame range `[start:end]` for each trial. This ensures temporal alignment is preserved.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI noted (CONVERSION_NOTES Step 1): "position is derived from DeepLabCut head tracking" and is "aligned with neural recording" at 30 Hz. The reference code confirms neural and behavioral streams were acquired simultaneously.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Unregistered cells (NaN in trace) are excluded per session via `get_valid_cell_mask`. (2) Sessions shorter than 40 minutes result in a shorter final trial (555 bins instead of 600 for three animals). (3) Frames beyond 40 minutes are discarded. (4) The discretization function guards against zero denominators with `np.where(denom <= 0, 1.0, denom)`. (5) Position bins are clipped to valid range [0, 2] to handle edge cases.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

# Short session handling:
usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
end = min(start + TRIAL_FRAMES, usable_frames)
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The AI noted (CONVERSION_NOTES Step 10): "Short recordings: final trial length is 555 bins for the three slightly short 39.93 min recordings and 600 bins elsewhere." The minimum trial length check (`>= TEMPORAL_BIN_FRAMES`) ensures no degenerate trials.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading/decompressing the large joblib animal files (each containing dense arrays for all days). The AI noted full conversion takes a few minutes total. Processing per session is ~0.73 seconds after the initial load. The full conversion (207 sessions) completes in under 10 minutes.

ii.
```python
animal_data = load_animal(data_dir, animal)  # Most time-consuming: decompressing large joblib
```

iii. The AI documented (CONVERSION_NOTES Step 6): "Full-data runtime will be dominated by decompressing the 7 large joblib animal files."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop iterating over trial slices (`for start, end in trial_slices`) could potentially be vectorized by reshaping the full session array into a 3D array of shape `(n_trials, n_neurons, n_frames_per_trial)` and applying binning operations in batch. However, this is complicated by the variable-length final trial.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy)
```

iii. The AI noted efficiency was acceptable and chose to keep the trial loop for clarity and to handle variable-length trials.

## 6-c. What processing does the code repeat multiple times?

i. The `open_mask.copy()` is called for every trial (40 times per session) even though the geometry is identical for all trials in a session. The `temporal_bin_mean` function is called separately for neural and position data for each trial, though these could theoretically be combined. The session-wise maximum computation is efficiently done once per session.

ii.
```python
input_trials.append(open_mask.copy())  # Repeated 40 times per session with same value
```

iii. The AI did not explicitly note this redundancy. The overhead is negligible since it's just copying a 9-element array.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `session_max_xy` in metadata but this is only used during conversion for discretization — it is not used by the downstream decoder. Similarly, `env_label`, `original_frames`, `usable_frames`, and `trial_slices` are stored in debug metadata that the decoder does not use. The position is temporally binned (averaged) before discretization, whereas discretizing directly on raw frames might produce a slightly different (arguably more accurate) result, though the reference code also bins position before discretization.

ii.
```python
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
converted["metadata"]["session_usable_frames"].append(debug_info["usable_frames"])
converted["metadata"]["session_trial_counts"].append(len(neural_trials))
converted["metadata"]["session_valid_neuron_counts"].append(int(brain_region_idx.shape[0]))
```

iii. The AI stored extra metadata for debugging and user inspection, which is good practice but adds some file size overhead to the pickle output.
