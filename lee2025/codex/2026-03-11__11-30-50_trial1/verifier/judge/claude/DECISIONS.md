# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-converted joblib files in the `data/` directory. Each per-animal joblib file is a dictionary keyed by animal ID containing `trace`, `position`, `blocked`, `envs`, and other fields. The AI finds animal files by matching filenames against the regex pattern `QLAK-CA1-\d+`.

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

iii. The reference code repository provides both MATLAB `.mat` files and pre-converted joblib files. The AI chose to load the joblib files, which is the same loading path used by the reference analysis code (`load_dat(..., format='joblib')` in `utils.py`). The AI documented this decision in CONVERSION_NOTES.md Step 1.

## 1-b. How are the data split into subjects?

i. Each joblib file in the data directory corresponds to one subject (mouse). The subject name is the filename itself (e.g., `QLAK-CA1-08`). Animals are sorted alphabetically to determine session ordering.

ii.
```python
animals = get_animals(data_dir)
subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
```

iii. Each file contains all recording sessions for one animal. The filename serves as the subject identifier. 7 subjects are identified.

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple recording days. Sessions are indexed by iterating over the first dimension of the `trace` array (shape `(n_days, n_cells, n_frames)`). Each day becomes a separate session in the output.

ii.
```python
animal_data = load_animal(data_dir, animal)
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    ...
```

iii. The data structure stores one recording session per day index. The total is 207 sessions (6 mice with 31 sessions + 1 mouse with 21 sessions).

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into up to 40 one-minute trials (1800 frames at 30 Hz). The session is first capped at `NOMINAL_SESSION_FRAMES = 72000` (40 minutes). Trial slices are created sequentially; the final trial may be shorter than 60 seconds if the session recording is slightly shorter than 40 minutes.

ii.
```python
NOMINAL_SESSION_FRAMES = NOMINAL_SESSION_SECONDS * FPS  # 72000

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

iii. The paper states all sessions were 40 minutes. The AI caps at 40 minutes to match. For sessions slightly shorter than 40 minutes (~39.93 min), the 40th trial gets fewer frames (1665 instead of 1800, becoming 555 time bins after 3-frame averaging instead of 600). The AI chose to keep these shorter trials rather than discard them.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial quality filtering is applied. All trials within the 40-minute window are kept, including shorter final trials.

ii. N/A — no trial filtering code.

iii. The reference paper does not describe trial-level quality controls for the continuous exploration sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the per-animal data dictionary, which contains binarized rising-phase calcium event traces with shape `(n_days, n_registered_cells, n_frames)`.

ii.
```python
trace_day = animal_data["trace"][day]
```

iii. The CONVERSION_NOTES.md documents that `trace` contains pre-processed rising-phase extracted binary event matrices, consistent with the paper's Methods description of calcium event extraction.

## 2-b. How is the `neural` data processed?

i. The neural data is: (1) filtered to keep only valid (non-NaN) cells for each session, (2) cast to float32, (3) temporally binned using non-overlapping 3-frame averaging to produce 100 ms time bins.

ii.
```python
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
# ... then per trial:
neural_raw = trace_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The AI applies 3-frame temporal averaging to match the reference code's decoder (`fit_decoder`/`test_decoder` in `utils.py`), which uses `temporal_bin_size=3` via `AvgPool1d`. This is documented in CONVERSION_NOTES.md Steps 1 and 5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are NaN at the first timepoint of a session are removed. Only cells with valid (non-NaN) values at timepoint 0 are kept.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. This matches the reference code's `get_valid_cell_mask` pattern in `utils.py`. The data uses NaN to mark cells not registered on a given day; checking the first timepoint is sufficient because registration status is constant across time within a session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials are consecutive 1-minute segments starting from the beginning of the recording session. The alignment event is the start of each 1-minute chunk.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
```

iii. The experiment is free exploration with no stimulus events. The AI sets `temporal_alignment_event` to "start of each consecutive 1-minute chunk within a session" in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms time bins (3-frame averaging of the native 30 Hz data). Temporal rebinning IS applied.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
# ...
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI chose 100 ms bins to match the reference decoder's temporal binning (`temporal_bin_size=3` in `decode_position_within`). This is documented in CONVERSION_NOTES.md Step 5 as Key Decision #4.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the `blocked` field in the per-animal data dictionary, which contains lists of blocked partition indices for each recording session.

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. The AI chose to use the raw `blocked` field rather than the `envs` label mapped through `get_env_mat()`, because the raw field preserves actual per-session blocked partitions without template simplifications. This is documented in CONVERSION_NOTES.md Step 4 and Step 5.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked partition indices are converted to a 9-element binary vector where 1 = open and 0 = blocked. The vector is then transposed (reshaped to 3x3, transposed, re-flattened) to convert from the raw y-major index convention to the AI's x-major output bin convention.

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

iii. The AI uses "open" polarity (1=open, 0=blocked) rather than "blocked" polarity. The transpose aligns the geometry input with the output position bin ordering (x-major). The AI documented this coordinate convention handling in CONVERSION_NOTES.md Step 10 where it found and fixed a geometry orientation bug.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Environment geometry is static per session — the same 9-element vector is used for all trials and all time bins within a session. No temporal alignment is needed.

ii.
```python
input_trials.append(open_mask.copy())
```

iii. Blocked partitions do not change within a recording session, so a static per-trial input is appropriate.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in the per-animal data, which contains 2D (x, y) coordinates with shape `(n_days, 2, n_frames)`.

ii.
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
```

iii. The `position` field records the animal's tracked location at each frame, derived from DeepLabCut head tracking as described in the paper.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is: (1) temporally binned using 3-frame averaging (same as neural), (2) discretized into a 3x3 grid using session-wise maximum-based binning with a small buffer, (3) encoded as a single categorical variable (0-8).

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

iii. The AI uses session-wise maximum position for normalization, matching the reference code's decoder logic in `fit_decoder`/`test_decoder`. The bin index formula is `x_bin * 3 + y_bin`. Documented in CONVERSION_NOTES.md Step 5 Key Decisions #6 and #7.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 classes (3x3 grid) using `np.floor(position / ((session_max + buffer) / 3))`, clipped to [0, 2]. The class label is `x_bin * 3 + y_bin`.

ii.
```python
binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
binned = np.clip(binned, 0, POSITION_BINS - 1)
classes = binned[0] * POSITION_BINS + binned[1]
```

iii. This mirrors the reference code's spatial binning logic adapted from 15x15 bins to 3x3 bins per the decoder task specification.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same native frame rate (30 Hz). Both are sliced into the same trial segments and then temporally binned with the same 3-frame averaging, ensuring frame-for-frame alignment.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy)
```

iii. Using the same trial slices and the same temporal binning ensures that neural and output arrays have identical time dimensions. The AI verifies this in sanity checks (CONVERSION_NOTES.md Step 10).

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms (3-frame bins). Temporal rebinning IS applied via non-overlapping 3-frame averaging of the native 30 Hz data.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. Matches the reference decoder's temporal bin size. Documented in metadata as `time_bin_size: 100.0`.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position data are aligned frame-for-frame at the native 30 Hz rate, then both are binned with the same 3-frame averaging operation and sliced using the same trial boundaries. Input geometry is static (no temporal dimension).

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. Common trial slices and common temporal binning ensure alignment. The AI verified alignment in processing plots and sanity checks.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Unregistered neurons (NaN at the first timepoint) are removed per session. Sessions slightly shorter than 40 minutes produce a shorter final trial rather than being discarded. The `blocked` field value of `[-1]` (indicating no blocked partitions) is handled as a special case producing an all-ones open mask.

ii.
```python
valid_cells = get_valid_cell_mask(trace_day)  # ~np.isnan(trace_day[:, 0])
# ...
if not (blocked_values.size == 1 and blocked_values[0] == -1):
    open_mask[blocked_values.astype(int)] = 0.0
```

iii. The AI documented NaN handling and edge cases in CONVERSION_NOTES.md Steps 2, 5, and 10.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and decompressing the per-animal joblib files. Conversion processing itself is fast (~0.1-0.4 seconds per session). Total conversion time for 207 sessions was approximately 467 seconds, dominated by I/O.

ii. N/A (timing is printed to stdout, not code logic).

iii. From conversion output: individual animal load + process times range from ~11s (QLAK-CA1-51, 21 sessions) to ~124s (QLAK-CA1-50, 31 sessions). The AI documented timing in CONVERSION_NOTES.md Step 7.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial processing loop (iterating over `trial_slices`) could potentially be vectorized for equal-length trials by reshaping the full session array. However, the last trial can be shorter, making full vectorization less straightforward.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    ...
```

iii. The AI vectorized temporal binning (reshape + mean) but kept the trial loop. The per-trial loop overhead is minimal compared to I/O.

## 7-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session's data is loaded once and processed in a single pass. The `session_max_xy` is computed once per session and reused for all trials.

ii.
```python
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
```

iii. The AI implemented per-animal streaming to avoid redundant loading.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive metadata (session IDs, geometry labels, frame counts, trial counts, neuron counts) that is not used by the downstream decoder. The `debug_info` dictionary per session is computed but only used for optional plotting and metadata storage.

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
```

iii. This metadata overhead is minimal and serves documentation/debugging purposes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6: unregistered neurons are removed (NaN check at first timepoint), shorter sessions produce shorter final trials, and `blocked=[-1]` is handled as a special case. The AI also found and fixed a geometry coordinate orientation bug during Step 10 review.

ii. See 6 above.

iii. The AI documented the geometry bug discovery and fix in CONVERSION_NOTES.md Step 10: the initial version stored the geometry vector in raw blocked-index layout while output classes used a different coordinate order, causing ~17% of output bins to appear in blocked partitions. After adding the transpose fix, blocked-bin occupancy dropped to ~0.004%.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading/decompressing joblib files dominates runtime.

ii. See 7-a above.

iii. See 7-a above.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-trial processing loop over `trial_slices`.

ii. See 7-b above.

iii. See 7-b above.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: no significant repeated processing identified.

ii. See 7-c above.

iii. See 7-c above.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: metadata computation and debug_info are not used by the downstream decoder.

ii. See 7-d above.

iii. See 7-d above.
