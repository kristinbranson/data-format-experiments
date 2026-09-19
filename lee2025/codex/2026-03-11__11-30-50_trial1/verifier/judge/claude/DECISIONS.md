# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files (not the `.mat` files). Each file is a dictionary keyed by animal ID containing `trace`, `position`, `blocked`, `envs`, `maps`, `SFPs`, and `centroids`. The AI streams one animal at a time using `joblib.load`.

ii.
```python
def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]
```

iii. The CONVERSION_NOTES (Step 1) document that the joblib files are the reference loading path used by `main.py` in the original code. The AI chose joblib over `.mat` because the reference code repository itself loads from joblib format.

## 1-b. How are the data split into subjects?

i. Each file in the data directory matching the pattern `QLAK-CA1-\d+` corresponds to one subject. Files are sorted alphabetically to establish subject order.

ii.
```python
def get_animals(data_dir: str) -> list[str]:
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and re.fullmatch(r"QLAK-CA1-\d+", name):
            animals.append(name)
    return animals
```

iii. The data directory contains per-animal files. The filename regex matches the animal ID naming convention from the paper (7 mice: QLAK-CA1-08, -30, -50, -51, -56, -74, -75).

## 1-c. How are the data split into sessions?

i. Each animal file contains multi-day data. The `trace` array has shape `(n_days, n_registered_cells, n_frames)`. Each day index becomes a separate session.

ii.
```python
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    ...
    trace_day = animal_data["trace"][day]
    position_day = animal_data["position"][day]
```

iii. Per the paper and data exploration, each animal has up to 31 recording days (one mouse has 21), totaling 207 sessions across all animals.

## 1-d. How are the data split into trials?

i. Each session is split into up to 40 one-minute trials. Sessions are capped at 40 minutes (72,000 frames at 30 Hz). Trial slices are computed by iterating over 40 possible 1800-frame windows; the last trial may be shorter if the recording is shorter than 40 minutes. A trial is kept if it contains at least TEMPORAL_BIN_FRAMES (3) frames.

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

iii. The paper states sessions are 40 minutes. The AI caps at this nominal duration and creates up to 40 one-minute trials. For recordings slightly shorter than 40 minutes (~39.93 min for 3 animals), the 40th trial is shorter (555 time bins instead of 600 after temporal binning). The CONVERSION_NOTES document this as an intentional decision to retain data rather than discard the final minute.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied beyond discarding frames beyond the 40-minute cap and requiring a minimum trial length of 3 frames.

ii. N/A (no explicit trial quality filter code)

iii. The paper and reference code do not describe trial-level quality filtering for continuous recording sessions. The only filtering is the session duration cap.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binarized rising-phase calcium event traces (binary 0/1 values).

ii.
```python
trace_day = animal_data["trace"][day]
```

iii. The CONVERSION_NOTES (Steps 1, 3) document that `trace` contains preprocessed binary rising-event data. The Methods section describes the rising-phase extraction pipeline (differentiation, smoothing, z-scoring, thresholding at z > 2.5). No further dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The neural data is filtered to keep only valid (non-NaN) neurons, cast to float32, and temporally binned using non-overlapping 3-frame (100 ms) mean pooling.

ii.
```python
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
...
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The CONVERSION_NOTES (Step 5, Key Decision 4) justify temporal binning by noting the reference decoder code (`fit_decoder`) uses `temporal_bin_size=3` frames via AvgPool1d. Using 100 ms bins matches this scale and reduces computational load for the downstream decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by checking if the first timepoint is NaN. Neurons that are NaN at the first timepoint (unregistered on that day) are excluded.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The CONVERSION_NOTES document that unregistered cells are represented as all-NaN across time. Checking the first timepoint is sufficient to detect these. No additional quality filter (e.g., place-cell classification) is applied, consistent with the paper's statement that all cells were retained for main analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The continuous recording is split into consecutive 1-minute chunks starting from the beginning of each session. The alignment event is the start of each chunk.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
```

iii. The experiment is free exploration with no discrete stimulus events. Trials are artificial temporal segments of the continuous recording. The metadata records: `"temporal_alignment_event": "start of each consecutive 1-minute chunk within a session"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 100 ms temporal resolution. Temporal rebinning is applied: non-overlapping 3-frame averages at the native 30 Hz rate yield 100 ms bins (600 bins per full 60-second trial).

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
...
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI chose 100 ms bins to match the reference decoder code's temporal binning (`temporal_bin_size=3` in `fit_decoder`/`test_decoder`). This reduces the data volume by 3x while preserving the temporal resolution used in the paper's analyses.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field in the raw data, which contains indices of blocked reward positions in the 3x3 grid for each session.

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. The CONVERSION_NOTES (Step 4) compare two geometry sources: the `blocked` field (direct blocked indices) and the `envs` field mapped through `get_env_mat()`. The AI chose raw `blocked` because it directly encodes per-session blocked partitions without the ambiguities of the helper template.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element open mask (1=open, 0=blocked). If no partitions are blocked (`[-1]`), the vector is all ones. The mask is then transposed from y-major to x-major ordering to match the position output bin convention.

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

iii. The CONVERSION_NOTES (Step 10) document that the AI found and fixed a geometry orientation bug. Initially the geometry vector was in y-major order while position classes used x-major order (`x_bin * 3 + y_bin`). The transposition `.reshape(3,3).T.reshape(-1)` corrects this misalignment. The open mask encoding (1=open) was chosen as it directly indicates where the animal can be.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` variable, which contains 2D (x, y) coordinates at 30 Hz.

ii.
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
```

iii. The position data comes from DeepLabCut head tracking, as described in the paper's Methods.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first temporally binned (3-frame mean, same as neural), then discretized into a 3x3 grid using session-wise maximum-based binning. The bin index formula is `x_bin * 3 + y_bin`. A small buffer (1e-5) is added to the session max to avoid edge-case out-of-range values.

ii.
```python
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
...
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy)

def discretize_position_3x3(position_xy_by_time, session_max_xy):
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    denom = np.where(denom <= 0, 1.0, denom)
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```

iii. The session-wise max normalization matches the reference code's approach (used in `fit_decoder`/`get_rate_maps`). The CONVERSION_NOTES (Step 5, Key Decision 6) justify this as preserving the same coordinate handling logic while adapting from 15x15 to 3x3 bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D position is discretized into 3x3 = 9 classes. Each axis is divided into 3 bins based on `floor(position / (session_max + buffer) * 3)`, then clipped to [0, 2]. The combined class label is `x_bin * 3 + y_bin`.

ii.
```python
denom = (session_max_xy + BUFFER) / POSITION_BINS
binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
binned = np.clip(binned, 0, POSITION_BINS - 1)
classes = binned[0] * POSITION_BINS + binned[1]
```

iii. The floor-based binning with session-wise max normalization follows the reference code's logic (integer flooring of normalized position). The 3x3 grid satisfies the instruction requirement for 9 spatial bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are temporally binned using the same 3-frame mean and split into trials using the same trial slices, ensuring frame-for-frame alignment.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. Both streams are acquired at 30 Hz and stored aligned in the source files. The same trial slicing and temporal binning is applied to both, preserving alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered neurons (NaN at first timepoint) are excluded from each session. For recordings shorter than 40 minutes, the final trial is kept but shorter. Frames beyond 40 minutes are discarded. The temporal binning trims any remainder frames that don't fill a complete 3-frame bin. The geometry transpose bug was identified and fixed during development.

ii.
```python
valid_cells = get_valid_cell_mask(trace_day)  # exclude NaN neurons
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)  # cap at 40 min
# temporal_bin_mean trims remainder frames
usable = (arr.shape[-1] // bin_size) * bin_size
```

iii. The CONVERSION_NOTES document that NaN filtering handles the main data quality issue (unregistered cells). The geometry orientation bug was caught by sanity checks comparing blocked-bin occupancy rates.

## 6-a. What are the most time-consuming steps of the code?

i. Loading and decompressing the joblib files is the dominant cost, as noted in the CONVERSION_NOTES. The full conversion took ~467 seconds, with file I/O dominating. Per-session processing is fast (~0.1-0.4 seconds).

ii. N/A (timing is from runtime output)

iii. The conversion_full_out.txt shows per-animal loading times (e.g., QLAK-CA1-08: 70.88s, QLAK-CA1-50: 123.98s), confirming I/O dominance.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop in `convert_session` iterates over trial slices applying `temporal_bin_mean` and `discretize_position_3x3` individually. This could theoretically be vectorized by processing all trials at once as a 3D array, though the savings would be modest given the already-fast per-trial processing.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    ...
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy)
```

iii. The AI's CONVERSION_NOTES note that per-session processing is fast (< 0.4s) and I/O dominates, so further vectorization would provide minimal speedup.

## 6-c. What processing does the code repeat multiple times?

i. The `temporal_bin_mean` function is called separately for neural and position data in each trial. These could be combined into a single operation if the arrays were stacked, though they have different dimensions (n_neurons vs 2).

ii. N/A (see 6-b code)

iii. The repetition is minor and does not meaningfully impact performance.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects extensive per-session metadata (environment labels, original frame counts, usable frame counts, trial counts, neuron counts) that is stored but not used by the downstream decoder. The `--show-processing` visualization code is included but only activated optionally.

ii.
```python
converted["metadata"]["session_ids"].append(session.session_id)
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
converted["metadata"]["session_usable_frames"].append(debug_info["usable_frames"])
converted["metadata"]["session_trial_counts"].append(len(neural_trials))
converted["metadata"]["session_valid_neuron_counts"].append(int(brain_region_idx.shape[0]))
```

iii. The extra metadata was included for debugging and documentation. While not used by the decoder, it helps verify the conversion's correctness.
