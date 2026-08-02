# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a joblib file in the data directory (e.g., `QLAK-CA1-08`). The AI uses `joblib.load()` to load each file, which contains a nested dictionary keyed by animal ID. The dictionary contains arrays for `trace`, `position`, `blocked`, and `envs`. This matches the paper code's `load_dat` function which supports both MATLAB and joblib formats. The data directory also contains `.mat` files, but the AI chose to use the pre-converted joblib files.

ii.
```python
def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

# In convert_dataset:
for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    envs = flatten_envs(dat["envs"])
    trace = np.asarray(dat["trace"], dtype=np.float32)
    position = np.asarray(dat["position"], dtype=np.float32)
```

iii. The AI identified that the data was available in both MATLAB (.mat) and joblib formats, and chose joblib as it matches the paper code's `load_dat` function with `format="joblib"`. The trajectory shows the AI explicitly inspected the data files and verified the structure before writing the converter.

## 1-b. How are the data split into subjects?

i. The AI hardcodes a list of 7 animal IDs (`ANIMALS`) and iterates over them. Each animal's data is loaded from a separate joblib file.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
```

iii. The AI identified the 7 subjects from the data directory and hardcoded them for reproducibility. The CONVERSION_NOTES confirms these are all 7 animals in the dataset.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal's data becomes a separate session. The AI iterates over the first axis of the `trace` array (shape `(n_days, n_cells, T)`), where each entry along that axis is one recording day/session.

ii.
```python
for day in range(trace.shape[0]):
    # ...process each day as a separate session...
    neural.append(session_trials_neural)
    decoder_input.append(session_trials_input)
    decoder_output.append(session_trials_output)
    subject_idx.append(subject_id)
```

iii. The AI's CONVERSION_NOTES states: "I treated each original recording day as one decoder session. This matches the paper code, which iterates over day/session axes in trace, position, and envs."

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into contiguous, non-overlapping 60-second windows. At 30 Hz, each trial is 1800 frames. Any trailing remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800
n_trials = T // TRIAL_FRAMES
# ...
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. The CONVERSION_NOTES explains: "The source recordings are continuous long sessions rather than pre-segmented trials. I split each session into contiguous full 60 s windows. Sessions produced either 39 or 40 complete one-minute trials depending on raw recording length."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. The AI only requires that each session has at least 2 full 1-minute trials (enforced with a ValueError check). Trailing incomplete trial segments are discarded.

ii.
```python
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
```

iii. No explicit justification for lack of trial filtering beyond the minimum trial count check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib data file. This contains binary rising-phase event data (binarized calcium transients) with shape `(n_days, n_cells, T)`.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
# ...
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The CONVERSION_NOTES states: "I used the provided binary rise-event traces directly from data/<animal>. I did not re-deconvolve, smooth, or re-threshold the calcium traces. This matches the paper/methods description: all analyses use the binarized rising phase of calcium transients."

## 2-b. How is the `neural` data processed?

i. The only processing is: (1) filtering out neurons not registered on that day (all-NaN columns), and (2) casting to float32. The binary rise-event traces are used as-is without additional processing.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The AI states in CONVERSION_NOTES: "I did not re-deconvolve, smooth, or re-threshold the calcium traces." The trajectory shows the AI verified the data is already in binary event format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by registration status: only neurons registered (not all-NaN) for a given day/session are included. No additional place-cell filtering or activity-threshold filtering is applied.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The CONVERSION_NOTES states: "No additional place-cell or activity-threshold filtering was applied to the exported neural arrays. This keeps the exported dataset faithful to the recorded session content while avoiding NaNs."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of each session. The alignment event is "trial start of each contiguous 60 second window."

ii.
```python
# metadata:
"temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

iii. There is no stimulus onset or behavioral event to align to in this free-exploration paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS  # ~33.33 ms
# metadata:
"time_bin_size": TIME_BIN_MS,
```

iii. The native frame rate matches the paper's recording rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the data files, which specifies which of the 9 positions in the 3x3 grid are blocked for each session. The AI converts this to a 3x3 open-bin mask.

ii.
```python
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_mask = mask_from_blocked(blocked_entry)
# ...
input_vec = env_mask.reshape(-1).astype(np.float32)
```

iii. The CONVERSION_NOTES explains the AI discovered that "env names alone are not enough to recover geometry orientation. The nested blocked field carries the authoritative per-session 3x3 occupancy pattern." The trajectory confirms this was a key insight during conversion.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are normalized from nested MATLAB-style lists, then converted to a 3x3 binary mask where 1=open and 0=blocked. This mask is flattened to a 9-element vector. The encoding is inverted compared to the reference: the AI uses 1=open/0=blocked, while the reference uses 1=blocked/0=open.

ii.
```python
def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask

# flattened for input:
input_vec = env_mask.reshape(-1).astype(np.float32)
```

iii. The CONVERSION_NOTES states: "Input is a static 9D vector per trial: a flattened 3x3 open-bin mask. 1 means the arena partition is open, 0 means blocked."

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per session (constant across all trials and timepoints within a session). Each trial receives a copy of the same 9D input vector.

ii.
```python
session_trials_input.append(input_vec.copy())
```

iii. Blocked positions don't change within a session, so the input is a per-session constant.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable in the data files, which contains 2D (x, y) coordinates of the animal at each timepoint, with shape `(n_days, 2, T)`.

ii.
```python
position = np.asarray(dat["position"], dtype=np.float32)
day_pos = position[day]
```

iii. Position data was recorded from behavioral video tracking via DeepLabCut at 30 Hz, matching the neural recording frame rate.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid using per-axis max normalization: `floor(position / ((per-axis max + eps) / 3))`. This matches the paper code's `get_rate_maps` binning logic. After binning, positions landing in blocked grid cells are reassigned to the nearest valid open bin.

ii.
```python
def bin_position_to_grid(position_xy: np.ndarray, n_bins: int = 3) -> np.ndarray:
    max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
    bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)

# Then reassignment to valid bins:
lookup = build_nearest_valid_lookup(env_mask)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. The CONVERSION_NOTES states: "I matched the paper code's binning style: no min subtraction, divide raw positions by per-axis maxima, then floor." The nearest-valid-bin cleanup is described as "in the spirit of the paper's within-session decoder code."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The position is discretized into 9 categories (0-8) using a 3x3 grid. The label is computed as `x_bin * 3 + y_bin`. After initial binning, positions in blocked cells are reassigned to the nearest valid open bin.

ii.
```python
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. The AI chose `x_bin * 3 + y_bin` ordering. The reference uses `y_bin * 3 + x_bin`. The AI also applies nearest-valid-bin reassignment which the reference does not.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same 30 Hz frame rate. Both are sliced into trials using the same frame indices, ensuring frame-by-frame alignment.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. Both arrays have the same number of timepoints and are sliced identically.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~33.33 ms (1/30 Hz). No temporal rebinning is applied; data is kept at native frame rate.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
```

iii. Same as 2-e.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural (trace) and output (position) are recorded at the same 30 Hz rate and share the same time axis. They are split into trials using identical frame indices. Input (geometry) is static per session and has no temporal dimension.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
    session_trials_input.append(input_vec.copy())
```

iii. The DAQ simultaneously acquired behavioral and cellular imaging at 30 Hz, so they are inherently aligned.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. (1) Neurons not registered on a given day (all-NaN) are excluded from that session. (2) The `blocked` field has nested MATLAB-style formatting; the AI wrote a `normalize_blocked_entry` function to robustly parse it. (3) Trailing frames that don't fill a complete 60s trial are discarded. (4) Positions in blocked grid cells are reassigned to nearest valid bins.

ii.
```python
def normalize_blocked_entry(entry) -> tuple[int, ...]:
    while isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    arr = np.asarray(entry, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return ()
    arr = arr.astype(int)
    if arr.size == 1 and arr[0] == -1:
        return ()
    return tuple(sorted(arr.tolist()))
```

iii. The trajectory shows the AI discovered the `blocked` field needed careful normalization. Validation checks confirmed NaN handling was correct.

## 7-a. What are the most time-consuming steps of the code?

i. Loading the large joblib files for each animal is the most I/O intensive step. Each file contains dense arrays (e.g., trace shape `(31, 515, 71866)` for one animal).

ii. N/A

iii. The trajectory shows the AI noted that the full conversion took significant time due to file sizes.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_nearest_valid_lookup` function uses nested loops over the 3x3 grid to compute nearest valid bin lookups. However, this is only a 9-iteration loop so vectorization would have negligible impact. The `mask_from_blocked` function also loops over blocked indices.

ii.
```python
def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    valid = np.argwhere(env_mask > 0)
    lookup = np.zeros((3, 3, 2), dtype=np.int64)
    for x in range(3):
        for y in range(3):
            # ...
```

iii. These loops are over at most 9 elements, so the impact is negligible.

## 7-c. What processing does the code repeat multiple times?

i. The `normalize_blocked_entry` function is called once per session (207 times), and `mask_from_blocked` and `build_nearest_valid_lookup` are called once per session. None of these are cached across sessions that share the same blocked pattern. However, the overhead is minimal.

ii. N/A

iii. No significant repeated processing was identified.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes extensive metadata and sanity statistics (environment counts, blocked patterns, frame lengths, reassignment counts, etc.) that are stored in the metadata dict but not used by the downstream decoder. The `ConversionStats` dataclass tracks multiple counters. The `summarize_dataset` function is called on both full and sample data.

ii.
```python
stats = ConversionStats()
per_animal_unique_neurons = {}
per_subject_sessions = Counter()
env_counts = Counter()
# ... extensive metadata collection
```

iii. This is for validation/documentation purposes rather than downstream use.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. Key handling: (1) NaN neurons filtered per session, (2) nested blocked entries normalized, (3) trailing frames discarded, (4) blocked-cell positions reassigned. The code also includes validation checks (raising ValueError) for unexpected NaN patterns.

ii. See question 6.

iii. See question 6.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Loading large joblib files is the bottleneck.

ii. N/A

iii. N/A

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b.

ii. N/A

iii. N/A

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c.

ii. N/A

iii. N/A

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d.

ii. N/A

iii. N/A
