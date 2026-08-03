# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-existing joblib files in `/app/data/`, one per animal. Each file is a dictionary keyed by animal name, containing fields: `trace` (neural activity), `position` (x-y tracking), `envs` (environment labels), `blocked` (blocked partition indices), `maps`, `centroids`, and `SFPs`. The AI iterates over a hardcoded list of 7 animal names, loads each joblib file with `joblib.load()`, and extracts the inner dictionary. All days (sessions) for each animal are stored in 3D arrays (days x neurons x timepoints for trace; days x 2 x timepoints for position).

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

# In convert_dataset:
for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    envs = flatten_envs(dat["envs"])
    trace = np.asarray(dat["trace"], dtype=np.float32)
    position = np.asarray(dat["position"], dtype=np.float32)
```

iii. The AI verified this loading approach by checking the data structure in the reference code's `load_dat` function and the Zenodo dataset README. The 7 animals match the reference code's `animals` list. The AI confirmed `trace` has shape `(n_days, n_cells, T)` and `position` has shape `(n_days, 2, T)` by examining the actual data files.

## 1-b. How are the data split into subjects (mice)?

i. Each of the 7 joblib files corresponds to one subject (mouse). The AI assigns a sequential `subject_id` (0-6) to each animal in the order they appear in the `ANIMALS` list. The `subjects` list stores the animal name strings, and `subject_idx` maps each session to its subject.

ii.
```python
for subject_id, animal in enumerate(ANIMALS):
    # ... process all days for this animal ...
    subject_idx.append(subject_id)

data = {
    "subjects": list(ANIMALS),
    "subject_idx": np.array(subject_idx, dtype=np.int64),
}
```

iii. The AI followed the reference code's `animals` list which defines the same 7 animals. The data files are organized one per animal, making subject splitting straightforward.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal is treated as one session. The AI iterates over the first axis of the `trace` array (days) for each animal. This produces 207 total sessions across all 7 animals (31 sessions each for 6 animals, 21 for QLAK-CA1-51).

ii.
```python
for day in range(trace.shape[0]):
    # ... process one day as one session ...
    stats.total_sessions += 1
```

iii. The AI confirmed this matches the reference paper's reported 207 sessions and the reference code which iterates over days as sessions. From CONVERSION_NOTES.md: "I treated each original recording day as one decoder session. This matches the paper code, which iterates over day/session axes."

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into contiguous non-overlapping 60-second windows. At 30 Hz, each trial is 1800 frames. Sessions produce either 39 or 40 complete 1-minute trials depending on total recording length. Any trailing incomplete remainder is discarded.

ii.
```python
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800

n_trials = T // TRIAL_FRAMES
dropped = T - n_trials * TRIAL_FRAMES

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. The AI justified this as a decoder-specific adaptation since the source data are continuous sessions, not pre-segmented trials. The instructions state "1-minute trials within each session." The AI verified sessions are ~40 minutes long at 30 Hz (~72000 frames), yielding 39-40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial-level quality filtering. All complete 60-second windows from all sessions are retained. The only "filtering" is discarding the trailing incomplete minute at the end of each session. The AI validates that each session produces at least 2 trials (required by the decoder format).

ii.
```python
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
```

iii. From CONVERSION_NOTES.md: no additional quality filtering is mentioned beyond discarding trailing frames. The reference paper does not describe trial-level quality filtering either (since it doesn't use trials in the same way).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in the joblib files. This contains binary rising-phase calcium event vectors (0s and 1s), already preprocessed by the paper's pipeline.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
# ...
day_trace = trace[day]
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. From CONVERSION_NOTES.md: "I used the provided binary rise-event traces directly from `data/<animal>`. I did not re-deconvolve, smooth, or re-threshold the calcium traces." The methods.txt confirms: "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5, and 0 otherwise. This binary vector was treated as the firing rate in all further analyses."

## 2-b. How is the `neural` data processed?

i. The AI applies minimal processing: it drops neurons that are unregistered on a given day (those with all-NaN values in the first timepoint), and casts the remaining traces to float32. No smoothing, temporal binning, velocity filtering, or activity thresholding is applied.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. From CONVERSION_NOTES.md: "For each session, I kept only neurons registered on that day. In the source files, unregistered neurons are all-NaN for that day; these were dropped from that session. No additional place-cell or activity-threshold filtering was applied."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural filtering is removing unregistered neurons (those with NaN traces on a given day). The reference code's `decode_position_within` applies two additional filters that the AI does NOT apply: (1) velocity filtering (excluding time points where mouse velocity < 5 cm/s), and (2) cell activity threshold (excluding neurons with fewer than 5 spikes in moving periods). Neither of these is applied by the AI.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The AI chose to keep all registered neurons and all time points, leaving filtering decisions to the downstream decoder. The reference code applies velocity and activity filtering within its own decoder pipeline, but the AI did not replicate this in the data conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial starts at the beginning of a contiguous 60-second window within the recording session. The first trial starts at frame 0 of the session, the second at frame 1800, etc. Neural, position, and output data are sliced using the same frame indices, so they are inherently aligned.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The metadata describes the alignment: `"temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session"`, `"off_start": 0.0`, `"off_end": 60.0`. Since the original data streams (trace and position) are already temporally aligned at 30 Hz, slicing at the same frame indices preserves alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native recording frame rate: 30 Hz, giving a time bin size of ~33.33 ms. No temporal rebinning is applied. The reference code's decoder uses `temporal_bin_size=3` (averaging 3 frames for ~10 Hz effective rate), but the AI does not replicate this.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS  # ~33.33 ms
```

iii. The AI's metadata records `"time_bin_size": 33.33...` ms and `"fps": 30`. The reference code's `fit_decoder` applies `AvgPool1d(kernel_size=3, stride=3)` for temporal binning, but the AI keeps the native resolution, presumably leaving rebinning to the downstream decoder.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field in the raw data, which specifies which of the 9 partitions (in a 3x3 grid) are blocked for each recording day. The AI does NOT use the `envs` field (environment name strings) to determine the mask.

ii.
```python
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_mask = mask_from_blocked(blocked_entry)
input_vec = env_mask.reshape(-1).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "I derived this from the `blocked` metadata, not only from the environment name. This matters because the `env` string alone does not fully specify orientation in this dataset, while `blocked` does." The `blocked` field is documented in the README as: "location of blocked (occluded) partitions in 3x3 design of environment."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The `blocked` field (a MATLAB-style nested list) is normalized to a tuple of blocked bin indices. A 3x3 binary mask is created where 1=open and 0=blocked. The mask is flattened to a 9-element vector serving as the per-trial static input. The mapping uses `idx // 3` for row and `idx % 3` for column.

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

def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask
```

iii. The AI handled edge cases: the `blocked` field may be `-1` (no blocked partitions, i.e., square environment), empty, or a nested list. The `normalize_blocked_entry` function robustly handles all these cases. The `ENV_TO_MASK` dictionary defined at the top of the code is never actually used; it appears to be dead code.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position output is derived from the `position` field in the raw data, which contains x-y coordinates from DeepLabCut head tracking at 30 Hz.

ii.
```python
position = np.asarray(dat["position"], dtype=np.float32)
day_pos = position[day]
binned_xy = bin_position_to_grid(day_pos, n_bins=3)
```

iii. The methods.txt states: "Position data were generated from tracking the head with DeepLabCut pose-estimation software." The position data has shape `(n_days, 2, T)` with x and y coordinates.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Raw x-y positions are binned into a 3x3 grid using max-based normalization (no minimum subtraction). The formula is `floor(position / ((per-axis-max + eps) / 3))`, clipped to [0, 2]. This matches the reference code's `get_rate_maps` binning logic (though the reference uses 15x15 bins).

ii.
```python
def bin_position_to_grid(position_xy: np.ndarray, n_bins: int = 3) -> np.ndarray:
    max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
    bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)
```

iii. From CONVERSION_NOTES.md: "I matched the paper code's binning style: no min subtraction, divide raw positions by per-axis maxima, then floor." The reference code's `get_rate_maps` uses: `position_binned = (position // ((np.nanmax(position, axis=0) + buffer) / n_bins)).astype(int)`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D binned position (x_bin, y_bin) is converted to a single categorical label: `position_bin = x_bin * 3 + y_bin`, yielding values 0-8 (9 categories). Positions landing in blocked bins are reassigned to the nearest valid (open) bin using Euclidean distance in grid space.

ii.
```python
lookup = build_nearest_valid_lookup(env_mask)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. The blocked-bin reassignment is described in CONVERSION_NOTES.md as: "in the spirit of the paper's within-session decoder code, which also snaps decoded/actual positions to valid bins for geometry-aware error measurement." In the reference code's `decode_position_within`, actual and predicted positions are cleaned to nearest valid bins using `temp_maps`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame indices (both recorded at 30 Hz and stored with the same time axis). The AI slices both using the same trial boundaries (`start:stop`), so they remain aligned frame-by-frame.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The methods.txt states: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz." The position and trace data have the same number of frames per day, confirmed by the AI's validation check: `if trace.shape[2] != position.shape[2]: raise ValueError(...)`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of data issues: (1) Unregistered neurons (all-NaN for a day) are dropped from that session's neural array. (2) Positions falling in blocked bins are reassigned to the nearest valid bin. (3) Trailing frames that don't complete a full 60-second trial are discarded. (4) The `blocked` field's MATLAB-style nesting is normalized robustly (handles nested lists, NaN values, -1 for "no blocks").

ii.
```python
# NaN neuron handling
registered_today = ~np.isnan(day_trace[:, 0])
session_neural = day_trace[registered_today].astype(np.float32, copy=True)

# Position NaN check
if np.isnan(day_pos).any():
    raise ValueError(f"{animal} day {day}: position contains NaNs")

# Blocked bin normalization
arr = arr[~np.isnan(arr)]
if arr.size == 1 and arr[0] == -1:
    return ()
```

iii. The AI validates data integrity with multiple assertions: no NaNs in registered neurons, no NaNs in position data, and consistent shapes between trace and position. The blocked-bin reassignment handled 2,555,747 total frames (~17% of all frames), which the AI notes is expected given the coarse 3x3 binning.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading the large joblib files for each animal (each file contains 3D arrays of shape ~(31, ~900, ~72000) for trace). (2) The per-day processing loop, particularly the position binning and nearest-valid-bin lookup for all ~72000 frames per day across 207 sessions.

ii.
```python
dat = load_animal(data_dir, animal)  # Loading large joblib files
# Per-day loop over 207 sessions:
for day in range(trace.shape[0]):
    binned_xy = bin_position_to_grid(day_pos, n_bins=3)
    lookup = build_nearest_valid_lookup(env_mask)
    projected_xy = lookup[binned_xy[0], binned_xy[1]].T
```

iii. The trajectory shows the full conversion runs took significant wall time (the agent ran it multiple times). The data loading dominates since each animal's data is hundreds of MB.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_nearest_valid_lookup` function uses nested Python loops over the 3x3 grid, but this is only 9 iterations so the impact is negligible. The per-trial slicing loop could potentially be vectorized with `np.split` or array reshaping, but again the overhead is small. The `mask_from_blocked` function loops over blocked bins.

ii.
```python
def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    for x in range(3):
        for y in range(3):
            # ...

def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
```

iii. These loops iterate over at most 9 elements, so vectorization would provide negligible speedup. The main computational cost is in array operations (slicing, copying) which are already vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The code loads each animal's full dataset once and processes all days in a single pass, so there is no redundant processing within a single run. However, during development, the agent ran the full conversion script at least 3 times (steps 50, 57, 83 in the trajectory). The `build_nearest_valid_lookup` is called once per session even though many sessions share the same environment geometry -- this could have been cached.

ii.
```python
for day in range(trace.shape[0]):
    blocked_entry = normalize_blocked_entry(dat["blocked"][day])
    env_mask = mask_from_blocked(blocked_entry)
    lookup = build_nearest_valid_lookup(env_mask)  # Rebuilt for each day even if same geometry
```

iii. The lookup table reconstruction is redundant when consecutive days have the same geometry, but since it operates on a 3x3 grid, the cost is negligible.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `ENV_TO_MASK` dictionary (lines 35-46) is defined but never used -- the code uses `mask_from_blocked` instead. (2) The code computes and stores extensive metadata statistics (per-animal neuron counts, environment counts, blocked patterns, frame lengths, etc.) which are useful for sanity checking but not consumed by the decoder. (3) The code tracks `total_frames_reassigned` and other detailed statistics that are only used for logging.

ii.
```python
ENV_TO_MASK = {  # Defined but never used in processing
    "square": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
    # ...
}

# Extensive metadata that isn't used by decoder:
"per_animal_unique_neurons": per_animal_unique_neurons,
"session_frame_lengths": {...},
"total_frames_reassigned_to_valid_bins": stats.total_frames_reassigned,
```

iii. The unused `ENV_TO_MASK` appears to be leftover from development. The extensive metadata is useful for documentation and sanity checking but adds code complexity and storage overhead without affecting decoder performance.
