# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (one per subject) in the `data/` directory using `joblib.load()`. Each joblib file contains a dictionary keyed by animal ID with fields `trace`, `position`, `blocked`, `envs`, etc. The animal list is hardcoded as the `ANIMALS` constant.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]

def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]

for animal in ANIMALS:
    dat = load_animal(animal)
```

iii. The CONVERSION_NOTES explains: "The joblib files are the repository's own converted form of the original MATLAB data, loaded by the reference helper `load_dat(..., format='joblib')`. They preserve the paper fields directly: `envs`, `blocked`, `position`, `trace`, `maps`, `SFPs`, and `centroids`."

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The subject list is hardcoded in the `ANIMALS` constant, and a lookup dictionary maps animal names to indices.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}
```

iii. The subject ordering follows the repository's animal IDs as documented in CONVERSION_NOTES.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject becomes a separate session. The AI iterates over the `envs` list for each animal, where each entry corresponds to one recording day.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]
for day_idx, env in enumerate(envs):
    position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
    trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The CONVERSION_NOTES states: "Sessions are recording days. This yields 207 sessions total, matching the paper and repository summaries."

## 1-d. How are the data split into trials?

i. Each session is split into exactly 40 consecutive 1-minute trials (1800 frames at 30 Hz). Sessions longer than 40 minutes (72000 frames) are truncated. Sessions shorter than 72000 frames have a shorter final trial. This differs from the reference approach which creates as many full 1800-frame trials as possible and drops any remainder.

ii.
```python
NOMINAL_SESSION_FRAMES = int(FPS * NOMINAL_SESSION_SECONDS)  # 72000
N_TRIALS_PER_SESSION = 40

keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
position = position[:keep_frames]
trace = trace[:, :keep_frames]

def build_trial_slices(n_timepoints: int) -> list[slice]:
    trial_slices: list[slice] = []
    for trial_idx in range(N_TRIALS_PER_SESSION):
        start = trial_idx * TRIAL_FRAMES
        end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
        trial_slices.append(slice(start, end))
    return trial_slices
```

iii. The CONVERSION_NOTES documents: "93 sessions have a short final trial. Minimum trial length: 1666 frames. Maximum trial length: 1800 frames. Total dropped trailing frames from recordings longer than 40 nominal minutes: 11,481."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All trials (including short final trials) are kept.

ii. N/A - no filtering code.

iii. No justification provided for not filtering trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib files, which contains calcium event traces (rise-extracted binary calcium events) per the paper's processing.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The CONVERSION_NOTES states: "Neural activity uses the paper's rise-extracted binary calcium event traces, not deconvolved or re-smoothed traces."

## 2-b. How is the `neural` data processed?

i. The trace is cast to float32. Neurons not registered on a given day (rows with any NaN) are removed. The code also validates that partial-NaN neurons don't exist (a neuron is either all-NaN or no-NaN). No additional processing (e.g., smoothing, normalization) is applied.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
```

iii. The CONVERSION_NOTES explains: "Unregistered neurons are removed session-by-session by dropping rows that are all NaN on that day."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are not registered on a given recording day (detected by any-NaN in their row) are removed. The code asserts that the any-NaN and all-NaN filters produce identical results.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
```

iii. Same as 2-b.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The recording is continuous, and trials are consecutive 1-minute segments starting from the beginning of each session.

ii.
```python
trial_slices = build_trial_slices(keep_frames)
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The metadata documents: `"temporal_alignment_event": "Start of each consecutive 1-minute segment within a recording day"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
```

iii. The metadata sets `'time_bin_size': TIME_BIN_MS` (~33.33 ms).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the joblib files, which contains blocked position indices for each session. The AI also cross-references against `envs` (environment labels) for validation.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. The CONVERSION_NOTES states: "Decoder input is a 9D static blocked-partition mask per trial."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary mask. If blocked is `[-1]`, the mask is all zeros. Otherwise, positions at the blocked indices are set to 1. The AI also implements `env_to_blocked_mask()` which derives the expected mask from environment labels for cross-validation (using `np.flipud` to convert from the display convention to bottom-up row-major indexing).

ii.
```python
def parse_blocked_mask(value: Any) -> np.ndarray:
    flat = np.array(flatten_numeric(value), dtype=float)
    if flat.size == 1 and np.isclose(flat[0], -1.0):
        return np.zeros(9, dtype=np.float32)
    blocked = np.unique(flat.astype(int))
    mask = np.zeros(9, dtype=np.float32)
    mask[blocked] = 1.0
    return mask
```

iii. The CONVERSION_NOTES explains the indexing: "The mask uses the dataset's bottom-up row-major partition numbering, which is the same convention encoded by `blocked`."

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The blocked mask is a static per-trial input (not time-varying). The same mask is replicated for all trials within a session.

ii.
```python
input_trial = blocked_mask.astype(np.float32).copy()
session_input.append(input_trial)
```

iii. The CONVERSION_NOTES confirms blocked positions are constant within a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in the joblib files, which contains 2D (x, y) coordinates of the animal in the arena at each timepoint.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T  # (n_timepoints, 2)
```

iii. The CONVERSION_NOTES documents that position data records the animal's location in a 75x75 cm open field arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI calibrates each animal's arena frame by finding the minimum position and maximum extent from square-session data. Position is then shifted by the arena minimum and divided by bin size to get 3x3 grid bin IDs (`y_bin * 3 + x_bin`). After binning, any frames that land in blocked partitions are "snapped" to the nearest open partition center.

ii.
```python
def calibrate_animal_arena(dat):
    square_days = [idx for idx, env in enumerate(envs) if env == "square"]
    square_positions = [np.asarray(dat["position"][day], dtype=np.float64).T for day in square_days]
    all_square = np.concatenate(square_positions, axis=0)
    arena_min = np.nanmin(all_square, axis=0)
    shifted = all_square - arena_min[None, :]
    arena_side = float(np.nanmax(shifted))
    return arena_min, arena_side / N_POSITION_BINS, arena_side

def position_to_bins(position_xy, arena_min, bin_size):
    shifted = position_xy - arena_min[None, :]
    coords = np.floor(shifted / bin_size).astype(np.int64)
    coords = np.clip(coords, 0, N_POSITION_BINS - 1)
    bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
    return coords, bin_ids
```

iii. The CONVERSION_NOTES explains: "I calibrated each animal's absolute 3x3 arena frame from that animal's square sessions... resulting side length: exactly 75.0 for all seven animals, resulting bin size: 25.0."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal bins of 25 cm each. The bin ID is `y_bin * 3 + x_bin` (bottom-up row-major). Additionally, positions that land in blocked bins are snapped to the nearest open bin center.

ii.
```python
coords = np.floor(shifted / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]

# Then clean blocked bins:
def clean_blocked_bins(position_xy, bin_ids, blocked_mask, arena_min, bin_size):
    blocked_ids = np.flatnonzero(blocked_mask > 0.5)
    bad = np.isin(bin_ids, blocked_ids)
    cleaned = bin_ids.copy()
    open_ids = np.flatnonzero(blocked_mask < 0.5)
    for idx in np.flatnonzero(bad):
        d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
        cleaned[idx] = int(open_ids[int(np.argmin(d2))])
    return cleaned, int(np.sum(bad))
```

iii. The CONVERSION_NOTES documents: "After absolute 3x3 binning, any frame that landed in a blocked partition was snapped to the nearest open partition center... Total corrected frames: 448 across the full dataset."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame rate and frame indices. Both are sliced using the same trial slices, ensuring frame-by-frame alignment.

ii.
```python
trial_slices = build_trial_slices(keep_frames)
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. Both arrays come from the same recording and are sliced identically.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e. The data is at 30 Hz (~33.33 ms per time bin). No rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS  # ~33.33 ms
```

iii. Native frame rate preserved.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data are aligned frame-by-frame since they share the same sampling rate and are sliced with the same trial indices. Input (blocked mask) is a static per-trial vector, not time-varying.

ii.
```python
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
    input_trial = blocked_mask.astype(np.float32).copy()
```

iii. The data streams originate from the same recording at 30 Hz, ensuring natural alignment.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Several strategies: (1) Unregistered neurons (NaN rows) are removed per session. (2) Sessions longer than 40 minutes are truncated. (3) Sessions shorter than 40 minutes produce a short final trial. (4) Positions that land in blocked bins are snapped to the nearest open bin (448 total frames affected). (5) Blocked indices are cross-validated against environment labels. (6) Partially NaN neurons trigger an error.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
```

iii. The CONVERSION_NOTES documents all these edge cases with quantified statistics.

## 7-a. What are the most time-consuming steps of the code?

i. Loading the joblib files and calibrating the arena from square sessions are likely the most time-consuming steps. The arena calibration concatenates all square-session positions for each animal.

ii.
```python
dat = load_animal(animal)  # joblib.load
arena_min, bin_size, arena_side = calibrate_animal_arena(dat)
```

iii. I/O-bound operations dominate. The per-animal arena calibration requires loading and concatenating multiple sessions' position data.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The `clean_blocked_bins` function loops over each bad frame individually to compute distances to open bin centers and snap it. This could be vectorized with matrix operations.

ii.
```python
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. Only 448 frames are affected, so the performance impact is minimal, but the loop could be vectorized.

## 7-c. What processing does the code repeat multiple times?

i. The `parse_blocked_mask` function is called once per session to parse blocked indices, and `env_to_blocked_mask` is also called per session to generate an expected mask from the environment label for cross-validation. This results in two independent computations of blocked masks per session. Additionally, `calibrate_animal_arena` re-loads and processes square session positions, which are also processed again in the main loop.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
```

iii. The duplication serves a validation purpose (cross-checking two independent sources).

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Arena calibration from square sessions is unnecessary since all animals yield exactly 75.0 cm - a fixed constant would suffice. (2) The blocked bin cleaning (snapping 448 frames) is extra processing not present in the reference. (3) Environment label cross-validation against blocked indices is extra validation. (4) Detailed statistics collection (`stats` dict) is not used in the output data.

ii.
```python
arena_min, bin_size, arena_side = calibrate_animal_arena(dat)  # yields 75.0 for all
stats["blocked_env_mismatches"].append(...)  # validation stats
```

iii. These add robustness but are not strictly necessary for the conversion.

## 8. How are minor mistakes in the data (e.g., missing data, malformed entries) handled?

i. Same as question 6. Unregistered neurons are removed (NaN filtering). Sessions are truncated at 40 minutes. Short final trials are allowed. Positions in blocked bins are snapped to nearest open bin. Blocked indices are cross-validated against environment labels. Partially NaN neurons cause an error.

ii. See question 6 code snippets.

iii. The CONVERSION_NOTES thoroughly documents all edge cases and their handling with quantified statistics.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Loading joblib files (I/O bound) and arena calibration from square sessions.

ii. See 7-a code snippets.

iii. I/O dominates compute time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The per-frame loop in `clean_blocked_bins` for snapping blocked positions.

ii. See 7-b code snippets.

iii. Minimal performance impact due to small number of affected frames (448).

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. Blocked mask computation is done twice per session (from `blocked` indices and from environment labels). Square session positions are processed during calibration and again in the main loop.

ii. See 7-c code snippets.

iii. Duplication serves validation purpose.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Arena calibration (could use fixed 75.0), blocked bin cleaning (not in reference), environment cross-validation, and statistics collection.

ii. See 7-d code snippets.

iii. Extra robustness at the cost of complexity.
