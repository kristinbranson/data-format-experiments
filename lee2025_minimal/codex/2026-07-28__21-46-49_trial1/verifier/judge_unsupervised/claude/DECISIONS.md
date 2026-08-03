# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the animal-level joblib files in `/app/data/QLAK-CA1-*` (without the `.mat` extension). These are the repository's pre-converted form of the original MATLAB data. Each file is loaded via `joblib.load()` and contains fields: `envs`, `blocked`, `position`, `trace`, `maps`, `SFPs`, and `centroids`. The AI iterates over a hardcoded list of 7 animal IDs (`ANIMALS`), loading each animal's data and then iterating over recording days within each animal.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]

def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]

# In build_dataset():
for animal in ANIMALS:
    dat = load_animal(animal)
    envs = [parse_env_label(v) for v in dat["envs"]]
    # ... iterate over days
    for day_idx, env in enumerate(envs):
        position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
        trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The AI confirmed that the joblib files match the reference code's `load_dat(..., format="joblib")` function in `utils.py`. It verified the files contain the expected fields and that the repository already ships these pre-converted files alongside the original `.mat` files.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats each of the 7 animal IDs as a separate subject. A `subject_lookup` dictionary maps animal names to integer indices (0-6). Each session's subject is tracked via `subject_idx`.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}
# ...
data["subject_idx"].append(subject_lookup[animal])
```

iii. The AI followed the repository's animal IDs directly. The 7 subjects match the paper's description. No subjects are excluded.

## 1-c. How are the data split into sessions?

i. Each recording day for each animal is treated as one session. The AI iterates over the `envs` list for each animal, where each entry corresponds to one recording day. This yields 207 sessions total (31 days x 6 animals + 21 days x 1 animal).

ii.
```python
for animal in ANIMALS:
    dat = load_animal(animal)
    envs = [parse_env_label(v) for v in dat["envs"]]
    for day_idx, env in enumerate(envs):
        # Each day_idx becomes one session
        # ...
        data["neural"].append(session_neural)
```

iii. The AI verified this yields 207 sessions, matching the paper and methods text. Sessions per animal: 6 animals with 31 sessions, 1 animal (QLAK-CA1-51) with 21 sessions.

## 1-d. How are the data split into trials?

i. Each session (~40 min recording at 30 Hz) is split into 40 consecutive 1-minute trials of 1800 frames each. Sessions shorter than 72000 frames keep a shorter final trial. Sessions longer than 72000 frames are truncated at the 40-minute boundary.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800
NOMINAL_SESSION_SECONDS = 40.0 * 60.0
NOMINAL_SESSION_FRAMES = int(FPS * NOMINAL_SESSION_SECONDS)  # 72000
N_TRIALS_PER_SESSION = 40

def build_trial_slices(n_timepoints: int) -> list[slice]:
    trial_slices = []
    for trial_idx in range(N_TRIALS_PER_SESSION):
        start = trial_idx * TRIAL_FRAMES
        end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
        trial_slices.append(slice(start, end))
    return trial_slices

# Truncation to nominal session length:
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
position = position[:keep_frames]
trace = trace[:, :keep_frames]
```

iii. The AI measured actual frame counts across sessions and found they varied (e.g., QLAK-CA1-08 has 71866 frames per day). It decided on fixed 40 x 1-minute trials as specified in the instructions ("split into 1-minute trials within each session"). 93 sessions have a shorter final trial (min 1666 frames). 11,481 trailing frames were dropped from sessions exceeding 72000 frames.

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered or excluded. All 40 trials per session are kept, even if the final trial is shorter than 1800 frames. The only quality check is that NaN values are not present in the final trial data.

ii.
```python
if np.isnan(neural_trial).any() or np.isnan(output_trial).any() or np.isnan(input_trial).any():
    raise ValueError(f"NaNs found after conversion for {animal} day {day_idx}")
```

iii. The reference paper does not mention trial-level quality filtering for neural data. The AI verified all trials pass the NaN check and kept all of them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in the joblib data files. This contains binary rise-extracted calcium event traces recorded at 30 Hz from CA1 neurons.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The AI confirmed that `trace` contains the paper's "rise-extracted binary calcium event traces," consistent with the methods description. It chose not to use deconvolved or re-smoothed traces.

## 2-b. How is the `neural` data processed?

i. The neural data processing is minimal: (1) Cast to float32. (2) Remove unregistered neurons (rows that are all-NaN for that day). (3) Truncate to 72000 frames if the session exceeds that. (4) Slice into 1-minute trial segments.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
# ...
trace = trace[:, :keep_frames]
# ...
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The AI verified the raw traces are binary rise events at 30 Hz and preserved them without additional smoothing, filtering, or normalization. The consistency check ensures no neuron has partial NaN values (it must be either all-NaN or all-valid).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only unregistered neurons (all-NaN rows for a given day) are removed. No filtering based on firing rate, spatial information, split-half reliability (SHR), or place cell identity is applied. All registered neurons are kept.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
trace = trace[registered]
```

iii. The AI inspected the precomputed SHR p-values in the repository but chose to keep all registered neurons rather than filtering for place cells. The AI's reasoning was that for a decoder task, all neurons could provide useful information. The paper's place cell filtering (via SHR) was used for specific representational analyses, not necessarily for decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the start of each consecutive 1-minute segment within a recording day. Since both neural (`trace`) and position data are recorded at the same 30 Hz frame rate, they are inherently aligned frame-by-frame. Each trial starts at frame `trial_idx * 1800` and ends at frame `(trial_idx + 1) * 1800` (or the end of the session for the last trial).

ii.
```python
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. The AI set `off_start = 0.0` and `off_end = 60.0` (seconds), meaning each trial spans 0 to 60 seconds relative to its alignment event (start of the 1-minute segment). No interpolation or resampling is needed since both streams share the same time base.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native 30 Hz recording rate, giving a time bin size of approximately 33.33 ms. No temporal rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS  # ≈ 33.33 ms
```

iii. The AI preserved the original temporal resolution without downsampling or rebinning. This matches the paper's recording frame rate of 30 Hz.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from two sources: (1) the `blocked` field in the joblib data (per-day blocked partition indices), and (2) the `envs` field (environment name labels like "square", "o", "t", etc.) used for validation.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. The AI cross-validated the `blocked` field against expected masks derived from the environment names using `get_env_mat()`, which encodes all 10 geometries as 3x3 binary matrices (matching the reference code's conventions).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The `blocked` field is parsed into a 9-element binary mask (1 = blocked, 0 = open) using bottom-up row-major indexing. This mask is static per trial (same for all trials within a session). The `get_env_mat()` function provides a top-down 3x3 matrix for each environment, which is then flipped vertically and flattened to match bottom-up row-major convention.

ii.
```python
def env_to_blocked_mask(env: str) -> np.ndarray:
    return (np.flipud(get_env_mat(env)).reshape(-1) == 0).astype(np.float32)

# Per trial:
input_trial = blocked_mask.astype(np.float32).copy()
```

iii. The AI chose to represent the input as a 9D binary vector indicating which bins are blocked, with input names `blocked_bin_0` through `blocked_bin_8`. This is consistent with the instructions requesting "Environment geometry to represent which part of the arena is blocked."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The mouse position output is derived from the `position` field in the joblib data, which contains 2D (x, y) tracking coordinates at 30 Hz.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. The position data is transposed from (2, T) to (T, 2) format for processing. The AI confirmed this is the same position data used by the reference code's `get_rate_maps()` function.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI implemented animal-level arena calibration using square-environment sessions: (1) Pool all position data from that animal's square sessions. (2) Compute the arena origin as the minimum position on each axis. (3) Compute the arena side length as the maximum extent. (4) Divide by 3 to get the bin size. Then for each session, positions are shifted by the arena origin and divided by the bin size using floor division, clipped to [0, 2].

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

iii. The AI chose animal-level calibration from square sessions rather than per-session normalization (as in the reference code's `get_rate_maps()`). The reasoning was that per-session normalization would misalign translated geometries like "rectangle" with the blocked mask. The resulting arena side length is 75.0 for all animals, yielding a bin size of 25.0.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Positions are discretized into 9 categories (3x3 grid) using `bin_id = y_bin * 3 + x_bin` (bottom-up row-major indexing). The bin assignment uses floor division after shifting by the arena origin. Values are clipped to the valid range [0, 2] on each axis.

ii.
```python
coords = np.floor(shifted / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]  # 0..8
```

iii. After binning, any frames that fall in blocked bins are snapped to the nearest open bin center (`clean_blocked_bins`). This corrected 448 frames across the full dataset.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The position data and neural data are already aligned frame-by-frame at 30 Hz. Both are sliced using the same trial slices, ensuring temporal correspondence.

ii.
```python
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. No interpolation or resampling is needed since both streams share the same 30 Hz time base and the same number of frames per session.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of data issues are handled:
- **Unregistered neurons**: All-NaN rows in `trace` are removed per session. A consistency check ensures no neuron has partial NaN values.
- **Positions in blocked bins**: After spatial binning, frames that land in blocked partitions are snapped to the nearest open partition center (448 frames total).
- **Sessions exceeding nominal length**: Truncated to 72000 frames (40 min).
- **Sessions shorter than nominal**: The final trial is allowed to be shorter (minimum 1666 frames observed).
- **Blocked field validation**: The `blocked` field values are cross-checked against expected masks from environment names.

ii.
```python
# NaN neuron check
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]

# Blocked bin snapping
def clean_blocked_bins(position_xy, bin_ids, blocked_mask, arena_min, bin_size):
    blocked_ids = np.flatnonzero(blocked_mask > 0.5)
    bad = np.isin(bin_ids, blocked_ids)
    # snap to nearest open bin center
    for idx in np.flatnonzero(bad):
        d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
        cleaned[idx] = int(open_ids[int(np.argmin(d2))])
    return cleaned, int(np.sum(bad))
```

iii. The AI documented all corrections in CONVERSION_NOTES.md with exact counts. Error handling raises exceptions for unexpected conditions (partially NaN neurons, blocked bins remaining after cleaning).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading joblib files**: Each animal's joblib file contains large trace matrices for all days.
2. **Arena calibration**: Concatenating all square-session position data per animal to compute arena bounds.
3. **Blocked-bin cleaning**: The `clean_blocked_bins` function iterates over individual bad frames with a Python for-loop, computing distances to all open bin centers.
4. **Trial construction**: Building trial slices and copying sub-arrays for all 8280 trials.

ii.
```python
# Blocked bin cleaning - per-frame Python loop
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. The blocked-bin cleaning loop is the most obviously inefficient step, using a Python for-loop over individual frames rather than vectorized operations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `clean_blocked_bins` function contains a Python for-loop that iterates over each frame landing in a blocked bin and computes distances to open bin centers. This could be fully vectorized using broadcasting.

ii.
```python
# Current: Python for-loop
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])

# Could be vectorized as:
# bad_positions = position_xy[bad]  # (n_bad, 2)
# d2 = np.sum((bad_positions[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (n_bad, n_open)
# cleaned[bad] = open_ids[np.argmin(d2, axis=1)]
```

iii. Only 448 frames needed correction across the full dataset, so the impact is negligible in practice, but the vectorization opportunity exists.

## 6-c. What processing does the code repeat multiple times?

i. The `parse_env_label()` and environment-related computations are done multiple times:
1. `envs` is parsed from `dat["envs"]` at the start of each animal.
2. `env_to_blocked_mask()` recomputes the expected mask for validation on every day.
3. Arena calibration re-reads and concatenates square-session position data (which is also read again in the main loop).
4. The `flatten_numeric` helper for parsing `blocked` uses a stack-based flattening approach on each day.

ii.
```python
# Position data read during calibration:
square_positions = [np.asarray(dat["position"][day], dtype=np.float64).T for day in square_days]

# Then read again in the main loop:
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. Square-session position data is loaded and processed twice: once during `calibrate_animal_arena()` and again when processing those sessions in the main loop. This redundancy has minimal practical impact since the data is in memory.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations produce data that is stored in `stats` but not used by the decoder:
1. **Arena side lengths and bin sizes**: Computed per animal and stored in stats/metadata, but the decoder only needs the binned output.
2. **Blocked mask validation**: Cross-checking `blocked` against `env_to_blocked_mask()` stores mismatches in stats but doesn't affect the output data.
3. **Frame count tracking**: `day_frame_counts`, `kept_frame_counts`, `dropped_trailing_frames` are computed per animal for statistics only.
4. **The `maps`, `SFPs`, `centroids` fields** are loaded from the joblib file but never used.

ii.
```python
# Stats tracking that doesn't affect output:
stats["arena_side_lengths"][animal] = arena_side
stats["dropped_trailing_frames"] += dropped_frames
animal_stats["snapped_blocked_frames"] += snapped

# Unused fields loaded from joblib:
dat = load_animal(animal)  # loads maps, SFPs, centroids too
```

iii. The statistics computation is useful for sanity checking and documentation but is not used by the decoder itself. The unused joblib fields (`maps`, `SFPs`, `centroids`) are loaded into memory but never accessed, wasting memory.
