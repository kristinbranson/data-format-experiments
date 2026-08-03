# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-processed joblib files (one per subject) in the `data/` directory, rather than from the raw `.mat` files. Each joblib file is loaded via `joblib.load()` and contains fields: `envs`, `blocked`, `position`, `trace`, `maps`, `SFPs`, `centroids`. The AI iterates over a hardcoded list of 7 animal IDs (`ANIMALS`).

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

iii. The AI's CONVERSION_NOTES.md states: "The joblib files are the repository's own converted form of the original MATLAB data, loaded by the reference helper `load_dat(..., format='joblib')`. They preserve the paper fields directly."

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject. The AI uses a hardcoded list of 7 animal IDs and creates a subject lookup dictionary.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}
```

iii. The animal IDs match those in the repository. The ordering follows the repository convention.

## 1-c. How are the data split into sessions?

i. Each subject has multiple recording days. The AI iterates over the `envs` list for each animal, where each entry corresponds to one recording day/session. Each day becomes a separate session in the output.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]
for day_idx, env in enumerate(envs):
    position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
    trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The CONVERSION_NOTES.md confirms "Sessions are recording days. This yields 207 sessions total, matching the paper and repository summaries."

## 1-d. How are the data split into trials?

i. The AI uses a fixed 40 trials per session. Each trial is 1800 frames (60 seconds at 30 Hz). Sessions longer than 72,000 frames (40 minutes) are truncated. The final trial of shorter sessions may be shorter than 1800 frames (minimum observed: 1666 frames).

ii.
```python
N_TRIALS_PER_SESSION = 40
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800
NOMINAL_SESSION_FRAMES = int(FPS * NOMINAL_SESSION_SECONDS)  # 72000

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

iii. CONVERSION_NOTES.md: "The recordings are nominally 40 min at 30 Hz. I split each day into 40 consecutive nominal 1 min trials of 1800 frames each. Sessions shorter than 72000 frames keep a shorter final trial. Sessions longer than 72000 frames are truncated at the nominal 40 min boundary."

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered or removed after creation. All 40 trials per session are kept, including short final trials.

ii. N/A - no filtering code.

iii. No justification provided; the AI keeps all trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib data, which contains calcium event traces with shape `(n_neurons, n_timepoints)`.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. CONVERSION_NOTES.md: "Neural activity uses the paper's rise-extracted binary calcium event traces, not deconvolved or re-smoothed traces."

## 2-b. How is the `neural` data processed?

i. The trace data is cast to float32. Neurons not registered on a given day (all-NaN rows) are removed. The data is truncated to the nominal session length if longer than 72,000 frames.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
registered = ~np.isnan(trace).any(axis=1)
trace = trace[registered]
trace = trace[:, :keep_frames]
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. CONVERSION_NOTES.md: "Unregistered neurons are removed session-by-session by dropping rows that are all NaN on that day."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses a stricter NaN filter: neurons with *any* NaN values (not just all-NaN) are removed. However, the code also verifies that any-NaN and all-NaN give the same result, raising an error if they differ ("Found partially NaN neurons").

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
```

iii. The code validates that no partially-NaN neurons exist, so the stricter check is equivalent in practice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous, and trials are consecutive 60-second segments starting from the beginning of the session.

ii. N/A - no alignment code.

iii. CONVERSION_NOTES.md metadata: `temporal_alignment_event: "Start of each consecutive 1-minute segment within a recording day"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS  # ~33.33 ms
```

iii. No justification needed; matches the source data rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the joblib data, which contains indices of blocked reward positions for each recording day.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
```

iii. CONVERSION_NOTES.md: "Decoder input is a 9D static blocked-partition mask per trial."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are parsed into a 9-element one-hot mask. A value of `[-1]` means no positions are blocked (all zeros). The AI also cross-validates blocked indices against expected masks derived from environment labels (`envs`) using a hardcoded geometry lookup (`get_env_mat` / `env_to_blocked_mask`).

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

def env_to_blocked_mask(env: str) -> np.ndarray:
    return (np.flipud(get_env_mat(env)).reshape(-1) == 0).astype(np.float32)

blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. CONVERSION_NOTES.md: "blocked values were checked against the environment labels and matched the expected geometry masks."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` field in the joblib data, which contains 2D (x, y) coordinates.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. The position variable records the animal's location in the arena at each frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI calibrates the arena bounds per-animal using square sessions only: computes `arena_min` as the minimum position across all square sessions, and `arena_side` as the maximum extent. Position is then shifted by `arena_min`, divided by `bin_size` (arena_side / 3), and floored to get bin coordinates.

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

iii. CONVERSION_NOTES.md: "I calibrated each animal's absolute 3x3 arena frame from that animal's square sessions... resulting side length: exactly 75.0 for all seven animals."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into a 3x3 grid (9 classes). Bin ID = `y_bin * 3 + x_bin` (bottom-up row-major). Additionally, any frame landing in a blocked bin is snapped to the nearest open bin center.

ii.
```python
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]

def clean_blocked_bins(position_xy, bin_ids, blocked_mask, arena_min, bin_size):
    blocked_ids = np.flatnonzero(blocked_mask > 0.5)
    bad = np.isin(bin_ids, blocked_ids)
    for idx in np.flatnonzero(bad):
        d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
        cleaned[idx] = int(open_ids[int(np.argmin(d2))])
    return cleaned, int(np.sum(bad))
```

iii. CONVERSION_NOTES.md: "After absolute 3x3 binning, any frame that landed in a blocked partition was snapped to the nearest open partition center. Total corrected frames: 448 across the full dataset."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are at the same frame rate. Both are truncated to the same `keep_frames` length and split into trials using the same slices.

ii.
```python
position = position[:keep_frames]
trace = trace[:, :keep_frames]
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. Both arrays have the same number of timepoints and are sliced identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Neurons with any NaN are removed (verified equivalent to all-NaN). (2) Sessions longer than 40 min are truncated. (3) Short final trials are kept rather than discarded. (4) Positions landing in blocked bins are snapped to nearest open bin. (5) NaN checks are performed after conversion to catch any remaining issues.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
spatial_bins, snapped = clean_blocked_bins(...)
if np.isnan(neural_trial).any() or np.isnan(output_trial).any():
    raise ValueError(...)
```

iii. CONVERSION_NOTES.md documents: 93 short final trials (min 1666 frames), 11,481 dropped trailing frames, 448 snapped blocked-bin frames.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading joblib files for each animal. (2) The `clean_blocked_bins` function which loops over individual bad frames. (3) Arena calibration which concatenates all square-session positions.

ii.
```python
dat = load_animal(animal)  # I/O bound
for idx in np.flatnonzero(bad):  # per-frame loop
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
```

iii. The blocked-bin cleaning loop processes 448 frames total across the dataset, so it's not a major bottleneck in practice.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `clean_blocked_bins` function contains a Python loop over individual bad frames that could be vectorized using broadcasting.

ii.
```python
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. With only 448 bad frames total, the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The `parse_blocked_mask` function is called once per session, and the `env_to_blocked_mask` function is also called once per session for cross-validation. The `flatten_numeric` helper does recursive flattening that could be simplified. Arena calibration loads square session positions separately from the main processing loop.

ii.
```python
for day_idx, env in enumerate(envs):
    blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
    expected_mask = env_to_blocked_mask(env)
```

iii. These are minor repetitions with no significant performance impact.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `env_to_blocked_mask` cross-validation is sanity checking only - the results are logged but not used for the actual blocked mask. (2) The `get_env_mat` function defines geometry matrices for 10 environments that are only used for validation. (3) Detailed statistics tracking (`stats` dict) throughout conversion. (4) The `coords` return value from `position_to_bins` is computed but never used.

ii.
```python
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
# expected_mask is never used for actual data processing

coords, bin_ids = position_to_bins(position, arena_min, bin_size)
# coords is never used
```

iii. These are validation/debugging steps that add robustness but are not part of the core data conversion pipeline.
