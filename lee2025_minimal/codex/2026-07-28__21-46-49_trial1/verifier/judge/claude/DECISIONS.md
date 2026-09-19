# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-converted joblib files in the data directory using `joblib.load()`. Each joblib file contains one animal's data as a dictionary with keys including `trace`, `position`, `blocked`, and `envs`. The AI iterates over a hardcoded list of 7 animal names (`ANIMALS`).

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
DATA_DIR = "/app/data"

def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]

for animal in ANIMALS:
    dat = load_animal(animal)
```

iii. The agent identified that the repository includes both `.mat` files and pre-converted joblib files. It chose joblib because the repository's own `main.py` uses `format="joblib"` exclusively, and the joblib files expose the original paper fields directly.

## 1-b. How are the data split into subjects?

i. Each animal name in the hardcoded `ANIMALS` list corresponds to one subject. The subject list is fixed and ordered.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}
```

iii. Each joblib file contains all recording sessions for one animal, so one file = one subject. The animal names are hardcoded to match the 7 animals in the dataset.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal's data becomes a separate session. The AI iterates over environment labels (`dat["envs"]`) which has one entry per recording day.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]
...
for day_idx, env in enumerate(envs):
    ...
    position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
    trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. Each day index in the data arrays corresponds to one recording session. The `envs` list provides the environment label for each day.

## 1-d. How are the data split into trials?

i. Each session is split into exactly 40 trials of 1 minute each (1800 frames at 30 Hz). Sessions are first truncated to a nominal 72,000 frames (40 minutes). The last trial may be shorter than 1800 frames if the session has fewer than 72,000 frames.

ii.
```python
NOMINAL_SESSION_FRAMES = int(FPS * NOMINAL_SESSION_SECONDS)  # 72000
N_TRIALS_PER_SESSION = 40
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800

keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
def build_trial_slices(n_timepoints: int) -> list[slice]:
    trial_slices: list[slice] = []
    for trial_idx in range(N_TRIALS_PER_SESSION):
        start = trial_idx * TRIAL_FRAMES
        end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
        trial_slices.append(slice(start, end))
    return trial_slices
```

iii. The paper states "All sessions were 40 min." The agent investigated actual frame counts and found they ranged from ~71,866 to ~72,219. Sessions are truncated to 72,000 and split into 40 fixed trials, with the last trial potentially being shorter.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All 40 trials per session are kept, including potentially shorter last trials.

ii. N/A (no filtering code)

iii. No justification provided for filtering; all trials are retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib data, which contains calcium rise-extracted traces with shape `(n_neurons, n_timepoints)`.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The agent verified these are "binary rise-extracted traces" (30 Hz) from the reference code.

## 2-b. How is the `neural` data processed?

i. The processing consists of: (1) removing unregistered (NaN) neurons per session, (2) truncating to the nominal session length (72,000 frames), and (3) casting to float32.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
...
trace = trace[:, :keep_frames]
```

iii. The agent verified that any-NaN and all-NaN masks are equivalent (no partially NaN neurons exist), providing a safety check.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with any NaN values (not registered in that session) are removed. The code verifies that any-NaN and all-NaN filtering produce the same result (i.e., there are no partially-NaN neurons). No other filtering (e.g., place cell selection) is applied.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
```

iii. The agent ran explicit verification that no partially NaN neurons exist, providing a stronger guarantee than just all-NaN filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials are consecutive 1-minute segments starting from the beginning of each recording day. The alignment event is described as "Start of each consecutive 1-minute segment within a recording day."

ii.
```python
"temporal_alignment_event": "Start of each consecutive 1-minute segment within a recording day",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

iii. There is no stimulus event to align to; the recording is continuous free exploration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz rate; no resampling is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in the joblib data, which contains indices of blocked positions for each session. The agent also uses the `envs` field for cross-validation.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. The agent cross-validates the `blocked` indices against the expected environment geometry derived from the `envs` label, using `get_env_mat()` with `np.flipud` to account for the coordinate convention.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are parsed from the data and converted to a 9-element binary mask (1 = blocked, 0 = open). If the value is `[-1]`, all positions are open (mask is all zeros). The mask is static per trial. The agent also validates the mask against the expected environment geometry.

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
...
input_trial = blocked_mask.astype(np.float32).copy()
```

iii. One-hot encoding allows the decoder to treat each blocked position independently. Cross-validation with environment labels provides quality assurance.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` field in the joblib data, which contains 2D coordinates (x, y) of the animal.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. The position variable records the animal's tracked location in the open field arena at each timepoint.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid. The AI calibrates the arena per-animal using square-environment sessions: it finds the global position minimum across all square sessions and derives the arena side length and bin size from the data. Position is shifted by the arena minimum and divided by the calibrated bin size, then floored and clipped to produce 0-2 bin coordinates.

ii.
```python
def calibrate_animal_arena(dat):
    square_positions = [np.asarray(dat["position"][day], dtype=np.float64).T for day in square_days]
    all_square = np.concatenate(square_positions, axis=0)
    arena_min = np.nanmin(all_square, axis=0)
    shifted = all_square - arena_min[None, :]
    arena_side = float(np.nanmax(shifted))
    return arena_min.astype(np.float64), arena_side / N_POSITION_BINS, arena_side

def position_to_bins(position_xy, arena_min, bin_size):
    shifted = position_xy - arena_min[None, :]
    coords = np.floor(shifted / bin_size).astype(np.int64)
    coords = np.clip(coords, 0, N_POSITION_BINS - 1)
    bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
    return coords, bin_ids
```

iii. The agent discovered that raw position coordinates are not zero-based and that per-session min-shifting fails for translated environments (e.g., rectangle). It adopted per-animal calibration from square sessions to preserve absolute spatial relationships.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is binned using `np.floor(shifted / bin_size)` where `bin_size = arena_side / 3`, derived from per-animal arena calibration. The bin label is `y_bin * 3 + x_bin`, giving 9 categories (0-8). Values are clipped to [0, 2] to handle edge cases.

ii.
```python
coords = np.floor(shifted / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
```

iii. The agent calibrated the bin size from data rather than assuming a fixed 75 cm arena size, which accounts for tracking-system-specific coordinate ranges.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are stored at the same 30 Hz frame rate. Both are truncated to the same `keep_frames` length and split into trials using identical slice indices.

ii.
```python
position = position[:keep_frames]
trace = trace[:, :keep_frames]
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. Both arrays have the same number of timepoints and are sliced identically, ensuring alignment. The code also verifies `trace.shape[1] != position.shape[0]`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple handling strategies: (1) Unregistered neurons (NaN rows) are removed. (2) Sessions are truncated to 72,000 frames to enforce nominal 40-minute length. (3) Positions that fall in blocked bins are snapped to the nearest open bin by Euclidean distance. (4) Blocked mask is cross-validated against environment label.

ii.
```python
# NaN neuron removal
registered = ~np.isnan(trace).any(axis=1)
trace = trace[registered]

# Session truncation
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)

# Blocked bin cleaning
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)

# Blocked mask validation
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. The agent tracked statistics for all data cleaning operations (dropped frames, snapped blocked-bin frames) for transparency.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are (1) loading the joblib files (I/O bound, containing large trace arrays), and (2) the `clean_blocked_bins` function, which uses a Python loop over individual timepoints falling in blocked bins.

ii.
```python
def clean_blocked_bins(...):
    ...
    for idx in np.flatnonzero(bad):
        d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
        cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. The data files are large, and the blocked-bin cleaning loop iterates over individual timepoints.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `clean_blocked_bins` function loops over individual blocked-bin timepoints, computing Euclidean distance to each open bin center. This could be vectorized using broadcasting over all blocked timepoints at once.

ii.
```python
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. The loop body involves simple array operations that could be vectorized with `scipy.spatial.distance.cdist` or NumPy broadcasting.

## 6-c. What processing does the code repeat multiple times?

i. The `parse_blocked_mask` is computed once, but `env_to_blocked_mask` is also computed for cross-validation — both derive the same blocked information from different sources. The `blocked_mask.astype(np.float32).copy()` is called for every trial even though the mask is identical within a session.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
...
input_trial = blocked_mask.astype(np.float32).copy()  # repeated per trial
```

iii. The cross-validation is intentional for quality assurance. The per-trial copy is a minor redundancy.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes extensive statistics (`stats` dict) including per-animal frame counts, arena side lengths, environment counts, etc. It also creates a sample dataset subset and tracks blocked-environment mismatches. These are informational but not part of the decoder data.

ii.
```python
stats: dict[str, Any] = {
    "animals": {}, "total_unique_neurons": 0, ...
}
...
sample_data = deep_subset_dataset(data, sample_session_indices)
save_pickle(args.sample_output, sample_data)
```

iii. The statistics and sample dataset are useful for debugging and verification but are not used by the decoder itself.
