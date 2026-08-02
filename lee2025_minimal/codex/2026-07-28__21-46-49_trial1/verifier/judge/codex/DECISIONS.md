# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the original `.mat` files. It hard-coded the seven animal IDs, loaded the repository's extensionless joblib files from `/app/data/QLAK-CA1-*`, and then iterated through each animal's per-day arrays to build sessions and trials.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]

def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]

for animal in ANIMALS:
    dat = load_animal(animal)
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by saying the joblib files are the repository's own converted form of the MATLAB data and preserve the paper fields directly. Trajectory step 57 also states that it chose the joblib animals as the "source of truth."

## 1-b. How are the data split into subjects?

i. Each subject is one hard-coded animal ID. The output `subjects` list is just the `ANIMALS` list, and each session gets a `subject_idx` pointing back to that list.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}

data: dict[str, Any] = {
    ...
    "subjects": subjects,
    "subject_idx": [],
    ...
}

data["subject_idx"].append(subject_lookup[animal])
```

iii. `CONVERSION_NOTES.md` says subject ordering follows the repository animal IDs. No additional justification was given beyond matching the repository's subject ordering.

## 1-c. How are the data split into sessions?

i. The AI treated each recording day in the joblib structure as one session. It iterated over `envs` day indices, and each day produced one session in the output dataset.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]
...
for day_idx, env in enumerate(envs):
    ...
    data["neural"].append(session_neural)
    data["input"].append(session_input)
    data["output"].append(session_output)
```

iii. `CONVERSION_NOTES.md` explicitly says "Sessions are recording days" and cites the resulting 207 sessions as matching the paper and repository summaries.

## 1-d. How are the data split into trials?

i. The AI forced each session into 40 nominal one-minute trials of 1800 frames each. It kept a shorter final trial if the session was under 72,000 frames and truncated frames beyond 72,000 if the session was longer.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
NOMINAL_SESSION_FRAMES = int(FPS * NOMINAL_SESSION_SECONDS)
N_TRIALS_PER_SESSION = 40

def build_trial_slices(n_timepoints: int) -> list[slice]:
    if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
        raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")
    trial_slices: list[slice] = []
    for trial_idx in range(N_TRIALS_PER_SESSION):
        start = trial_idx * TRIAL_FRAMES
        end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
        ...
        trial_slices.append(slice(start, end))
    return trial_slices

keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
trial_slices = build_trial_slices(keep_frames)
```

iii. `CONVERSION_NOTES.md` says the recordings are nominally 40 minutes at 30 Hz and that the AI therefore split each day into 40 consecutive nominal one-minute trials, keeping a short final trial for shorter sessions and truncating longer sessions at 40 minutes.

## 1-e. How are trials filtered based on quality controls?

i. The AI did not apply trial-quality filtering. Every nominal trial slice was kept once the session passed the minimum-length check, including short final trials.

ii.
```python
trial_slices = build_trial_slices(keep_frames)
session_neural: list[np.ndarray] = []
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
    input_trial = blocked_mask.astype(np.float32).copy()
    ...
    session_neural.append(neural_trial)
```

iii. No explicit trial-quality justification was given. The notes only justify the nominal 40-trial segmentation scheme.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from `dat["trace"][day_idx]` in the joblib animal files.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says the joblib files preserve the paper fields directly and that neural activity uses the paper's `trace` data.

## 2-b. How is the `neural` data processed?

i. The AI kept the session's `trace` values at native frame resolution, cast them to `float32`, removed unregistered neurons, truncated to at most 40 nominal minutes, and then sliced into per-trial `(neurons, time)` arrays. Because the joblib source is already stored as `(neurons, time)`, it did not transpose.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
registered = ~np.isnan(trace).any(axis=1)
trace = trace[registered]
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
trace = trace[:, :keep_frames]
...
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says neural activity uses the paper's rise-extracted binary calcium event traces and that no re-smoothing was applied. Trajectory step 39 says it verified 30 Hz position plus binary rise-extracted traces before finalizing the design.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removed neurons that were not registered on that day by dropping any row containing `NaN`, and it asserted that such rows were entirely NaN rather than partially missing.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
```

iii. `CONVERSION_NOTES.md` says unregistered neurons are removed session-by-session by dropping rows that are all `NaN` on that day. Trajectory step 39 says the AI verified that per-day neuron registration masks are encoded as `NaN`s.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI did not align neural data to an experimental event. It treated the start of each consecutive one-minute segment as the alignment event and reported that in metadata.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "Start of each consecutive 1-minute segment within a recording day",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
    ...
}
...
trial_slices = build_trial_slices(keep_frames)
```

iii. The notes say these one-minute trials are decoder-specific formatting decisions rather than paper-defined events. No stronger event-based justification was given.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at 30 Hz, so the time bin size is `1000/30` ms, about 33.33 ms. No temporal rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"time_bin_size": TIME_BIN_MS,
```

iii. `CONVERSION_NOTES.md` says the recordings are 30 Hz and describes trialization only, not any temporal downsampling or rebinning.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The actual decoder input is derived from `dat["blocked"][day_idx]`. The AI also read `dat["envs"][day_idx]`, but only to sanity-check that the blocked mask matches the named geometry.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]
...
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. `CONVERSION_NOTES.md` says the decoder input is a blocked-partition mask and that `blocked` values were checked against the environment labels and matched the expected geometry masks.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI flattened the raw blocked entry, interpreted `[-1]` as "no blocked bins," converted the remaining blocked partition indices into a 9D binary mask, and used bottom-up row-major numbering.

ii.
```python
def parse_blocked_mask(value: Any) -> np.ndarray:
    flat = np.array(flatten_numeric(value), dtype=float)
    if flat.size == 1 and np.isclose(flat[0], -1.0):
        return np.zeros(9, dtype=np.float32)
    blocked = np.unique(flat.astype(int))
    ...
    mask = np.zeros(9, dtype=np.float32)
    mask[blocked] = 1.0
    return mask
```

iii. `CONVERSION_NOTES.md` says the mask uses the dataset's bottom-up row-major partition numbering and is intended to represent which partitions are blocked in the 3x3 arena.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The input is not time-varying. The AI copied the same 9D blocked-mask vector into every trial for that session, so alignment is at the session/trial level rather than frame-by-frame.

ii.
```python
for trial_slice in trial_slices:
    ...
    input_trial = blocked_mask.astype(np.float32).copy()
    session_input.append(input_trial)
```

iii. `CONVERSION_NOTES.md` explicitly describes the geometry input as a static per-trial mask.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from `dat["position"][day_idx]` in the joblib source.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. `CONVERSION_NOTES.md` says the spatial output is derived from the position stream and transformed into a 3x3 spatial-bin readout.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first calibrated an animal-specific absolute arena frame from that animal's square sessions, then shifted positions into that frame, divided each axis into three equal bins, clipped the result to `0..2`, converted `(x_bin, y_bin)` into a single class `y * 3 + x`, and finally snapped any positions that fell into blocked bins to the nearest open-bin center.

ii.
```python
def calibrate_animal_arena(dat: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    envs = [parse_env_label(v) for v in dat["envs"]]
    square_days = [idx for idx, env in enumerate(envs) if env == "square"]
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

spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
```

iii. `CONVERSION_NOTES.md` says this was done to preserve translated geometries such as `rectangle`, which the AI believed would be misaligned by per-session min-shifting. It further justified snapping blocked-bin frames as mirroring the reference code's geometry masking.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each frame is assigned to one of 9 categories by thresholding x and y into three bins each and converting the 2D bin pair into a single bottom-up row-major class ID. Frames assigned to blocked categories are reassigned to the nearest open category.

ii.
```python
coords = np.floor(shifted / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
...
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. The notes justify the class IDs as bottom-up row-major 3x3 bins and justify the blocked-bin reassignment as a geometry-consistency cleaning step.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI kept neural and position data frame-synchronous by truncating both to the same `keep_frames`, converting position to bin IDs at frame resolution, and then slicing both streams with the same trial slices.

ii.
```python
position = position[:keep_frames]
trace = trace[:, :keep_frames]
if trace.shape[1] != position.shape[0]:
    raise ValueError(f"Trace/position length mismatch for {animal} day {day_idx}")
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. The notes present the good sample-decoder accuracy as a sanity check that the neural/activity alignment and spatial output formatting are coherent.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is still the native 30 Hz sampling rate, so each bin is about 33.33 ms. No rebinning is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
```

iii. No separate justification beyond preserving the native frame rate appears in the notes or trajectory.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output streams are aligned frame-by-frame by applying the same truncation and the same one-minute `slice` objects. The input stream is static per trial and is copied once for each trial in the same session.

ii.
```python
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
position = position[:keep_frames]
trace = trace[:, :keep_frames]
...
trial_slices = build_trial_slices(keep_frames)
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
    input_trial = blocked_mask.astype(np.float32).copy()
```

iii. The AI relied on shared frame indices and verification/training success as justification that alignment was coherent.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI added defensive handling for multiple cases: flattening nested blocked entries, treating `[-1]` as no blocked bins, checking blocked indices for range errors, asserting that NaN neurons are entirely NaN rather than partially missing, truncating overly long sessions, keeping short final trials, and snapping blocked-bin position labels to nearby open bins.

ii.
```python
if flat.size == 0:
    raise ValueError("Blocked entry was empty")
...
if np.any((blocked < 0) | (blocked > 8)):
    raise ValueError(f"Blocked indices out of range: {blocked.tolist()}")
...
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
```

iii. The notes justify these changes as decoder-specific formatting and geometry-consistency cleaning. There is no separate justification specifically framed as "minor issue" handling.

## 7-a. What are the most time-consuming steps of the code?

i. The most expensive parts appear to be loading each large joblib animal file and then scanning many frame-wise arrays during arena calibration and blocked-bin cleaning. Those are the dominant whole-dataset operations in the script.

ii.
```python
def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]

square_positions = [np.asarray(dat["position"][day], dtype=np.float64).T for day in square_days]
all_square = np.concatenate(square_positions, axis=0)
...
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. The AI did not explicitly discuss runtime hotspots in the notes or trajectory. This is inferred from the implementation.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loop is the blocked-bin reassignment loop over every bad frame. The repeated Python loops over trial creation and over trial/session copying could also be reduced.

ii.
```python
for open_id in open_ids:
    ...
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])

for trial_idx in range(N_TRIALS_PER_SESSION):
    ...
for trial_slice in trial_slices:
    ...
```

iii. No explicit efficiency justification was given. This is inferred from the code structure.

## 7-c. What processing does the code repeat multiple times?

i. The AI repeatedly copies the same static blocked mask once per trial, repeatedly builds and appends trial lists session by session, and repeatedly deep-copies session objects again when constructing the sample dataset.

ii.
```python
for trial_slice in trial_slices:
    ...
    input_trial = blocked_mask.astype(np.float32).copy()
    session_input.append(input_trial)

def deep_subset_dataset(data: dict[str, Any], session_indices: list[int]) -> dict[str, Any]:
    ...
    if key in {"neural", "input", "output", "brain_region_idx"}:
        subset[key] = [deepcopy(value[idx]) for idx in session_indices]
```

iii. No explicit justification was given for these repeated copies. The notes only say the sample dataset is a subset of the full dataset.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes extensive stats and session metadata, checks environment/blocked consistency, calibrates square-session arena bounds, and performs blocked-bin snapping. Most of that is used only for reporting and validation rather than by the decoder itself.

ii.
```python
stats: dict[str, Any] = {
    "animals": {},
    ...
    "blocked_env_mismatches": [],
    "session_env_counts": Counter(),
    "arena_side_lengths": {},
}
...
data["metadata"]["session_info"].append(
    {
        "animal": animal,
        "day_index_within_animal": day_idx,
        "environment": env,
        ...
        "snapped_blocked_frames": snapped,
        "arena_min_xy": arena_min.tolist(),
        "arena_side_length": arena_side,
        "bin_size": bin_size,
    }
)
```

iii. The notes justify these computations as sanity checks against the paper and code. They are not justified as necessary for downstream decoding.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same handling as 6: the AI validates malformed blocked entries, removes NaN neurons, checks for partially missing neuron rows, truncates overly long sessions, keeps short final trials, and snaps blocked-bin labels to open bins.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
...
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(...)
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
```

iii. The justification is the same as in 6: decoder-specific formatting choices and geometry-consistency cleaning, not an explicit paper-derived missing-data policy.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading the large joblib animal files and then doing whole-session array scans for arena calibration and blocked-bin reassignment are the most expensive steps.

ii.
```python
def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
all_square = np.concatenate(square_positions, axis=0)
...
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
```

iii. No explicit runtime justification was given; this is inferred from the implementation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the blocked-bin reassignment loops and trial construction loops are the most obvious candidates for vectorization.

ii.
```python
for open_id in open_ids:
    ...
for idx in np.flatnonzero(bad):
    ...
for trial_idx in range(N_TRIALS_PER_SESSION):
    ...
```

iii. No explicit efficiency justification was given.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: the code repeats blocked-mask copies per trial, session-by-session list construction, and deep-copying when materializing the sample subset.

ii.
```python
input_trial = blocked_mask.astype(np.float32).copy()
...
subset[key] = [deepcopy(value[idx]) for idx in session_indices]
```

iii. No explicit justification was given.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: it computes extensive stats and session metadata, geometry sanity checks, and blocked-bin snapping that are not required by the decoder's input/output interface.

ii.
```python
stats["blocked_env_mismatches"].append(...)
...
data["metadata"]["session_info"].append(...)
```

iii. The notes frame these as validation and sanity checks rather than decoder necessities.
