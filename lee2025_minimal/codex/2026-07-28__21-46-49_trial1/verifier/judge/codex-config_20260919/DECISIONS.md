# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads seven explicitly listed animal-level joblib files from `/app/data`. Each file is loaded with `joblib.load`, the dictionary for that animal is selected, and every recording day in its `envs`, `blocked`, `position`, and `trace` collections is processed.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]

def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]

for animal in ANIMALS:
    dat = load_animal(animal)
    envs = [parse_env_label(v) for v in dat["envs"]]
    for day_idx, env in enumerate(envs):
        position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
        trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The trajectory says the joblib files are the repository's own converted animal-level representation, expose the original paper fields directly, and are the “right source.” The AI also checked that processing them yields the paper-level totals of seven mice and 207 sessions.

## 1-b. How are the data split into subjects?

i. Each named joblib file is one mouse. The fixed `ANIMALS` order defines `subjects`, and a lookup maps each recording-day session back to that mouse.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}
...
data["subject_idx"].append(subject_lookup[animal])
```

iii. The AI justified this by inspecting the repository and confirming that each animal-level file contains all days for one animal and that the seven IDs match the paper/repository subject set.

## 1-c. How are the data split into sessions?

i. One recording day is one output session. The code loops over every environment/day entry for each animal and appends one set of neural, input, and output trial lists per day.

ii.
```python
for day_idx, env in enumerate(envs):
    ...
    data["neural"].append(session_neural)
    data["input"].append(session_input)
    data["output"].append(session_output)
```

iii. The trajectory states that “sessions are recording days” and notes that this produces 207 sessions, matching the paper and repository summaries.

## 1-d. How are the data split into trials?

i. Each nominal 40-minute day is split into 40 consecutive one-minute segments at 30 Hz. Trials normally contain 1,800 frames; a recording shorter than 72,000 frames retains a shorter 40th trial, while frames beyond the nominal 40 minutes are dropped.

ii.
```python
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
N_TRIALS_PER_SESSION = 40

for trial_idx in range(N_TRIALS_PER_SESSION):
    start = trial_idx * TRIAL_FRAMES
    end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
    trial_slices.append(slice(start, end))
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
```

iii. The AI measured actual frame-count variation and chose a fixed 40-trial representation because recordings are nominally 40 minutes. It explicitly documented that short sessions keep a short final trial and long recordings are truncated at the nominal boundary.

## 1-e. How are trials filtered based on quality controls?

i. No individual trials are filtered. Instead, the code requires enough frames to construct all 40 nominal trials, validates every slice, and retains even a short final trial. Session frames beyond 40 minutes are excluded before trialization.

ii.
```python
if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
    raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")
...
if end <= start:
    raise ValueError(f"Invalid trial slice {trial_idx}: {start}:{end}")
```

iii. The trajectory contains no decision to reject trials on behavioral or neural quality. Its stated goal was to preserve 40 nominal one-minute trials per session while handling small recording-length deviations.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the animal file's per-day `trace` field, described by the AI as the paper's rise-extracted binary calcium-event traces.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The AI inspected the repository loading and decoding functions and the data values, then concluded that `trace` contains the paper's 30 Hz binary rise-extracted activity and should be used without substituting rate maps or further deconvolution.

## 2-b. How is the `neural` data processed?

i. The trace is cast to `float32`, restricted to neurons registered on the current day, truncated with position to at most 72,000 frames, and sliced into trials. No smoothing, rebinning, normalization, or event inference is applied.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
registered = ~np.isnan(trace).any(axis=1)
trace = trace[registered]
...
trace = trace[:, :keep_frames]
...
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The trajectory says the source already contains rise-extracted event traces, so retaining those values most directly mirrors the paper. It also notes that NaN rows encode cross-day registration absence rather than neural measurements.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained for a session only if it has no NaNs. The code verifies that rows are either entirely valid or entirely NaN, rejects partially NaN neurons as malformed data, and checks that no NaNs remain in trial arrays.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
...
if np.isnan(neural_trial).any():
    raise ValueError(...)
```

iii. The AI identified NaNs as per-day neuron-registration masks and therefore removed unregistered rows session by session. It verified that this yields 69,744 session-cell observations, matching the paper and repository output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Time zero is the start of each consecutive one-minute segment, and neural data and position use the same trial slice.

ii.
```python
"temporal_alignment_event": "Start of each consecutive 1-minute segment within a recording day",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
...
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. The AI treated the decoder-requested one-minute segmentation, rather than a stimulus onset, as the alignment event because the underlying experiment consists of continuous recording days.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data remain at the native 30 Hz resolution, or 33.333 ms per frame. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"time_bin_size": TIME_BIN_MS,
```

iii. The trajectory confirms that both neural and position streams are already sampled at 30 Hz and says trialization should not invent interpolation that the paper did not use.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the per-day `blocked` field. The `envs` label is also parsed and converted to an expected geometry mask, but only as a consistency check; the saved input uses `blocked`.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. The AI inspected the session sequence and concluded that `blocked` is the exact 3×3 occlusion descriptor intended to drive decoder input, while `envs` provides a useful cross-check.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Nested numeric values are flattened. `[-1]` becomes an all-zero mask; otherwise, valid unique blocked indices 0–8 are set to one in a nine-element `float32` vector. That static mask is copied into every trial of the session.

ii.
```python
if flat.size == 1 and np.isclose(flat[0], -1.0):
    return np.zeros(9, dtype=np.float32)
blocked = np.unique(flat.astype(int))
mask = np.zeros(9, dtype=np.float32)
mask[blocked] = 1.0
...
input_trial = blocked_mask.astype(np.float32).copy()
```

iii. The AI states that the dataset's blocked indices use bottom-up row-major partition numbering and that a 9D static blocked-partition mask directly represents which parts of the arena are unavailable.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from each day's `position` field, transposed to a time-by-two array of x/y coordinates.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. The AI inspected the raw shapes and coordinate ranges and identified this field as the 30 Hz two-dimensional behavioral stream corresponding to the neural trace.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. For each animal, all square-session positions are combined to estimate a fixed arena origin and side length. Positions from every environment are shifted by that origin, converted to 3×3 coordinates, clipped to valid bins, and mapped to `y * 3 + x`. Rare samples assigned to geometrically blocked bins are reassigned to the nearest open-bin center.

ii.
```python
all_square = np.concatenate(square_positions, axis=0)
arena_min = np.nanmin(all_square, axis=0)
shifted = all_square - arena_min[None, :]
arena_side = float(np.nanmax(shifted))
...
coords = np.floor((position_xy - arena_min[None, :]) / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
...
spatial_bins, snapped = clean_blocked_bins(...)
```

iii. The AI reasoned that per-animal calibration on square sessions preserves the common absolute frame across translated geometries, whereas per-session minimum shifting would misalign shapes such as the rectangle. It described nearest-open-bin correction as mirroring the reference code's masking of impossible geometry bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is floor-divided by one third of the calibrated arena side, clipped to 0–2, and combined into nine bottom-up row-major categories. The observed calibrated side is 75, so thresholds occur at 25 and 50 coordinate units after subtracting the arena origin.

ii.
```python
bin_size = arena_side / N_POSITION_BINS
coords = np.floor(shifted / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
```

iii. The AI selected a 3×3 grid because the requested output has nine spatial classes and verified that all animals' square-session calibration produces a 75-unit arena and 25-unit bin width.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace are truncated to the same `keep_frames` count, their lengths are explicitly compared, and identical trial slices are used for both streams.

ii.
```python
position = position[:keep_frames]
trace = trace[:, :keep_frames]
if trace.shape[1] != position.shape[0]:
    raise ValueError(...)
...
neural_trial = trace[:, trial_slice]
output_trial = spatial_bins[trial_slice][None, :]
```

iii. The AI verified that both streams are 30 Hz and emphasized using the same indices rather than interpolation. It also ran format verification and sample decoding as alignment sanity checks.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Known all-NaN neuron rows are removed. Partial-NaN rows, empty or invalid blocked entries, unknown environments, invalid arena calibration, trace/position length mismatches, too-short sessions, and residual NaNs trigger errors. Small recording-length differences are handled by truncating after 40 minutes or retaining a short final trial. Position samples falling in blocked bins are snapped to the nearest open-bin center, with counts recorded in statistics.

ii.
```python
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(...)
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
```

iii. The trajectory distinguishes expected missingness (unregistered neurons and small duration variation) from malformed data. It reports 448 impossible-bin frames corrected and validates final arrays rather than silently accepting unexpected missing values.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the large animal joblib files and constructing/copying all per-session trial arrays dominate conversion time. The trajectory also found that downstream full-dataset decoder setup, especially per-session SVD initialization, is far more expensive than conversion, though that is outside `convert_data.py` itself.

ii.
```python
dat = load_animal(animal)
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. During exploration, the AI observed that full animal loads were slow and switched to the lighter `behav_dict` for frame-count analysis. Later it explicitly identified full decoder SVD initialization across 207 sessions as the validation bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop that repairs samples assigned to blocked bins could be vectorized by computing distances from all bad samples to all open centers at once. Trial construction could also use reshaping for full-length trials, although the variable-length last trial makes the current loop simpler. The animal/day loops are structurally necessary because arrays and neuron sets vary by session.

ii.
```python
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])

for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The trajectory does not explicitly discuss vectorizing these loops. This assessment follows from the implemented operations; the correction loop affects only 448 frames, so vectorization would have little practical impact.

## 6-c. What processing does the code repeat multiple times?

i. It parses environment labels once during arena calibration and again in the main animal loop. It repeatedly casts/copies the same session-level blocked mask for all 40 trials. It also computes trial statistics and validity checks for every session, intentionally repeating checks to validate the full dataset.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]  # calibrate_animal_arena
...
envs = [parse_env_label(v) for v in dat["envs"]]  # build_dataset
...
input_trial = blocked_mask.astype(np.float32).copy()
```

iii. The trajectory gives no explicit justification for these small repetitions. They arise from keeping calibration self-contained and ensuring each trial owns its input array; their cost is minor relative to loading trace data.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Conversion computes extensive diagnostic statistics, expected geometry masks, mismatch records, coordinate-bin intermediates, and a separate sample subset. These support validation and reporting but are not part of decoder training from `converted_data.pkl`. The environment consistency check does not alter saved inputs unless an impossible position bin is cleaned.

ii.
```python
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
...
coords, spatial_bins = position_to_bins(position, arena_min, bin_size)
...
sample = deep_subset_dataset(data, sample_indices)
```

iii. The trajectory shows that these diagnostics were deliberate sanity checks used to compare against paper-level counts, inspect geometry consistency, create a representative test subset, and validate the decoder before finalizing the full output.
