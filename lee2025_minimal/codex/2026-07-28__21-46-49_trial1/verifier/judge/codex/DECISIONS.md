# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the original `.mat` files. It hard-coded the seven animal IDs, then loaded the repository's preconverted animal-level `joblib` files from `/app/data/<animal>`. It then iterated over each animal and over each day/session inside each loaded object.

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

DATA_DIR = "/app/data"

def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]

for animal in ANIMALS:
    dat = load_animal(animal)
```

iii. In trajectory step 34, the AI said "The joblib animals are the right source". In step 57 it explicitly planned to "Reuse the repository’s animal-level joblib files as the source of truth". In step 82 it repeated that the script "will use the animal-level joblib files".

## 1-b. How are the data split into subjects?

i. Subjects are the seven hard-coded animal IDs in `ANIMALS`. Each loaded animal-level file is treated as one subject, and `subject_idx` is built from the position of that ID in the hard-coded list.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}

for animal in ANIMALS:
    ...
    data["subject_idx"].append(subject_lookup[animal])
```

iii. In step 57 the AI stated that it would "Reuse the repository’s animal-level joblib files as the source of truth", which implies one subject per animal-level file.

## 1-c. How are the data split into sessions?

i. Each day/recording inside an animal-level object is treated as one session. The split is driven by iterating through `envs` with `enumerate(envs)` and indexing the corresponding `blocked`, `position`, and `trace` entries with `day_idx`.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]

for day_idx, env in enumerate(envs):
    blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
    position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
    trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
    ...
    data["neural"].append(session_neural)
    data["input"].append(session_input)
    data["output"].append(session_output)
```

iii. In step 57 the AI planned to "Convert each recording day into one decoder session". Its notes in `/app/CONVERSION_NOTES.md` say the same thing.

## 1-d. How are the data split into trials?

i. The AI imposed a fixed nominal session structure of 40 one-minute trials per session at 30 Hz. It built 40 consecutive slices of 1800 frames each, but if a session was shorter than 72000 frames it kept a shorter final trial, and if it was longer it truncated the session to 72000 frames first.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
NOMINAL_SESSION_SECONDS = 40.0 * 60.0
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

iii. In step 39 the AI said the remaining design choice was "trialization". In step 57 it committed to "split the retained 30 Hz stream into 40 nominal 1-minute trials". In its notes it justified this from the recordings being nominally 40 minutes long.

## 1-e. How are trials filtered based on quality controls?

i. The AI did not implement a separate trial-quality filter. Instead, it validated that sessions were long enough to support 40 nominal trials, checked for NaNs after conversion, and counted short final trials. Trials themselves were not selectively removed based on behavioral or neural quality.

ii.
```python
def build_trial_slices(n_timepoints: int) -> list[slice]:
    if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
        raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")

for trial_slice in trial_slices:
    ...
    if np.isnan(neural_trial).any() or np.isnan(output_trial).any() or np.isnan(input_trial).any():
        raise ValueError(f"NaNs found after conversion for {animal} day {day_idx}")
```

iii. The trajectory does not show a distinct trial-QC policy beyond validating session length and NaN-free converted arrays. The AI framed the main issue as trialization rather than trial filtering in steps 39 and 57.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived `neural` from the `trace` field in each animal-level object, using one `trace` array per day/session.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. In step 39 the AI described the raw session streams as "30 Hz position plus binary rise-extracted traces". In step 57 it said it would mirror the repository’s conventions for "trace/position handling".

## 2-b. How is the `neural` data processed?

i. The AI assumed the loaded `trace` arrays were already in neuron-by-time orientation and already represented the desired event-like neural signal. It cast them to `float32`, removed unregistered neurons, truncated them to the retained session length, and then sliced them into trials. It did not apply further smoothing, deconvolution, or temporal rebinning.

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

iii. Step 39 says the AI believed the repository already contained "binary rise-extracted traces". Step 57 says it would "remove unregistered neurons (`NaN` rows) on that day" and then trialize the retained 30 Hz stream.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Session-specific unregistered neurons are removed. The AI marks neurons as registered only if a neuron row contains no NaNs at all for that day, and it raises an error if a neuron is only partially NaN rather than fully absent.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
```

iii. In step 39 the AI said neuron registration masks were "encoded as `NaN`s". In step 57 it explicitly planned to "remove unregistered neurons (`NaN` rows) on that day".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experiment-defined stimulus or behavioral event alignment. The AI aligned trials to the start of each consecutive artificial one-minute segment within a recording day, and recorded that segment start as the alignment event in metadata.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "Start of each consecutive 1-minute segment within a recording day",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
    ...
}
```

iii. In step 57 the AI said it would split each day into 40 nominal one-minute trials. The chosen metadata reflects that same artificial segmentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI kept the native 30 Hz sampling rate, corresponding to 33.33 ms bins, and did not rebin or resample the neural stream.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    ...
}
```

iii. In step 39 the AI described the source as "30 Hz position plus binary rise-extracted traces". Its plan in step 57 preserved that 30 Hz stream and only trialized it.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The actual decoder input is derived from the `blocked` field for each session. The AI also read the `envs` labels and converted them to expected masks as a consistency check, but the saved input comes from `blocked`.

ii.
```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. In step 50 the AI said "`blocked` is the exact 3x3 occlusion descriptor that should drive the decoder input". It used `envs` only to infer/check geometry conventions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI flattened the raw blocked-position representation into a 9-element binary mask, treating `-1` as "no blocked bins". The mask uses bottom-up row-major indexing. The same static 9D mask is copied into every trial from that session.

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

input_trial = blocked_mask.astype(np.float32).copy()
session_input.append(input_trial)
```

iii. In step 50 the AI justified this by saying `blocked` is the exact 3x3 occlusion descriptor. Steps 63, 70, and 75 show that it also spent time aligning the blocked-mask indexing with the position-bin indexing.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the per-session `position` field in the animal-level data.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. Step 39 identifies the source streams as "30 Hz position plus binary rise-extracted traces". Step 57 says the decoder output would be a time-varying 9-class spatial bin derived from position.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first calibrated an animal-specific arena origin and bin size from that animal’s square sessions. It then shifted each position trace into that arena frame, converted coordinates into 3x3 bins by floor-division and clipping, and finally repaired any frames that landed in blocked bins by snapping them to the nearest open bin center.

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

def position_to_bins(position_xy: np.ndarray, arena_min: np.ndarray, bin_size: float):
    shifted = position_xy - arena_min[None, :]
    coords = np.floor(shifted / bin_size).astype(np.int64)
    coords = np.clip(coords, 0, N_POSITION_BINS - 1)
    bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
    return coords, bin_ids

_, spatial_bins = position_to_bins(position, arena_min, bin_size)
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
```

iii. Steps 63, 67, 70, 75, and 79 document the reasoning. The AI said it needed the blocked geometry and position bins to share a coordinate frame, first tried per-session min-shifting, then switched to "an animal-level arena frame inferred from that animal’s square sessions" because per-session min-shifting broke translated geometries.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Positions are thresholded into 9 categories by computing x and y bin coordinates in a 3x3 grid, clipping each axis to `{0,1,2}`, and then converting the pair to a single class ID with `y_bin * 3 + x_bin`. Because the AI used calibrated `arena_min` and `bin_size`, the thresholds are animal-specific rather than fixed global edges.

ii.
```python
shifted = position_xy - arena_min[None, :]
coords = np.floor(shifted / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
```

iii. In step 75 the AI states the category rule explicitly: "`class_id = y_bin * 3 + x_bin`". Steps 67, 70, and 79 explain why it chose calibrated thresholds instead of using the raw coordinates directly.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI kept position and neural data on the same 30 Hz frame grid, truncated both to the same retained session length, and then sliced both with the same `trial_slices`. That preserves frame-wise temporal alignment.

ii.
```python
position = position[:keep_frames]
trace = trace[:, :keep_frames]
if trace.shape[1] != position.shape[0]:
    raise ValueError(f"Trace/position length mismatch for {animal} day {day_idx}")

trial_slices = build_trial_slices(keep_frames)
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. Step 39 describes the source as shared 30 Hz streams. Step 57 says the AI would derive the 9-class output from session-level position binning after choosing the trialization scheme.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled several edge cases explicitly: it removed unregistered neurons encoded as NaN rows; it raised an error if NaNs were partial rather than whole-row; it truncated overlong sessions to a nominal 40 minutes; it allowed underlength final trials rather than dropping them; and it corrected position samples that fell in blocked bins by snapping them to the nearest open bin.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]

keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
...
if np.isnan(neural_trial).any() or np.isnan(output_trial).any() or np.isnan(input_trial).any():
    raise ValueError(f"NaNs found after conversion for {animal} day {day_idx}")
```

iii. Step 39 says the AI saw registration masks encoded as NaNs. Step 79 says it thought per-session min-shifting created geometry mismatches, and step 82 says it would "clean impossible blocked-bin assignments". Its notes justify these as geometry-consistency repairs and sanity checks.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading and iterating through all seven large animal files, calibrating/binning every session, trializing the full dataset, and serializing the resulting pickle. The blocked-bin cleaning loop can also add per-frame overhead for sessions with mismatched geometry bins.

ii.
```python
for animal in ANIMALS:
    dat = load_animal(animal)
    ...
    for day_idx, env in enumerate(envs):
        ...
        _, spatial_bins = position_to_bins(position, arena_min, bin_size)
        spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
        ...
        for trial_slice in trial_slices:
            ...

save_pickle(args.output, data)
```

iii. In step 88 the AI said "The heavy part is the seven large animal files". In step 92 it said conversion was slow likely because it was "serializing the full trialized dataset".

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is `clean_blocked_bins`, which loops over blocked frames and recomputes nearest open-bin centers frame by frame. The per-trial append loop could also be replaced with list comprehensions or bulk slicing, and static `input_trial` masks need not be copied inside the loop.

ii.
```python
centers = []
for open_id in open_ids:
    ...
    centers.append(...)
centers = np.stack(centers, axis=0)
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])

for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
    input_trial = blocked_mask.astype(np.float32).copy()
```

iii. The trajectory does not contain an explicit efficiency justification for these loops. This assessment is inferred from the final implementation.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeatedly converts the same session-level blocked mask to `float32` and copies it once per trial, repeatedly appends one trial at a time, and repeatedly traverses session metadata for the sample subset. It also rebuilds open-bin center lists every time `clean_blocked_bins` is called.

ii.
```python
for trial_slice in trial_slices:
    ...
    input_trial = blocked_mask.astype(np.float32).copy()
    ...
    session_input.append(input_trial)

def clean_blocked_bins(...):
    ...
    centers = []
    for open_id in open_ids:
        ...
        centers.append(...)
```

iii. The trajectory does not show the AI discussing these repetitions directly; they are visible in the code structure itself.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only for sanity checks or reporting, not for the saved decoder dataset: `expected_mask` comparisons against `envs`, the `stats` bookkeeping, the unused `coords` output of `position_to_bins`, per-session metadata such as `snapped_blocked_frames`, and the separate sample dataset generation. The nearest-open-bin snapping also changes outputs in a way that is not required by the downstream decoder interface.

ii.
```python
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)

_, spatial_bins = position_to_bins(position, arena_min, bin_size)

stats["max_trial_length"] = max(stats["max_trial_length"], int(neural_trial.shape[1]))
...
data["metadata"]["session_info"].append(...)

sample_session_indices = choose_sample_sessions(data)
sample_data = deep_subset_dataset(data, sample_session_indices)
save_pickle(args.sample_output, sample_data)
```

iii. The trajectory frames these as sanity checks and validation artifacts. Step 82 says the script would emit "built-in sanity checks", and later steps 100-166 show the AI explicitly using those counts and extra outputs for verification and documentation rather than for the decoder dataset itself.
