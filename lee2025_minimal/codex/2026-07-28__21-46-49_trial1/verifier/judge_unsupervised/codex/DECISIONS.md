# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven mouse IDs, loads one joblib file per mouse from `/app/data`, and then iterates through every day in each loaded animal object. Trials are not loaded directly from disk; they are created later by slicing each day-long position/trace stream into nominal 1-minute chunks.

ii. ```python
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
    envs = [parse_env_label(v) for v in dat["envs"]]
    ...
    for day_idx, env in enumerate(envs):
        ...
        trial_slices = build_trial_slices(keep_frames)
```

iii. In `CONVERSION_NOTES.md`, the agent says the joblib files are "the repository's own converted form of the original MATLAB data" and treats them as the source of truth because they already expose the paper fields directly (`envs`, `blocked`, `position`, `trace`, etc.). The trajectory also states that it chose joblib after confirming those files preserved the original paper fields.

## 1-b. How are the data split into subjects?

i. Subjects are split by file: one hard-coded animal ID corresponds to one subject. The final dataset stores the ordered subject list in `subjects`, and appends one `subject_idx` entry for every recording day/session belonging to that animal.

ii. ```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}
...
for animal in ANIMALS:
    ...
    for day_idx, env in enumerate(envs):
        ...
        data["subject_idx"].append(subject_lookup[animal])
```

iii. `CONVERSION_NOTES.md` explicitly says "Subject ordering follows the repository animal IDs" and lists the seven `QLAK-CA1-*` animals in that order.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal is treated as one decoder session. The code loops over `envs` by day index, and for each `day_idx` creates one session entry in `neural`, `input`, `output`, `brain_region_idx`, and `metadata["session_info"]`.

ii. ```python
for animal in ANIMALS:
    dat = load_animal(animal)
    envs = [parse_env_label(v) for v in dat["envs"]]
    ...
    for day_idx, env in enumerate(envs):
        ...
        data["neural"].append(session_neural)
        data["input"].append(session_input)
        data["output"].append(session_output)
        data["brain_region_idx"].append(np.zeros(session_neurons, dtype=np.int64))
        data["metadata"]["session_info"].append(
            {
                "animal": animal,
                "day_index_within_animal": day_idx,
                "environment": env,
                ...
            }
        )
```

iii. `CONVERSION_NOTES.md` says "Sessions are recording days" and claims this yields 207 sessions, matching the paper/repository summaries.

## 1-d. How are the data split into trials?

i. The agent imposes 40 nominal 1-minute trials per session at 30 Hz. It computes `TRIAL_FRAMES = 1800`, truncates any session longer than 72,000 frames to that nominal 40-minute length, and for shorter sessions still forces 40 trial slices by allowing the final slice to be shorter than 1800 frames.

ii. ```python
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
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
trial_slices = build_trial_slices(keep_frames)
```

iii. In the notes, the agent justifies this as a decoder-task choice: recordings are "nominally 40 min at 30 Hz," so it split each day into 40 consecutive nominal 1-minute trials, kept a shorter final trial for sessions below 72,000 frames, and truncated sessions above 72,000 frames.

## 1-e. How are trials filtered based on quality controls?

i. There is effectively no trial-level quality filter beyond basic integrity checks. The agent keeps all 40 nominal trials per session, including short final trials. It only rejects impossible cases such as too-short sessions, NaNs after conversion, or trace/output length mismatches.

ii. ```python
def build_trial_slices(n_timepoints: int) -> list[slice]:
    if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
        raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
    input_trial = blocked_mask.astype(np.float32).copy()

    if np.isnan(neural_trial).any() or np.isnan(output_trial).any() or np.isnan(input_trial).any():
        raise ValueError(f"NaNs found after conversion for {animal} day {day_idx}")
    if neural_trial.shape[1] != output_trial.shape[1]:
        raise ValueError(f"Trial length mismatch for {animal} day {day_idx}")
```

iii. The trajectory says the agent intentionally avoided "inventing interpolation," but chose to keep a fixed 40 x 1-minute split. `CONVERSION_NOTES.md` documents that 93 sessions have a short final trial rather than being dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the raw `trace` field for each day. No other raw field contributes to `neural`.

ii. ```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
...
trace = trace[registered]
...
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` states that the joblib files preserve the paper fields directly and that neural activity uses the paper's rise-extracted calcium traces from `trace`.

## 2-b. How is the `neural` data processed?

i. The agent converts `trace` to `float32`, removes unregistered neurons on each day, truncates the time axis to at most 72,000 frames, and slices the remaining neuron-by-time array into per-trial matrices. It does not smooth, rebin, or otherwise transform the neural events.

ii. ```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)

registered = ~np.isnan(trace).any(axis=1)
...
trace = trace[registered]

keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
trace = trace[:, :keep_frames]
...
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The notes say the script uses the paper's "rise-extracted binary calcium event traces, not deconvolved or re-smoothed traces," and the trajectory says the agent verified the raw session streams were 30 Hz position plus binary rise-extracted traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC/filter is removal of neurons whose entire row is `NaN` on a given day. The code also asserts there are no partially-NaN neurons: if a neuron has some but not all NaNs, it raises an error instead of trying to repair it.

ii. ```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
```

iii. `CONVERSION_NOTES.md` says "Unregistered neurons are removed session-by-session by dropping rows that are all `NaN` on that day." The trajectory also mentions that per-day neuron registration masks are encoded as NaNs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of each consecutive 1-minute segment within a recording day. There is no event extracted from the animal's behavior; the alignment event is an artificial segmentation boundary.

ii. ```python
"metadata": {
    ...
    "temporal_alignment_event": "Start of each consecutive 1-minute segment within a recording day",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
    ...
}
...
for trial_idx in range(N_TRIALS_PER_SESSION):
    start = trial_idx * TRIAL_FRAMES
    end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
```

iii. The notes say the decoder task introduces artificial 1-minute trials, and the metadata explicitly names the alignment event as the start of each 1-minute segment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at 30 Hz, so the time bin size is `1000 / 30 = 33.333... ms`. No temporal rebinning is applied.

ii. ```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "fps": FPS,
    ...
}
```

iii. The trajectory says the agent verified that the raw session streams are 30 Hz, and the notes describe the traces as 30 Hz rise-extracted activity.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the raw `blocked` field for each day. The raw `envs` field is also read, but only to validate that the `blocked` mask matches the expected named environment geometry.

ii. ```python
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. `CONVERSION_NOTES.md` says the decoder input is the "3x3 blocked-partition geometry" and that `blocked` was checked against `envs` as a geometry sanity check.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent flattens the possibly nested `blocked` entry, interprets `-1` as "no blocked bins," converts the blocked indices into a length-9 binary mask, and stores the same static mask for every trial in the session. The mask follows bottom-up row-major bin numbering.

ii. ```python
def parse_blocked_mask(value: Any) -> np.ndarray:
    flat = np.array(flatten_numeric(value), dtype=float)
    if flat.size == 1 and np.isclose(flat[0], -1.0):
        return np.zeros(9, dtype=np.float32)
    blocked = np.unique(flat.astype(int))
    ...
    mask = np.zeros(9, dtype=np.float32)
    mask[blocked] = 1.0
    return mask
...
input_trial = blocked_mask.astype(np.float32).copy()
```

iii. The notes say the mask uses the dataset's bottom-up row-major partition numbering and is a decoder-specific 9D static input per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw `position` field for each day.

ii. ```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
...
_, spatial_bins = position_to_bins(position, arena_min, bin_size)
...
output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. The notes say spatial output is derived from the tracked mouse position and converted into 3x3 spatial bins.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first calibrates an animal-level square-arena frame from that animal's square sessions, then subtracts the inferred arena minimum, bins position into 3 equal spatial bins per axis, clips to the valid range, and finally snaps any frame that falls in a blocked partition to the nearest open-bin center.

ii. ```python
def calibrate_animal_arena(dat: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    ...
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

def clean_blocked_bins(...):
    ...
    for idx in np.flatnonzero(bad):
        d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
        cleaned[idx] = int(open_ids[int(np.argmin(d2))])
    return cleaned, int(np.sum(bad))
```

iii. The trajectory shows the justification clearly: the agent first tried per-session min-shifting, decided that translated geometries like `rectangle` would be misaligned, switched to animal-level arena calibration from square sessions, and then added blocked-bin snapping to keep output labels consistent with geometry masks.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into a 3 x 3 grid. After coordinate normalization, each axis is floored into bin indices `0, 1, 2`, clipped into range, and then combined into a single categorical class with `class_id = y_bin * 3 + x_bin`.

ii. ```python
coords = np.floor(shifted / bin_size).astype(np.int64)
coords = np.clip(coords, 0, N_POSITION_BINS - 1)
# Output classes use bottom-up row-major indexing: y * 3 + x.
bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
```

iii. The trajectory says the agent brute-forced the flip/swap convention and settled on `class_id = y_bin * 3 + x_bin` with 25 cm bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by keeping the same frame range for both streams, checking equal frame counts, deriving `spatial_bins` from the retained position frames, and then slicing both streams with identical `trial_slice`s.

ii. ```python
position = position[:keep_frames]
trace = trace[:, :keep_frames]
if trace.shape[1] != position.shape[0]:
    raise ValueError(f"Trace/position length mismatch for {animal} day {day_idx}")
...
_, spatial_bins = position_to_bins(position, arena_min, bin_size)
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. The notes emphasize that the same underlying traces, session definitions, and geometry conventions are used, and the code enforces trace/position length equality before trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent adds several defensive checks and repairs. It raises errors for malformed environment labels, empty or out-of-range `blocked` entries, partially-NaN neurons, post-conversion NaNs, and trace/position length mismatches. It removes neurons whose whole row is NaN, truncates sessions longer than 72,000 frames, keeps short final trials for shorter sessions, and snaps blocked-bin position labels to nearby open bins. It does not interpolate position, and the raw data inspection found no NaN positions.

ii. ```python
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
spatial_bins, snapped = clean_blocked_bins(...)
...
if np.isnan(neural_trial).any() or np.isnan(output_trial).any() or np.isnan(input_trial).any():
    raise ValueError(...)
```

iii. The trajectory explicitly says the agent checked for missing position samples, chose not to interpolate, and added geometry cleanup after finding blocked-bin mismatches. `CONVERSION_NOTES.md` documents the counts for dropped trailing frames and snapped blocked-bin frames.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps in the conversion code itself are loading each large animal file, scanning every day/frame to bin positions, and especially the blocked-bin cleanup loop, which computes nearest open-bin centers frame by frame for every flagged sample. Writing the full trialized pickle is also expensive.

ii. ```python
for animal in ANIMALS:
    dat = load_animal(animal)
    ...
    for day_idx, env in enumerate(envs):
        ...
        _, spatial_bins = position_to_bins(position, arena_min, bin_size)
        spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
        ...
        for trial_slice in trial_slices:
            neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
            ...
...
save_pickle(args.output, data)
```

iii. The trajectory repeatedly notes that the full conversion spends most of its time on the "seven large animal files" and serializing the full trialized dataset.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious vectorization opportunities are inside `clean_blocked_bins`: building open-bin centers in a Python loop and snapping every bad frame in a second Python loop. Trial creation also loops over 40 slices per session even though many operations are simple views/copies that could be batched.

ii. ```python
centers = []
for open_id in open_ids:
    ...
    centers.append(np.array([...], dtype=np.float64))
centers = np.stack(centers, axis=0)
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. This is an inference from the code structure rather than something the agent documented directly.

## 6-c. What processing does the code repeat multiple times?

i. The script repeatedly copies the same static blocked mask once per trial, repeatedly computes per-trial array wrappers inside every session, and repeatedly performs validation/statistics bookkeeping that is only used for notes and summaries.

ii. ```python
for trial_slice in trial_slices:
    ...
    input_trial = blocked_mask.astype(np.float32).copy()
    ...
    stats["max_trial_length"] = max(stats["max_trial_length"], int(neural_trial.shape[1]))
    stats["min_trial_length"] = min(stats["min_trial_length"], int(neural_trial.shape[1]))
    if neural_trial.shape[1] < TRIAL_FRAMES:
        stats["short_final_trials"] += 1
```

iii. This is visible from the implementation. The agent did not frame it as a performance problem, but the repeated per-trial copies and counters are explicit in the code.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only used for sanity checks or metadata and not by the decoder: `expected_mask` comparison against `envs`, `blocked_env_mismatches`, `coords` from `position_to_bins`, arena calibration statistics, environment/session summaries, and the extensive `session_info`/`stats` bookkeeping. More importantly, the blocked-bin snapping step changes labels even though those repaired frame identities are not part of the reference pipeline.

ii. ```python
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
...
def position_to_bins(...):
    ...
    coords = np.floor(shifted / bin_size).astype(np.int64)
    coords = np.clip(coords, 0, N_POSITION_BINS - 1)
    bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
    return coords, bin_ids
...
data["metadata"]["session_info"].append({...})
```

iii. `CONVERSION_NOTES.md` shows that many of these statistics are there for documentation and sanity checks. The trajectory also describes the snapping logic as a cleanup added after exploratory debugging rather than a requirement from the reference pipeline.
