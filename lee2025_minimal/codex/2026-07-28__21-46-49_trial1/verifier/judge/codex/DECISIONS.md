# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the original `.mat` files. It hardcodes the seven animal IDs, loads one joblib file per animal from `/app/data`, and then iterates through each animal's per-day arrays inside the loaded dictionary. Sessions and trials are constructed later inside `build_dataset()`.

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
...
DATA_DIR = "/app/data"
...
def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
for animal in ANIMALS:
    dat = load_animal(animal)
```

iii. The justification given in `CONVERSION_NOTES.md` and the trajectory is that the repository's animal-level joblib files were treated as the "source of truth" because they already expose the paper fields directly. The agent explicitly said: "Reuse the repository’s animal-level joblib files as the source of truth" and described them as preserving `envs`, `blocked`, `position`, and `trace`.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hardcoded animal IDs in `ANIMALS`. The script copies that list into `subjects` and uses a lookup table to assign each session to a subject index.

ii.
```python
subjects = list(ANIMALS)
subject_lookup = {animal: idx for idx, animal in enumerate(subjects)}
...
data: dict[str, Any] = {
    ...
    "subjects": subjects,
    "subject_idx": [],
    ...
}
...
data["subject_idx"].append(subject_lookup[animal])
```

iii. The justification in `CONVERSION_NOTES.md` is that subject ordering should follow the repository animal IDs, which it lists explicitly. The trajectory also says the conversion should "mirror its conventions for ... session ordering."

## 1-c. How are the data split into sessions?

i. Each recording day within one animal file becomes one session. The script iterates over `envs` by `day_idx`, and for each day it reads the corresponding `blocked`, `position`, and `trace` entries and appends one session to the output lists.

ii.
```python
for animal in ANIMALS:
    dat = load_animal(animal)
    envs = [parse_env_label(v) for v in dat["envs"]]
    ...
    for day_idx, env in enumerate(envs):
        blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
        position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
        trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
        ...
        data["neural"].append(session_neural)
        data["input"].append(session_input)
        data["output"].append(session_output)
```

iii. The justification in the notes is: "Sessions are recording days. This yields 207 sessions total, matching the paper and repository summaries." The trajectory also says: "Convert each recording day into one decoder session."

## 1-d. How are the data split into trials?

i. The AI imposes 40 nominal 1-minute trials per session at 30 Hz. It defines `TRIAL_FRAMES = 1800`, truncates sessions longer than 40 minutes to 72,000 frames, and creates 40 consecutive slices. If a session is shorter than 72,000 frames, the final slice may be shorter than 1,800 frames rather than being dropped.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
NOMINAL_SESSION_SECONDS = 40.0 * 60.0
NOMINAL_SESSION_FRAMES = int(FPS * NOMINAL_SESSION_SECONDS)
N_TRIALS_PER_SESSION = 40
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
def build_trial_slices(n_timepoints: int) -> list[slice]:
    if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
        raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")
    trial_slices: list[slice] = []
    for trial_idx in range(N_TRIALS_PER_SESSION):
        start = trial_idx * TRIAL_FRAMES
        end = min((trial_idx + 1) * TRIAL_FRAMES, n_timepoints)
        trial_slices.append(slice(start, end))
    return trial_slices
```

iii. The justification from `CONVERSION_NOTES.md` is that recordings are "nominally 40 min at 30 Hz" and therefore each day was split into "40 consecutive nominal 1 min trials." The trajectory says the agent wanted to "split the retained 30 Hz stream into 40 nominal 1-minute trials" and avoid interpolation.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter based on behavior or signal quality. Instead, the code enforces a length-based convention: sessions longer than 40 nominal minutes are truncated, sessions too short to produce 40 nominal slices raise an error, and shorter-than-full final trials are kept if they exist.

ii.
```python
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
dropped_frames = original_frames - keep_frames
...
if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
    raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")
...
if neural_trial.shape[1] < TRIAL_FRAMES:
    stats["short_final_trials"] += 1
```

iii. The notes justify this as a decoder-formatting choice rather than a paper-driven quality control: sessions are nominally 40 minutes, shorter final trials are kept, and long sessions are truncated at the nominal boundary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `trace` field inside each loaded animal dictionary, specifically `dat["trace"][day_idx]` for each day/session.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
```

iii. The justification in `CONVERSION_NOTES.md` is that the conversion uses the repository's preserved paper fields directly, and that neural activity should use the dataset's "rise-extracted binary calcium event traces."

## 2-b. How is the `neural` data processed?

i. The script casts each session trace to `float32`, removes unregistered neurons, truncates the time axis to the nominal session length, and then slices the `(neurons, time)` matrix into per-trial matrices. It does not rebin or otherwise transform the neural time series.

ii.
```python
trace = np.asarray(dat["trace"][day_idx], dtype=np.float32)
registered = ~np.isnan(trace).any(axis=1)
...
trace = trace[registered]
...
trace = trace[:, :keep_frames]
...
neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The notes justify this by claiming the stored `trace` values are already the desired processed neural signal, so the main remaining work is dropping unregistered rows and packaging the data into decoder trials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered session-by-session using NaN-based registration masks. Any neuron row containing NaNs is treated as unregistered and removed. The code also asserts that there are no partially missing rows by checking that `any NaN` and `all NaN` give the same mask.

ii.
```python
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
trace = trace[registered]
...
if np.isnan(neural_trial).any() or np.isnan(output_trial).any() or np.isnan(input_trial).any():
    raise ValueError(f"NaNs found after conversion for {animal} day {day_idx}")
```

iii. The trajectory says the agent "verified the raw session streams are ... traces, with per-day neuron registration masks encoded as `NaN`s." The notes say: "Unregistered neurons are removed session-by-session by dropping rows that are all `NaN` on that day."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use a biological or task event. It aligns each trial to the start of each artificial consecutive 1-minute segment within a recording day, and records that in metadata.

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
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The justification is implicit in the notes and trajectory: trials were artificial 1-minute segments, so the agent chose the segment start as the alignment event for metadata purposes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data remain at 30 Hz, with `time_bin_size = 1000 / 30` ms (`33.333...` ms). No temporal rebinning is applied.

ii.
```python
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

iii. The trajectory explicitly says the agent wanted to "split into true 1-minute trials without inventing interpolation," and the notes describe the recordings as 30 Hz streams used directly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment-geometry input is derived from the per-day `blocked` field. The script also consults `envs` to check that the blocked mask matches the named environment geometry, but the actual decoder input comes from `blocked_mask`.

ii.
```python
envs = [parse_env_label(v) for v in dat["envs"]]
...
blocked_mask = parse_blocked_mask(dat["blocked"][day_idx])
expected_mask = env_to_blocked_mask(env)
if not np.array_equal(blocked_mask, expected_mask):
    stats["blocked_env_mismatches"].append(...)
```

iii. The trajectory says "`blocked` is the exact 3x3 occlusion descriptor that should drive the decoder input." The notes say the mask uses the dataset's partition numbering and is the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The script flattens the blocked specification, converts `[-1]` to an all-zero 9-vector, maps blocked indices to a 9D float mask, and then copies that same static mask into every trial for the session.

ii.
```python
def parse_blocked_mask(value: Any) -> np.ndarray:
    flat = np.array(flatten_numeric(value), dtype=float)
    if flat.size == 0:
        raise ValueError("Blocked entry was empty")
    if flat.size == 1 and np.isclose(flat[0], -1.0):
        return np.zeros(9, dtype=np.float32)
    blocked = np.unique(flat.astype(int))
    ...
    mask = np.zeros(9, dtype=np.float32)
    mask[blocked] = 1.0
    return mask
...
input_trial = blocked_mask.astype(np.float32).copy()
session_input.append(input_trial)
```

iii. The notes justify this as a decoder-formatting decision: "Decoder input is a 9D static blocked-partition mask per trial."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position is derived from the `position` field for each day/session, i.e. `dat["position"][day_idx]`.

ii.
```python
position = np.asarray(dat["position"][day_idx], dtype=np.float64).T
```

iii. The trajectory says the agent verified the raw session streams were "30 Hz position plus binary rise-extracted traces," and the notes describe the spatial output as being computed from the tracked position stream.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first calibrates an animal-specific absolute arena frame from all square sessions, using the minimum square-session coordinates as origin and the maximum shifted extent as arena size. For each session it shifts positions into that frame, bins them into a 3x3 grid, clips out-of-range values, and then snaps any frame that lands in a blocked bin to the nearest open bin center.

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
...
def position_to_bins(position_xy, arena_min, bin_size):
    shifted = position_xy - arena_min[None, :]
    coords = np.floor(shifted / bin_size).astype(np.int64)
    coords = np.clip(coords, 0, N_POSITION_BINS - 1)
    bin_ids = coords[:, 1] * N_POSITION_BINS + coords[:, 0]
    return coords, bin_ids
...
_, spatial_bins = position_to_bins(position, arena_min, bin_size)
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
```

iii. The trajectory justifies this as a geometry-alignment fix: per-session min-shifting was said to break translated geometries such as `rectangle`, so the agent switched to an animal-level square-session calibration. The notes further justify snapping blocked-bin frames as a geometry-consistency cleaning step.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into three bins per axis, then converted to a single class ID with bottom-up row-major indexing: `class_id = y_bin * 3 + x_bin`. The thresholds come from the calibrated animal-specific arena origin and bin size rather than direct fixed `[0, 25, 50, 75]` edges. Values outside the arena are clipped into the nearest valid bin, and blocked-bin assignments may then be reassigned to the nearest open class.

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

iii. The notes say the output classes use bottom-up row-major indexing and that the animal-level square-session calibration produces an effective 25 cm bin size while preserving translated geometries. The trajectory says the agent "brute-forc[ed] the simple flip/swap conventions" to make input and output share partition IDs.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are aligned by truncating both to the same `keep_frames` value, then slicing both with the same `trial_slices`. The code also checks that each neural trial and output trial have matching time lengths.

ii.
```python
position = position[:keep_frames]
trace = trace[:, :keep_frames]
if trace.shape[1] != position.shape[0]:
    raise ValueError(f"Trace/position length mismatch for {animal} day {day_idx}")
...
trial_slices = build_trial_slices(keep_frames)
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
    ...
    if neural_trial.shape[1] != output_trial.shape[1]:
        raise ValueError(f"Trial length mismatch for {animal} day {day_idx}")
```

iii. The justification is operational rather than textual: both arrays are sliced using the same frame count and the same per-trial indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases explicitly. Empty `blocked` entries raise an error; `blocked == [-1]` becomes an all-open mask; out-of-range blocked indices raise an error; neurons with NaN-marked registration gaps are removed; partially NaN neuron rows raise an error; sessions longer than the nominal 40-minute length are truncated; sessions too short for 40 nominal slices raise an error; blocked/environment mismatches are recorded in stats; and position samples that land in blocked bins are snapped to the nearest open bin.

ii.
```python
if flat.size == 0:
    raise ValueError("Blocked entry was empty")
if flat.size == 1 and np.isclose(flat[0], -1.0):
    return np.zeros(9, dtype=np.float32)
...
if np.any((blocked < 0) | (blocked > 8)):
    raise ValueError(f"Blocked indices out of range: {blocked.tolist()}")
...
registered = ~np.isnan(trace).any(axis=1)
if not np.array_equal(registered, ~np.isnan(trace).all(axis=1)):
    raise ValueError(f"Found partially NaN neurons in {animal} day {day_idx}")
...
keep_frames = min(original_frames, NOMINAL_SESSION_FRAMES)
...
if n_timepoints < (N_TRIALS_PER_SESSION - 1) * TRIAL_FRAMES + 2:
    raise ValueError(f"Session too short for 40 nominal trials: {n_timepoints} frames")
...
spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
```

iii. The notes justify the truncation and snapping as decoder-formatting and geometry-consistency choices. The trajectory shows the agent actively looking for missing position samples, NaNs, and geometry mismatches before deciding on these rules.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are loading the large animal joblib files, looping over all animals and days to process traces and positions, snapping blocked-bin frames inside `clean_blocked_bins`, and serializing the full converted dataset. The trajectory also says the conversion runtime was dominated by the heavy animal files and serializing the full trialized dataset.

ii.
```python
def load_animal(animal: str) -> dict[str, Any]:
    return joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
for animal in ANIMALS:
    dat = load_animal(animal)
    ...
    for day_idx, env in enumerate(envs):
        ...
        spatial_bins, snapped = clean_blocked_bins(position, spatial_bins, blocked_mask, arena_min, bin_size)
        ...
        for trial_slice in trial_slices:
            neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
```

iii. The trajectory explicitly mentions that "the heavy part is the seven large animal files" and later that the conversion was slow partly because it was "serializing the full trialized dataset."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is `clean_blocked_bins`, which loops once over open-bin IDs to build centers and then once over every blocked frame to compute nearest-center distances. Trial construction also uses Python loops over slices even though much of the session data could be reshaped or split more directly.

ii.
```python
centers = []
for open_id in open_ids:
    y_bin = open_id // N_POSITION_BINS
    x_bin = open_id % N_POSITION_BINS
    centers.append(...)
...
for idx in np.flatnonzero(bad):
    d2 = np.sum((centers - position_xy[idx][None, :]) ** 2, axis=1)
    cleaned[idx] = int(open_ids[int(np.argmin(d2))])
...
for trial_slice in trial_slices:
    neural_trial = np.asarray(trace[:, trial_slice], dtype=np.float32)
    output_trial = np.asarray(spatial_bins[trial_slice][None, :], dtype=np.int64)
```

iii. This is inferred from the code structure rather than directly argued in the notes. The agent's runtime comments about the conversion being heavy are consistent with these loops being avoidable overhead.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly copies the same static blocked mask into every trial of a session, recomputes per-session trial slicing and trial packaging, and recomputes open-bin centers each time `clean_blocked_bins` is called. It also deep-copies the already-built full dataset when constructing the sample subset.

ii.
```python
for trial_slice in trial_slices:
    ...
    input_trial = blocked_mask.astype(np.float32).copy()
    session_input.append(input_trial)
...
def clean_blocked_bins(...):
    ...
    centers = []
    for open_id in open_ids:
        ...
...
sample_data = deep_subset_dataset(data, sample_session_indices)
```

iii. This is mostly evident from the implementation. The trajectory indicates the agent cared about validation and artifact generation more than minimizing repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed values are not used by the decoder itself: `coords` returned from `position_to_bins` are immediately discarded; extensive `stats` bookkeeping and `metadata["session_info"]` are accumulated only for reporting; blocked/environment mismatch tracking is diagnostic only; and the built-in verification inside the conversion script is not part of the saved dataset contents.

ii.
```python
def position_to_bins(...):
    ...
    return coords, bin_ids
...
_, spatial_bins = position_to_bins(position, arena_min, bin_size)
...
stats["blocked_env_mismatches"].append(...)
...
data["metadata"]["session_info"].append(...)
...
valid, errors, warnings = run_decoder_verify(data)
stats["verify_data_format"] = {"valid": valid, "errors": errors, "warnings": warnings}
```

iii. This follows directly from the code. The notes make clear that many of these computations were added as sanity checks and documentation support rather than as decoder inputs.
