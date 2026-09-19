# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset from the primary per-animal `joblib` files in `data/`, not from the `.mat` files. It first enumerates animal IDs from filenames like `QLAK-CA1-*`, builds a flat list of `(animal, day_index)` session references, and then processes each referenced session in a second pass.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )

def load_animal_dataset(data_dir: str, animal: str) -> dict:
    return joblib.load(os.path.join(data_dir, animal))[animal]

def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose the "primary joblib animal files" because this matches the paper code path `load_dat(..., format="joblib")`. It also justifies the choice as a memory-management decision: process one animal file at a time and reuse it across sessions before releasing it.

## 1-b. How are the data split into subjects (mice)?

i. Each `joblib` file corresponds to one mouse. The subject ID is the filename, and the sorted list of those filenames becomes `subjects`.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )

animals, session_refs = iter_session_refs(data_dir, sample=sample)
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}

converted = {
    ...
    "subjects": animals,
    ...
}
```

iii. The notes say the raw data are organized "per animal" and that one subject file corresponds to one animal. The trajectory and notes both treat the filename as the natural subject identifier.

## 1-c. How are the data split into sessions?

i. Each subject file contains multiple recording days/sessions. The AI treats each `day_index` along the session axis as one output session.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

```python
day = session_ref.day_index
env_name = str(dat["envs"][day, 0])
position_day = np.asarray(dat["position"][day], dtype=np.float64)
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
```

iii. `CONVERSION_NOTES.md` says "keep one target session per original recording day" because the raw data are organized by day/session and the target format supports multiple trials within each session.

## 1-d. How are the data split into trials?

i. Trials are artificial, non-overlapping 1-minute chunks. At 30 Hz, each trial is 1800 frames. The code uses floor division, so any trailing partial minute is dropped.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

```python
n_frames = position_day.shape[1]
slices = trial_slices(n_frames)
...
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes explicitly say trialization is task-specific rather than native to the experiment: continuous 40-minute sessions are split into non-overlapping 1800-frame windows to satisfy the decoder format.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a separate per-trial quality-control filter. It keeps every full 1-minute trial. The only gating is that sessions with fewer than 2 complete trials are rejected, and incomplete trailing data are discarded.

ii.
```python
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

```python
def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. The notes say the source data are continuous sessions without native trials, so trial QC is not part of the original curation. The "at least two trials" check is justified as a target-format/decoder requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data are derived from the raw per-day `trace` array in the `joblib` subject structure.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The notes say the released `trace` variable is the paper's rise-extracted calcium-event signal and that no new calcium feature needs to be computed.

## 2-b. How is the `neural` data processed?

i. The AI takes the session's `trace` matrix, filters it to session-present cells, keeps the native frame rate, and slices it into 1-minute trials. It does not compute delta-F/F, deconvolution, smoothing, or temporal pooling. It downcasts the stored trials to `float16`.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
...
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. In the notes, the AI justifies this by saying the released `trace` is already the binary rising-phase representation used by the paper/code, so "do not compute new calcium features." It separately justifies `float16` as a storage/memory reduction to make the 207-session converted dataset tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered only by day/session presence: a cell is kept if the first frame of that day's trace is not `NaN`. There is no place-cell filter and no decoder-time velocity/activity filter carried into the converted file.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

```python
present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. The notes say "cells absent on a given day appear as `NaN` in per-day traces/maps" and that the conversion should "use all session-present cells" rather than pre-filtering to place cells. The AI also explicitly argues that the paper's position decoder is not place-cell restricted.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological or task event alignment. The AI aligns each trial to the start of the artificial 1-minute segment created within a session.

ii.
```python
"metadata": {
    "task_description": "Decode 3x3-binned mouse position from CA1 calcium-event activity with static 3x3 environment geometry input.",
    "time_bin_size": 1000.0 / FPS,
    "temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
    ...
}
```

iii. The notes repeatedly say the source recordings are continuous 40-minute sessions, so the 1-minute segmentation is an imposed target-format transformation rather than an event-based alignment from the original experiment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native 30 Hz frame rate, i.e. `1000 / 30 = 33.33...` ms per bin. No temporal rebinning is applied.

ii.
```python
FPS = 30.0
...
"metadata": {
    ...
    "time_bin_size": 1000.0 / FPS,
    ...
}
```

iii. The notes justify this as preserving the raw synchronized 30 Hz position/trace time base used in the paper.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from the raw `blocked` field for each day/session.

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)
```

iii. The notes say the `blocked` field is the authoritative raw description of which 3x3 arena partitions are unavailable in each session.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI builds a 3x3 geometry mask from `blocked`: start with all ones, set blocked positions to zero, reshape to 3x3, transpose to match the position/map coordinate frame, flatten back to length 9, and reuse that same static vector for every trial in the session. It also validates this geometry against the non-`NaN` footprint of `maps["smoothed"]`.

ii.
```python
def blocked_to_geometry_vector(blocked_list: list, day_index: int) -> tuple[np.ndarray, np.ndarray]:
    blocked = extract_day_blocked_entry(blocked_list, day_index)
    geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
    if not (blocked.size == 1 and blocked[0] == -1):
        geometry[blocked] = 0.0
    geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
    return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)
```

```python
geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )
...
input_trials.append(geometry_vector.copy())
```

iii. The notes justify the transpose explicitly: `blocked.reshape(3,3).T` was chosen because it matched the valid support of `maps["smoothed"]` for all 207 sessions. The AI says this was necessary to align geometry with the position/map axes for asymmetric environments.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The decoder output is derived from the raw per-day `position` array.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
```

iii. The notes say the paper's raw behavioral stream is the synchronized 2D position trace, so this is the natural source for the decoded output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI transposes the session position into `(time, 2)`, computes a session-specific scale for each axis from that session's maximum coordinate plus a small buffer, bins each frame by floor division into a 3x3 grid, clips bin indices to `[0, 2]`, and converts `(x_bin, y_bin)` into a single class index.

ii.
```python
def compute_position_bins(position_day: np.ndarray, n_bins: int = POSITION_BINS) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(position_day, dtype=np.float64).T
    scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
    binned = np.floor(coords / scale).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
    return binned, class_idx
```

iii. The notes justify this as using the "same session-wide position floor-division rule as the reference code" while replacing the original 15x15 spatial map binning with the required 3x3 decoder bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded by thirds of that session's observed coordinate range, not by fixed global 75 cm boundaries. The final categorical label is `x_bin * 3 + y_bin`, producing classes `0..8`.

ii.
```python
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The notes say this choice preserves one session-wide spatial partition across all derived trials and follows the floor-division convention used in the paper code's rate-map construction.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI computes the session-long position class labels once, then slices those labels with the exact same `trial_slice` boundaries used for the neural matrix. That keeps neural and output frame-aligned within every 1-minute trial.

ii.
```python
position_bins, output_class = compute_position_bins(position_day)
...
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes justify this by saying the raw `position` and `trace` streams are already aligned at 30 Hz, so identical slicing preserves alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid data are handled conservatively. The AI removes cells absent on a given day via NaN-based masking, discards incomplete trailing trial fragments, errors out if a session has no present cells or fewer than 2 full trials, and errors out if the `blocked`-derived geometry disagrees with the map support. It also stores no remaining NaNs in converted trials.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

```python
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
...
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )
...
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The notes describe these as "edge-case checks" and "sanity checks" added to make mismatches visible immediately and to ensure the converted dataset is internally consistent with the raw data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading the large per-animal `joblib` files and reading `maps["smoothed"]` for every session to validate the geometry orientation. Optional plotting also adds extra work, but the notes identify file loading as the main bottleneck.

ii.
```python
def load_animal_dataset(data_dir: str, animal: str) -> dict:
    return joblib.load(os.path.join(data_dir, animal))[animal]
```

```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
...
valid_grid = aggregate_valid_map(smoothed_day)
```

iii. In Step 6 of `CONVERSION_NOTES.md`, the AI says "Loading the full animal files is the main runtime cost because each file contains all sessions and registered cells for one subject."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loops are the Python loop that appends each trial one by one and the plotting-only occupancy accumulation loop. The session enumeration loop is structural, but the inner slicing/copying work could be batched more aggressively.

ii.
```python
def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    input_trials.append(geometry_vector.copy())
    output_trials.append(output_trial)
```

```python
occupancy = np.zeros((POSITION_BINS, POSITION_BINS), dtype=np.float32)
for xbin, ybin in position_bins:
    occupancy[xbin, ybin] += 1.0
```

iii. The AI did not explicitly call these loops out in the notes, but its own speedup discussion shows it was thinking about runtime and memory. These are the main remaining Python-level loops that could be replaced by reshape/split/counting operations.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats subject-file loading and some lightweight per-trial copying. Each animal is loaded once in `iter_session_refs()` just to count sessions and then loaded again in `convert_dataset()` to process those same sessions. Inside each processed session, the same static geometry vector is copied into every trial.

ii.
```python
def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

```python
for session_index, session_ref in enumerate(session_refs):
    if session_ref.animal not in animal_cache:
        animal_cache.clear()
        gc.collect()
        animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

```python
for trial_slice in slices:
    ...
    input_trials.append(geometry_vector.copy())
```

iii. The notes mention the animal-by-animal cache as an optimization, but the code still does a first-pass load for session enumeration and a second-pass load for actual conversion. That repeated work is visible directly in the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The heaviest discarded processing is the per-session geometry validation against `maps["smoothed"]`, since the downstream decoder never uses the spatial maps. Optional plotting also uses extra computations that are not saved into the final decoder inputs/outputs. Metadata like `env_name` and rich `session_info` are useful for bookkeeping but not consumed by the decoder.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
...
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )
```

```python
if show_processing and session_ref.session_id in plot_session_ids:
    plot_processing_figure(...)
```

```python
session_meta = {
    "session_id": session_ref.session_id,
    "animal": session_ref.animal,
    "day_index": day,
    "environment": env_name,
    ...
}
```

iii. The AI justified these steps as sanity checks and documentation rather than as part of the actual converted representation. The notes explicitly say the geometry-vs-map comparison was used to resolve an orientation ambiguity before full conversion.
