# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the primary per-animal joblib files from `data/`, discovers animal IDs by filename, enumerates one session per recording day from `dat["envs"].shape[0]`, and then reloads one animal at a time during conversion while caching the current animal in memory.

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

```python
animal_cache: dict[str, dict] = {}
for session_index, session_ref in enumerate(session_refs):
    if session_ref.animal not in animal_cache:
        animal_cache.clear()
        gc.collect()
        animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose the primary joblib animal files to match the reference `load_dat(..., format="joblib")` path and to avoid inheriting downstream cached-analysis assumptions.

## 1-b. How are the data split into subjects (mice)?

i. Each subject is defined by one top-level animal file such as `QLAK-CA1-08`. Subjects are sorted lexicographically, stored in `subjects`, and each session is assigned a `subject_idx` through `subject_lookup`.

ii.
```python
animals, session_refs = iter_session_refs(data_dir, sample=sample)
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}

converted = {
    "subjects": animals,
    "subject_idx": [],
    ...
}
```

```python
session_data = {
    ...
    "subject_idx": subject_lookup[session_ref.animal],
}
```

iii. The notes say the raw file organization is the subject definition and that session order should follow subject file order and in-file day order.

## 1-c. How are the data split into sessions?

i. The agent defines one converted session per original recording day. Day indices are taken from the first axis of `envs`, and `SessionRef(animal, day_index)` is the session key.

ii.
```python
@dataclass(frozen=True)
class SessionRef:
    animal: str
    day_index: int

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
```

iii. In the notes, the agent explicitly states "Keep one target session per original recording day" because the raw data are organized by day/session and the target format supports multiple trials inside each session.

## 1-d. How are the data split into trials?

i. Trials are artificial task-specific trials created by splitting each continuous session into non-overlapping 1-minute windows at 30 Hz. Each trial is 1800 frames, and any trailing partial minute is dropped.

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
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    input_trials.append(geometry_vector.copy())
    output_trials.append(output_trial)
```

iii. The notes justify this as a task-required transformation: the reference experiment is continuous 40-minute sessions, but the decoder task explicitly asked for 1-minute trials within each session.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no trial-level quality control. The agent keeps every full 1-minute slice, drops only the trailing incomplete slice at session end, and rejects a session only if it has fewer than 2 full 1-minute trials.

ii.
```python
def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

```python
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The notes say the reference code has no native trial structure, so the agent treated 1-minute trialization as a task-specific transformation and did not add trial rejection beyond the format requirement of at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken directly from the per-day `trace` array, after selecting cells present on that day.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)

def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
```

iii. The notes say the released `trace` field is already the binary rising-phase calcium-event representation used in the paper/code, so no new neural feature was computed from another raw source variable.

## 2-b. How is the `neural` data processed?

i. The agent does very little processing: it keeps the released event traces, removes absent cells, splits into 1-minute trials, and casts the saved arrays to `float16`. It does not apply the reference decoder's temporal smoothing or temporal pooling before saving.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The notes justify this by saying the published `trace` is already the binary event signal used by the authors and that no delta-F/F or new calcium preprocessing should be added. Later notes also say `float16` was chosen to reduce storage pressure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality filter is day-specific cell presence: cells whose first frame is `NaN` are removed. There is no place-cell filter, no low-activity cell filter, and no timepoint masking based on movement speed.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. The notes say the agent intentionally used "all session-present cells" and "do not pre-filter to place cells," citing the paper's statement that all cells were included in subsequent analyses. The notes also acknowledge that the reference decoder applies velocity and cell-activity filters online.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns each trial to the start of its non-overlapping 1-minute session segment. In metadata this is described as `"start of each non-overlapping 1-minute within-session segment"`.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    ...
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
}
```

iii. The notes say there is no native trial event in the reference experiment, so the 1-minute segment start is the task-specific alignment event introduced for the target format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the raw 30 Hz frame rate, so the time bin is `1000 / 30 = 33.33 ms`. No temporal rebinning is applied before saving.

ii.
```python
FPS = 30.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
```

```python
"metadata": {
    ...
    "time_bin_size": 1000.0 / FPS,
    "raw_fps": FPS,
    "trial_length_frames": TRIAL_FRAMES,
}
```

iii. The notes explicitly state that the converter "preserves the raw 30 Hz synchronized time base" and uses the aligned `position` and `trace` streams directly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The geometry input is derived from the raw `blocked` field for each day, not from the `envs` string label.

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)
```

```python
geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
input_trials.append(geometry_vector.copy())
```

iii. The notes say `blocked` is the authoritative raw source because it directly encodes which 3x3 arena partitions are blocked, while `envs` was retained only as a cross-check and metadata label.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent starts from an all-ones 9-element vector, sets blocked partitions to `0`, reshapes to `3 x 3`, transposes that grid to match the position/map coordinate frame, and then flattens it back to a 9D vector. It also validates this grid against the support of `maps["smoothed"]`.

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
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )
```

iii. The notes say the transpose was chosen after resolving an orientation ambiguity: `blocked.reshape(3,3).T` matched the valid support of the reference spatial maps for all sessions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the per-day `position` array.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
position_bins, output_class = compute_position_bins(position_day)
```

iii. The notes map `dat[animal]["position"][day, :, frame_start:frame_end]` directly to the converted output variable.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent transposes `position_day` to frames-by-2 coordinates, computes a session-wide per-axis scale from `(nanmax + buffer) / 3`, floors positions into bin indices, clips them to `[0, 2]`, and converts `(xbin, ybin)` to a single class index.

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

iii. The notes say this was chosen to match the reference floor-division binning rule, but adapted from the paper code's `15 x 15` maps to the task's required `3 x 3` decoder output.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous position is discretized into `3 x 3 = 9` categories. Each axis is thresholded into three bins using floor division against the session-wide scale, and the two axis bins are collapsed into one class `x * 3 + y`.

ii.
```python
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

```python
output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes say the task explicitly requested nine spatial bins, so the agent chose one time-varying 9-class variable rather than separate x and y outputs.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The output uses the same session/day, raw frame rate, and trial slices as the neural data. For each `trial_slice`, `output_class[trial_slice]` is saved next to `session_trace[:, trial_slice]`.

ii.
```python
position_bins, output_class = compute_position_bins(position_day)
...
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes say the converter preserves the released aligned `position` and `trace` streams directly, so the alignment is frame-for-frame within each 1-minute segment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing cells are handled by dropping neurons whose day-specific trace row is `NaN`. If a session has no present cells, if blocked geometry disagrees with the spatial-map support, or if there are fewer than two full 1-minute trials, the script raises an error instead of trying to repair the data. Incomplete tail frames at session end are discarded.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

```python
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )

if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The notes say absent cells marked by `NaN` are the main day-specific validity indicator in the dataset. They also describe the geometry-vs-map check and discarded tail frames as explicit edge-case handling.

## 6-a. What are the most time-consuming steps of the code?

i. The main runtime cost is loading large per-animal joblib datasets, which include all sessions and large map arrays. The rest of the per-session conversion is relatively light array slicing and casting.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

```python
if session_ref.animal not in animal_cache:
    animal_cache.clear()
    gc.collect()
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. In Step 6 of the notes, the agent explicitly says loading the full animal files is the main runtime cost because each file contains all sessions and registered cells for one subject.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still uses Python loops that could be vectorized or otherwise reduced: the per-trial append loop in `process_session`, the occupancy-count loop inside `plot_processing_figure`, and repeated per-session iteration in `iter_session_refs` just to count day indices. These loops are not catastrophic, but they are avoidable.

ii.
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

iii. The notes emphasize I/O as the main bottleneck and do not claim these loops were eliminated; the retained Python loops are a pragmatic rather than fully optimized implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code reloads each animal once in `iter_session_refs` and again during actual conversion. It also casts `session_trace` to `float16` once and then calls `astype(np.float16, copy=False)` again inside each trial, and it copies the same static geometry vector into every trial of a session.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

```python
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    ...
    input_trials.append(geometry_vector.copy())
```

iii. The notes mention speedups such as caching one loaded animal at a time, but the final code still repeats these smaller operations because the agent prioritized straightforward session-wise conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads the full `maps["smoothed"]` array and computes `valid_grid` only to validate geometry orientation and optionally support plotting; those map-derived objects are not saved into the converted dataset. The plotting-only occupancy calculation is also discarded after figure generation.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
...
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
```

```python
if show_processing and session_ref.session_id in plot_session_ids:
    plot_processing_figure(...)
```

iii. The notes justify this as a sanity check to resolve geometry orientation ambiguity, not as part of the final decoder dataset. They explicitly say the transpose was verified against the non-NaN support of `maps["smoothed"]` for all sessions.
