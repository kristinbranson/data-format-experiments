# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads subject files from `data/` by listing filenames beginning with `QLAK-CA1-`, loading each animal-level joblib file, enumerating sessions from `dat["envs"]`, then processing each session into trials. It keeps only one animal dataset cached at a time.

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

iii. In `CONVERSION_NOTES.md`, the agent says this matches the reference `load_dat(..., format="joblib")` path and avoids relying on precomputed analysis outputs.

## 1-b. How are the data split into subjects?

i. Subjects are split by animal file. The sorted file names become `subjects`, and each session stores an integer `subject_idx` into that subject list.

ii.
```python
animals, session_refs = iter_session_refs(data_dir, sample=sample)
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}

converted = {
    ...
    "subjects": animals,
    "subject_idx": [],
    ...
}

session_data = {
    ...
    "subject_idx": subject_lookup[session_ref.animal],
    ...
}
```

iii. The notes say one target subject corresponds to one raw animal file, matching the raw data organization and the reference animal list.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal file is treated as one session. The session list is created by iterating over `range(dat["envs"].shape[0])`.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))

day = session_ref.day_index
env_name = str(dat["envs"][day, 0])
position_day = np.asarray(dat["position"][day], dtype=np.float64)
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The notes justify this as “one target session per original recording day,” because the raw data are already organized day-by-day and the paper describes one session per day.

## 1-d. How are the data split into trials?

i. Sessions are trialized into non-overlapping 1-minute windows at 30 Hz, so each trial is 1800 frames. Any incomplete final remainder is dropped.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]

slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes say native sessions are continuous 40-minute recordings, so 1-minute trials are a task-specific transformation required by the decoder instructions.

## 1-e. How are trials filtered based on quality controls?

i. The code does not apply behavioral or neural quality filtering at the trial level. It only requires at least two complete 1-minute trials and discards trailing partial frames.

ii.
```python
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")

session_meta = {
    ...
    "discarded_tail_frames": int(n_frames - len(slices) * TRIAL_FRAMES),
}
```

iii. The notes explicitly say the reference experiment has no native trial structure, so the agent did not invent trial-level rejection beyond the target-format minimum of two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from the raw per-session calcium-event matrix `dat["trace"][day]`, after selecting cells present on that day.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)

def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
```

iii. The notes say the released `trace` field already contains the binary rising-phase calcium-event signal used in the paper/code, so no new fluorescence preprocessing is computed.

## 2-b. How is the `neural` data processed?

i. The agent keeps the released binary event traces, removes absent cells via a NaN-based mask, slices them into 1-minute trials, and stores them as `float16`. It does not smooth, pool, or otherwise transform them.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The notes justify this by saying the reference data already provide the event representation used downstream, and that `float16` was chosen later to reduce dataset size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural filtering is limited to removing session-absent cells whose trace rows are NaN. The code does not filter to place cells, does not drop low-activity cells, and does not remove low-velocity time bins.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. The notes say this matches the paper statement that all cells were included in subsequent analyses and that place-cell labels were not used for the position decoder; the agent treated day-specific NaNs as the main session-level quality indicator.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural trials are aligned to the start of each non-overlapping 1-minute segment. There is no event other than the segment boundary.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
}

for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
```

iii. The notes explain that the raw sessions are continuous, so the only alignment event introduced by the conversion is the start of each required 1-minute trial window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the raw 30 Hz frame rate, so each bin is 33.33 ms. No temporal rebinning is applied.

ii.
```python
FPS = 30.0

"metadata": {
    ...
    "time_bin_size": 1000.0 / FPS,
    "raw_fps": FPS,
}
```

iii. The notes say the converter preserves the synchronized raw time base from the source data and slices it directly into trials.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the raw `dat["blocked"][day]` field rather than directly from `envs`.

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)

geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
```

iii. The notes say the agent chose `blocked` as the authoritative raw geometry source after cross-checking it against the map support and the environment labels.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code converts the blocked partition IDs into a 3x3 binary geometry: start from all ones, set blocked indices to zero, reshape to 3x3, transpose to match the position/map axes, then flatten to a 9D vector.

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

iii. The notes justify the transpose by saying `blocked.reshape(3,3).T` matched the valid support of `maps["smoothed"]` for all sessions.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Geometry is treated as static within each session/trial. The same 9D vector is copied once per 1-minute trial, with no time axis.

ii.
```python
for trial_slice in slices:
    ...
    input_trials.append(geometry_vector.copy())

converted = {
    ...
    "input_names": [f"geometry_bin_{i}" for i in range(GEOMETRY_BINS * GEOMETRY_BINS)],
}
```

iii. The notes say the decoder input is intended to be static per trial because the blocked environment geometry does not vary within a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw per-session position stream `dat["position"][day]`.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
position_bins, output_class = compute_position_bins(position_day)
```

iii. The notes describe the output as a task-specific discretization of the continuous position variable used in the reference paper/code.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is transposed to frame-by-coordinate form, scaled by the session-wide coordinate maxima plus a small buffer, floor-binned separately in x and y, and clipped into `[0, 2]`.

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

iii. The notes say this follows the reference position-binning rule, except using `3 x 3` bins rather than the paper code’s finer spatial maps.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The binned x and y coordinates are converted into one categorical 9-class label using `xbin * 3 + ybin`.

ii.
```python
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes say the task explicitly asks for `3 x 3 = 9` spatial bins, so a single 9-class output variable was used.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output labels are cut with the same frame slices as the neural data, so each trial’s position labels correspond frame-by-frame to the neural trial.

ii.
```python
slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes emphasize that `position` and `trace` are already synchronized at 30 Hz in the source data, so direct shared slicing preserves alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e: 33.33 ms bins at the raw 30 Hz sampling rate, with no temporal pooling or rebinning.

ii.
```python
FPS = 30.0
"time_bin_size": 1000.0 / FPS,
```

iii. The notes say the agent intentionally preserved the original synchronized time base.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output use identical frame indices from each 1-minute trial slice. Input is a static geometry vector copied for every trial in that same session, so it is session-aligned rather than frame-varying.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    input_trials.append(geometry_vector.copy())
    output_trials.append(output_trial)
```

iii. The notes justify this by saying the original position and neural streams were recorded simultaneously at 30 Hz, while geometry is constant across each trial/session.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The code unwraps singleton `blocked` entries, uses `np.atleast_1d` for malformed blocked values, removes NaN-marked absent cells, discards partial trailing frames, and raises errors for sessions with no present cells, fewer than two full trials, or geometry mismatches.

ii.
```python
entry = blocked_list[day_index]
if isinstance(entry, list) and len(entry) == 1:
    entry = entry[0]
return np.atleast_1d(np.asarray(entry)).astype(int)

present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")

if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)

if len(slices) < 2:
    raise ValueError(...)
```

iii. The notes describe these as defensive checks and simple normalization steps rather than major imputation or repair.

## 7-a. What are the most time-consuming steps of the code?

i. The agent identifies loading the large per-animal joblib files as the main runtime cost; per-session processing and serialization are secondary costs.

ii.
```python
for session_index, session_ref in enumerate(session_refs):
    if session_ref.animal not in animal_cache:
        animal_cache.clear()
        gc.collect()
        animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. In Step 6 and Step 7 notes, the agent explicitly says full-animal file loading is the main runtime bottleneck.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop appending `neural`, `input`, and `output` trial objects could be reduced, and the occupancy plot loop is also scalar. The agent did not vectorize these loops.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    input_trials.append(geometry_vector.copy())
    output_trials.append(output_trial)

for xbin, ybin in position_bins:
    occupancy[xbin, ybin] += 1.0
```

iii. The notes say the agent focused on reducing reloads and memory pressure rather than deeper vectorization.

## 7-c. What processing does the code repeat multiple times?

i. It loads each animal once during session enumeration and again during conversion, and it copies the same static geometry vector once per trial.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))

for trial_slice in slices:
    ...
    input_trials.append(geometry_vector.copy())
```

iii. The notes mention repeated animal loading indirectly and explicitly note repeated per-trial geometry copies as part of the chosen data structure.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes `valid_grid` only to verify geometry orientation, generates optional plotting products, and records extensive per-session metadata used for documentation rather than decoding.

ii.
```python
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)

if show_processing and session_ref.session_id in plot_session_ids:
    plot_processing_figure(...)

converted["metadata"]["session_info"].append(session_meta)
```

iii. The notes describe these checks and plots as sanity checks to prove consistency with the reference processing.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: singleton `blocked` entries are normalized, absent cells are removed by NaN masking, incomplete tails are dropped, and structurally bad sessions trigger errors instead of being silently kept.

ii.
```python
if isinstance(entry, list) and len(entry) == 1:
    entry = entry[0]
return np.atleast_1d(np.asarray(entry)).astype(int)

present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(...)
```

iii. The notes frame this as conservative handling: normalize simple irregularities, but fail fast on sessions that violate assumptions.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: subject-file loading is the main bottleneck, with conversion and writing the large pickle also contributing.

ii.
```python
if session_ref.animal not in animal_cache:
    animal_cache.clear()
    gc.collect()
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)

with open(outpicklefile, "wb") as f:
    pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes say these were the dominant runtime/storage costs and motivated the caching and dtype changes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-trial append loop and scalar occupancy accumulation loop are the clearest vectorization candidates.

ii.
```python
for trial_slice in slices:
    ...

for xbin, ybin in position_bins:
    occupancy[xbin, ybin] += 1.0
```

iii. The notes focus on file-level caching instead of these smaller loop-level optimizations.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: it enumerates sessions by loading every animal, then reloads animals during conversion, and repeats the same static geometry copy for each trial.

ii.
```python
dat = load_animal_dataset(data_dir, animal)
...
animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)

input_trials.append(geometry_vector.copy())
```

iii. The notes justify the repeated geometry copies as necessary to satisfy the target per-trial format.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: geometry-vs-map verification, optional plotting, and verbose metadata creation are not used by the downstream decoder itself.

ii.
```python
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)

plot_processing_figure(...)
converted["metadata"]["session_info"].append(session_meta)
```

iii. The notes say these were retained as sanity checks and documentation rather than decoder inputs.
