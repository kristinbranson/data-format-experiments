# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads per-animal `joblib` files from `data/`, not the `.mat` files. It first enumerates animal IDs from filenames, then loads one animal dataset at a time with `joblib.load`, iterates over day/session indices, and processes each session into trials.

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

animals, session_refs = iter_session_refs(data_dir, sample=sample)
```

iii. In `CONVERSION_NOTES.md`, the agent says this matches the reference loading path in `load_dat(..., format="joblib")` and avoids cached analysis outputs. The trajectory also notes it deliberately chose the primary animal joblib files rather than cached results.

## 1-b. How are the data split into subjects?

i. Each subject is one animal file. Subject IDs are the sorted filenames beginning with `QLAK-CA1-`, and `subject_idx` is assigned by a lookup table from those IDs.

ii.
```python
animals = get_animal_ids(data_dir)
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}
...
"subjects": animals,
...
"subject_idx": subject_lookup[session_ref.animal],
```

iii. The notes justify this by stating that the raw data are organized as one primary file per animal and that session order should follow subject file order and in-file day order.

## 1-c. How are the data split into sessions?

i. Each original recording day becomes one session. The code builds `SessionRef(animal, day_index)` for every row in `dat["envs"]`.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. The notes say “keep one target session per original recording day,” because the raw data are organized by day/session and the target format supports multiple trials inside each session.

## 1-d. How are the data split into trials?

i. Each continuous session is split into non-overlapping 1-minute trials of `1800` frames at `30 Hz`. Any tail shorter than a full minute is discarded.

ii.
```python
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. The notes explicitly call this a task-specific transformation required by the decoder format, not part of the original experiment.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial behavioral or signal-quality filter. The only trial-level rule is that a session must yield at least two full 1-minute trials; otherwise the converter raises an error.

ii.
```python
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The notes justify this from the target-format requirement that each session must contain at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken from the raw `trace` array for each day/session.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The notes state that `trace` is the released rise-extracted binary calcium-event representation used by the paper/code, so no new fluorescence preprocessing is needed.

## 2-b. How is the `neural` data processed?

i. The code keeps the released event traces directly, filters session-present cells, casts the session matrix to `float16`, and slices it into per-trial `(neurons, time)` arrays. No delta-F/F, deconvolution, smoothing, or rebinning is applied.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The notes justify this by citing the README/paper description that `trace` already stores binary rising-phase events and that the reference decoder uses those traces directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered only by day-specific presence: a cell is kept if its first frame is not `NaN`. There is no place-cell filtering and no decoder-style velocity/activity filtering.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The notes justify this as matching the paper’s statement that all session-present cells were included in downstream analyses, while absent cells appear as `NaN` on a given day.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent treats the start of each non-overlapping 1-minute segment as the alignment event. Neural trials are simply frame slices from the continuous session.

ii.
```python
"temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
...
neural_trial = session_trace[:, trial_slice]
```

iii. The notes describe trialization as a task-specific transformation of continuous sessions. The trajectory says there is no natural event, so the segment start is used for metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native `30 Hz` frame rate, with `33.33 ms` bins and no temporal rebinning.

ii.
```python
FPS = 30.0
...
"time_bin_size": 1000.0 / FPS,
```

iii. The notes cite the paper’s 30 Hz synchronized acquisition and state the converter preserves that raw time base.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the raw `blocked` field for each day/session, with `maps["smoothed"]` used only as a validation cross-check.

ii.
```python
geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
valid_grid = aggregate_valid_map(smoothed_day)
```

iii. The notes say the agent used `blocked` as the authoritative source and used the map valid mask to resolve orientation ambiguity.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code creates a 9-element geometry vector initialized to ones, sets blocked indices to zero, reshapes to `3x3`, transposes to match the position/map coordinate frame, then flattens back to 9 values. This is an “open-space mask,” not a blocked-position one-hot vector.

ii.
```python
geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
if not (blocked.size == 1 and blocked[0] == -1):
    geometry[blocked] = 0.0
geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)
```

iii. The notes justify the transpose by saying `blocked.reshape(3,3).T` matches the valid support of `maps["smoothed"]` across sessions, so geometry and position share a coordinate frame.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Geometry is static within a session. The same 9-element geometry vector is copied into every trial for that session.

ii.
```python
for trial_slice in slices:
    ...
    input_trials.append(geometry_vector.copy())
```

iii. The notes state that blocked geometry does not vary within a session, so a constant per-trial input is sufficient.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output comes from the raw `position` stream for each day/session.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
```

iii. The notes say this is the synchronized behavioral position stream used by the paper’s decoder and rate-map code.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code bins continuous position into a `3x3` grid using session-wide maxima on each axis: `scale = (session_max + buffer) / 3`, then `floor(position / scale)`, clip to `[0,2]`, and convert to a single categorical class.

ii.
```python
coords = np.asarray(position_day, dtype=np.float64).T
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The notes justify this as following the reference code’s session-wide position normalization rule, adapted from finer spatial bins to `3x3` bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded into three bins by the per-session scale above; the two axis bins are then combined into one class index from `0` to `8` using `x_bin * 3 + y_bin`.

ii.
```python
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes say the task required a single `3x3=9` categorical output, so the two spatial dimensions were collapsed into one 9-class label.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position labels are computed on the full raw session, then sliced with the same trial boundaries used for neural data, so alignment is frame-for-frame.

ii.
```python
position_bins, output_class = compute_position_bins(position_day)
...
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice]
    output_trial = output_class[trial_slice][np.newaxis, :]
```

iii. The notes say position and calcium-event traces share the raw 30 Hz time base, so identical slicing preserves alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. `33.33 ms` per frame at `30 Hz`; no temporal rebinning is applied.

ii.
```python
FPS = 30.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
...
"time_bin_size": 1000.0 / FPS,
```

iii. The notes repeatedly describe the converter as preserving the raw synchronized 30 Hz time base.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are aligned by using the same raw-session frame indices and the same `trial_slice` objects. Input geometry is static and repeated once per trial rather than time-resolved.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice]
    input_trials.append(geometry_vector.copy())
    output_trial = output_class[trial_slice][np.newaxis, :]
```

iii. The notes justify this by saying geometry is session-constant, while neural and position were already synchronized in the source data.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The converter unwraps nested blocked entries, coerces them to arrays, removes absent cells via `NaN` filtering, drops trailing partial trials, and raises hard errors for sessions with no present cells, geometry mismatches, or fewer than two trials.

ii.
```python
if isinstance(entry, list) and len(entry) == 1:
    entry = entry[0]
return np.atleast_1d(np.asarray(entry)).astype(int)
...
if not np.any(present_mask):
    raise ValueError(...)
...
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
...
if len(slices) < 2:
    raise ValueError(...)
```

iii. The notes describe these as sanity checks added to verify reference consistency and to enforce decoder-format assumptions.

## 7-a. What are the most time-consuming steps of the code?

i. The agent identifies loading the large per-animal joblib files as the main runtime cost. Optional processing plots and the per-session geometry check add some overhead, but I/O dominates.

ii.
```python
if session_ref.animal not in animal_cache:
    animal_cache.clear()
    gc.collect()
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. `CONVERSION_NOTES.md` explicitly says full-animal file loading is the main runtime bottleneck and motivated the animal-level cache.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorizable loop is the per-trial append loop in `process_session`. The plotting helper also has an explicit occupancy-count loop over all time points.

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

iii. The notes discuss runtime bottlenecks and speedups, but do not present additional vectorization beyond caching and precomputing session-wide outputs.

## 7-c. What processing does the code repeat multiple times?

i. The code repeatedly copies the same geometry vector into every trial and re-casts already `float16` neural slices inside the trial loop. It also reloads one animal dataset whenever iteration moves to a new animal.

ii.
```python
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    input_trials.append(geometry_vector.copy())
```

iii. The notes frame these as acceptable implementation costs, and specifically mention the repeated static input and dtype/storage choices as part of the runtime/storage tradeoff.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded work is computing `valid_grid` from `maps["smoothed"]` solely to validate geometry orientation. Optional plot generation is also unrelated to downstream decoding. The converter also stores session metadata and environment names that the decoder does not use.

ii.
```python
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
...
if show_processing and session_ref.session_id in plot_session_ids:
    plot_processing_figure(...)
```

iii. The notes explicitly describe the geometry-vs-map comparison as a consistency check rather than a decoder input, and describe plots as optional diagnostics.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as section 6: nested blocked entries are normalized, absent cells are dropped via `NaN` filtering, trailing partial trials are dropped, and structurally bad sessions raise errors.

ii.
```python
return np.atleast_1d(np.asarray(entry)).astype(int)
...
present_mask = session_present_cell_mask(trace_day)
...
slices = trial_slices(n_frames)
```

iii. The notes justify these as consistency and validation checks rather than attempts to repair corrupted sessions.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading the large animal files is the dominant cost.

ii.
```python
animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. The notes explicitly call this the main runtime cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-trial append loop and the plotting occupancy loop are the clearest candidates.

ii.
```python
for trial_slice in slices:
    ...

for xbin, ybin in position_bins:
    occupancy[xbin, ybin] += 1.0
```

iii. The notes focus on caching and reuse rather than additional vectorization.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: repeated geometry copies, repeated `astype(np.float16, copy=False)` on sliced neural trials, and repeated per-animal loads when the animal changes.

ii.
```python
neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
input_trials.append(geometry_vector.copy())
```

iii. The notes acknowledge these repeated operations but accept them as implementation tradeoffs.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: geometry validation from the smoothed maps, optional processing plots, and decoder-irrelevant metadata.

ii.
```python
valid_grid = aggregate_valid_map(smoothed_day)
...
plot_processing_figure(...)
```

iii. The notes justify them as sanity checks and diagnostics.
