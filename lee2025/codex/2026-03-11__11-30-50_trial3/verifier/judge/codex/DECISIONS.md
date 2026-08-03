# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads extensionless per-animal `joblib` files from `data/`, not the `.mat` files. It first enumerates animal IDs from filenames, then loads each animal dictionary with `joblib.load(...)`, and later iterates over `dat["envs"]` to enumerate recording days. Trial slices are created only after the full per-session arrays are in memory.

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

iii. In `CONVERSION_NOTES.md`, the agent says it chose the “primary joblib animal files” because `load_dat(..., format="joblib")` in the reference code uses that path by default. The notes and trajectory also emphasize that these files are large, so the agent treated per-animal joblib loading as the main raw-data access path.

## 1-b. How are the data split into subjects?

i. Each extensionless `QLAK-CA1-*` file is treated as one subject. The subject list is the sorted list of those filenames, and the filename string itself becomes the subject ID.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )

converted = {
    ...
    "subjects": animals,
    "subject_idx": [],
    ...
}
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}
```

iii. The notes describe the dataset as “7 primary subject datasets,” each stored as one per-animal file. The trajectory repeatedly refers to “per-animal files” and reports per-animal statistics, so the agent clearly treated one file as one mouse.

## 1-c. How are the data split into sessions?

i. Within each animal file, every `day_index` from `0` to `dat["envs"].shape[0] - 1` is treated as one recording session. Each `SessionRef(animal, day_index)` becomes one output session.

ii.
```python
@dataclass(frozen=True)
class SessionRef:
    animal: str
    day_index: int

def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. In the notes, the agent states “Keep one target session per original recording day,” and Step 2 documents `position` and `trace` as arrays indexed by session/day in their first dimension.

## 1-d. How are the data split into trials?

i. Each continuous session is split into non-overlapping 1-minute windows at 30 Hz, so each trial is `1800` frames. Any trailing remainder shorter than 1800 frames is discarded.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]

...
slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes explicitly call trialization “a task-specific transformation” required by the decoder task: 1-minute non-overlapping windows, with discarded partial tails.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply per-trial behavioral or neural quality filtering. The only trial-level exclusions are implicit: trailing partial windows are dropped, and a session is rejected if it produces fewer than two full 1-minute trials.

ii.
```python
def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]

...
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The notes say the reference experiment is continuous rather than trial-based, so the agent did not invent extra trial QC. The added “at least 2 trials” guard is justified there as a target-format requirement for downstream decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-session `trace` array in the animal dataset, specifically `dat["trace"][day]`.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The notes say `trace` is the released rise-extracted calcium-event signal used directly by the paper’s analyses, and that no raw fluorescence or delta-F/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The agent treats `trace` as already-processed binary calcium-event activity. It keeps only session-present cells, leaves the per-session orientation as `(neurons, time)`, converts the stored trial data to `float16`, and slices the session into 1800-frame trials. It does not compute any new neural features.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
...
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. In the notes, the agent argues that the released `trace` is already the binary rising-phase representation described in the paper and should be used directly. Later notes justify the `float16` cast as a storage optimization to keep the full converted dataset tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is kept if its first frame is not `NaN`; cells with `NaN` in the first frame are treated as absent from that session and removed. There is no additional place-cell or activity-threshold filtering in the converter.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

...
present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. The notes say the paper’s main analyses include “all session-present cells” and that absent cells appear as `NaN` on that day. The agent therefore rejected place-cell pre-filtering, but used a first-frame `NaN` test as the day-specific presence mask.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align to an experimental stimulus or behavior event. Instead, it defines the alignment event as the start of each non-overlapping 1-minute segment and slices neural and behavioral streams on the same frame indices.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
    ...
}

...
slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes explicitly say the reference data are continuous 40-minute recordings with no native trial events, so the agent introduced a synthetic segment-start alignment only to satisfy the target metadata fields.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native 30 Hz frame rate, corresponding to `1000 / 30 = 33.33...` ms per time bin. No temporal rebinning is applied.

ii.
```python
FPS = 30.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

"metadata": {
    ...
    "time_bin_size": 1000.0 / FPS,
    "raw_fps": FPS,
    ...
}
```

iii. The notes repeatedly cite the paper’s 30 Hz synchronized acquisition and say the converter “preserves the raw 30 Hz synchronized time base.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The actual input vector is derived from `dat["blocked"][day]`. The agent also loads `dat["maps"]["smoothed"][:, :, :, day]` and `dat["envs"][day, 0]`, but only to validate the orientation and identity of the geometry, not to construct the saved input.

ii.
```python
env_name = str(dat["envs"][day, 0])
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)

geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )
```

iii. The notes say the `blocked` field is “authoritative” for decoder input construction, but the agent resolved an orientation ambiguity by checking it against the non-NaN support of the smoothed 15x15 maps.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent converts the blocked-partition list into a 9-element float vector of open-versus-blocked geometry, not a one-hot blocked-position vector. It starts with all ones, sets blocked partitions to `0`, reshapes to `3 x 3`, transposes that grid to match the position/map axes, flattens it back to 9 values, and copies the same vector into every trial of the session.

ii.
```python
def blocked_to_geometry_vector(blocked_list: list, day_index: int) -> tuple[np.ndarray, np.ndarray]:
    blocked = extract_day_blocked_entry(blocked_list, day_index)
    geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
    if not (blocked.size == 1 and blocked[0] == -1):
        geometry[blocked] = 0.0
    geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
    return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)

...
for trial_slice in slices:
    ...
    input_trials.append(geometry_vector.copy())
```

iii. The notes justify this as a geometry-alignment decision: use `blocked`, but transpose it so asymmetric layouts match the coordinate frame used by position and by the smoothed spatial maps.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from `dat["position"][day]`, the per-session 2D position stream.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
```

iii. The notes identify `position` as the DeepLabCut-derived continuous x-y location stream acquired at the same 30 Hz rate as neural activity.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent transposes the per-session position array to `(time, 2)`, computes one scale value per axis from the session-wide coordinate maxima plus a small buffer, floors the normalized coordinates into 3 bins per axis, clips to `[0, 2]`, and collapses the two axis bins to a single categorical class `x_bin * 3 + y_bin`.

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

iii. The notes say this was meant to mirror the reference code’s session-wise position normalization rule while adapting the output to the required `3 x 3 = 9` classes.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholding is dynamic and session-specific. For each axis, the bin width is `(session_max + 1e-5) / 3`; category boundaries are therefore based on the observed session maxima rather than fixed 75 cm edges. The final categories are named `x0_y0` through `x2_y2`.

ii.
```python
POSITION_BUFFER = 1e-5

def compute_position_bins(position_day: np.ndarray, n_bins: int = POSITION_BINS) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(position_day, dtype=np.float64).T
    scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
    binned = np.floor(coords / scale).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
    return binned, class_idx

"output_values": [[f"x{x}_y{y}" for x in range(POSITION_BINS) for y in range(POSITION_BINS)]],
```

iii. The notes explicitly describe the “session-wide maxima and buffer” rule and defend it as keeping one consistent partition for all derived trials within a session.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The agent assumes position and neural traces are already aligned frame-for-frame in the raw session arrays. It computes the full-session position classes once, then applies the same trial slices to `output_class` and to the neural trace.

ii.
```python
position_bins, output_class = compute_position_bins(position_day)
...
slices = trial_slices(n_frames)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes say the converter “preserves the raw 30 Hz synchronized time base” and uses the released aligned `position` and `trace` streams directly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent does not impute or repair raw values. Instead, it drops absent neurons using a `NaN`-based mask, drops trailing frames that do not fill a full minute, rejects sessions with zero present cells or fewer than two complete trials, and raises an error if the `blocked`-derived geometry disagrees with the support of the smoothed maps.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

...
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")

valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(
        f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask"
    )

slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The notes describe these as sanity checks rather than scientific preprocessing. They also emphasize that no NaN/Inf values remained in the final saved arrays after filtering.

## 6-a. What are the most time-consuming steps of the code?

i. The agent identified per-animal raw-file loading as the main runtime bottleneck. The conversion notes say the subject files are large and that the expensive part is loading/reloading them, not trial slicing or position discretization.

ii.
```python
def load_animal_dataset(data_dir: str, animal: str) -> dict:
    return joblib.load(os.path.join(data_dir, animal))[animal]

def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    ...
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        ...

animal_cache: dict[str, dict] = {}
for session_index, session_ref in enumerate(session_refs):
    if session_ref.animal not in animal_cache:
        animal_cache.clear()
        gc.collect()
        animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. In the trajectory, the agent says a first full-subject load took about 30 seconds and later notes that “loading the full animal files is the main runtime cost.” The code’s only explicit runtime optimization is caching one animal file at a time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent did not document any major vectorization rewrite. The code still uses Python loops to enumerate animals, sessions, and trials, and even the optional occupancy plot accumulates counts with a Python loop over timepoints.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))

...
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
    input_trials.append(geometry_vector.copy())
    output_trials.append(output_trial)

...
for xbin, ybin in position_bins:
    occupancy[xbin, ybin] += 1.0
```

iii. The notes focus on I/O reductions and memory footprint instead of vectorizing the Python loops. No separate vectorization rationale is given beyond saying raw-file loading dominates runtime.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats per-animal loading twice: once in `iter_session_refs(...)` just to discover session IDs, and again in `convert_dataset(...)` to do the actual conversion. It also recomputes the geometry-vs-map consistency check for every session even though that check is not used to build the saved arrays.

ii.
```python
def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    ...
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))

...
for session_index, session_ref in enumerate(session_refs):
    ...
    if session_ref.animal not in animal_cache:
        animal_cache.clear()
        gc.collect()
        animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)

...
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
```

iii. The notes acknowledge repeated subject-file loading as part of the runtime problem and explain that the later one-animal cache was added to reduce, but not eliminate, that overhead.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter loads smoothed rate maps only to derive `valid_grid` for a geometry sanity check, not to populate the final dataset. When `--show-processing` is used, it also computes occupancy summaries and saves figures that are not used by downstream decoder training. The saved metadata also include diagnostic fields such as `discarded_tail_frames` that are not consumed by the decoder.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
...
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)

if show_processing and session_ref.session_id in plot_session_ids:
    plot_processing_figure(...)

session_meta = {
    ...
    "discarded_tail_frames": int(n_frames - len(slices) * TRIAL_FRAMES),
}
```

iii. The notes explicitly describe the map-based geometry check and the processing plots as validation/sanity-check machinery rather than part of the converted signal representation used by the decoder.
