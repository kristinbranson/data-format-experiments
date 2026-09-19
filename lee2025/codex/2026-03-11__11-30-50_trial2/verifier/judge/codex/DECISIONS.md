# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads per-animal `joblib` files from `data/`, not the MATLAB `.mat` files used in the human reference solution. It enumerates every non-hidden, non-`.mat` file except `behav_dict`, then loads each selected animal file with `joblib.load(...)`. Trials are not loaded directly; they are derived later from each session.

ii.
```python
def get_animal_files(data_dir):
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if (
            os.path.isfile(path)
            and not name.endswith(".mat")
            and name not in {"behav_dict"}
            and not name.startswith(".")
        ):
            animals.append(name)
    return animals
...
animal_path = os.path.join(data_dir, animal)
dat = joblib.load(animal_path)[animal]
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by saying the joblib files are the “primary raw source” because they match the reference code’s storage path and already expose “analysis-ready axis ordering,” which it argued reduces transpose mistakes relative to MATLAB HDF5.

## 1-b. How are the data split into subjects?

i. The AI treats each animal file as one subject. Subject IDs are the filenames returned by `get_animal_files`, and each processed session gets a `subject_idx` pointing back to that filename list.

ii.
```python
animals = get_animal_files(data_dir)
...
for animal in selected_animals:
    animal_path = os.path.join(data_dir, animal)
    dat = joblib.load(animal_path)[animal]
...
subject_idx.append(animals.index(animal))
...
"subjects": animals,
"subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. The notes say the joblib animal files correspond to one animal each, and that session order should follow animal-file order, then session order within animal.

## 1-c. How are the data split into sessions?

i. Within each animal file, the AI treats axis 0 of the session-wise arrays as the session index. It loops over `dat["position"].shape[0]`, and each iteration becomes one session in the converted output.

ii.
```python
for session_idx in range(dat["position"].shape[0]):
    session_id = f"{animal}_s{session_idx:02d}"
    env_name = str(dat["envs"][session_idx, 0])
    ...
    session_neural, session_input, session_output, ... = preprocess_session(
        trace_session=dat["trace"][session_idx],
        position_session=dat["position"][session_idx],
        blocked_entry=dat["blocked"][session_idx],
        ...
    )
```

iii. The AI’s notes describe the joblib content as `trace (n_sessions, n_cells, n_frames)` and `position (n_sessions, 2, n_frames)`, so one slice along the first dimension is treated as one recording session.

## 1-d. How are the data split into trials?

i. The AI first splits each continuous session into consecutive non-overlapping 60 s chunks of `1800` raw frames using floor division, then processes each chunk independently as a trial candidate. Any leftover frames at the end of the session are discarded implicitly.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
...
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
...
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
```

iii. In the notes, the AI explicitly says there are no native trials and that one-minute trials must be derived from continuous sessions to satisfy the requested decoder format.

## 1-e. How are trials filtered based on quality controls?

i. The AI adds trial filtering that is not present in the human reference solution. A one-minute chunk is dropped if it has fewer than 3 movement-valid raw frames, fewer than 10 pooled samples after processing, or all-zero processed neural activity. Entire sessions are rejected if fewer than 2 trials remain after this curation.

ii.
```python
chunk_mask = velocity_mask[start:end]
if int(chunk_mask.sum()) < POOL_SIZE:
    continue
...
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue

if not np.any(pooled_trace):
    continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")
```

iii. The AI justified this in the notes as “trial-quality curation.” It documented that an initial sample run produced an all-zero neural trial warning and a minimum pooled length of 4 samples, then said it fixed this by excluding trials with too few pooled samples or no processed neural signal.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data comes from the per-session `trace` array in the joblib animal file.

ii.
```python
session_neural, session_input, session_output, ... = preprocess_session(
    trace_session=dat["trace"][session_idx],
    ...
)
```

iii. The notes say `trace` is already the reference paper’s binary rising-phase calcium-event signal, so the AI treats it as the neural activity source rather than recomputing fluorescence-derived quantities.

## 2-b. How is the `neural` data processed?

i. The AI applies substantial preprocessing before export. For each session it keeps only finite cells, restricts activity to movement-valid frames, smooths the resulting traces with a Gaussian (`sigma=3` frames), average-pools every 3 retained frames, and saves the pooled matrix as `float32`.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
...
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
...
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. The justification in `CONVERSION_NOTES.md` is that this mirrors the reference decoder’s within-session preprocessing more closely than exporting raw 30 Hz traces, and that it is also more memory-efficient.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removes any cell with any non-finite value in the session, then removes cells with `<= 5` summed events during movement-valid frames. This is stricter than the human reference solution, which only drops all-NaN/unrecorded neurons.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
...
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
```

iii. The AI’s notes justify this as matching the reference decoder’s “session-level low-activity cell filter” and as preventing NaNs from entering the exported session-centric dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use a biological or task event. It defines each trial relative to the start of a consecutive one-minute chunk from the continuous recording, then further keeps only movement-valid samples within that chunk.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "Start of each consecutive one-minute chunk from a continuous recording session",
    "off_start": 0.0,
    "off_end": 60.0,
    ...
}
```

iii. The notes repeatedly state that the source data are continuous 40 min recordings with no native trial event, so chunk boundaries are used as the alignment convention.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data by average-pooling 3 frames at 30 Hz, and records the exported time bin size as `100.0` ms. Because movement-invalid frames are removed before pooling, the resulting samples correspond to pooled valid frames rather than an unfiltered regular 30 Hz grid.

ii.
```python
FPS = 30
POOL_SIZE = 3
...
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
...
"metadata": {
    "time_bin_size": 100.0,
    "native_frame_rate_hz": FPS,
    "temporal_pool_size_frames": POOL_SIZE,
}
```

iii. The AI justified this as following the reference decoder’s “3-frame average pooling” step rather than exporting the native frame rate directly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the raw/session `blocked` field, not from `envs`.

ii.
```python
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
...
geometry_open, blocked_idx = blocked_to_open_vector(blocked_entry)
```

iii. The notes say `blocked` directly encodes which 3x3 partitions are occluded, so it was chosen as the most direct source for the requested geometry input, while `envs` was kept only as metadata/cross-check.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts blocked-partition indices into a static length-9 “open” vector in row-major 3x3 order, using `1=open` and `0=blocked`. The same vector is copied into every trial of the session.

ii.
```python
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
...
input_trials.append(geometry_open.copy())
...
"input_names": [f"partition_{idx}_open" for idx in range(GEOMETRY_SIZE * GEOMETRY_SIZE)],
```

iii. The AI justified this encoding by saying one binary feature per partition gives the decoder the full static geometry, and that `1=open, 0=blocked` matches the convention used by the reference geometry masks.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw per-session `position` array.

ii.
```python
session_neural, session_input, session_output, ... = preprocess_session(
    ...
    position_session=dat["position"][session_idx],
    ...
)
```

iii. The notes identify `position` as frame-aligned x-y position data and say it should be converted into the requested 3x3 categorical decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI bins x and y coordinates into a 3x3 grid using a per-animal scale `np.nanmax(dat["position"]) / 3`, infers a coordinate transform per animal to align occupancy with blocked geometry, snaps samples that fall in blocked bins to the nearest open bin, keeps only movement-valid frames, pools 3 valid frames at a time, then converts the pooled row/column pairs into a single categorical label stream.

ii.
```python
scale_cm = float(np.nanmax(dat["position"]))
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
coord_map = build_coord_map(transform_name)
...
raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
raw_ids_all = raw_rows_all * GEOMETRY_SIZE + raw_cols_all
mapped_ids_all = coord_map[raw_ids_all]
...
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
...
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
...
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. The AI justified this as adapting the reference decoder’s position-binning logic to the required 3x3 output while trying to reconcile blocked geometry orientation and reduce label noise from inaccessible partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. After obtaining pooled row/column indices in the 3x3 grid, the AI clips them to `[0, 2]`, optionally re-snaps invalid blocked bins, and maps each sample to a row-major category `row * 3 + col`, yielding 9 possible values.

ii.
```python
pooled_rows = np.clip(pooled_rows, 0, GEOMETRY_SIZE - 1)
pooled_cols = np.clip(pooled_cols, 0, GEOMETRY_SIZE - 1)
invalid = geometry_mat[pooled_rows, pooled_cols] == 0
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. The notes say the output uses a single 9-class categorical variable in row-major order so that the geometry input and position output share a common partition indexing scheme.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output and neural data by applying the same per-session movement mask, the same one-minute chunk boundaries, and the same 3-frame pooling to both streams before exporting each trial.

ii.
```python
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
...
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. The AI’s notes justify this as preserving the simultaneous neural/behavioral frame axis while applying the same decoder-inspired preprocessing to both modalities.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly: `blocked = -1` becomes no blocked bins; empty blocked lists become an empty index array; cells with any NaN/non-finite values are removed; short remainder frames at the end of sessions are dropped via floor division; low-information trials are discarded; and sessions with fewer than 2 surviving trials raise an error.

ii.
```python
if isinstance(blocked_entry, list):
    if len(blocked_entry) == 0:
        return np.array([], dtype=np.int64)
...
if arr.size == 1 and arr[0] < 0:
    return np.array([], dtype=np.int64)
...
finite_cells = np.isfinite(trace_session).all(axis=1)
...
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
...
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")
```

iii. The AI justified these choices as avoiding NaNs in exported session data, directly handling the `blocked` sentinel conventions, and removing pathological trials that caused decoder-format warnings during sample validation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading each animal’s large joblib file and the per-animal/session preprocessing that follows, especially transform inference, movement filtering, smoothing, and pooling across long recordings.

ii.
```python
animal_load_start = time.time()
dat = joblib.load(animal_path)[animal]
load_seconds = time.time() - animal_load_start
print(f"  loaded in {load_seconds:.2f}s")
...
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
...
session_neural, session_input, session_output, ... = preprocess_session(...)
```

iii. `CONVERSION_NOTES.md` explicitly says “Animal joblib loads are relatively slow,” gives a measured first-animal load of about 62 s, and says the AI added vectorized preprocessing and one transform inference per animal to reduce total runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest remaining vectorization target is `snap_to_open_bins`, which loops over every retained sample and computes distances to all open bins one sample at a time. The transform-scoring loop in `infer_transform` also repeatedly loops over transforms and sessions in Python instead of pushing more work into NumPy.

ii.
```python
def snap_to_open_bins(row_idx, col_idx, geometry_open):
    ...
    for i in range(row_idx.shape[0]):
        if geometry_mat[row_idx[i], col_idx[i]] == 1:
            continue
        distances = np.sum((open_coords - np.array([row_idx[i], col_idx[i]])) ** 2, axis=1)
        nearest = open_coords[np.argmin(distances)]
        row_idx[i] = nearest[0]
        col_idx[i] = nearest[1]
```

iii. The AI did not call this loop out directly in its notes, but it did claim to have vectorized the “heavy preprocessing.” This sample-wise snapping loop is the main place where that optimization was left incomplete.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly copies the same static geometry vector into every trial, repeatedly recomputes nearest-open-bin snapping for both full-session and per-trial arrays, and repeatedly performs per-trial smoothing/pooling inside the trial loop rather than pooling once for the whole session and then slicing.

ii.
```python
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
...
for trial_idx in range(n_full_trials):
    ...
    chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
    pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
    ...
    if np.any(invalid):
        pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
    neural_trials.append(pooled_trace)
    input_trials.append(geometry_open.copy())
```

iii. The notes acknowledge only some repeated work indirectly, mainly by saying that one transform inference per animal was introduced to avoid recomputing alignment per session. Other repeated trial-level work remains in the final implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several diagnostics that are not part of the saved decoder dataset: occupancy matrices, transform scores for all candidate transforms, rich plotting payloads, timing counters, and verbose `session_info` fields. Even when `--show-processing` is not used, some plot-only intermediates such as occupancy maps are still computed during preprocessing.

ii.
```python
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)

occupancy_aligned = np.zeros_like(occupancy_raw)
np.add.at(occupancy_aligned, (mapped_rows_all, mapped_cols_all), 1)
occupancy_snapped = np.zeros_like(occupancy_raw)
np.add.at(occupancy_snapped, (snapped_rows_all, snapped_cols_all), 1)
...
plot_payload = SessionPlotPayload(...)
...
session_info.append(
    {
        "session_id": session_id,
        ...
        "movement_valid_fraction": float(velocity_mask.mean()),
        "trial_lengths_pooled": trial_lengths,
    }
)
```

iii. The AI justified this extra work as supporting processing plots, sanity checks, and validation, but these diagnostics are not used by downstream decoder training itself.
