# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads per-animal joblib files from `data/`, not the `.mat` files used in the human reference. It scans `data/` for regular files that are not `.mat`, skips `behav_dict`, loads each animal with `joblib.load`, then iterates over every session in `dat["position"]`. Trials are created later by splitting each session into consecutive 60 s chunks and keeping only chunks that survive additional preprocessing.

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
for animal in selected_animals:
    animal_path = os.path.join(data_dir, animal)
    dat = joblib.load(animal_path)[animal]
...
for session_idx in range(dat["position"].shape[0]):
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose the joblib files because they “match the reference code storage path” and expose “analysis-ready axis ordering,” which it viewed as safer than reading the MATLAB HDF5 layout directly.

## 1-b. How are the data split into subjects?

i. Each joblib animal file is treated as one subject. Subject IDs are the filenames returned by `get_animal_files`, and each processed session gets a `subject_idx` pointing to that filename’s position in the full `animals` list.

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
data = {
    ...
    "subjects": animals,
    "subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. The notes describe the dataset as “one dataset per animal” in both joblib and `.mat` formats, so the AI kept the file-per-animal organization as the mouse split.

## 1-c. How are the data split into sessions?

i. Within each animal file, the AI treats the first axis of `position` and `trace` as session/day index. Every `session_idx` becomes one output session.

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
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. In the notes, the AI states that the joblib files expose shapes like `trace (n_sessions, n_cells, n_frames)` and `position (n_sessions, 2, n_frames)`, so it used the session axis directly.

## 1-d. How are the data split into trials?

i. The AI first computes the number of complete 60 s chunks per session using floor division by `1800` frames, then loops over those chunks. However, the exported “trials” are not the raw 1800-frame segments: within each chunk it keeps only movement-valid frames, smooths neural data, average-pools in 3-frame groups, and may drop the chunk entirely.

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
...
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
...
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
    chunk_mask = velocity_mask[start:end]
    ...
    chunk_trace = trace_active[:, start:end][:, chunk_mask]
    ...
    pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
    ...
    neural_trials.append(pooled_trace)
```

iii. The notes say the AI intentionally “split sessions into one-minute chunks” but then “within each chunk keep only movement-valid frames and pool in groups of 3” to mimic the reference decoder rather than preserve raw continuous samples.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered aggressively. A 60 s chunk is dropped if it has fewer than 3 movement-valid frames, if the pooled neural data has fewer than 10 pooled samples, or if the pooled neural matrix is all zero. After this, the whole session is rejected if fewer than 2 valid trials remain.

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

iii. The AI justified this in the notes as “trial-quality curation,” saying it removed all-zero and too-short pooled trials after seeing warnings during sample validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural output is derived from `dat["trace"][session_idx]` in the per-animal joblib structure.

ii.
```python
session_neural, session_input, session_output, ... = preprocess_session(
    trace_session=dat["trace"][session_idx],
    position_session=dat["position"][session_idx],
    blocked_entry=dat["blocked"][session_idx],
    ...
)
```

iii. The notes identify `trace` as the released “rise-event calcium traces” and state that this is the neural signal used by the reference analyses.

## 2-b. How is the `neural` data processed?

i. The AI applies several processing steps to the raw session trace: it removes non-finite cells, restricts to movement-valid frames, keeps only cells with more than 5 events during movement, smooths each retained cell with a Gaussian filter (`sigma=3` frames), then average-pools non-overlapping groups of 3 frames. The exported per-trial neural arrays are these pooled matrices.

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

iii. The notes say this was intended to mirror `decode_position_within`, `fit_decoder`, and `test_decoder` from the reference code, which the AI interpreted as requiring movement filtering, activity filtering, Gaussian smoothing, and 3-frame pooling before export.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural quality control happens in two stages. First, a cell must be finite at every time point within the session (`np.isfinite(...).all(axis=1)`). Second, among those finite cells, it must have more than 5 total events during movement-valid frames. Chunks/trials with too little movement or no pooled activity are then dropped separately.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
...
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
...
if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
    continue
if not np.any(pooled_trace):
    continue
```

iii. The notes explicitly justify this as matching the reference within-session decoder’s “movement / activity” filtering rather than the broader paper analyses that used all cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to an experimental event from the paper. It defines the alignment event as the start of each artificial 60 s chunk from the continuous session. Neural data are then further aligned sample-for-sample with the movement-filtered and pooled behavioral output from the same chunk.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
    chunk_mask = velocity_mask[start:end]
    chunk_trace = trace_active[:, start:end][:, chunk_mask]
...
"metadata": {
    ...
    "temporal_alignment_event": "Start of each consecutive one-minute chunk from a continuous recording session",
    "off_start": 0.0,
    "off_end": 60.0,
```

iii. The notes describe the sessions as continuous recordings with no native trial events, and the AI chose chunk-start alignment to satisfy the target format while keeping neural and behavior on the same filtered time axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are exported at an effective 100 ms bin size. Yes, temporal rebinning is applied: after movement filtering, the AI average-pools non-overlapping groups of 3 native 30 Hz frames.

ii.
```python
FPS = 30
POOL_SIZE = 3
...
def trial_average_pool(values, pool_size=POOL_SIZE):
    n_full = values.shape[-1] // pool_size
    ...
    return trimmed.reshape(new_shape).mean(axis=-1)
...
"metadata": {
    ...
    "time_bin_size": 100.0,
    ...
    "temporal_pool_size_frames": POOL_SIZE,
```

iii. The notes say the AI adopted the reference decoder’s 3-frame pooling and therefore exported the pooled representation instead of native 30 Hz samples.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the raw `blocked` field for each session, not from `envs`. The AI parses the blocked partition indices and converts them to a 9-element geometry vector.

ii.
```python
def parse_blocked_indices(blocked_entry):
    ...
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
...
geometry_open, blocked_idx = blocked_to_open_vector(blocked_entry)
```

iii. The notes say the AI chose `blocked` because it “directly encodes which 3x3 partitions are occluded,” whereas `envs` would require an additional mapping step.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts each session’s blocked-bin list into a static length-9 vector in row-major 3x3 order with `1=open` and `0=blocked`. The same geometry vector is copied into every exported trial from that session.

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

iii. The notes justify this as a direct geometry encoding “consistent with the reference code’s `get_env_mat` convention” and as a decoder-friendly static context variable.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived primarily from raw `position`, but they are also conditioned by `blocked` through geometry snapping and by an inferred coordinate transform built from all sessions of the same animal.

ii.
```python
scale_cm = float(np.nanmax(dat["position"]))
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
coord_map = build_coord_map(transform_name)
...
session_neural, session_input, session_output, ... = preprocess_session(
    trace_session=dat["trace"][session_idx],
    position_session=dat["position"][session_idx],
    blocked_entry=dat["blocked"][session_idx],
    ...
)
```

iii. The notes say the AI wanted the output labels to respect the blocked geometry and therefore used `blocked` plus an inferred transform to align spatial bins with the session’s environment layout.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI computes a per-animal spatial scale from the maximum position value, converts x/y coordinates into 3x3 row/column bins, infers and applies a global coordinate transform per animal, snaps any blocked-bin samples to the nearest open bin, applies the same movement filter used for neural data, average-pools row/column values in groups of 3 frames, floors them back to integer bins, clips to `[0, 2]`, and finally converts row/column to a single class index.

ii.
```python
def raw_position_to_rc(position_xy, bin_size_cm):
    x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
    y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
    x = np.clip(x, 0, GEOMETRY_SIZE - 1)
    y = np.clip(y, 0, GEOMETRY_SIZE - 1)
    return y, x
...
mapped_ids_all = coord_map[raw_ids_all]
mapped_rows_all = mapped_ids_all // GEOMETRY_SIZE
mapped_cols_all = mapped_ids_all % GEOMETRY_SIZE
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
...
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
...
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. The notes justify this as adapting the reference decoder’s position preprocessing to the required 9-class output while reducing label noise from samples that fall in blocked regions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into categories by dividing each axis into 3 equal-width bins determined by `bin_size_cm = scale_cm / 3`, where `scale_cm` is the per-animal maximum coordinate value. After pooling, rows and columns are clipped to `0, 1, 2` and combined into a row-major category `row * 3 + col`.

ii.
```python
bin_size_cm = (scale_cm + 1e-6) / GEOMETRY_SIZE
...
x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
x = np.clip(x, 0, GEOMETRY_SIZE - 1)
y = np.clip(y, 0, GEOMETRY_SIZE - 1)
...
pooled_rows = np.clip(pooled_rows, 0, GEOMETRY_SIZE - 1)
pooled_cols = np.clip(pooled_cols, 0, GEOMETRY_SIZE - 1)
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. The notes say this was chosen to mirror the reference decoder’s use of a per-animal global spatial scale while adapting it to the required 3x3 coarse grid.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and output data are aligned by using the same session boundaries, the same 60 s chunk boundaries, the same per-frame movement mask within each chunk, and the same non-overlapping 3-frame pooling logic. The output labels are therefore on the same filtered pooled sample axis as the neural data.

ii.
```python
chunk_mask = velocity_mask[start:end]
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
...
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. The notes explicitly say the conversion “preserves the simultaneous neural/behavior streams and applies filtering on the same frame axis.”

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles irregularities by dropping or coercing them rather than imputing them. Empty blocked lists become empty arrays, `[-1]` means “no blocked bins,” cells with any non-finite values are removed, coordinates are clipped to the 3x3 range, samples in blocked bins are snapped to the nearest open bin, and bad chunks are discarded. Sessions with no finite cells, no active cells, or fewer than 2 valid trials trigger errors.

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
x = np.clip(x, 0, GEOMETRY_SIZE - 1)
y = np.clip(y, 0, GEOMETRY_SIZE - 1)
...
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")
```

iii. The notes frame these choices as pragmatic cleanup to keep geometry-consistent labels and remove sessions/trials that caused validation warnings or had too little usable movement data.

## 6-a. What are the most time-consuming steps of the code?

i. The code’s main expensive steps are loading each joblib animal file, inferring the per-animal coordinate transform by scoring multiple 3x3 transforms across all sessions, and then per-session preprocessing with movement masking, smoothing, and pooling.

ii.
```python
dat = joblib.load(animal_path)[animal]
...
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
...
speed = gaussian_filter1d(speed, sigma=VELOCITY_SMOOTH_SIGMA, mode="nearest")
...
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. The notes explicitly say the animal joblib loads are “relatively slow,” give a per-animal load-time estimate, and mention transform inference and preprocessing as the main deliberate speed optimizations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several Python loops that could have been vectorized: the per-sample loop in `snap_to_open_bins`, the nested transform/session loops in `infer_transform`, the per-trial loop in `preprocess_session`, and repeated list-building loops such as `build_output_values`.

ii.
```python
for i in range(row_idx.shape[0]):
    if geometry_mat[row_idx[i], col_idx[i]] == 1:
        continue
    distances = np.sum((open_coords - np.array([row_idx[i], col_idx[i]])) ** 2, axis=1)
    nearest = open_coords[np.argmin(distances)]
    row_idx[i] = nearest[0]
    col_idx[i] = nearest[1]
...
for name, transform_fn in TRANSFORMS.items():
    ...
    for sess, occ in enumerate(occupancies):
        ...
for trial_idx in range(n_full_trials):
    ...
```

iii. The notes acknowledge the code is only partially optimized and emphasize vectorized NumPy/Scipy where convenient, but they do not claim these loops were eliminated.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several operations: it copies the same static geometry vector into every trial, looks up `animals.index(animal)` inside the session loop, scores all 8 candidate coordinate transforms for every animal, snaps position bins twice (once session-wide and again after pooling if any pooled bins are blocked), and creates plot payload data for the first valid trial of every session.

ii.
```python
for name, transform_fn in TRANSFORMS.items():
    for sess, occ in enumerate(occupancies):
        ...
...
input_trials.append(geometry_open.copy())
...
subject_idx.append(animals.index(animal))
...
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
...
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
```

iii. The notes justify some of this repetition, especially transform inference and per-trial geometry copies, as the cost of the AI’s geometry-alignment approach and of storing self-contained trial objects.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores plotting-only intermediates that are discarded unless `--show-processing` is used: occupancy summaries, raw trial copies, transform score summaries, and the `SessionPlotPayload`. It also constructs `plot_payload` during session preprocessing even when plotting is disabled, and `full_mode` is computed but never used.

ii.
```python
full_mode = not args.sample
...
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
occupancy_aligned = np.zeros_like(occupancy_raw)
np.add.at(occupancy_aligned, (mapped_rows_all, mapped_cols_all), 1)
occupancy_snapped = np.zeros_like(occupancy_raw)
np.add.at(occupancy_snapped, (snapped_rows_all, snapped_cols_all), 1)
...
if plot_payload is None:
    plot_payload = SessionPlotPayload(
        ...
        raw_trial_neural=trace_active[:, start:end].copy(),
        raw_trial_position=position_session[:, start:end].copy(),
        ...
        occupancy_raw=occupancy_raw.copy(),
        occupancy_aligned=occupancy_aligned.copy(),
        occupancy_snapped=occupancy_snapped.copy(),
        ...
    )
...
if len(processing_payloads) < 2 and args.show_processing:
    processing_payloads.append(plot_payload)
```

iii. The notes describe the plots as sanity checks, but these computations are not part of the exported dataset and are unnecessary for downstream decoding when plotting is off.
