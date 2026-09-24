# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the per-animal **joblib** files in `data/` (the files without a `.mat` extension, e.g. `data/QLAK-CA1-08`), which is the default format used by the reference code's `load_dat`. `get_animal_files` enumerates `data/`, skipping `*.mat`, `behav_dict`, dot-files and directories (`precomputed_results/`), yielding the 7 animal files. Each file is loaded in full with `joblib.load(path)[animal]`, giving a dict whose relevant fields are `trace` (n_sessions, n_cells, n_frames), `position` (n_sessions, 2, n_frames), `blocked` (list per session) and `envs` (n_sessions, 1). Animals are processed sequentially and `del dat; gc.collect()` frees each animal before the next. Trials are not present in the raw data; they are derived (see 1-d). The full run loaded 7 subjects / 207 sessions / 8,109 exported trials.

ii.
```python
def get_animal_files(data_dir):
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if (os.path.isfile(path) and not name.endswith(".mat")
            and name not in {"behav_dict"} and not name.startswith(".")):
            animals.append(name)
    return animals
...
    dat = joblib.load(animal_path)[animal]
...
    for session_idx in range(dat["position"].shape[0]):
        ...
        session_neural, ... = preprocess_session(
            trace_session=dat["trace"][session_idx],
            position_session=dat["position"][session_idx],
            blocked_entry=dat["blocked"][session_idx], ...)
...
        del dat
        gc.collect()
```

iii. From CONVERSION_NOTES.md Step 5, Key Decision 1: *"Use joblib files as the primary raw source: They expose the same field names as the reference code and already match the analysis-ready axis ordering, reducing the risk of silent transpose errors relative to the MATLAB HDF5 layout."* Step 4 also records that `load_dat` treats joblib as the authors' default path, and that the joblib and `.mat` contents are equivalent up to axis order.

## 1-b. How are the data split into subjects?

i. One joblib file = one mouse. The file name is the subject ID; `subjects` is the sorted list of the 7 animal IDs (`QLAK-CA1-08/30/50/51/56/74/75`) and `subject_idx` stores `animals.index(animal)` for every exported session. The verification log confirms 7 subjects with 31/31/31/21/31/31/31 sessions.

ii.
```python
animals = get_animal_files(data_dir)
...
    subject_idx.append(animals.index(animal))
...
    "subjects": animals,
    "subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 2/Step 4: the dataset contains "one dataset per animal" and the reference `main.py` lists exactly these 7 animal IDs, so the per-animal file is the natural subject unit; session order in the converted data follows animal-file order and then day order within an animal.

## 1-c. How are the data split into sessions?

i. Every recording day stored in an animal file becomes one output session: the code iterates `for session_idx in range(dat["position"].shape[0])` and slices `dat["trace"][session_idx]`, `dat["position"][session_idx]`, `dat["blocked"][session_idx]`, `dat["envs"][session_idx, 0]`. No sessions are dropped (207 in = 207 out), and `session_info` metadata records animal, day index, environment label, blocked bins, cell counts and movement fraction per session.

ii.
```python
for session_idx in range(dat["position"].shape[0]):
    session_id = f"{animal}_s{session_idx:02d}"
    env_name = str(dat["envs"][session_idx, 0])
    ...
    session_info.append({"session_id": session_id, "animal": animal,
                         "session_in_animal": session_idx, "environment": env_name, ...})
```
(A session is only rejected if it yields fewer than 2 usable trials: `raise ValueError(f"{session_id}: fewer than 2 valid trials after preprocessing")` — this never triggered.)

iii. CONVERSION_NOTES.md Steps 2–4: each animal file holds 31 daily sessions (21 for `QLAK-CA1-51`), for 207 sessions total, matching the paper's "5,413 unique neurons across 207 sessions in 10 geometries". Each session is one continuous 40-min free-exploration recording, i.e. the natural session unit.

## 1-d. How are the data split into trials?

i. There is no native trial structure. Following the decoder task instruction, each session is cut into consecutive non-overlapping 1-minute chunks of `RAW_TRIAL_FRAMES = 30 * 60 = 1800` raw frames; the trailing remainder (< 1800 frames) is dropped (`n_full_trials = n_frames // 1800`, giving 39–40 chunks per session, 8,187 candidate chunks). Within each chunk, only movement-valid frames are kept and those frames are average-pooled in groups of 3, so exported trials are **variable length** (mean T = 313, median 307, min 10, max 560 pooled samples).

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS   # 30 * 60 = 1800
...
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
    chunk_mask = velocity_mask[start:end]
    ...
    chunk_trace = trace_active[:, start:end][:, chunk_mask]
    chunk_rows = snapped_rows_all[start:end][chunk_mask]
    chunk_cols = snapped_cols_all[start:end][chunk_mask]
    chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
    pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md Step 2: *"There are no native trial objects in the raw dataset. For the requested decoder format, one-minute trials will need to be derived by splitting each session time series into consecutive 1800-frame chunks."* Step 5 Key Decision 3 justifies combining the chunking with reference-style sample selection: *"Split sessions into full one-minute chunks first, but within each chunk keep only movement-valid frames and pool in groups of 3: This satisfies the user's trial-format requirement while remaining consistent with the reference decoder's sample-selection logic."*

## 1-e. How are trials filtered based on quality controls?

i. Three trial-level rejections are applied, all consequences of the movement filter: (1) chunks with fewer than `POOL_SIZE = 3` movement-valid frames are skipped; (2) chunks yielding fewer than `MIN_POOLED_SAMPLES_PER_TRIAL = 10` pooled samples are skipped; (3) chunks whose processed neural matrix is entirely zero are skipped. A session with fewer than 2 surviving trials raises an error (never triggered). Net effect: 8,187 candidate chunks → 8,109 exported trials (0.95% dropped), concentrated in four low-movement sessions (`QLAK-CA1-74_s17/s26/s27`, `QLAK-CA1-75_s01`, movement fractions 0.217–0.351).

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

iii. CONVERSION_NOTES.md Step 7/Step 10: *"Initial sample run produced one all-zero neural trial warning and a minimum pooled length of 4 samples. Fixed by excluding trials with fewer than 10 pooled samples or no processed neural activity."* Step 10 edge-case review confirms the dropped trials coincide with the lowest movement-valid fractions rather than an indexing bug. The paper/reference code prescribe no trial curation (sessions are continuous), so these rules exist only to guarantee non-degenerate trials for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `trace` field of each animal file: frame-aligned, rise-extracted calcium event traces of shape `(n_sessions, n_cells, n_frames)`, binary 0/1 (verified: `np.unique(trace)` = {0, 1}). Unregistered cells for a given day are stored as all-NaN rows. No dF/F or deconvolution is recomputed.

ii.
```python
            session_neural, ... = preprocess_session(
                trace_session=dat["trace"][session_idx], ...)
...
def preprocess_session(trace_session, ...):
    finite_cells = np.isfinite(trace_session).all(axis=1)
    trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md Step 1/Step 3/Step 4: *"Neural data are not raw fluorescence and no delta-F/F computation appears in the reference code. The README states `trace` is already 'rise-extracted calcium traces' where `1` marks a significant event."* and *"The binary rising-phase vector is the neural signal used in all subsequent analyses; this is the signal that should be treated as neural activity for conversion. … Do not recompute dF/F or deconvolution."*

## 2-b. How is the `neural` data processed?

i. The AI reproduces the reference within-session decoder's preprocessing rather than exporting raw 30 Hz binary events:
1. drop cells with any non-finite value in the session, then drop low-activity cells (see 2-c);
2. keep only **movement-valid frames** (smoothed speed > 5 cm/s) inside each 1-minute chunk;
3. Gaussian-smooth each cell's masked trace along time with `sigma = 3` frames (`mode="nearest"`);
4. non-overlapping average pooling over `POOL_SIZE = 3` frames (trailing incomplete group dropped);
5. store as `float32`, shape `(n_active_neurons, n_pooled_timepoints)`.
The exported values are therefore smoothed event rates in [0, 1], not binary events, and ~51% of raw frames survive (`raw_to_valid_frame_fraction=0.5124`).

ii.
```python
def compute_velocity_mask(position_xy, fps=FPS, threshold_cm_s=VELOCITY_THRESHOLD_CM_S):
    delta = np.diff(position_xy, axis=1)
    speed = np.linalg.norm(delta, axis=0) * fps
    speed = gaussian_filter1d(speed, sigma=VELOCITY_SMOOTH_SIGMA, mode="nearest")
    velocity_mask[1:] = speed > threshold_cm_s
    return velocity_mask
...
        chunk_trace = trace_active[:, start:end][:, chunk_mask]
        chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
        pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)

def trial_average_pool(values, pool_size=POOL_SIZE):
    n_full = values.shape[-1] // pool_size
    trimmed = values[..., : n_full * pool_size]
    new_shape = values.shape[:-1] + (n_full, pool_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2: *"Apply the reference decoder's session-level preprocessing before trial export: The paper/code decode position after filtering to movement-valid samples, dropping very-low-activity cells, smoothing traces, and average-pooling every 3 frames. Exporting this processed representation is both closer to the reference decoder and substantially more memory-efficient."* The constants mirror the reference `decode_position_within(..., fps=30, v_filt_size=5, v_thresh=5, cell_threshold=5)` and `fit_decoder(..., temporal_bin_size=3)`, which applies `gaussian_filter1d(traces, sigma=temporal_bin_size)` followed by 3-frame average pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two per-session neuron filters: (1) cells with any non-finite sample in the session are removed (equivalent in this dataset to removing cells not registered that day — verified that all-NaN and any-NaN counts are identical, e.g. 330/515 cells in `QLAK-CA1-08_s00`); (2) the reference decoder's activity filter, keeping cells with more than `CELL_EVENT_THRESHOLD = 5` events summed over movement-valid frames. Result: mean 332.67 active cells/session (range 112–562) vs 336.93 registered non-NaN cells/session, i.e. the activity filter removes ~1.3% of registered cells. Place-cell selection is deliberately *not* applied.

ii.
```python
    finite_cells = np.isfinite(trace_session).all(axis=1)
    trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
    if trace_finite.size == 0:
        raise ValueError(f"{session_id}: no finite cells after NaN filtering")

    activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
    trace_active = trace_finite[activity_mask]
    if trace_active.shape[0] == 0:
        raise ValueError(f"{session_id}: no active cells after activity filtering")
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 4–5: *"Retain all session-valid registered cells initially, then apply the reference decoder's low-activity cell filter per session: This matches the code path in `decode_position_within`"* and *"Drop cells with any NaNs within a session before applying the activity filter … downstream decoder code should not receive NaNs for nonexistent cells."* Step 4 records the paper's statement that high reliability "motivated the inclusion of all cells in subsequent analyses", hence no place-cell restriction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event: the recording is continuous free exploration. Trials are aligned to the start of each consecutive 1-minute chunk of the session, counting from the first frame of the recording. Neural, input and output streams are cut with exactly the same frame indices and the same movement mask, so they are aligned sample-for-sample. The metadata states `temporal_alignment_event = "Start of each consecutive one-minute chunk from a continuous recording session"`, `off_start = 0.0`, `off_end = 60.0` (nominal raw-chunk span; the exported samples cover only the movement-valid subset of that minute).

ii.
```python
        start = trial_idx * RAW_TRIAL_FRAMES
        end = start + RAW_TRIAL_FRAMES
        chunk_mask = velocity_mask[start:end]
        chunk_trace = trace_active[:, start:end][:, chunk_mask]
        chunk_rows = snapped_rows_all[start:end][chunk_mask]
...
        "temporal_alignment_event": "Start of each consecutive one-minute chunk from a continuous recording session",
        "off_start": 0.0,
        "off_end": 60.0,
```

iii. CONVERSION_NOTES.md Steps 3–4: *"Each session is a continuous 40 min free-exploration recording"*, *"Native vs derived trials … Not a contradiction. Conversion will derive 1-minute trials solely to satisfy the target decoder format."* The Gaussian trace smoothing is symmetric, so it introduces no lag between neural and behavior.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning from the native 30 Hz (33.33 ms) to **100 ms bins** by non-overlapping 3-frame average pooling, matching the reference decoder's `temporal_bin_size=3`. `metadata['time_bin_size'] = 100.0` ms, with `native_frame_rate_hz = 30` and `temporal_pool_size_frames = 3` also recorded. Because pooling is applied *after* the movement mask, each bin always aggregates 3 movement-valid frames (100 ms of moving time) but consecutive bins are not necessarily contiguous in wall-clock time.

ii.
```python
FPS = 30
POOL_SIZE = 3
...
        pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
        pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
...
            "time_bin_size": 100.0,
            "temporal_pool_size_frames": POOL_SIZE,
            "native_frame_rate_hz": FPS,
```

iii. CONVERSION_NOTES.md Step 1: *"Temporal binning is by average pooling over 3 frames in both traces and position"* (reference `fit_decoder`/`test_decoder`), and Step 5 Key Decision 2 adopts that binning for the exported data so the converted dataset matches the reference decoder's input representation while also reducing dataset size.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field: a per-session list whose single element is either `-1` (nothing blocked, i.e. the open square) or an array of blocked partition indices in the 3×3 layout. `envs` (the geometry name, e.g. `square`, `o`, `t`, `u`, `+`) is loaded as well but used only as metadata / cross-check, not as the decoder input.

ii.
```python
def parse_blocked_indices(blocked_entry):
    if isinstance(blocked_entry, list):
        if len(blocked_entry) == 0:
            return np.array([], dtype=np.int64)
        blocked_entry = blocked_entry[0]
    arr = np.array(blocked_entry, dtype=np.float64).reshape(-1)
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
    return np.sort(arr.astype(np.int64))
...
                    "environment": env_name,
                    "blocked_bins": parse_blocked_indices(dat["blocked"][session_idx]).tolist(),
```

iii. CONVERSION_NOTES.md Step 4/Step 5 Key Decision 6: *"Use `blocked` for decoder input construction because it directly represents blocked partitions in the raw dataset. Keep `envs` as session metadata / cross-check against code conventions"* and *"avoids ambiguities from canonical geometry labels and plotting/masking orientation conventions"* (the reference `get_env_mat` matrices require a transpose/flip before masking, which the agent explicitly investigated in trajectory step 117).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked` is converted to a 9-element **open-partition** binary vector in row-major 3×3 order (`1 = open`, `0 = blocked`); `-1` maps to all-ones (fully open square). The vector is static per trial (shape `(9,)`), repeated for every trial of the session, and named `partition_0_open … partition_8_open`. Verification shows all nine features in [0, 1] except `partition_7_open`, which is constant 1.0 (partition 7 is never blocked in the dataset).

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
    "geometry_input_encoding": "9 binary features in row-major 3x3 order, 1=open 0=blocked",
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: *"Encode geometry as 9 binary partition features: One feature per 3x3 partition provides the decoder with the full static environmental context. The planned feature semantics are `1 = open`, `0 = blocked`, consistent with the reference code's `get_env_mat` convention."* Step 10 records a sanity check that the converted geometry vectors reconstruct the raw `blocked` lists exactly.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, `(n_sessions, 2, n_frames)` x–y head-tracking coordinates (DeepLabCut) frame-aligned with `trace` at 30 Hz. The per-animal spatial scale is taken as the maximum tracked coordinate across all of that animal's sessions (`np.nanmax(dat["position"])`), which equals 75.0 cm for every animal.

ii.
```python
        scale_cm = float(np.nanmax(dat["position"]))
...
            session_neural, ... = preprocess_session(
                position_session=dat["position"][session_idx], ..., scale_cm=scale_cm, ...)
```

iii. CONVERSION_NOTES.md Step 3: *"Position was derived from DeepLabCut head tracking"*; Step 5 Key Decision 9: *"Derive 3x3 position bins from the same arena scale used by the reference decoder logic … compute a per-animal spatial scale from the maximum tracked coordinate across that animal's sessions … This mirrors the reference code's use of a global max over sessions for position binning"* (reference: `bin_down = (behav_max.max() + buffer) / n_bins`).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Pipeline per session: (1) discretize x and y into 3 equal bins of `(scale_cm + 1e-6)/3 ≈ 25 cm` by flooring and clipping, giving a row-major id `row*3 + col`; (2) apply a per-animal 3×3 **orientation transform** chosen from the 8 dihedral transforms by minimizing occupancy of blocked partitions (`identity` was selected for all 7 animals, confirming the raw position indexing already matches the `blocked` indexing); (3) **snap** any sample falling in a blocked partition to the nearest open partition (Euclidean nearest in grid coordinates); (4) within each trial, keep movement-valid frames, average-pool the row and column indices over 3 frames and floor them, re-clip and re-snap if the pooled bin is blocked; (5) store as `(1, n_timepoints)` int64 class labels. Measured on the raw data, only ~1.3e-6 of frames fall in blocked partitions, so step (3) is effectively a no-op safeguard.

ii.
```python
def raw_position_to_rc(position_xy, bin_size_cm):
    x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
    y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
    return np.clip(y, 0, GEOMETRY_SIZE - 1), np.clip(x, 0, GEOMETRY_SIZE - 1)
...
    bin_size_cm = (scale_cm + 1e-6) / GEOMETRY_SIZE
    raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
    raw_ids_all = raw_rows_all * GEOMETRY_SIZE + raw_cols_all
    mapped_ids_all = coord_map[raw_ids_all]
    mapped_rows_all = mapped_ids_all // GEOMETRY_SIZE
    mapped_cols_all = mapped_ids_all % GEOMETRY_SIZE
    snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
...
        pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
        pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
        invalid = geometry_mat[pooled_rows, pooled_cols] == 0
        if np.any(invalid):
            pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
        pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 9–11 and Step 6: the per-animal scale mirrors the reference `bin_down`; the orientation inference exists because *"the raw `blocked` lists and the canonical matrices in `get_env_mat` differ … if you compare them naively, but the reference code applies a transpose/left-right flip before using those matrices for masking"* (trajectory step 117); the snapping is justified as *"The reference decoder cleans positions/predictions to the nearest valid bin when evaluating decoding. Performing the analogous cleanup at the coarse 3x3 level should reduce label noise from tracking jitter in physically inaccessible regions."* Pooling of the position stream mirrors `fit_decoder`, which pools behavior in the same 3-frame groups as the traces.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. A single categorical output `position_bin` with 9 classes: each axis of the 75 cm arena is split into 3 equal 25 cm bins by `floor(coord / 25.0000003)` with clipping to [0, 2], and the class index is `row*3 + col` (row = y, col = x), i.e. the same row-major indexing as the geometry input. `output_names = ['position_bin']`, `output_values[0] = ['bin_0_r0c0', …, 'bin_8_r2c2']`. Observed class fractions in the full converted data: [0.098, 0.110, 0.113, 0.089, 0.071, 0.086, 0.109, 0.157, 0.166], range [0, 8].

ii.
```python
GEOMETRY_SIZE = 3
...
    bin_size_cm = (scale_cm + 1e-6) / GEOMETRY_SIZE
    x = np.clip(np.floor(position_xy[0] / bin_size_cm).astype(np.int64), 0, GEOMETRY_SIZE - 1)
    y = np.clip(np.floor(position_xy[1] / bin_size_cm).astype(np.int64), 0, GEOMETRY_SIZE - 1)
...
def build_output_values():
    values = []
    for row in range(GEOMETRY_SIZE):
        for col in range(GEOMETRY_SIZE):
            values.append(f"bin_{row * GEOMETRY_SIZE + col}_r{row}c{col}")
    return [values]
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 8 and 11: *"Encode mouse location as one categorical output over time with 9 possible values: This matches the requested decoder task and keeps the output compact and explicitly categorical"*, and *"Use row-major partition labels for outputs … so geometry input and position output share a common indexing scheme."* The `+1e-6` buffer replicates the reference code's `buffer=1e-15` guard so the maximum coordinate does not spill into a 4th bin.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace are natively frame-aligned at 30 Hz. The code applies the *same* `velocity_mask`, the *same* chunk boundaries and the *same* 3-frame pooling grouping to both streams inside `preprocess_session`, so pooled sample *k* of the output corresponds exactly to pooled sample *k* of the neural matrix. Trial lengths are therefore identical for `neural` and `output` by construction, and the verification script reports no dimension warnings.

ii.
```python
        chunk_mask = velocity_mask[start:end]
        chunk_trace = trace_active[:, start:end][:, chunk_mask]
        chunk_rows = snapped_rows_all[start:end][chunk_mask]
        chunk_cols = snapped_cols_all[start:end][chunk_mask]
        pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
        pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Output is time-varying and uses the same temporally pooled samples as the neural data"*; Step 10 sanity check: pooled/snapped 9-class outputs independently reconstructed from raw `position` for three spot-checked (session, trial) pairs matched the converted arrays with `np.allclose()`. The smoothing applied to traces is symmetric (`gaussian_filter1d`), so it does not shift the neural stream relative to position.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) Cells that are not registered in a session appear as all-NaN rows and are removed by the finite-cell mask; an empty result raises an explicit error. (b) `blocked` sentinel `-1` (and an empty list) is mapped to "nothing blocked" instead of being treated as an index. (c) Trailing frames that do not fill a 1800-frame chunk are dropped, as are trailing frames that do not fill a 3-frame pooling group. (d) Degenerate trials (too few movement frames, < 10 pooled samples, all-zero neural) are dropped, and a session with < 2 usable trials raises. (e) Position samples outside the arena bounds or inside blocked partitions are clipped and snapped to valid bins. (f) Session frame-count variability (71,866–72,219 frames) is handled naturally by the floor-division chunking.

ii.
```python
    finite_cells = np.isfinite(trace_session).all(axis=1)
    if trace_finite.size == 0:
        raise ValueError(f"{session_id}: no finite cells after NaN filtering")
...
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
...
    x = np.clip(x, 0, GEOMETRY_SIZE - 1)
...
        if pooled_trace.shape[1] < MIN_POOLED_SAMPLES_PER_TRIAL:
            continue
        if not np.any(pooled_trace):
            continue
```

iii. CONVERSION_NOTES.md Step 4/5: *"Unregistered cells are represented by NaN entries for a given day/session"* and Key Decision 5 (drop NaN cells so *"downstream decoder code should not receive NaNs for nonexistent cells"*). Step 7/10 document the all-zero-trial warning found in the sample run and the resulting minimum-length / non-zero-activity trial rules, and Step 10 documents that the four sessions with reduced trial counts are explained by low movement fractions, not by indexing errors.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the per-animal joblib files dominates: 20–113 s per animal, ~501 s of the 675 s total run (74%). Per-session processing is ~0.6 s (174 s for 207 sessions), split between the session-level Gaussian smoothing/pooling of ~500 × 72,000 traces, the `np.add.at` occupancy accumulations, and the Python-level `snap_to_open_bins` loop. The script prints per-animal load time, per-animal total time and overall elapsed time, and animals are processed sequentially with `gc.collect()` between them to bound memory.

ii.
```python
        animal_load_start = time.time()
        dat = joblib.load(animal_path)[animal]
        load_seconds = time.time() - animal_load_start
        print(f"  loaded in {load_seconds:.2f}s")
...
        print(f"Finished {animal} in {animal_seconds:.2f}s")
...
    print(f"Elapsed time: {elapsed:.2f}s")
```
Observed: `loaded in 61.94s` / `Finished QLAK-CA1-08 in 82.09s` … `Elapsed time: 675.01s`.

iii. CONVERSION_NOTES.md Step 6: *"Animal joblib loads are relatively slow (roughly tens of seconds per animal), so the script processes animals sequentially and frees memory between animals."* Step 7 estimated ~13 min for the full run from the sample timings; the actual run took 11.3 min, inside the 15-minute budget, so no further optimization was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clear candidate is `snap_to_open_bins`, a per-sample Python loop executed over every frame of every session (~72,000 iterations × 207 sessions) even though there are only 9 possible (row, col) values — a 9-entry lookup table computed once per geometry would make it O(1) per session. Secondary candidates: the per-trial loop in `preprocess_session` (masking, smoothing and pooling could be done once per session and then split, since the movement mask and pooling group boundaries are known in advance); the `for new_idx, old_idx in enumerate(transformed.ravel())` loop in `build_coord_map` (trivial, but a direct `argsort`/fancy-index would do); and the nested transform × session loop in `infer_transform`. The rest of the pipeline (velocity, smoothing, pooling, discretization) is already vectorized NumPy/SciPy.

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
    return row_idx, col_idx
```

iii. CONVERSION_NOTES.md Step 6 claims *"Moved heavy preprocessing to vectorized NumPy/Scipy operations"* and *"Performs one transform inference per animal instead of per session"*; the remaining per-frame snapping loop is not identified as a bottleneck in the notes, presumably because the measured runtime (11.3 min) already met the instructions' 15-minute target.

## 6-c. What processing does the code repeat multiple times?

i. (1) 3×3 occupancy is accumulated repeatedly: once per session inside `infer_transform`, then again as `occupancy_raw`, `occupancy_aligned` and `occupancy_snapped` inside `preprocess_session` — four `np.add.at` passes over ~72,000 frames per session. (2) `raw_position_to_rc` is recomputed inside `occupancy_matrix` and again in `preprocess_session`. (3) `parse_blocked_indices` is called 8× per session inside `infer_transform` (once per candidate transform), again in `preprocess_session` via `blocked_to_open_vector`, and once more in `main` for `session_info`. (4) Bin snapping is applied twice — once to all session frames, then again per trial after pooling. (5) `bin_size_cm` is recomputed in `infer_transform` and in `preprocess_session`.

ii.
```python
def infer_transform(position_all, blocked_all, scale_cm):
    occupancies = [occupancy_matrix(position_all[sess], bin_size_cm) for sess in range(position_all.shape[0])]
    for name, transform_fn in TRANSFORMS.items():
        for sess, occ in enumerate(occupancies):
            open_vec, _ = blocked_to_open_vector(blocked_all[sess])   # re-parsed 8x per session
...
    occupancy_raw = occupancy_matrix(position_session, bin_size_cm)   # recomputes raw_position_to_rc
    occupancy_aligned = np.zeros_like(occupancy_raw); np.add.at(occupancy_aligned, (mapped_rows_all, mapped_cols_all), 1)
    occupancy_snapped = np.zeros_like(occupancy_raw); np.add.at(occupancy_snapped, (snapped_rows_all, snapped_cols_all), 1)
```

iii. Not discussed in CONVERSION_NOTES.md. The repeated work is a by-product of the diagnostic machinery (orientation inference and processing plots) being computed unconditionally rather than being gated; the notes only claim the coarser optimization of running transform inference once per animal.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `joblib.load` reads the **entire** animal dict — `SFPs`, `centroids`, `maps['smoothed'/'unsmoothed'/'sampling']` — although only `trace`, `position`, `blocked`, `envs` are used; this is the single largest cost of the run (~500 s and several GB of RAM), and a lazy read (e.g. h5py on the `.mat` files, as the expert solution does) would avoid it. (2) `infer_transform` scores all 8 dihedral transforms over all sessions of every animal and always selects `identity`, so the coordinate remapping (`coord_map`, `mapped_ids_all`) is a no-op in the final dataset. (3) The three occupancy matrices are computed for all 207 sessions but only rendered for at most 2 sessions with `--show-processing`. (4) A `SessionPlotPayload` is constructed for **every** session — copying the full session position array, the velocity mask, a full `(n_active, 1800)` raw trace slice and three occupancy maps — and then discarded unless `--show-processing` was passed. (5) Session-level discretization/snapping is done for all frames, including the ~49% of frames that the movement mask later removes. (6) `session_info` metadata (per-trial pooled lengths for all 8,109 trials) is stored but unused by the decoder.

ii.
```python
        dat = joblib.load(animal_path)[animal]            # loads SFPs / maps / centroids too
        transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
...
        if plot_payload is None:
            plot_payload = SessionPlotPayload(
                ...,
                raw_position=position_session.copy(),
                raw_trial_neural=trace_active[:, start:end].copy(),
                occupancy_raw=occupancy_raw.copy(), ... )
...
            if len(processing_payloads) < 2 and args.show_processing:
                processing_payloads.append(plot_payload)
```

iii. CONVERSION_NOTES.md justifies the orientation inference as a safeguard against a mirrored geometry encoding (the reference code transposes/flips `get_env_mat` before masking) and reports that `identity` was chosen for all animals — i.e. the check was worth running once, but its per-animal, per-session cost is not needed in production. The unconditional plot-payload construction and the full-dict joblib load are not discussed in the notes.
