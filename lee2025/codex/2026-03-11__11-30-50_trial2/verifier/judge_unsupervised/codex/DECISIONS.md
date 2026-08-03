# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script enumerates animal-level joblib files from `data/`, excluding `.mat` files and `behav_dict`, then loads each animal with `joblib.load(animal_path)[animal]`. It does not read the MATLAB files.

ii. ```python
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

iii. In `CONVERSION_NOTES.md`, the agent says it chose joblib because it matches the reference code’s storage path and avoids MATLAB axis-order mistakes.

## 1-b. How are the data split into subjects?

i. Subjects are the animal files. Sessions are grouped by animal-file order, and `subject_idx` is built by appending the index of the current animal for every exported session.

ii. ```python
animals = get_animal_files(data_dir)
...
for animal in selected_animals:
    ...
    for session_idx in range(dat["position"].shape[0]):
        ...
        subject_idx.append(animals.index(animal))
...
"subjects": animals,
"subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. The notes say session order follows animal-file order, then day/session order within each animal, to mirror the reference dataset organization.

## 1-c. How are the data split into sessions?

i. Each raw session/day is one index along the first axis of `position`, `trace`, `blocked`, and `envs`. The script loops `session_idx` over `dat["position"].shape[0]` and exports one converted session per raw session.

ii. ```python
for session_idx in range(dat["position"].shape[0]):
    session_id = f"{animal}_s{session_idx:02d}"
    env_name = str(dat["envs"][session_idx, 0])
    ...
    preprocess_session(
        trace_session=dat["trace"][session_idx],
        position_session=dat["position"][session_idx],
        blocked_entry=dat["blocked"][session_idx],
```

iii. The notes describe the raw dataset as continuous 40 min recordings, one session per day, with no native trial objects.

## 1-d. How are the data split into trials?

i. Trials are derived, not native. Each session is cut into consecutive 1-minute chunks of `1800` frames (`30 Hz * 60 s`) using floor division; any remainder frames at the end of a session are discarded.

ii. ```python
FPS = 30
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
...
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
```

iii. The notes say this was done only because the target format required trials while the reference dataset is continuous-session based.

## 1-e. How are trials filtered based on quality controls?

i. A derived 1-minute chunk is dropped if it has fewer than 3 movement-valid frames, fewer than 10 pooled samples after preprocessing, or an all-zero processed neural matrix. A whole session is rejected if fewer than 2 valid trials remain.

ii. ```python
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

iii. The notes justify this as fixing verification warnings and enforcing usable trials after movement/neural-signal curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from `trace`, one session at a time.

ii. ```python
preprocess_session(
    trace_session=dat["trace"][session_idx],
    position_session=dat["position"][session_idx],
```

iii. The notes say `trace` is already the reference paper’s binary rising-phase event signal, so no dF/F or deconvolution is recomputed.

## 2-b. How is the `neural` data processed?

i. Per session, the script drops non-finite cells, keeps only cells with more than 5 events during movement-valid frames, applies a Gaussian temporal filter (`sigma=3`) to each chunk, then average-pools in non-overlapping 3-frame bins.

ii. ```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
...
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. The notes say this was meant to mirror the reference within-session decoder’s preprocessing and reduce dataset size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered twice: first by finite-valued traces within the session, then by activity threshold during movement-valid samples (`>5` events). Trials with no usable processed neural signal are also dropped.

ii. ```python
finite_cells = np.isfinite(trace_session).all(axis=1)
...
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
...
if not np.any(pooled_trace):
    continue
```

iii. The notes explicitly tie this to the reference decoder’s movement and low-activity-cell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata says trials are aligned to the start of each consecutive 1-minute chunk of the continuous session. In practice, only movement-valid frames inside each chunk are retained before pooling, so the stored time axis is a filtered subset of that chunk.

ii. ```python
"temporal_alignment_event": "Start of each consecutive one-minute chunk from a continuous recording session",
"off_start": 0.0,
"off_end": 60.0,
...
chunk_trace = trace_active[:, start:end][:, chunk_mask]
pooled_trace = trial_average_pool(chunk_trace)
```

iii. The notes frame the chunk start as the derived trial boundary required by the target format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported data are at 100 ms resolution because the script pools every 3 native 30 Hz frames. Yes, temporal rebinning is applied.

ii. ```python
FPS = 30
POOL_SIZE = 3
...
pooled_trace = trial_average_pool(chunk_trace)
...
"time_bin_size": 100.0,
"native_frame_rate_hz": FPS,
"temporal_pool_size_frames": POOL_SIZE,
```

iii. The notes say this follows the reference decoder’s 3-frame temporal pooling.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry input is derived from the raw `blocked` field, not from `envs`.

ii. ```python
def blocked_to_open_vector(blocked_entry):
    blocked_idx = parse_blocked_indices(blocked_entry)
    open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
    open_vec[blocked_idx] = 0.0
    return open_vec, blocked_idx
...
blocked_entry=dat["blocked"][session_idx],
```

iii. The notes say `blocked` directly encodes which 3x3 partitions are occluded and avoids ambiguity from environment-label orientation conventions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The script parses `blocked`, treats `-1` as no blocked bins, and converts each session’s geometry into a static 9-element binary vector with `1=open` and `0=blocked`. The same vector is copied into every trial of the session.

ii. ```python
def parse_blocked_indices(blocked_entry):
    ...
    if arr.size == 1 and arr[0] < 0:
        return np.array([], dtype=np.int64)
...
open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
open_vec[blocked_idx] = 0.0
...
input_trials.append(geometry_open.copy())
```

iii. The notes describe this as the decoder input most directly requested by the task.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output position is derived from the raw `position` array.

ii. ```python
preprocess_session(
    ...
    position_session=dat["position"][session_idx],
```

iii. The notes say the raw position comes from DeepLabCut head tracking and is frame-aligned with `trace`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The script bins x/y coordinates into a 3x3 grid using a per-animal global max coordinate as scale, applies an inferred coordinate transform chosen by occupancy-versus-blocked-bin scoring, snaps blocked-bin samples to the nearest open bin, filters to movement-valid frames, pools rows/cols in groups of 3 frames, then converts pooled row/col back to a 9-bin categorical output.

ii. ```python
scale_cm = float(np.nanmax(dat["position"]))
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
coord_map = build_coord_map(transform_name)
...
raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
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

iii. The notes justify the transform as aligning occupancy to raw blocked geometry and the snapping as reducing label noise from tracking jitter.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 row-major categories, indexed as `row * 3 + col`.

ii. ```python
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :].astype(np.int64)
...
def build_output_values():
    values = []
    for row in range(GEOMETRY_SIZE):
        for col in range(GEOMETRY_SIZE):
            idx = row * GEOMETRY_SIZE + col
            values.append(f"bin_{idx}_r{row}c{col}")
```

iii. The notes say the task required `3 x 3 = 9` spatial bins and that geometry/input/output should share one indexing convention.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output uses the same chunk boundaries, the same movement-valid mask, and the same 3-frame pooling as `neural`, so each trial’s position stream has the same number of time bins as the neural matrix.

ii. ```python
chunk_mask = velocity_mask[start:end]
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
...
pooled_trace = trial_average_pool(chunk_trace)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_cols = np.floor(trial_average_pool(chunk_cols[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
```

iii. The notes say this was chosen to mirror the reference decoder’s sample-selection logic.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid cells are dropped via finite-value filtering; `blocked=-1` becomes “no blocked bins”; empty/short/no-signal trials are dropped; clipped position bins are forced into `[0, 2]`; and samples in blocked bins are reassigned to the nearest open bin.

ii. ```python
if arr.size == 1 and arr[0] < 0:
    return np.array([], dtype=np.int64)
...
finite_cells = np.isfinite(trace_session).all(axis=1)
...
if int(chunk_mask.sum()) < POOL_SIZE:
    continue
...
pooled_rows = np.clip(pooled_rows, 0, GEOMETRY_SIZE - 1)
pooled_cols = np.clip(pooled_cols, 0, GEOMETRY_SIZE - 1)
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
```

iii. The notes describe this as edge-case handling and, for snapping, as a way to reduce tracking-jitter label noise.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is loading the animal joblib files; per-animal logs show loads on the order of 20 to 113 seconds. The notes also mention large-pickle I/O as a bottleneck during sanity checking.

ii. ```python
animal_load_start = time.time()
dat = joblib.load(animal_path)[animal]
load_seconds = time.time() - animal_load_start
print(f"  loaded in {load_seconds:.2f}s")
```

iii. In the notes, the agent explicitly says sequential joblib loading is the main slowdown; `conversion_full_out.txt` shows this in practice.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest remaining scalar loops are: per-sample nearest-open snapping in `snap_to_open_bins`, the per-transform/per-session scoring loop in `infer_transform`, and the per-trial chunk loop in `preprocess_session`.

ii. ```python
for name, transform_fn in TRANSFORMS.items():
    ...
    for sess, occ in enumerate(occupancies):
...
for i in range(row_idx.shape[0]):
    if geometry_mat[row_idx[i], col_idx[i]] == 1:
        continue
...
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
```

iii. The notes claim the heavy parts were vectorized where possible, but these loops remain.

## 6-c. What processing does the code repeat multiple times?

i. The code snaps positions twice (once at frame level, once again after pooling), smooths and pools separately inside every trial chunk rather than once session-wide, and recomputes transform scores across all sessions for every animal.

ii. ```python
snapped_rows_all, snapped_cols_all = snap_to_open_bins(mapped_rows_all, mapped_cols_all, geometry_open)
...
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(pooled_rows, pooled_cols, geometry_open)
...
chunk_trace = gaussian_filter1d(chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest")
pooled_trace = trial_average_pool(chunk_trace)
...
for name, transform_fn in TRANSFORMS.items():
    for sess, occ in enumerate(occupancies):
```

iii. The notes call out one transform inference per animal as a speedup, which implies this scoring pass was recognized as nontrivial work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter always computes raw/aligned/snapped occupancy summaries and creates a `plot_payload` object for the first valid trial of every session even when `--show-processing` is off. It also computes transform scores for all candidate transforms even though only the winning transform is used downstream.

ii. ```python
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
occupancy_aligned = np.zeros_like(occupancy_raw)
occupancy_snapped = np.zeros_like(occupancy_raw)
...
plot_payload = SessionPlotPayload(...)
...
if len(processing_payloads) < 2 and args.show_processing:
    processing_payloads.append(plot_payload)
...
for name, transform_fn in TRANSFORMS.items():
    ...
```

iii. The notes present these computations as support for processing plots and sanity checks rather than for the final exported dataset.
