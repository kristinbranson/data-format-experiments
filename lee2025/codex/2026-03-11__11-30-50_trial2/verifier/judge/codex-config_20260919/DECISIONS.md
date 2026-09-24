# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sorted, extensionless per-animal joblib files in `data/` (excluding `.mat`, `behav_dict`, and hidden files), loads each with `joblib.load`, and extracts the dictionary keyed by animal ID. Full mode visits every discovered animal and every session; sample mode uses the first animal and stops after two sessions.

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

dat = joblib.load(animal_path)[animal]
for session_idx in range(dat["position"].shape[0]):
```

iii. The AI says the joblib files expose the same analysis-ready fields and axis order as the paper code, avoiding MATLAB/HDF5 transpose errors. It validated 7 animals and 207 sessions.

## 1-b. How are the data split into subjects?

i. Each discovered joblib filename is treated as one mouse ID. `subjects` is the complete sorted filename list, and each exported session receives the corresponding index.

ii.
```python
animals = get_animal_files(data_dir)
for animal in selected_animals:
    dat = joblib.load(os.path.join(data_dir, animal))[animal]
    ...
    subject_idx.append(animals.index(animal))
...
"subjects": animals,
```

iii. The AI found that the reference analysis uses seven per-animal joblib objects and states that output order follows animal-file order, then day/session order.

## 1-c. How are the data split into sessions?

i. The first axis of each animal's `position`, `trace`, `blocked`, and `envs` arrays is treated as the session/day axis. Each index becomes one output session.

ii.
```python
for session_idx in range(dat["position"].shape[0]):
    session_neural, session_input, session_output, ... = preprocess_session(
        trace_session=dat["trace"][session_idx],
        position_session=dat["position"][session_idx],
        blocked_entry=dat["blocked"][session_idx],
        env_name=str(dat["envs"][session_idx, 0]),
        ...
    )
```

iii. The AI justified this using the analysis-ready shapes and reproduced the expected total of 207 sessions.

## 1-d. How are the data split into trials?

i. Each continuous session is first divided into non-overlapping 1-minute, 1,800-frame chunks; trailing incomplete frames are discarded. Within each chunk, only movement-valid frames are retained and groups of three retained samples are pooled. Chunks can then be discarded for insufficient movement, fewer than 10 pooled samples, or all-zero processed activity, so exported trial lengths vary.

ii.
```python
n_full_trials = trace_session.shape[1] // RAW_TRIAL_FRAMES
for trial_idx in range(n_full_trials):
    start = trial_idx * RAW_TRIAL_FRAMES
    end = start + RAW_TRIAL_FRAMES
    chunk_mask = velocity_mask[start:end]
    if int(chunk_mask.sum()) < POOL_SIZE:
        continue
    chunk_trace = trace_active[:, start:end][:, chunk_mask]
    pooled_trace = trial_average_pool(chunk_trace)
```

iii. The one-minute chunks satisfy the requested derived trial structure. The AI additionally chose movement selection and three-frame pooling to imitate the paper's within-session decoder, yielding 8,109 exported trials rather than 8,187 raw full chunks.

## 1-e. How are trials filtered based on quality controls?

i. A chunk is rejected if it has fewer than three movement-valid frames, produces fewer than ten pooled samples, or has no nonzero pooled neural activity. A whole session raises an error if fewer than two trials survive.

ii.
```python
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

iii. The AI added these checks after sample validation found an all-zero trial and a trial with only four pooled samples. It describes reduced trial counts in low-movement sessions as intentional quality curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural output is derived from each session of the joblib `trace` array, which the AI interprets as the paper's already-extracted binary rising-phase calcium-event signal.

ii.
```python
trace_session=dat["trace"][session_idx]
```

iii. The AI concluded from the paper, code, and sampled values that `trace` is already the final event representation; it therefore does not recompute fluorescence, dF/F, or deconvolution.

## 2-b. How is the `neural` data processed?

i. After session-level cell filtering, the AI selects only movement-valid frames separately inside each minute, Gaussian-smooths each neuron's retained sequence with sigma 3 samples, average-pools non-overlapping groups of three, and stores float32 neuron-by-pooled-sample matrices.

ii.
```python
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_trace = gaussian_filter1d(
    chunk_trace, sigma=TRACE_SMOOTH_SIGMA, axis=1, mode="nearest"
)
pooled_trace = trial_average_pool(chunk_trace).astype(np.float32, copy=False)
```

iii. The AI says movement filtering, smoothing, and three-frame pooling mirror the reference paper code's position-decoder path and reduce storage and sample noise.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells must be finite at every frame in the session, then must have more than five events over movement-valid frames. Sessions with no surviving cells fail.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
trace_finite = trace_session[finite_cells].astype(np.float32, copy=False)
activity_mask = np.sum(trace_finite[:, velocity_mask], axis=1) > CELL_EVENT_THRESHOLD
trace_active = trace_finite[activity_mask]
```

iii. The AI treats NaNs as unrecorded registered cells and says the additional activity threshold matches the within-session decoder more closely than exporting all session-valid cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. The alignment event is defined as the start of each consecutive one-minute chunk. Neural and position use the same raw chunk boundaries, movement mask, and pooling groups, although removed stationary frames make the exported time axis discontinuous relative to wall-clock time.

ii.
```python
start = trial_idx * RAW_TRIAL_FRAMES
end = start + RAW_TRIAL_FRAMES
chunk_mask = velocity_mask[start:end]
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
```

iii. The AI notes that the source sessions are continuous and have no native trials or stimulus onset, so the instruction-mandated minute boundary is the only alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native data are 30 Hz, but the AI reports 100 ms output bins after average-pooling groups of three movement-valid frames. Because stationary frames are removed before pooling, these samples are not necessarily contiguous 100 ms wall-clock intervals and trial lengths vary.

ii.
```python
FPS = 30
POOL_SIZE = 3
pooled_trace = trial_average_pool(chunk_trace)
...
"time_bin_size": 100.0,
"temporal_pool_size_frames": POOL_SIZE,
```

iii. The AI justified three-frame pooling as matching the reference decoder and reducing memory. Its notes call the result 100 ms pooled resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the session's raw `blocked` entry; `envs` is retained only as metadata/cross-check information.

ii.
```python
geometry_open, blocked_idx = blocked_to_open_vector(blocked_entry)
...
blocked_entry=dat["blocked"][session_idx],
env_name = str(dat["envs"][session_idx, 0])
```

iii. The AI chose `blocked` because it directly specifies which of the nine partitions are occluded, avoiding reliance on environment-name conventions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Negative sentinel values mean no blocked bins. Otherwise indices are sorted and used to turn a length-nine all-ones vector into an open-space mask (`1=open`, `0=blocked`). The static vector is copied into every surviving trial.

ii.
```python
if arr.size == 1 and arr[0] < 0:
    return np.array([], dtype=np.int64)
...
open_vec = np.ones(GEOMETRY_SIZE * GEOMETRY_SIZE, dtype=np.float32)
open_vec[blocked_idx] = 0.0
...
input_trials.append(geometry_open.copy())
```

iii. The AI says this follows the reference code's canonical geometry-mask convention and supplies independent binary features in the same row-major index system as position.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each session's two-coordinate `position` array.

ii.
```python
position_session=dat["position"][session_idx]
...
raw_rows_all, raw_cols_all = raw_position_to_rc(position_session, bin_size_cm)
```

iii. The AI identifies this as frame-aligned DeepLabCut head position and uses it as the behavioral target.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI computes a per-animal maximum coordinate as arena scale, bins x/y into a raw 3x3 grid, infers one of eight rotations/reflections per animal by minimizing occupancy in blocked bins, maps coordinates into that orientation, snaps labels in blocked bins to the nearest open bin, selects movement frames, averages row and column coordinates in groups of three, floors them, re-snaps if needed, and emits one row-major class stream.

ii.
```python
scale_cm = float(np.nanmax(dat["position"]))
transform_name, transform_scores = infer_transform(dat["position"], dat["blocked"], scale_cm)
...
mapped_ids_all = coord_map[raw_ids_all]
snapped_rows_all, snapped_cols_all = snap_to_open_bins(...)
...
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0]).astype(np.int64)
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :]
```

iii. The AI intended the global per-animal scale and three-frame pooling to mirror reference decoder logic. Transform inference and snapping were added to reconcile storage orientation and suppress tracking labels in physically inaccessible partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into three equal-width intervals based on `(per-animal maximum + 1e-6) / 3`; floor-and-clip produces indices 0–2, and the final category is `row * 3 + column` (0–8). After orientation mapping, blocked categories are replaced by the nearest open category.

ii.
```python
bin_size_cm = (scale_cm + 1e-6) / GEOMETRY_SIZE
x = np.floor(position_xy[0] / bin_size_cm).astype(np.int64)
y = np.floor(position_xy[1] / bin_size_cm).astype(np.int64)
x = np.clip(x, 0, GEOMETRY_SIZE - 1)
y = np.clip(y, 0, GEOMETRY_SIZE - 1)
...
pooled_bins = (pooled_rows * GEOMETRY_SIZE + pooled_cols)[np.newaxis, :]
```

iii. The AI says the scheme gives the requested nine categories and shares row-major indexing with the geometry input; snapping is intended to remove boundary/tracking jitter.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position begin with identical minute boundaries, are indexed by the same movement mask, and are pooled in groups of three. Thus every output label corresponds to the same retained/pooled samples as a neural column.

ii.
```python
chunk_trace = trace_active[:, start:end][:, chunk_mask]
chunk_rows = snapped_rows_all[start:end][chunk_mask]
chunk_cols = snapped_cols_all[start:end][chunk_mask]
pooled_trace = trial_average_pool(chunk_trace)
pooled_rows = np.floor(trial_average_pool(chunk_rows[np.newaxis, :].astype(np.float32))[0])
```

iii. The AI independently reconstructed neural and output trials from raw data and reports exact spot-check agreement, supporting sample-for-sample alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Cells containing any nonfinite session value are removed. Negative blocked sentinels become an empty blocked list. Coordinates are clipped to grid bounds and labels falling in blocked space are snapped to the nearest open cell. Incomplete minute tails and low-quality chunks are discarded; fatal empty-cell or too-few-trial cases raise errors.

ii.
```python
finite_cells = np.isfinite(trace_session).all(axis=1)
...
if arr.size == 1 and arr[0] < 0:
    return np.array([], dtype=np.int64)
...
x = np.clip(x, 0, GEOMETRY_SIZE - 1)
...
if geometry_mat[row_idx[i], col_idx[i]] == 0:
    ...  # replace with nearest open coordinate
```

iii. The AI characterizes NaNs as absent registered cells, clipping/snapping as protection from tracking boundary noise, and chunk rejection as necessary to avoid invalid/all-zero decoder samples.

## 6-a. What are the most time-consuming steps of the code?

i. Per-animal joblib loading is the dominant measured cost (tens to more than 100 seconds per animal); session preprocessing is secondary. Full conversion took about 675 seconds.

ii.
```python
animal_load_start = time.time()
dat = joblib.load(animal_path)[animal]
load_seconds = time.time() - animal_load_start
...
started = time.time()
```

iii. The notes explicitly identify joblib loads as relatively slow and report measured sample/full timings. Sequential processing and `del dat; gc.collect()` limit memory pressure rather than improving I/O latency.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most heavy operations are vectorized, but `snap_to_open_bins` loops over every sample and could vectorize blocked-sample detection and nearest-open-distance assignment. `build_coord_map`'s nine-element loop is also vectorizable but immaterial. Session/trial loops reflect heterogeneous output nesting and are less straightforward to eliminate.

ii.
```python
for i in range(row_idx.shape[0]):
    if geometry_mat[row_idx[i], col_idx[i]] == 1:
        continue
    distances = np.sum((open_coords - np.array([row_idx[i], col_idx[i]])) ** 2, axis=1)
    nearest = open_coords[np.argmin(distances)]
```

iii. The AI's notes emphasize that NumPy/SciPy vectorization was used for heavy preprocessing, but they do not explicitly identify this remaining per-position snapping loop.

## 6-c. What processing does the code repeat multiple times?

i. Position-to-grid occupancy is computed once for transform inference and again in each session. Snapping is performed for the complete session and conditionally repeated after coordinate pooling. The three-frame pooling helper is separately invoked for neural activity, rows, and columns in every trial.

ii.
```python
occupancies = [occupancy_matrix(position_all[sess], bin_size_cm) ...]
...
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
snapped_rows_all, snapped_cols_all = snap_to_open_bins(...)
...
if np.any(invalid):
    pooled_rows, pooled_cols = snap_to_open_bins(...)
```

iii. The AI documents one transform inference per animal as a speedup over per-session inference, but does not explicitly discuss these remaining repeated computations.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Even without `--show-processing`, every session computes three occupancy matrices and constructs a large first-trial `SessionPlotPayload`; only up to two payloads are retained when plotting is requested, so most/all of this work and copied data are discarded. Transform scores and several metadata statistics are diagnostic rather than decoder inputs.

ii.
```python
occupancy_raw = occupancy_matrix(position_session, bin_size_cm)
occupancy_aligned = np.zeros_like(occupancy_raw)
occupancy_snapped = np.zeros_like(occupancy_raw)
...
if plot_payload is None:
    plot_payload = SessionPlotPayload(... raw_position=position_session.copy(), ...)
...
if len(processing_payloads) < 2 and args.show_processing:
    processing_payloads.append(plot_payload)
```

iii. The notes describe plots and occupancy checks as validation aids but do not acknowledge that their payloads are built unconditionally. They otherwise claim heavy preprocessing was vectorized and memory was reduced through filtering and pooling.
