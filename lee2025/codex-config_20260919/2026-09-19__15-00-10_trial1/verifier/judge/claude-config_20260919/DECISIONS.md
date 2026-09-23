# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs taken from the reference repository's `main.py` (`QLAK-CA1-08, -30, -50, -51, -56, -74, -75`) and loads each animal's **extensionless joblib file** from `/app/data/` with `joblib.load`, which is the default format of the reference loader `utils.load_dat` (`format="joblib"`). Animals are loaded strictly one at a time inside the outer loop, and `del source; gc.collect()` is called after each animal to bound peak memory. The `.mat` (HDF5) twins of these files are deliberately not used, although the AI separately cross-checked sampled `.mat` values against the joblib values with `h5py` + `np.allclose` as a loading sanity check. Only the `trace`, `position`, `blocked` and `envs` fields are touched; `SFPs`, `centroids` and `maps` are ignored. The result is 7 subjects → 207 sessions → 8,187 one-minute trials → 69,744 session-neuron rows.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_index, animal in enumerate(ANIMALS):
    if stop:
        break
    load_start = time.perf_counter()
    source = joblib.load(DATA_DIR / animal)[animal]
    print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f} s", flush=True)
    ndays = source["trace"].shape[0]
    for day in range(ndays):
        ...
        trace = source["trace"][day]
        position = source["position"][day]
        ...
        blocked = blocked_mask_x_y(source["blocked"][day])
    del source
    gc.collect()
```

iii. From CONVERSION_NOTES Step 1/2: "The seven animals are explicitly listed in `main.py`"; "Joblib is the reference code's default and preserves a top-level `{animal_id: dataset}` dictionary." Step 10 Check 6: "Used `h5py` to dereference sampled original `.mat` trace and position arrays and compared them to joblib values with `np.allclose` (including NaN handling). PASS, confirming reference default joblib loading preserves source values." Sequential loading is justified in Step 6 as a memory bound: "Loading every animal simultaneously would additionally materialize tens of GB of compressed joblib arrays."

## 1-b. How are the data split into subjects?

i. One subject per source file / animal ID. `subjects` is the fixed 7-element `ANIMALS` list, and `subject_idx` records the index of the animal for each emitted session. The subject list is always all 7 names even in `--sample` mode (where only animal 0 contributes sessions). Each session's `metadata.session_info` entry also stores `subject` and `session_id = f"{animal}_day{day:02d}"`.

ii.
```python
subject_idx.append(animal_index)
...
"subjects": ANIMALS.copy(),
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The subject identity is the animal ID; the AI took the canonical list from the reference `main.py` rather than globbing the directory, noting in the Step 5 mapping table: "Seven canonical IDs; per-day session points to subject … `main.py` animal list. Session order: listed animal order, then chronological day." Verification confirms 31/31/31/21/31/31/31 sessions per subject, matching the reference count of 207.

## 1-c. How are the data split into sessions?

i. One target session per **recording day** per animal. The number of days is read from `source["trace"].shape[0]` (the leading axis of the `(n_days, n_cells, n_frames)` array), and the loop emits one session per day, in chronological order within an animal. The environment name for the day (`envs`) is stored as descriptive metadata only. This yields 207 sessions (31 for six animals, 21 for QLAK-CA1-51).

ii.
```python
ndays = source["trace"].shape[0]
for day in range(ndays):
    ...
    trace = source["trace"][day]
    position = source["position"][day]
    ...
    session_id = f"{animal}_day{day:02d}"
    ...
    environment = str(np.asarray(source["envs"])[day].squeeze())
    session_info.append({
        "session_id": session_id, "subject": animal, "source_day_index": day,
        "environment": environment, ...
    })
```

iii. CONVERSION_NOTES Step 4: "every recording day becomes one target session". This matches the experimental design documented in methods.txt ("All sessions were 40 min, and one session was recorded per day") and reproduces the paper's headline count of 207 sessions.

## 1-d. How are the data split into trials?

i. Per the decoder-task instruction, each continuous session is cut from frame 0 into consecutive, non-overlapping 60-second segments. Because the AI pools 3 source frames into one output bin (see 2-e), a trial is 1,800 source frames = **600 output bins**. The number of trials is `floor(n_source_frames / 1800)`; the terminal partial minute (2.0–55.5 s depending on session) is dropped. Neural, input, and output are sliced on exactly the same boundaries. Result: 39 trials for the three animals with 71,866 frames and 40 for the rest; 8,187 trials total.

ii.
```python
RAW_TRIAL_FRAMES = SOURCE_HZ * TRIAL_SECONDS          # 1800
TRIAL_TIMEPOINTS = int(TARGET_HZ * TRIAL_SECONDS)     # 600
...
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
pooled_neural = pool_neural(trace, registered, n_raw_keep)
pooled_xy, position_classes = pool_position(position, n_raw_keep)
...
assert pooled_neural.shape[1] == position_classes.size == ntrials * TRIAL_TIMEPOINTS
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
    input_trials.append(blocked.copy())
    output_trials.append(position_classes[None, start:end].copy())
```

iii. Step 5 Key Decision 2: "**Exact one-minute trials**: Start at session frame zero, use consecutive 1,800-frame (600 pooled-bin) chunks, and drop only the terminal 2.0-55.5 s partial segment. Fixed-length trials satisfy the requirement and preserve 8,187/8,187 possible complete minutes." Step 10 Edge cases: "terminal partial frames are the only excluded samples and are recorded per session" (`dropped_terminal_frames` in `session_info`).

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality filtering is applied.** The only trials removed are the incomplete terminal segments (a length requirement, not a quality control). The AI explicitly identified and then explicitly rejected the reference code's within-session decoder filters (`decode_position_within`: speed > 5 cm/s and > 5 events per cell), because applying a running-speed mask would destroy contiguous one-minute trials and would delete exactly the stationary position labels the categorical output is supposed to represent. An assertion guarantees that every session contributes at least 2 trials, satisfying the format requirement.

ii.
```python
# no trial rejection; only complete segments are emitted
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
...
"trial_filter": "all complete consecutive 60-s segments; terminal partial minute omitted",
...
def validate_built_data(data, expected_sessions):
    ...
        assert len(neural_trials) >= 2
```

iii. Step 3 Curation: "No trial rejection is described. Each session is a continuous 40-minute free-exploration recording." Step 4 discrepancy table: "Requested time-varying categorical output should retain all frames and registered cells so trials represent full location occupancy; otherwise outputs would cease to be contiguous one-minute trials." Step 5 Key Decision 3 repeats: "Running-only filtering would destroy contiguous one-minute timing and remove stationary position labels."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely the `trace` field, i.e. the authors' rise-extracted, **binarized** calcium event traces, shape `(n_days, n_global_cells, n_frames)` for each animal. Unregistered cell/day pairs are stored as all-NaN rows. No raw fluorescence, no ΔF/F, no deconvolution, and no rate maps (`maps`) are used; the AI explicitly rejected exporting `maps` because those are derived from the same `trace`/`position` and would leak the decoding target.

ii.
```python
trace = source["trace"][day]
registered = np.isfinite(trace[:, 0])
assert registered.any()
assert np.isfinite(trace[registered]).all()
assert np.isin(trace[registered], (0.0, 1.0)).all()
...
"neural_signal": "binarized significant calcium-transient rising phases",
```

iii. Step 1 notes: "Native neural data are already processed **rise-extracted calcium event traces**, with 1 denoting a significant event and NaN denoting a cell not registered that day. Delta-F/F must not be recomputed; the raw fluorescence needed for that operation is not the reference decoder input." Step 4: methods.txt states "This binary vector was treated as the firing rate in all further analyses", confirming exact agreement. Step 5 mapping: `maps` "Do not export; derived from the same trace/position and would leak target spatial information if used as inputs."

## 2-b. How is the `neural` data processed?

i. Three operations, applied once per session over the whole continuous recording *before* trial slicing:
1. select the registered (finite) cell rows and cast to `float32`;
2. `scipy.ndimage.gaussian_filter1d(..., sigma=3 source frames, axis=time)`;
3. non-overlapping 3-frame mean pooling (equivalent to `torch.nn.AvgPool1d(kernel_size=3, stride=3)`), after truncating to `n_raw_keep`.

This reproduces exactly the preprocessing that the reference repository's own position decoder applies (`fit_decoder`/`test_decoder` in `utils.py:1776/1806`: `gaussian_filter1d(traces, sigma=temporal_bin_size=3, axis=0)` followed by `AvgPool1d(kernel_size=3, stride=3)`). Smoothing is done over the full session so that trial boundaries do not create edge artifacts. Output is `float32`, contiguous, `(n_registered, 600)` per trial. I independently reproduced this from the raw joblib files: `np.allclose` passes exactly for both sample sessions.

ii.
```python
SMOOTH_SIGMA_FRAMES = 3.0
POOL_FRAMES = 3

def pool_neural(trace, registered, n_raw_keep):
    """Apply the reference decoder's sigma-3 smoothing and three-frame pooling."""
    source = np.asarray(trace[registered], dtype=np.float32)
    smoothed = np.empty_like(source, dtype=np.float32)
    gaussian_filter1d(source, sigma=SMOOTH_SIGMA_FRAMES, axis=1, output=smoothed)
    del source
    # Smooth over the full acquired session so trial boundaries do not create artifacts,
    # then omit the terminal incomplete minute and pool exactly as AvgPool1d does.
    pooled = smoothed[:, :n_raw_keep].reshape(
        smoothed.shape[0], -1, POOL_FRAMES
    ).mean(axis=2, dtype=np.float32)
    return np.ascontiguousarray(pooled, dtype=np.float32)
```

iii. Step 5 Key Decision 1: "**Temporal bins are 100 ms**: The reference position decoder Gaussian-smooths trace events with sigma equal to 3 frames and applies non-overlapping 3-frame average pooling to both neural and position streams. Applying this once over each full session before trial slicing exactly matches that processing, avoids artificial trial-edge smoothing discontinuities, and keeps the full dataset tractable (~6.7 GB neural float32 rather than ~20 GB at 30 Hz)." Step 10 comparison table row (d) records "Exact logic; 100 ms bins".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only unregistered cells are removed. A cell is "registered" for a day iff its trace is finite at frame 0; the code then asserts that the whole selected block is finite and strictly binary. All registered cells are kept — no place-cell selection and no activity-sparsity threshold. `brain_region_idx` is a zero vector (all `CA1`) of length `registered.sum()`, and `session_info` stores the source global cell indices for auditability. This reproduces the paper's 69,744 registered session-neuron instances exactly (mean 336.93/session, range 113–564).

ii.
```python
registered = np.isfinite(trace[:, 0])
assert registered.any()
assert np.isfinite(trace[registered]).all()
assert np.isin(trace[registered], (0.0, 1.0)).all()
...
brain_region_idx.append(np.zeros(int(registered.sum()), dtype=np.int64))
registered_ids = np.flatnonzero(registered).astype(np.int32)
...
"cell_filter": "all and only cells registered (finite trace) in each session",
```

iii. Step 5 Key Decision 3: "**All registered cells, no place-cell or running filter**: A registered cell is finite at the first frame (and verified finite throughout). This gives the paper's exact 69,744 rate-map count. Place-cell labels are downstream analyses." Step 2 quantifies why the reference decoder's `cell_threshold=5` filter is immaterial: "Only 112 of 69,744 registered instances have <=5 events over the full session, confirming that the reference decoder's activity filter is mild before its separate running filter." Step 4: "paper decoding explicitly uses all registered cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the recordings are continuous 40-minute free exploration. The AI therefore defines the alignment event as the start of each consecutive 60-second segment, counted from session frame 0, and records this honestly in metadata: `temporal_alignment_event = "start of each consecutive 60-second recording segment"`, `off_start = 0.0`, `off_end = 60.0`. Neural, input and output for a trial are sliced with the identical `[start:end]` indices derived from the same pooled time axis, so the three streams are aligned by construction; the AI additionally asserted, at every 60 s boundary, that adjacent last-bin/first-bin pairs match independently recomputed raw indices (ruling out skipped or duplicated bins).

ii.
```python
"temporal_alignment_event": "start of each consecutive 60-second recording segment",
"off_start": 0.0,
"off_end": 60.0,
...
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
    input_trials.append(blocked.copy())
    output_trials.append(position_classes[None, start:end].copy())
```

iii. Step 4: "Continuous sessions are cut from frame zero into exact 1,800-frame trials; only the final incomplete segment is excluded." Step 10 Check 4: "Every adjacent last-bin/first-bin pair at all one-minute boundaries was separately asserted against raw indices, ruling out skipped/duplicated bins." Step 3 confirms the two streams were hardware-synchronized: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz … timestamped for post-hoc alignment."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms bins (10 Hz)**; yes, temporal rebinning is applied. The source is 30 Hz (33.33 ms). The AI rebins by a factor of 3 via Gaussian smoothing (σ = 3 source frames) plus non-overlapping 3-frame mean pooling, giving `TARGET_HZ = 10` and 600 bins per 60 s trial. The identical pooling is applied to the position stream, so bin edges coincide across streams. `metadata['time_bin_size'] = 100.0` (ms), and the bin size is identical for every trial and session. The decoder verification confirms `T: mean 600.00, min 600, max 600` across all 207 sessions.

ii.
```python
SOURCE_HZ = 30
POOL_FRAMES = 3
TARGET_HZ = SOURCE_HZ / POOL_FRAMES          # 10 Hz
TRIAL_TIMEPOINTS = int(TARGET_HZ * TRIAL_SECONDS)   # 600
...
"time_bin_size": 100.0,
"source_sampling_rate_hz": SOURCE_HZ,
"target_sampling_rate_hz": TARGET_HZ,
"timepoints_per_trial": TRIAL_TIMEPOINTS,
```

iii. Step 5 Key Decision 1 (quoted in 2-b): the 100 ms bin is chosen because it "exactly reproduce[s] reference decoder temporal preprocessing" (`temporal_bin_size=3` in `fit_decoder`/`test_decoder`), and secondarily because it reduces the exported dataset from ~20 GB to ~6.2 GB. Step 7 lists "100 ms reference pooling | Approximately 3x lower output I/O/training volume than 30 Hz" as a speed-up.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field: a per-day nested list of indices of the occluded partitions in the 3×3 partitioning of the 75×75 cm arena, with the sentinel `-1` meaning "nothing blocked" (i.e. the open square). The `envs` string (`square`, `o`, `t`, `u`, `rectangle`, …) is *not* used as the decoder input — it is kept only as descriptive metadata, and no environment-identity one-hot is added.

ii.
```python
blocked = blocked_mask_x_y(source["blocked"][day])
...
source_indices = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
...
environment = str(np.asarray(source["envs"])[day].squeeze())   # metadata only
```

iii. Step 1: "`blocked` is described in the README as indices in row-major 3x3 geometry, with -1 for none." Step 5 Key Decision 5: "**Geometry input is blockedness, not named condition**: Nine float32 binary dimensions retain exact geometry and generalize across names. No environment identity one-hot is added because the explicit requested input is geometry."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are scattered into a 3×3 row-major matrix (1 = blocked, 0 = accessible), the matrix is **transposed** from the source `(y, x)` convention into `(x, y)`, and then flattened x-major, giving a 9-element `float32` vector with `input_names = ['blocked_x0_y0', 'blocked_x0_y1', …, 'blocked_x2_y2']`. The `-1` sentinel is handled by only scattering non-negative indices, yielding an all-zero vector for the square. The same 9-vector (a `.copy()`) is repeated for every trial of the session, i.e. static per trial as the task requires; shape is `(9,)`, which the target format permits.

The transpose is the one non-obvious step, and it is correct: I verified independently on QLAK-CA1-08 that for the `t`, `u` and `rectangle` geometries the set of grid cells with exactly zero occupancy in `[x_bin, y_bin]` coordinates equals the *transposed* mask, not the raw row-major mask. The effect of the transpose is to make input dimension `3*x+y` refer to the same physical grid cell as output class `3*x+y`.

ii.
```python
def blocked_mask_x_y(blocked_entry: object) -> np.ndarray:
    """Return 9 features in position-compatible x-major order; 1 means blocked.

    Source blocked indices are row-major in (y, x). Position is stored as (x, y),
    hence the transpose before flattening. The square sentinel -1 means no blocks.
    """
    source_indices = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    source_y_x = np.zeros((N_POSITION_BINS, N_POSITION_BINS), dtype=np.float32)
    valid = source_indices[source_indices >= 0].astype(np.int64)
    if valid.size:
        source_y_x.reshape(-1)[valid] = 1.0
    return source_y_x.T.reshape(-1).copy()
...
input_trials.append(blocked.copy())
...
"input_names": [f"blocked_x{x}_y{y}" for x in range(3) for y in range(3)],
"input_encoding": "blocked_x_y is 1 for blocked and 0 for accessible",
```

iii. Step 4 discrepancy table: "`blocked` row-major indices correspond to `(y,x)`; position arrays are `(x,y)` … Transpose source blocked matrices to `(x_bin,y_bin)` before flattening, so input dimension `x*3+y` matches output class. Zero-occupancy bins empirically match this transform for all 10 geometries." Step 10 Issue 1 restates it: "Empirical raw occupancy showed source row-major blocked indices are `(y,x)` while position/rate-map indexing is `(x,y)`. Resolution: transpose before flattening. Independent checks across the geometry set show zero occupancy in blocked output classes."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field only: DeepLabCut head-tracking x–y coordinates, shape `(n_days, 2, n_frames)`, in centimetres spanning ~0–75 cm on both axes, with `position[0] = x` and `position[1] = y`. The code asserts that position is entirely finite and that its frame count equals the trace frame count.

ii.
```python
position = source["position"][day]
assert np.isfinite(position).all()
assert trace.shape[1] == position.shape[1]
...
pooled_xy, position_classes = pool_position(position, n_raw_keep)
```

iii. Step 3: "Position was obtained using DeepLabCut head tracking." Step 2: "Position coordinates span approximately 0 to 75 cm on both axes." Position appears only under `output`, never as a decoder input — Step 12 "Leakage/alignment review": "Position is present only under `output`, never `input`."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is truncated to `n_raw_keep` frames (same truncation as the neural stream), then averaged in the *same* non-overlapping 3-frame bins as the neural data (in `float64`, then discretized). The pooled continuous `(2, n_bins)` trajectory is discretized (see 4-c) into a single categorical channel of dtype `int8` with shape `(1, 600)` per trial. `output_names = ['position_3x3_bin']` and `output_values = [['x0_y0', …, 'x2_y2']]`. No smoothing, interpolation, or speed gating is applied to position.

ii.
```python
def pool_position(position: np.ndarray, n_raw_keep: int):
    """Average aligned raw x/y samples in non-overlapping three-frame bins."""
    trimmed = np.asarray(position[:, :n_raw_keep], dtype=np.float64)
    pooled_xy = trimmed.reshape(2, -1, POOL_FRAMES).mean(axis=2)
    xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
    np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
    classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
    return pooled_xy, classes
...
output_trials.append(position_classes[None, start:end].copy())
```

iii. Step 5 mapping table: "Average x/y over the same non-overlapping 3 source frames … Categorical int8, shape `(1,600)`, classes 0..8", attributed to "Behavioral pooling in `fit_decoder`/`test_decoder`; spatial binning in `decode_position_within`". This mirrors the reference decoder, which pools the behavioural stream with the same `AvgPool1d(3)` before converting to a discrete one-hot bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into three equal 25 cm bins over the fixed 0–75 cm arena (edges `[0, 25, 50, 75]`), using `floor(coord / 25)` followed by `clip(0, 2)`. The clip is what handles the exact boundary value `x = 75.00 cm` that occurs in the data (floor gives 3). The 9 classes are then `class = 3 * x_bin + y_bin`, i.e. 0…8, stored as `int8`. Bin edges are recorded in metadata. Over the full dataset the class distribution is `[0.0997, 0.0754, 0.1157, 0.0985, 0.0573, 0.1412, 0.1351, 0.0768, 0.2003]` — non-uniform because many geometries block some partitions and because mice over-occupy one corner; the decoder script uses balanced accuracy, and all 9 classes are present.

ii.
```python
ARENA_SIZE_CM = 75.0
N_POSITION_BINS = 3
POSITION_BIN_CM = ARENA_SIZE_CM / N_POSITION_BINS      # 25.0
...
xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
...
"output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
"position_bin_edges_cm": [0.0, 25.0, 50.0, 75.0],
```

iii. Step 5 Key Decision 4: "**Position class order**: Class `3*x_bin+y_bin`, where each bin is [0,25), [25,50), or [50,75] cm. Output labels are `x0_y0` ... `x2_y2`. The source blocked mask is transposed so its flattened dimensions use the same class coordinate system; raw occupancy confirmed blocked/accessible correspondence." Step 10 Edge cases: "Position coordinate 75 cm is clipped safely into bin 2." The 3×3 grid is dictated by the Decoder Task section and also matches the arena's physical 3×3 partitioning described in the paper.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Alignment is structural rather than computed: the two streams are already frame-synchronous at 30 Hz in the source (asserted via `trace.shape[1] == position.shape[1]`), both are truncated at the same `n_raw_keep`, both are pooled with the same non-overlapping 3-frame grouping (so bin *k* of the output covers the same source frames as bin *k* of the neural data), and both are sliced with identical `[start:end]` indices per trial. An assertion checks `pooled_neural.shape[1] == position_classes.size == ntrials * TRIAL_TIMEPOINTS`. No lag or offset is introduced. I verified this end-to-end against the raw joblib files: concatenating the converted trials reproduces the independently recomputed pooled class sequence exactly.

ii.
```python
assert trace.shape[1] == position.shape[1]
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
pooled_neural = pool_neural(trace, registered, n_raw_keep)
pooled_xy, position_classes = pool_position(position, n_raw_keep)
...
assert pooled_neural.shape[1] == position_classes.size == ntrials * TRIAL_TIMEPOINTS
```

iii. Step 4: "Passes same time indices from `position` and `trace`; both 30 Hz … Exact agreement." Step 10 comparison row (c): "Same raw indices, identically grouped by 3 before slicing | Exact; raw allclose and boundary tests pass." The `--show-processing` figure panel 3 overlays raw and pooled x/y against the trial boundary specifically to make misalignment visible.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Five classes of imperfection are handled, all explicitly:
- **Unregistered cells (all-NaN rows)**: excluded per session via the `registered` mask; the code then asserts the retained block is fully finite, so NaNs can never reach the output.
- **Sessions not an exact multiple of 60 s** (71,866–72,219 frames vs the nominal 72,000): the terminal partial minute is dropped, and the number of dropped frames is recorded per session in `session_info['dropped_terminal_frames']`. No padding or fabrication.
- **`blocked == -1` sentinel** for the open square: filtered by `source_indices >= 0`, producing an all-zero geometry vector.
- **Boundary position value `x = 75.00 cm`**: `np.clip` folds it into bin 2 rather than creating a spurious 10th class.
- **Position gaps**: none exist; the code asserts `np.isfinite(position).all()` rather than assuming it, so a violation would fail loudly instead of silently corrupting output.

A final `validate_built_data` pass re-checks every trial's shape, dtype, finiteness, non-negativity, input binarity, and output range before the pickle is written.

ii.
```python
registered = np.isfinite(trace[:, 0])
assert registered.any()
assert np.isfinite(trace[registered]).all()
assert np.isfinite(position).all()
...
valid = source_indices[source_indices >= 0].astype(np.int64)
...
np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
...
"source_frames": int(trace.shape[1]),
"retained_source_frames": int(n_raw_keep),
"dropped_terminal_frames": int(trace.shape[1] - n_raw_keep),
...
assert neural.dtype == np.float32 and np.isfinite(neural).all()
assert (neural >= 0).all()
assert np.isin(inputs, (0, 1)).all()
assert outputs.min() >= 0 and outputs.max() <= 8
```

iii. Step 10 Issue 2: "Paper says 40 min, while three subjects have 39 min 55.5 s and others slightly exceed 40 min. Resolution: preserve every exact 60 s segment, drop only incomplete tails, and record dropped frames; this yields fixed trial lengths without padding or fabricating data." Step 10 Edge cases: "Position coordinate 75 cm is clipped safely into bin 2. Square sentinel -1 produces no blocked inputs. Unregistered all-NaN rows are excluded, while low-activity registered cells remain … No missing/nonfinite converted value exists."

## 6-a. What are the most time-consuming steps of the code?

i. From the AI's own printed timings in `conversion_full_out.txt`: full conversion took 194.13 s before serialization plus 5.70 s to write the 6.173 GiB pickle.
- **Loading the seven compressed joblib files dominates**: 16.33 + 15.01 + 16.01 + 6.46 + 14.73 + 12.17 + 16.15 ≈ **96.9 s, ~50 % of total runtime**, and it is largely wasted work because `joblib.load` inflates the *entire* animal dictionary (`SFPs`, `centroids`, `maps`) even though only `trace`/`position`/`blocked`/`envs` are used.
- **Per-session processing** is ~0.26–0.75 s × 207 ≈ 80 s, dominated by `gaussian_filter1d` over `(n_cells ≈ 337, ~72,000)` float32 arrays, plus the three separate fancy-index copies of `trace[registered]`.
- **Pickle write** of 6.173 GiB: 5.70 s.
- Plot generation (`--show-processing`) is a noticeable per-session cost but capped at 2 sessions and disabled for the full run.

ii.
```python
load_start = time.perf_counter()
source = joblib.load(DATA_DIR / animal)[animal]
print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f} s", flush=True)
...
print(f"Processed session {len(neural_all):3d}: {session_id}, "
      f"cells={registered.sum()}, trials={ntrials}, "
      f"time={time.perf_counter()-session_start:.2f} s", flush=True)
...
print(f"Conversion before pickle: {stats['conversion_seconds_before_pickle']:.2f} s", flush=True)
print(f"Wrote {args.outpicklefile} (...) in {time.perf_counter()-pickle_start:.2f} s", flush=True)
```

iii. Step 6: "Loading every animal simultaneously would additionally materialize tens of GB of compressed joblib arrays"; Step 7's timing table attributes "Source loading | 9.77 s for first animal" and Step 9 reports "Full conversion took 194.13 s before serialization plus 5.70 s to write (3.33 minutes total), substantially below the conservative estimate and optimization threshold." Because the total was ~3.3 min, well under the instruction's 15-minute threshold, the AI did not optimise further.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The numerically heavy work is already fully vectorized: smoothing is a single `gaussian_filter1d` over the whole `(cells, time)` array, pooling is a single `reshape(...).mean(axis=2)`, and position binning is a vectorized `floor`/`clip`/arithmetic. The remaining loops are structural and cannot be removed without violating the required list-of-lists output format:
- `for trial in range(ntrials)` (≈8,187 iterations total) slices and copies each trial. It could be written as `np.split`/`np.array_split` or a `reshape → list(...)` one-liner, but each iteration is a large memcpy, so the Python overhead is negligible; the copies themselves are needed only because the format expects independent arrays.
- `input_trials.append(blocked.copy())` makes 39–40 identical 9-element copies per session; a single shared array reference would do, but the cost is trivial.
- The per-trial loop inside `validate_built_data` (another ≈8,187 iterations, each scanning a full trial) is a second complete pass over the 6 GB dataset. It is vectorized *within* each trial but could be reduced to per-session checks.
- `for animal` / `for day` are inherently sequential given the memory strategy; genuine wall-clock gains would come from multiprocessing across animals (the AI chose not to, to bound RAM) rather than from vectorization.

ii.
```python
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
    input_trials.append(blocked.copy())
    output_trials.append(position_classes[None, start:end].copy())
...
for s, (neural_trials, input_trials, output_trials) in enumerate(
    zip(data["neural"], data["input"], data["output"])
):
    ...
    for neural, inputs, outputs in zip(neural_trials, input_trials, output_trials):
        assert neural.shape == (nneurons, TRIAL_TIMEPOINTS)
        assert neural.dtype == np.float32 and np.isfinite(neural).all()
```

iii. Step 6 "Code speedups added": "Gaussian filtering writes directly to a preallocated float32 array; filtering/pooling is vectorized over cells and time; the 3-frame reshape/mean replaces Python time loops; full-session processing is done once … and static 1D geometry avoids needless time replication." The AI's stance is that the residual loops are format-imposed, and the measured 3.3-minute runtime supported leaving them alone.

## 6-c. What processing does the code repeat multiple times?

i. Three genuine repetitions, all cheap-to-moderate and all deliberate:
- **`trace[registered]` fancy indexing is materialized three times per session** — once in `assert np.isfinite(trace[registered]).all()`, once in `assert np.isin(trace[registered], (0.0, 1.0)).all()`, and once inside `pool_neural`. Each is a full ~100–300 MB copy of the session's registered block, and `np.isin` on floats is the most expensive of the three. Caching `trace[registered]` once would remove two large copies and the redundant scans.
- **A second full pass over the entire converted dataset** in `validate_built_data`, which re-checks finiteness and non-negativity of all ~6 GB of neural data that was just asserted finite at the source.
- **`blocked.copy()` per trial**, duplicating the same 9-element vector 39–40 times per session.
- When `--show-processing` is on, `plot_processing` recomputes `raw_trace[registered]` a fourth time.

ii.
```python
registered = np.isfinite(trace[:, 0])
assert registered.any()
assert np.isfinite(trace[registered]).all()          # copy 1
assert np.isin(trace[registered], (0.0, 1.0)).all()  # copy 2
...
pooled_neural = pool_neural(trace, registered, n_raw_keep)   # copy 3 inside
...
def validate_built_data(data, expected_sessions):   # full second pass over 6 GB
    ...
        for neural, inputs, outputs in zip(neural_trials, input_trials, output_trials):
            assert neural.dtype == np.float32 and np.isfinite(neural).all()
            assert (neural >= 0).all()
```

iii. The AI does not flag these as redundancy; it presents them as intentional defensive validation. Step 6: the converter "validates every array"; Step 9: "Internal validation passed before writing." The instructions explicitly demand shape/type validation at each step and sanity checks, so the duplication buys fail-fast guarantees at a cost the AI measured to be acceptable (~80 s of processing total for 207 sessions).

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Largest item: `joblib.load` decompresses each animal's full dictionary, including `SFPs` (35×35×n_cells×n_days), `centroids` and the three `maps` arrays, none of which are used.** These dominate the ~97 s of load time and the transient memory footprint. Reading only the four needed fields (e.g. lazily from the `.mat`/HDF5 twin with `h5py`) would cut total runtime roughly in half.
- **`pooled_xy`** (the pooled continuous x/y trajectory) is computed and returned on every session but consumed only by `plot_processing`; in the default `--full` run it is discarded immediately. It is cheap and is on the critical path anyway to produce `classes`, but returning it is unnecessary work in non-plot mode.
- **`validate_built_data`'s full re-scan** of the dataset (see 6-c) produces `output_counts`/`output_fractions` used only for one printed line.
- **Per-session provenance stored in the pickle but unused by the decoder**: `registered_source_cell_indices` (an `int32` array per session), `environment`, `source_frames`, `dropped_terminal_frames`, etc. These are small relative to 6.2 GiB and were retained intentionally for auditability.
- Conversely, the AI *avoided* several plausible wastes: it never computes ΔF/F, never exports `maps`, never replicates the static 9-vector across 600 timepoints, and keeps everything in `float32`/`int8`.

ii.
```python
source = joblib.load(DATA_DIR / animal)[animal]   # inflates SFPs/centroids/maps too
...
def pool_position(position, n_raw_keep):
    ...
    return pooled_xy, classes        # pooled_xy used only by plot_processing
...
pooled_xy, position_classes = pool_position(position, n_raw_keep)
...
session_info.append({... "registered_source_cell_indices": registered_ids, ...})
```

iii. Step 5 mapping table explains the deliberate omissions: `SFPs`/`centroids` are "Not decoder task variables and too large to duplicate", and `maps` "Do not export; derived from the same trace/position and would leak target spatial information if used as inputs." Step 5 Key Decision 8 justifies the retained metadata: "Preserve subject, zero-based source day, environment, source/retained/dropped frames, registered global cell indices, source and target sampling rates, and transforms for auditability." The AI never identified the joblib whole-file inflation as waste — it chose joblib because it is the reference loader's default format.
