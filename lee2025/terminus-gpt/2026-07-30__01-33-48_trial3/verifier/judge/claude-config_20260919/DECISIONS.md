# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset from the **per-animal joblib files** in `/app/data` (the extension-less files `QLAK-CA1-08`, `QLAK-CA1-30`, ...), not from the parallel `.mat` files. The list of 7 animals is hard-coded in a module-level constant `ANIMALS`. Each joblib file unpickles to a dict keyed by the animal name, whose value is a record with fields `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`. The AI pulls only `envs`, `trace`, `position`, `blocked`. Within an animal, `trace` is `(n_days, n_cells, n_frames)`, `position` is `(n_days, 2, n_frames)`, and `envs`/`blocked` are per-day; the code loops over days to produce sessions, then chops each day into 1-minute trials. The `SFPs`, `centroids` and precomputed `maps` fields are ignored.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]
...
    for subj_idx, animal in enumerate(animals):
        print(f'Loading {animal}...')
        rec = load_animal(data_dir, animal)
        envs = np.array(rec['envs']).reshape(-1)
        trace = np.asarray(rec['trace'])
        position = np.asarray(rec['position'])
        blocked = rec['blocked']
        n_days = len(envs)
        for day in range(n_days):
            ...
```

iii. From CONVERSION_NOTES.md Step 5, Key Decision 1: *"Use native joblib files as source: `load_dat` in reference code directly consumes the joblib files; no need to recompute from MATLAB."* The AI had read `/app/code/georepca1/src/utils.py` and noted (Step 1) that `load_dat` "Load[s] per-animal joblib or mat data and convert[s] envs/position/trace fields to numpy-friendly arrays", i.e. the reference repository treats the joblib file as a first-class source and the `.mat` path as an alternative that needs extra MATLAB-object unwrapping. Step 2 records that "Each animal is available both as a joblib file (no extension) and a MATLAB `.mat` file."

## 1-b. How are the data split into subjects (mice)?

i. One subject per animal file. `subjects` is set directly to the hard-coded `ANIMALS` list (7 mice), and `subject_idx` is the index of the animal whose file produced each session. In `--sample` mode the AI takes `ANIMALS[:2]`, i.e. it samples by *animal*, not by session.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
data = {
    ...
    'subjects': animals.copy(),
    'subject_idx': [],
    ...
}
for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    for day in range(n_days):
        ...
        data['subject_idx'].append(subj_idx)
...
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 2: "`data/` contains 7 animal datasets: QLAK-CA1-08, QLAK-CA1-30, QLAK-CA1-50, QLAK-CA1-51, QLAK-CA1-56, QLAK-CA1-74, QLAK-CA1-75." Step 4 records the AI's check that the 7 files imply 7 mice ("Data files show 7 animals ... Tentatively use 7 subjects from dataset files"), cross-checked against the methods statement of 207 sessions / 5,413 unique neurons.

## 1-c. How are the data split into sessions?

i. One session per animal-day. Each animal file holds 31 recording days (21 for QLAK-CA1-51), and every day becomes one entry in `neural`/`input`/`output`/`subject_idx`/`brain_region_idx`, giving 207 sessions. The AI also builds a rich `session_info` metadata record per session with a session id of the form `<animal>_day<NN>_<env>`, the day index, environment name, neuron count, number of neurons dropped, raw/used frame counts, trial count, and the blocked vector.

ii.
```python
n_days = len(envs)
for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
    blocked_vec = blocked_to_vec(blocked[day])
    trace_day = np.asarray(trace[day], dtype=np.float32)
    pos_day = np.asarray(position[day], dtype=np.float32)
    ...
    data['neural'].append(neural_trials)
    data['input'].append(input_trials)
    data['output'].append(output_trials)
    data['subject_idx'].append(subj_idx)
    data['brain_region_idx'].append(np.zeros(trace_day.shape[0], dtype=np.int64))
    session_info.append({
        'session_id': session_id, 'animal': animal, 'day_index': int(day),
        'environment': flatten_env_name(envs[day]),
        'n_neurons': int(trace_day.shape[0]),
        'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
        'n_frames_raw': int(n_frames), 'n_frames_used': int(used),
        'n_trials': int(n_trials),
        'blocked_vector': blocked_vec.astype(int).tolist(),
    })
```

iii. CONVERSION_NOTES.md Step 2: "Sessions correspond to recording days / environments within each animal file"; Step 3 quotes the methods: "All sessions were 40 min, and one session was recorded per day". Step 2/3 both record "31 for six animals; 21 for QLAK-CA1-51" summing to the paper's 207 sessions, which the AI used as a consistency check.

## 1-d. How are the data split into trials?

i. Each 40-minute continuous session is cut into contiguous, non-overlapping 60-second windows of 1800 frames (30 Hz × 60 s), starting at frame 0. The number of trials is `min(n_trace_frames, n_pos_frames) // 1800`; both streams are first truncated to that common usable length, so the trailing partial minute is dropped. This yields 39 or 40 trials per session and 8,187 trials in total.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS   # 1800

def split_session_into_trials(trace_day, pos_day, blocked_vec):
    n_frames = min(trace_day.shape[1], pos_day.shape[1])
    n_trials = n_frames // FRAMES_PER_TRIAL
    used = n_trials * FRAMES_PER_TRIAL
    trace_day = trace_day[:, :used]
    pos_day = pos_day[:, :used]
    pos_bins = position_to_bins_3x3(pos_day)

    neural_trials, input_trials, output_trials = [], [], []
    for i in range(n_trials):
        s = i * FRAMES_PER_TRIAL
        e = (i + 1) * FRAMES_PER_TRIAL
        neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
        input_trials.append(blocked_vec.copy())
        output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
    return neural_trials, input_trials, output_trials, n_frames, used, n_trials
```

iii. CONVERSION_NOTES.md Step 3 Trial curation rules: "No explicit trial concept in native data. Sessions are 40 min continuous exploration recordings, one per day. For our conversion, trials will be derived by splitting each session into 1-minute chunks while preserving frame alignment." Step 5 Key Decision 4: "Derive 1-minute trials from continuous 40 min sessions: Native data have no trial structure; task specification requires at least two trials per session, so each day/session will become ~40 contiguous 1-minute trials." The `min(...)` over the two streams and the truncation to `used` frames are defensive measures so neural and behavioural streams can never drift apart if a session had unequal lengths.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality control is applied.** Every complete 1800-frame window is kept; only the trailing incomplete window is discarded. The single filter that exists is at the *session* level: a session producing fewer than 2 complete trials is skipped entirely (with a printed message). In practice this guard never fires, because all 207 sessions are ~40 min and yield 39–40 trials. Notably, the AI deliberately did **not** port the reference paper's running-speed/velocity frame mask or its active-cell mask from `decode_position_within`.

ii.
```python
neural_trials, input_trials, output_trials, n_frames, used, n_trials = split_session_into_trials(
    trace_day, pos_day, blocked_vec
)
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The `< 2` guard implements the target-format requirement quoted in the instructions: "There needs to be at least two trials within each session in order to evaluate the decoder performance" (echoed in CONVERSION_NOTES.md Step 5 Key Decision 4). For not applying velocity/activity filtering, trajectory step 193/196 records the AI's reading of the reference decoder: "per day/session, they smooth velocity, keep frames above a velocity threshold, keep cells with enough activity within those valid frames, perform 5-fold cross-validation ... For our required target task, we must adapt this to 3x3 output bins and 1-minute derived trials, but keep the native 30 Hz aligned traces/positions ... the target format should probably store native framewise data and let train_decoder handle its own modeling." I.e. the AI treated velocity thresholding as part of the paper's *decoder*, not part of the *dataset*, and dropping frames would break the fixed-length trial structure required by the target format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely the `trace` field of each animal record, indexed by day: `trace[day]` has shape `(n_cells, n_frames)`. This is the authors' rise-extracted, binarized calcium event vector (values are exactly {0, 1}, with NaN for cells not registered on that day). No use is made of `SFPs`, `centroids`, or the precomputed `maps`.

ii.
```python
trace = np.asarray(rec['trace'])
...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. CONVERSION_NOTES.md Step 1: "README states raw neural signal used by authors is rise-extracted calcium trace; value 1 indicates significant event." Step 3 Processing Details: "Neural signal used for analysis is not dF/F; it is a binarized rising-phase event vector extracted from filtered calcium traces. Rising-phase extraction: smooth derivative with Gaussian kernel (SD 5 frames), estimate noise from negative derivative half-normal distribution, z-score, threshold at 2.5, then binarize to 0/1. This binary vector is treated as the firing rate in all subsequent analyses." Step 5 Key Decision 2: "Use `trace` as neural activity: Reference methods/code treat the binarized rising-phase vector as firing rate for all analyses."

## 2-b. How is the `neural` data processed?

i. Essentially no processing of the values. The per-day trace is cast to `float32`, invalid (non-finite) neurons are dropped, a `np.nan_to_num` pass is applied (a no-op after the drop), and the surviving array is sliced into trials and stored as **`uint8`**. The data are already in `(n_neurons, n_timepoints)` orientation in the joblib file, so no transpose is needed. **No ΔF/F is computed, no smoothing, no z-scoring, no normalisation, no temporal rebinning, no place-cell selection.**

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```
Metadata records the interpretation:
```python
'source_signal': 'Binarized rising-phase event vector treated as firing rate',
```

iii. CONVERSION_NOTES.md Step 4 Discrepancies table: "Neural signal representation | README/methods say binary rising-phase vector treated as firing rate | `trace` arrays are float64 but represent binarized event activity per frame | ... | Use `trace` as neural data; do not compute dF/F". Step 5 Key Decision 7: "Do not restrict to place cells by default: The paper's decoding code filters by velocity and cell activity, not place-cell status, and the task asks to decode from neural activity broadly." The `uint8` cast was introduced in trajectory step 203 for size reasons: "the 12G sample file indicates we should optimize dtypes now ... `trace` is binary and could be stored as uint8/bool to drastically reduce file size".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Per session, a neuron is kept only if **every** frame of that day's trace is finite; neurons with any NaN/Inf are dropped. Because a cell that is not registered on a given day is NaN for the whole day, this is equivalent in practice to dropping all-NaN rows (verified: for QLAK-CA1-08 day 0, `any-NaN` and `all-NaN` both select the same 330 of 515 cells). The count of dropped neurons is stored per session in `session_info['n_neurons_dropped_nonfinite']`. This yields a mean of 336.9 neurons/session, min 113, max 564, and **69,744 total session-neuron entries — exactly the paper's reported 69,744 rate maps.** No place-cell / split-half-reliability filtering and no active-cell filtering are applied.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
...
'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
```

iii. Trajectory step 203: "neural data contain NaN or Inf values for all listed trials in session 61 ... This is consistent with the dataset containing unregistered cells represented as NaN on some days ... The correct fix is likely to remove neurons that are invalid for a given session/day (e.g., rows with NaN in trace for that day) before splitting into trials. This matches the reference notion that unregistered cells appear as NaN for a given day." CONVERSION_NOTES.md Step 10: "Edge-case handling: unregistered neurons with non-finite values are dropped per session/day before trial splitting" and "7 subjects, 207 sessions, 5,413 unique cells across animals, and 69,744 session-neuron entries are consistent with reference materials." For skipping place-cell curation, Step 5 Key Decision 7 (quoted above) plus Step 4: "Place-cell curation ... Optional for decoder conversion unless matching reference filtering requires it."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is **no external alignment event** — the task is continuous free exploration. Trials are contiguous 60 s windows tiled from the start of the session, so the nominal "alignment event" is the start of each 1-minute window: `off_start = 0.0`, `off_end = 60.0`. Neural and behavioural streams are never re-indexed relative to each other; the native 30 Hz frame correspondence established by the authors' DAQ is preserved, and the same frame indices `[s:e]` slice both `trace_day` and `pos_bins`.

ii.
```python
'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
'source_sessions_are_continuous': True,
...
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. CONVERSION_NOTES.md Step 3: "Behavioral and cellular imaging streams were simultaneously acquired at 30 Hz and timestamped for post-hoc alignment." Step 4: "Temporal alignment | Code/methods say behavior and calcium are aligned at 30 Hz | `position` and `trace` have matching frame counts within each animal/day | Consistent | Preserve native frame alignment when splitting into 1-min trials." Step 5 Key Decision 3: "Preserve native 30 Hz frame alignment in converted data: Position and traces are already aligned framewise; splitting into 1-minute trials should preserve this alignment exactly."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native acquisition resolution is kept: 30 Hz, i.e. **33.333 ms per bin**, 1800 bins per trial. **No temporal rebinning, downsampling, smoothing, or spike-count aggregation is performed**, even though the reference paper's own decoder averages traces over 3-frame (~100 ms) windows after Gaussian smoothing.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
'fps': FPS,
'trial_seconds': TRIAL_SECONDS,
```

iii. CONVERSION_NOTES.md Step 3: "Neural data time bin | 33.3 ms native frame rate | 'The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz...'". Trajectory step 196 shows the rebinning question was explicitly considered and rejected: "the reference within-session decoder uses ... temporal binning by averaging over 3-frame windows (~100 ms), Gaussian smoothing of traces before pooling ... we must adapt this ... but keep the native 30 Hz aligned traces/positions and likely preserve the 3-frame temporal binning only if needed by downstream decoder — however the target format should probably store native framewise data and let train_decoder handle its own modeling." I.e. the AI treats 3-frame binning as a decoder-side modelling choice rather than a dataset property, and preserves maximal temporal resolution for the downstream decoder.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field of the animal record, indexed by day (`blocked[day]`). This is a nested MATLAB-derived object array listing which of the nine 3×3 arena partitions were walled off in that day's geometry; a single value of `-1` means nothing was blocked (the open square). The `envs` field (geometry name, e.g. `square`, `o`, `t`, `u`, `rectangle`, `+`) is read too, but only for the human-readable `session_id` and `session_info['environment']` — it is not part of the decoder input vector.

ii.
```python
blocked = rec['blocked']
...
blocked_vec = blocked_to_vec(blocked[day])
...
envs = np.array(rec['envs']).reshape(-1)
session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
```

iii. CONVERSION_NOTES.md Step 2: "`blocked` is a per-day nested list/array of blocked partition indices, with `-1` meaning no blocked partition." Step 3: "square arena 75x75 cm partitioned into a 3x3 grid" (trajectory step 190). Step 5 Key Decision 5: "Use blocked geometry as decoder input: This is static contextual information specified by the task and directly available from `blocked`."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked[day]` is flattened out of its nested object-array wrapper, NaN entries are skipped, each remaining value is cast to `int`, and the result is multi-hot encoded into a length-9 `float32` vector over the nine 3×3 partitions. The sentinel `[-1]` maps to an all-zero vector. Indices are range-checked (`0 <= v <= 8`) before being set. The same vector is attached to every trial of the session (a fresh `.copy()` per trial), so the input is static within a session and shape `(9,)` per trial. The nine inputs are named `blocked_partition_0 ... blocked_partition_8`.

ii.
```python
def blocked_to_vec(blocked_entry):
    vec = np.zeros(9, dtype=np.float32)
    flat = np.array(blocked_entry, dtype=object).reshape(-1)
    vals = []
    for item in flat:
        arr = np.array(item).reshape(-1)
        for v in arr:
            if np.isnan(v):
                continue
            vals.append(int(v))
    if len(vals) == 1 and vals[0] == -1:
        return vec
    for v in vals:
        if 0 <= v <= 8:
            vec[v] = 1.0
    return vec
...
input_trials.append(blocked_vec.copy())
...
'input_names': [f'blocked_partition_{i}' for i in range(9)],
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "`blocked` per animal/day → input[0:9] → Encode blocked partitions as 9-d binary vector over 3x3 arena partitions; static per trial ... `-1` means no blocked partition so all zeros." The instruction's Decoder Inputs section specifies "Environment geometry, representing which parts of the arena are blocked. Static per-trial." The multi-hot form was chosen because a session can block several partitions at once (e.g. the `+` and `o` geometries block 4 partitions), so a single categorical label over geometry names would not expose the spatial structure.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field of the animal record, indexed by day: `position[day]` has shape `(2, n_frames)`, giving DeepLabCut-tracked (x, y) head position in centimetres within the 75 × 75 cm arena, one sample per imaging frame.

ii.
```python
position = np.asarray(rec['position'])
...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. CONVERSION_NOTES.md Step 3 Processing Details: "Position was obtained from DeepLabCut head tracking"; Step 2 lists `position` with shape `(31, 2, 71866)` for QLAK-CA1-08, matching `trace`'s frame count exactly.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is cast to `float32`, any non-finite values are replaced with `0.0`, then discretized into a single categorical label per frame (see 4-c). The label array is reshaped to `(1, n_timepoints)` per trial and stored as `uint8`. There is **no smoothing, no interpolation, no velocity filtering, and no re-referencing of coordinates** — the raw arena coordinates are used directly. The single output is named `position_bin_3x3` with value names `bin_0 ... bin_8`.

ii.
```python
pos_day = np.asarray(position[day], dtype=np.float32)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
...
pos_bins = position_to_bins_3x3(pos_day)
...
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
...
'output_names': ['position_bin_3x3'],
'output_values': [[f'bin_{i}' for i in range(9)]],
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "`position` per animal/day → output[0] → Discretize x-y position into 3x3 spatial bins per frame; flatten 2D bin to 9-class categorical output ... Time-varying output at native frame rate." Step 5 Key Decision 6: "Use 3x3 discretized position as decoder output: Required by task; this is a coarser version of the paper's within-session spatial decoding." The instruction's Decoder Outputs section requires "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into 3 equal 25 cm bins over the fixed 75 cm arena extent by floor-division, after clipping to `[0, 75 − 1e−6]` so that a coordinate exactly at the far wall (x = 75.0 occurs in the data) falls into bin 2 rather than a phantom bin 3. A redundant second `np.clip(..., 0, 2)` guards the result. The 2-D bin is flattened **row-major with y as the slow axis**: `label = ybin * 3 + xbin`, giving labels 0–8. The arena extent is hard-coded (75 cm) rather than derived from the data's min/max, so the grid is identical across all sessions and geometries — including blocked geometries where some bins are unvisitable.

ii.
```python
ARENA_SIZE_CM = 75.0
N_POS_BINS = 3

def position_to_bins_3x3(position_xy):
    pos = np.asarray(position_xy, dtype=np.float32)
    x = pos[0]
    y = pos[1]
    eps = 1e-6
    xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    xbin = np.clip(xbin, 0, N_POS_BINS - 1)
    ybin = np.clip(ybin, 0, N_POS_BINS - 1)
    return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 3 (via trajectory step 190) records the methods statement "square arena 75x75 cm partitioned into a 3x3 grid" — the AI reused the paper's own 3×3 partitioning of the arena, which is also the partitioning the `blocked` indices refer to, so output bins and input blocked-partition indices share one coordinate system. Step 5 Planned Sanity Checks include "Check that 3x3 position bins are within 0-8 and distribute sensibly across sessions" and "Check that output class occupancy excludes impossible bins only when animal never visits them, not due to mis-binning." The resulting occupancy is non-degenerate: bins 0–8 at fractions 0.100, 0.099, 0.135, 0.075, 0.057, 0.077, 0.116, 0.141, 0.200.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, with no shift or lag. Both streams are truncated to the same usable frame count `used = n_trials * 1800` (using `min()` of the two stream lengths as a guard), binned/sliced with the identical index arithmetic, and the code asserts per trial that the neural and output time dimensions agree. The 9-element blocked input is the same for every trial in a session.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
pos_bins = position_to_bins_3x3(pos_day)
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
...
for s in range(len(data['neural'])):
    assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s])
    for tr in range(len(data['neural'][s])):
        assert data['neural'][s][tr].shape[1] == data['output'][s][tr].shape[1]
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 3: "Preserve native 30 Hz frame alignment in converted data: Position and traces are already aligned framewise; splitting into 1-minute trials should preserve this alignment exactly." Step 5 Planned Sanity Check 1: "Check that per-trial neural and position timepoints match exactly after splitting sessions into 1-minute chunks." Step 10 reports: "Raw-data sanity checks: session 0 / trial 0 spot-check against original `data/QLAK-CA1-08` passed for neural, input, and output using `np.allclose()`."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct guards:
- **Unregistered neurons (NaN rows):** dropped per session before trial splitting (`np.all(np.isfinite(...), axis=1)`), with the drop count logged to `session_info`.
- **Residual non-finite neural values:** `np.nan_to_num(..., nan=0, posinf=0, neginf=0)` after the drop (a no-op in practice, since the drop already removes every non-finite row).
- **Non-finite position:** `np.nan_to_num(pos_day, nan=0.0, ...)` — missing coordinates would silently become (0, 0), i.e. bin 0. This never fires on this dataset (no session has a NaN in `position`).
- **Unequal neural/behavioural lengths:** `n_frames = min(trace.shape[1], pos.shape[1])` truncates both to the common length. Also a no-op here (all 207 sessions have matching lengths).
- **Trailing partial minute:** the remainder frames after the last complete 1800-frame window are discarded (≈0–40 s of each 40-min session).
- **Too-short sessions:** any session yielding < 2 complete trials is skipped.
Malformed/ragged `blocked` entries are also tolerated: the nested object array is flattened, NaNs skipped, and out-of-range indices ignored.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
...
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
...
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
...
for v in vals:
    if 0 <= v <= 8:
        vec[v] = 1.0
```

iii. CONVERSION_NOTES.md Step 8 Format Validation: "initial sample had non-finite neural values from unregistered cells; fixed by dropping non-finite neurons per session/day and regenerating sample." Step 10 Issues Found and Resolved: "non-finite neural values from unregistered cells caused sample verification failures; resolved by dropping non-finite neurons per session/day and regenerating datasets." Trajectory step 203 gives the diagnosis ("the dataset contain[s] unregistered cells represented as NaN on some days; our current conversion passes those NaNs straight through"). The remaining guards are not individually justified in the notes; they read as generic defensive coding added alongside the NaN fix.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the per-animal joblib files dominates: the whole 207-session full conversion took **198.78 s** end to end, and the only heavy operations are `joblib.load` of seven 70–150 MB pickles plus the per-session `np.asarray(trace[day], dtype=np.float32)` / `np.nan_to_num` materialisation of ~500 MB float32 arrays. The actual per-trial work (slicing, position binning, multi-hot encoding) is negligible. The AI instrumented the run with a single wall-clock timer stored in metadata; it did not add per-stage timing. The only genuinely slow downstream step is not in `convert_data.py` at all: the resulting 4.98 GB pickle makes `train_decoder.py` load and SVD-initialisation slow (the AI ran training on CPU for this reason).

ii.
```python
t0 = time.time()
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
print(f'Converted {len(data["neural"])} sessions in {data["metadata"]["conversion_runtime_sec"]:.2f}s')
```
`conversion_full_out.txt`:
```
Loading QLAK-CA1-08...
...
Converted 207 sessions in 198.78s
```

iii. The AI never wrote an explicit bottleneck analysis — CONVERSION_NOTES.md Step 6 ("Code inefficiencies identified", "Code speedups added") and Step 7 ("Run Time Estimates") are left as unfilled `[Note]` / empty-table placeholders. The only efficiency reasoning recorded is about *output size* rather than time: trajectory step 202, "the file is huge at 12G for only 62 sessions ... `trace` is binary and could be stored as uint8/bool to drastically reduce file size", leading to the `uint8` cast in step 203. Since the run comfortably beat the instructions' 15-minute budget, no further optimisation was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not document this. Reviewing the code, the candidates are all minor:
- The `for i in range(n_trials)` loop in `split_session_into_trials` builds 39–40 slices one at a time; because the trial length divides `used` exactly, it could be a single reshape/`np.split` (e.g. `trace_day.reshape(n_neurons, n_trials, 1800).transpose(1, 0, 2)`) or `np.array_split`. The slices are views, so the cost is small.
- The doubly-nested Python loop in `blocked_to_vec` walks an object array element by element; it runs once per session over ≤4 values, so it is irrelevant to runtime (though it is the only place a vectorised `np.concatenate` would also be cleaner).
- The outer `for animal` / `for day` loops are inherently sequential I/O and could have been parallelised across animals with `joblib.Parallel` (the instructions explicitly suggest parallel processing), but the per-animal working set is ~0.5–1 GB so parallelism would have traded runtime for memory.
None of these are the bottleneck; loading is.

ii.
```python
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    input_trials.append(blocked_vec.copy())
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```
```python
for item in flat:
    arr = np.array(item).reshape(-1)
    for v in arr:
        if np.isnan(v):
            continue
        vals.append(int(v))
```

iii. No justification is recorded. CONVERSION_NOTES.md Step 6 "Code inefficiencies identified: [Note]" and "Code speedups added: [Note]" were never filled in, and Step 7's "Run Time Estimates" tables are empty, so the instruction to estimate runtime and identify bottlenecks was not carried out in writing. The implicit justification is that the 198.78 s total runtime is far under the 15-minute threshold at which the instructions require optimisation.

## 6-c. What processing does the code repeat multiple times?

i. Two real repeats, both small:
- **Redundant dtype round-trip on the trace.** `trace[day]` is materialised as a full `float32` copy, then `nan_to_num` produces a *second* full copy, and then the data are immediately down-cast to `uint8` per trial. For an animal with ~950 cells × 72k frames that is two ~270 MB temporaries per session to produce ~68 MB of `uint8`. Converting to `uint8` (or `bool`) right after masking would have avoided both. `nan_to_num` on the trace is doubly redundant: the `valid_neurons` mask has already removed every row containing a non-finite value.
- **Position binning computed twice** when `--show-processing` is active: once inside `split_session_into_trials` and again in `convert_dataset` for the plot, over the same `pos_day[:, :used]`. The already-computed `pos_bins` could have been returned and reused.
- Minor: `flatten_env_name(envs[day])` is called twice per session (for `session_id` and for `session_info['environment']`).

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)   # copy 1
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]                   # copy 2
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)  # copy 3, no-op
...
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))  # down-cast
```
```python
    pos_bins = position_to_bins_3x3(pos_day)          # inside split_session_into_trials
...
            if show_processing and plotted < 2:
                pos_bins = position_to_bins_3x3(pos_day[:, :used])   # recomputed for the plot
```
```python
session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
...
    'environment': flatten_env_name(envs[day]),
```

iii. Not documented — the Step 6 inefficiency notes are placeholders. The `float32` intermediate is a leftover from the pre-`uint8` version of the script: trajectory step 203/204 shows the `uint8` cast was retro-fitted at the trial-append site ("patched `convert_data.py` to (1) drop non-finite neurons per session/day, (2) convert neural trials to uint8, and (3) convert outputs to uint8") without revisiting the upstream `float32` load, and the `nan_to_num` was added in the same patch without noticing the row-drop already made it unnecessary.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, all cheap:
- **`np.nan_to_num` on `trace_day` and on `pos_day`** — provably no-ops on this dataset (the neuron mask already removes all non-finite trace rows; no session has a non-finite `position`). Likewise `min(trace.shape[1], pos.shape[1])` and the second `np.clip(xbin, 0, 2)` in `position_to_bins_3x3` (the earlier clip on the coordinate already bounds the result) can never change anything here. These are defensive, not wasted in any meaningful sense.
- **`envs` / `session_id` / `session_info`** — the full per-session metadata record (environment name, day index, dropped-neuron count, raw vs used frame counts, blocked vector) is computed and pickled but never read by `train_decoder.py`. It is useful provenance for a human, so this is arguably worth the cost.
- **The `float32` intermediate** described in 6-c is genuinely discarded work: the values end up as `uint8`.
- **`uint8` storage itself is partially wasted downstream**: `train_decoder.py` emits one warning per trial (8,187 of them in `verification_full_out.txt`) saying "neural dtype is uint8, expected float32. Will be converted during training", so the decoder re-materialises float32 copies at training time anyway. The dataset is still 4.98 GB on disk.
- The reference-format fields `SFPs`, `centroids`, and the precomputed `maps` are correctly *not* loaded, so no work is wasted there.

ii.
```python
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
```
```python
    xbin = np.clip(xbin, 0, N_POS_BINS - 1)   # already bounded by the clip on x
    ybin = np.clip(ybin, 0, N_POS_BINS - 1)
```
```python
            session_info.append({
                'session_id': session_id, 'animal': animal, 'day_index': int(day),
                'environment': flatten_env_name(envs[day]),
                'n_neurons': int(trace_day.shape[0]),
                'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
                'n_frames_raw': int(n_frames), 'n_frames_used': int(used),
                'n_trials': int(n_trials),
                'blocked_vector': blocked_vec.astype(int).tolist(),
            })
...
data['metadata']['session_info'] = session_info
```

iii. Not documented as unnecessary. The `session_info` block is implicitly justified by the instructions' target-format note "Add other relevant fields, e.g. `session_info`". The `uint8` decision is justified in trajectory step 202/203 on file-size grounds ("`trace` is binary and could be stored as uint8/bool to drastically reduce file size"), but the resulting per-trial dtype warnings were never addressed — CONVERSION_NOTES.md Step 8 claims "Warnings: none after regeneration; verification completed successfully" and Step 10 Check 1 claims "`verification_full_out.txt` completed with no errors", neither of which acknowledges the 8,187 dtype warnings actually present in the log.
