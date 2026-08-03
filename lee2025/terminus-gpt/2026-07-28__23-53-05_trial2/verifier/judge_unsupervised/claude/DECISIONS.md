# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files in the `data/` directory. For each of the 7 animals, it calls `joblib.load(Path('data') / animal)` and indexes into the result with the animal ID key to get a dictionary containing `trace`, `position`, `envs`, and other fields. It iterates over all environments/sessions within each animal and converts each session into trials.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']

def convert_dataset(sample=False):
    animals = ANIMALS[:2] if sample else ANIMALS
    # ...
    for animal in animals:
        rec = joblib.load(Path('data') / animal)[animal]
        envs = rec['envs'].ravel().tolist()
        traces = rec['trace']
        positions = rec['position']
        for s, env_name in enumerate(envs):
            nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The AI identified from the reference code (`load_dat`, `mat2joblib`) that the data were stored as joblib-serialized per-animal dictionaries converted from MATLAB files. The AI confirmed this by exploring the data directory structure in Step 2 of its workflow.

## 1-b. How are the data split into subjects (mice)?

i. Each animal is a separate joblib file. The AI iterates over a hardcoded list of 7 animal IDs. Each animal corresponds to one subject. The `subjects` list and `subject_idx` array map sessions to their respective animals.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']

subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
# ...
subject_idx.append(subject_to_idx[animal])
```

iii. The AI noted 7 subjects from the data files, consistent with the paper's dataset. Each animal's data is stored in a separate file keyed by its ID.

## 1-c. How are the data split into sessions?

i. Within each animal, sessions correspond to entries in the `envs` array. The AI iterates over all environment entries (sessions) for each animal. Sessions with fewer than 2 valid 1-minute trials are skipped. The result is 207 sessions total (31 sessions each for 6 animals, 21 for QLAK-CA1-51).

ii.
```python
envs = rec['envs'].ravel().tolist()
traces = rec['trace']
positions = rec['position']
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
    if len(nt) < 2:
        continue
    neural.append(nt)
```

iii. The AI confirmed from both data exploration and the paper that there are 207 sessions across 7 animals. The `envs` array indexes sessions, with each session recorded in a different environment geometry on a different day.

## 1-d. How are the data split into trials?

i. Each 40-minute session is split into consecutive 1-minute (1800-frame at 30 Hz) non-overlapping chunks. The number of complete trials is determined by integer division of valid frames by 1800. Most sessions yield 39 trials (due to ~71,866 valid frames being slightly less than 40*1800=72,000); some yield 40.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS  # 1800

def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    if n_trials < min_trials:
        return [], [], []
    n_keep = n_trials * FRAMES_PER_TRIAL
    trace_sxn = trace_sxn[:, :n_keep]
    # ...
    for t in range(n_trials):
        sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
        neural_trials.append(trace_sxn[:, sl])
```

iii. The AI followed the task instructions which specified "1-minute trials within each session." The AI investigated the 39-trial result and confirmed that session frame counts (~71,866) genuinely cannot fit 40 complete 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only filtered implicitly: (1) sessions with fewer than 2 complete 1-minute trials are excluded entirely, and (2) the last partial minute of each session (with incomplete 1800 frames) is discarded. No explicit trial-level quality filtering (e.g., based on behavior, running speed, or tracking quality) is applied.

ii.
```python
def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    if n_trials < min_trials:
        return [], [], []
```

```python
def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
    valid = pos_valid & neural_valid
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1] + 1)
```

iii. The AI uses `infer_valid_frames` to find the last frame where both position and neural data are valid (finite). This determines the usable session length. No additional trial-level quality filtering was applied beyond this. The reference paper does not describe explicit trial-level filtering since it uses full sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field in each animal's data dictionary (`rec['trace'][session]`). This is indexed as `traces[s]` per session, with shape `(n_neurons, n_frames)`.

ii.
```python
traces = rec['trace']
# ...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The AI confirmed that `trace` contains already-preprocessed binary rising-phase vectors (values only 0 and 1, ~1.5% sparsity), consistent with the paper's description that traces were binarized after calcium derivative processing, Gaussian smoothing, and z-scoring with threshold > 2.5.

## 2-b. How is the `neural` data processed?

i. The neural data (binary traces) is used directly without any further processing (no dF/F computation, no smoothing, no rate map computation, no temporal rebinning). The only processing is: (1) filtering out neurons with any non-finite values across the session, (2) casting to float32, and (3) slicing into 1-minute trial chunks.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
# ...
neural_trials.append(trace_sxn[:, sl])
```

iii. The AI verified empirically that stored traces are already binary (0/1 values), matching the methods description of binarized rising-phase vectors. No recomputation of calcium preprocessing was needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Per session, neurons with any non-finite (NaN/Inf) values across the entire session are removed. No place-cell filtering is applied — all recorded neurons that pass the finite-value check are included.

ii.
```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The AI recognized the reference code's place-cell filtering pipeline (split-half reliability, 1000 shuffles, 99th percentile threshold) but chose not to apply it. The agent never explicitly articulated why place-cell filtering was skipped. The CONVERSION_NOTES mention that "place-cell filtering is analysis-specific and may not be appropriate for a general decoder unless required by reference decoding code." The reference decoding function (`decode_position_within`) does optionally use place-cell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to session start time. Since behavioral and neural data were acquired simultaneously at 30 Hz with shared frame indices, no additional temporal alignment is needed. Trials are consecutive 1-minute chunks from session start.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
```

Metadata:
```python
'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),  # 60.0
```

iii. The AI noted that DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz, meaning frame indices are already aligned between position and trace data. The trial alignment event is simply the session start, with trials being consecutive non-overlapping windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native acquisition rate of 30 Hz (33.33 ms per frame). No temporal rebinning is applied.

ii.
```python
FPS = 30
# ...
'time_bin_size': 1000.0 / FPS,  # 33.33 ms
```

iii. The AI preserved the native 30 Hz sampling rate, consistent with the reference paper's acquisition rate. Each trial has exactly 1800 timepoints (60 seconds * 30 Hz).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `envs` field in each animal's data dictionary, which contains string labels for each session's environment (e.g., 'square', 'o', 't', 'u', 'rectangle', '+', 'i', 'l', 'bit donut', 'glenn').

ii.
```python
envs = rec['envs'].ravel().tolist()
# ...
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The AI also explored the `blocked` field as a potential alternative/cross-check but decided to use `envs` mapped through `get_env_mat()` as the primary source.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Each environment name string is mapped to a 3x3 binary matrix (1=accessible, 0=blocked) via the `get_env_mat()` function, then flattened to a 9-dimensional vector. This vector is static (same for every trial within a session).

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    # ... (10 environments total)

env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
input_trials.append(env_vec.copy())
```

iii. The AI reimplemented a `get_env_mat()` function based on the reference code's environment-to-geometry mapping. The 10 environment names and their 3x3 blocked/open patterns are hardcoded.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The mouse position output is derived from the `position` field in each animal's data dictionary (`rec['position'][session]`), which contains x/y coordinates over time with shape `(2, n_frames)`.

ii.
```python
positions = rec['position']
# ...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The AI identified from the reference code and data exploration that `position` contains DeepLabCut-tracked x/y trajectories at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x/y position is discretized into a 3x3 spatial grid over a 75x75 cm arena. Each position at each timepoint is assigned a single bin index (0-8) representing one of the 9 grid cells.

ii.
```python
def discretize_position_3x3(position_xy, arena_size=75.0):
    x = np.clip(position_xy[0], 0, arena_size - 1e-6)
    y = np.clip(position_xy[1], 0, arena_size - 1e-6)
    xb = np.floor((x / arena_size) * 3).astype(int)
    yb = np.floor((y / arena_size) * 3).astype(int)
    xb = np.clip(xb, 0, 2)
    yb = np.clip(yb, 0, 2)
    return (yb * 3 + xb).astype(np.int64)
```

iii. The AI followed the task instruction to discretize position into a 3x3 grid and the paper's description of the arena as 75x75 cm partitioned into a 3x3 grid space.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into 9 categories (bins 0-8) using floor division of the x and y coordinates by one-third of the arena size (25 cm per bin). The bin index is computed as `yb * 3 + xb`, giving a row-major ordering. Values are clipped to [0, arena_size) before binning to handle edge cases.

ii.
```python
x = np.clip(position_xy[0], 0, arena_size - 1e-6)
y = np.clip(position_xy[1], 0, arena_size - 1e-6)
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
xb = np.clip(xb, 0, 2)
yb = np.clip(yb, 0, 2)
return (yb * 3 + xb).astype(np.int64)
```

iii. The AI chose a straightforward floor-based binning into equal-sized thirds of the arena. The bin ordering follows row-major convention (y*3 + x).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame indices (both acquired at 30 Hz simultaneously), so no separate alignment step is needed. Both are sliced using the same trial boundaries.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    # ...
    output_trials.append(pos_bins[sl][None, :])
```

iii. The AI confirmed from the paper that "DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz," ensuring frame-by-frame alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing/invalid data in several ways: (1) Non-finite position values (NaN/Inf) are used to determine the valid frame range via `infer_valid_frames`, which finds the last frame where both position and neural data are finite; (2) Neurons with any non-finite values across the valid session range are entirely removed; (3) Sessions yielding fewer than 2 valid trials are skipped; (4) Position values are clipped to [0, arena_size) before discretization to handle out-of-bounds coordinates.

ii.
```python
def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
    valid = pos_valid & neural_valid
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1] + 1)
```

iii. The AI documented in CONVERSION_NOTES that non-finite neuron filtering was added after encountering issues during the conversion process. The valid-frame inference approach assumes all frames up to the last valid frame are usable, which could include isolated invalid frames in the middle.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib data files for each animal, which are large (containing multi-session arrays of traces, positions, maps, spatial footprints, etc.). The full conversion took ~291 seconds (about 5 minutes). No explicit timing breakdown per step is printed; only the total elapsed time is reported.

ii.
```python
t0 = time.time()
data = convert_dataset(sample=sample)
# ...
print('elapsed_sec', round(time.time() - t0, 3))
```

iii. The AI noted elapsed time of 291 seconds for full conversion. The CONVERSION_NOTES indicate that timing information was planned but the detailed "Speed-ups Implemented" and "Time / Session" tables were left blank.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over each trial within a session, creating slices of the trace and position arrays. This could potentially be vectorized using `np.reshape` to split the arrays into trial-sized chunks in one operation rather than looping.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The AI's CONVERSION_NOTES left the "Code inefficiencies identified" and "Code speedups added" sections blank, indicating no explicit analysis of vectorization opportunities was performed.

## 6-c. What processing does the code repeat multiple times?

i. The `get_env_mat()` function is called once per session, and `env_vec.copy()` is called once per trial within that session. Since the environment geometry is the same for all trials in a session, the `env_vec` computation is not repeated but copies are made for each trial. There is no significant repeated computation in the code.

ii.
```python
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
# ...
for t in range(n_trials):
    input_trials.append(env_vec.copy())
```

iii. The code is relatively simple and does not have major instances of repeated processing. The `env_vec.copy()` per trial is a minor inefficiency but negligible.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads entire animal data dictionaries via `joblib.load()`, which includes large arrays like `SFPs` (spatial footprints), `centroids`, `maps` (rate maps), and `blocked` that are never used in the conversion. Only `trace`, `position`, and `envs` are extracted. Loading these unused fields wastes memory and I/O time.

ii.
```python
rec = joblib.load(Path('data') / animal)[animal]
envs = rec['envs'].ravel().tolist()
traces = rec['trace']
positions = rec['position']
# SFPs, centroids, maps, blocked are loaded but never used
```

iii. The AI did not explicitly address this inefficiency. The joblib files contain the full per-animal data structure including spatial footprints, centroids, precomputed rate maps, and blocked indices, none of which are used in the conversion. Selective loading would reduce memory usage significantly.
