# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files in the `data/` directory. It uses a hardcoded list of 7 animal IDs (`ANIMALS`). For each animal, it loads the joblib file, accesses the nested dictionary keyed by animal ID, and extracts `trace`, `position`, and `envs` arrays.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    envs = rec['envs'].ravel().tolist()
    traces = rec['trace']
    positions = rec['position']
```

iii. The AI chose to use the pre-converted joblib files rather than raw `.mat` files, matching the reference code's `load_dat` function which also loads from joblib format.

## 1-b. How are the data split into subjects?

i. Each entry in the hardcoded `ANIMALS` list corresponds to one subject. The subject name is the animal ID string.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
```

iii. Each joblib file contains all recording sessions for one animal. The 7 animals correspond to the 7 subjects in the dataset.

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple recording sessions indexed along the first dimension of the `trace`, `position`, and `envs` arrays. Each recording session becomes a separate session in the output.

ii.
```python
envs = rec['envs'].ravel().tolist()
traces = rec['trace']
positions = rec['position']
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The data structure has sessions as the first dimension of the arrays. The AI iterates over each session index.

## 1-d. How are the data split into trials?

i. Each 40-minute session is split into consecutive 1-minute (1800-frame) non-overlapping trials. A valid-frame detection step (`infer_valid_frames`) first determines how many frames contain valid (finite) data, then only complete 1-minute chunks within that range are kept.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS  # 30 * 60 = 1800
...
def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    ...
    for t in range(n_trials):
        sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
        neural_trials.append(trace_sxn[:, sl])
```

iii. The instruction specifies 1-minute trials. The valid-frame detection ensures trailing invalid frames are excluded before trial splitting.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 valid trials are entirely excluded. The valid-frame inference discards trailing frames where position or neural data are non-finite. No per-trial quality filtering is applied beyond this.

ii.
```python
def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    if n_trials < min_trials:
        return [], [], []
```

iii. The minimum trial check ensures every session has enough data for train/test split in the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains calcium imaging traces (binary rising-phase vectors) with shape `(n_neurons, n_frames_max)` per session.

ii.
```python
traces = rec['trace']
...
trace_sxn = traces[s]  # shape: (n_neurons, n_frames)
```

iii. The `trace` arrays are already preprocessed rising-phase binary vectors, as confirmed by the paper's methods.

## 2-b. How is the `neural` data processed?

i. The trace data is cast to float32, truncated to valid frames, and filtered to remove neurons with any non-finite values. No additional processing (e.g., smoothing, rebinning) is applied.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
```

iii. The stored traces are already binarized rising-phase vectors. The AI uses them directly, which matches the reference code approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that have any non-finite (NaN/Inf) values across the valid time range are removed on a per-session basis. This filters out neurons not actually recorded in that session.

ii.
```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. This ensures only neurons with complete data for the session are included. Sessions with zero valid neurons are skipped.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are consecutive 1-minute segments starting from the beginning of the session.

ii. N/A - no alignment code.

iii. There is no stimulus onset or behavioral event to align to. Trials are arbitrary temporal segments of a continuous recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The acquisition rate is 30 Hz for both neural and behavioral streams, so no rebinning is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable, which contains environment name strings (e.g., 'square', 'o', 't', 'u') for each session. These are converted to geometry matrices using `get_env_mat()`.

ii.
```python
envs = rec['envs'].ravel().tolist()
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
```

iii. The reference code contains a `get_env_mat` function that maps environment names to 3x3 binary matrices indicating open/blocked positions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via `get_env_mat()`, then flattened to a 9-element vector. A value of 1 means the grid cell is accessible; 0 means blocked.

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    ...
```

iii. This mapping is taken directly from the reference code, ensuring consistency with the original analysis.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The input is static per session (same geometry for all trials within a session), so no temporal alignment is needed. The same 9-element vector is replicated for every trial.

ii.
```python
input_trials.append(env_vec.copy())
```

iii. Environment geometry doesn't change within a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena over time.

ii.
```python
positions = rec['position']
pos_sxn = positions[s]  # shape: (2, n_frames)
```

iii. Position was tracked with DeepLabCut at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes) by dividing each axis into 3 equal bins over the 75 cm arena. The grid label is `y_bin * 3 + x_bin`.

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

iii. The 3x3 discretization matches the task instructions requiring position discretized into 3x3=9 spatial bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is divided into 9 categories using equal-width bins. Each 25 cm segment of the 75 cm arena defines one bin edge. Values at/beyond the boundary are clipped to the nearest valid bin.

ii. See 4-b code snippet above. `np.floor((x / arena_size) * 3)` creates bins [0, 25), [25, 50), [50, 75).

iii. Equal-width binning across the arena is the natural choice matching the paper's 3x3 grid partition.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are sampled at the same 30 Hz rate and share the same frame indices. Both are split into trials using the same slice indices, ensuring frame-for-frame alignment.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    ...
    output_trials.append(pos_bins[sl][None, :])
```

iii. The DAQ acquired behavioral and neural streams simultaneously at 30 Hz, so no interpolation or resampling is needed.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e: 30 Hz native (~33.33 ms bins), no rebinning.

ii.
```python
FPS = 30
'time_bin_size': 1000.0 / FPS,
```

iii. Both streams are already at the same temporal resolution.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position data are frame-aligned from the source (same 30 Hz acquisition). Input (geometry) is static per trial. Both neural and output are sliced using identical frame indices within each trial.

ii. See trial splitting code in 1-d and 4-d.

iii. Simultaneous acquisition ensures inherent alignment.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI handles: (1) trailing invalid frames via `infer_valid_frames`, which finds the last frame where both position and neural data are finite; (2) non-finite neurons are removed per-session; (3) sessions with <2 valid trials or zero valid neurons are skipped entirely; (4) remainder frames that don't fill a complete trial are discarded.

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

iii. This approach handles sessions that may have been terminated early or have trailing NaN data.

## 7-a. What are the most time-consuming steps of the code?

i. Loading the joblib files containing large trace arrays is the most time-consuming step (I/O bound). The full conversion took ~291 seconds.

ii. N/A

iii. The trace arrays are large (hundreds of neurons x ~72000 frames per session) and there are 207 sessions across 7 animals.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials to create slices, but this is already lightweight. The main per-session loop is inherently sequential due to I/O.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
```

iii. The trial splitting could theoretically be done with `np.split` or reshaping, but the performance impact is minimal.

## 7-c. What processing does the code repeat multiple times?

i. No significant repeated processing is evident. Each session is processed once.

ii. N/A

iii. The code processes each session independently in a single pass.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No obviously unnecessary processing. The code is fairly streamlined. The `--show-processing` flag is accepted but not implemented ("plotting not yet implemented").

ii.
```python
if args.show_processing:
    print('show-processing requested; plotting not yet implemented')
```

iii. The code does the minimum required processing.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6: trailing invalid frames detected and excluded, non-finite neurons removed, sessions with insufficient data skipped.

ii. See code in question 6.

iii. The approach is conservative, preferring to exclude suspect data rather than impute.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading joblib files is the bottleneck.

ii. N/A

iii. N/A

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b.

ii. N/A

iii. N/A

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: no significant repeated processing.

ii. N/A

iii. N/A

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: no obviously unnecessary processing.

ii. N/A

iii. N/A
