# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the original `.mat` files. It hardcodes the seven animal IDs, then loads one extensionless per-animal joblib file from `data/` for each animal. Each loaded object is a dictionary keyed by animal ID and contains per-session arrays such as `trace`, `position`, and `envs`. Trials are not loaded directly; they are created later by splitting each session.

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

iii. The notes and trajectory justify this by saying the reference repository works with per-animal joblib-converted files, and the agent treated those as the native working format used by the reference code.

## 1-b. How are the data split into subjects?

i. Each hardcoded animal ID is treated as one subject. The `subjects` list is just the hardcoded animal list, and `subject_idx` records which sessions came from which animal.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
...
for animal in animals:
    ...
    subject_idx.append(subject_to_idx[animal])
```

iii. The trajectory and notes state that each animal file corresponds to one mouse, so the file/animal identity serves as the subject identifier.

## 1-c. How are the data split into sessions?

i. Within each animal, the code iterates over `envs` and uses the same session index to pull `trace[s]` and `position[s]`. Each index `s` becomes one output session.

ii.
```python
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
    if len(nt) < 2:
        continue
    neural.append(nt)
    inp.append(it)
    out.append(ot)
```

iii. The notes describe the joblib files as per-animal arrays with a session axis, so iterating over `envs` was the agent’s way of enumerating sessions consistently across arrays.

## 1-d. How are the data split into trials?

i. Each session is split into consecutive non-overlapping 1-minute trials at 30 Hz. The code first infers how many valid frames exist, computes `n_trials = valid_frames // 1800`, keeps only whole trials, and slices the neural and behavioral arrays into trial-length chunks.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
...
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
n_trials = valid_frames // FRAMES_PER_TRIAL
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The notes say the decoder task requires at least two trials per session and that 1-minute trialization was chosen to match the task instructions while preserving native 30 Hz alignment.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Instead, incomplete tail segments are dropped by integer division into full 1-minute chunks, and sessions are skipped if they yield fewer than two full trials or no remaining finite neurons.

ii.
```python
n_trials = valid_frames // FRAMES_PER_TRIAL
if n_trials < min_trials:
    return [], [], []
...
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The trajectory explicitly ties the `min_trials` requirement to decoder evaluation constraints. No separate justification for per-trial QC was documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-session `trace` arrays in the joblib animal files.

ii.
```python
traces = rec['trace']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The notes identify `trace` as the stored neural variable and state that the reference analyses use `trace` directly.

## 2-b. How is the `neural` data processed?

i. The code keeps the stored trace values directly, truncates to the kept frame window, casts to `float32`, and preserves the `(neurons, time)` orientation already present in the joblib data. It does not recompute calcium preprocessing.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
...
neural_trials.append(trace_sxn[:, sl])
```

iii. The notes and trajectory justify this by saying the stored traces already appear to be binary rising-phase calcium-event vectors, matching the paper’s preprocessing endpoint.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code removes any neuron that has any non-finite value over the retained part of the session. If no neurons survive, the session is discarded.

ii.
```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The only explicit rationale in the trajectory is that verification initially complained about non-finite neural values, so the agent patched the converter to “drop non-finite neurons per session.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. Trials are aligned only to session time, by cutting the continuous recording into consecutive 1-minute chunks.

ii.
```python
'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The notes explicitly say there is no stimulus/event alignment available and that native within-session alignment between trace and position should be preserved.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native 30 Hz frame rate, with a time bin size of `1000 / 30` ms. No temporal rebinning is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes cite the paper’s statement that behavioral and imaging streams were acquired simultaneously at 30 Hz, so the agent kept the native frame resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. In the implemented code, environment geometry is derived from the per-session environment label `env_name`, which comes from `rec['envs']`. The notes also say `blocked` would be used as a cross-check, but that cross-check is not implemented in `convert_data.py`.

ii.
```python
envs = rec['envs'].ravel().tolist()
...
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The trajectory says the agent preferred the named geometry labels because the decoder input should represent blocked parts of the arena, and it wanted to follow the reference `get_env_mat` mapping.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment label is mapped through `get_env_mat` to a 3×3 binary template of open/blocked bins, then flattened to a 9-element static vector. That same vector is copied into every trial from that session.

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    ...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
input_trials.append(env_vec.copy())
```

iii. The notes say the decoder input should be a 9-dimensional geometry/block mask over the conceptual 3×3 partition, derived from the reference environment mapping.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the per-session `position` arrays in the joblib animal files.

ii.
```python
positions = rec['position']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The notes identify `position` as the 2D tracked mouse location stream and tie it to the paper’s DeepLabCut-based position tracking.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code clips `x` and `y` into the 75 cm arena, converts each coordinate to one of three bins per axis by scaling and flooring, then combines them into a single class index `y_bin * 3 + x_bin`.

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

iii. The notes justify this by citing the paper’s conceptual 3×3 partition of the 75×75 cm arena and the decoder task’s required 9-way position output.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholding is done by uniform 3-bin partitioning along each axis of the 75 cm arena. Category labels run from 0 to 8 after combining the `x` and `y` bins.

ii.
```python
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
xb = np.clip(xb, 0, 2)
yb = np.clip(yb, 0, 2)
return (yb * 3 + xb).astype(np.int64)
```

iii. The trajectory says the agent wanted the output to match the requested 3×3 spatial discretization directly.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned frame-for-frame with neural data by slicing both arrays with the same per-trial `slice` object after truncation to the retained session window.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    output_trials.append(pos_bins[sl][None, :])
```

iii. The notes explicitly state that native 30 Hz alignment between trace and position should be preserved when trializing sessions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missingness by finding the last frame where both position is finite and at least one neuron is finite, truncating the session there, removing neurons with any non-finite values in the retained window, and skipping sessions that do not leave enough data for at least two full trials.

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
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The clearest explicit justification is from the trajectory after verification failures: the agent patched the code to remove non-finite neurons so the verifier would pass. It did not document a broader missing-data policy beyond that.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the large per-animal joblib files and iterating serially over all sessions to slice large neural/position arrays into trials. Full conversion took about 291 seconds according to `conversion_full_out.txt`.

ii.
```python
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    ...
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. There is almost no explicit efficiency analysis in the notes. The only direct evidence is the measured runtime written to stdout and the fact that the code is dominated by file loading plus large array slicing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate is the per-trial Python loop inside `session_to_trials`, which repeatedly slices arrays and appends trial objects. The code could also avoid repeated `env_vec.copy()` calls by constructing the static trial input list differently.

ii.
```python
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The agent did not explicitly justify leaving this loop in Python. This is inferred directly from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. It repeatedly creates identical static geometry vectors for every trial within a session, re-runs finite-value scans for every session, and recomputes the same per-session trial-splitting logic separately for sample and full runs.

ii.
```python
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
input_trials.append(env_vec.copy())
```

iii. No explicit justification was documented. The repetition is visible in the code structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does not perform any large unused transformation, but it does allocate a separate copy of the same static geometry vector for every trial, keeps unused CLI/import scaffolding (`math`, `--show-processing` placeholder), and computes metadata fields such as `SESSION_SECONDS` that do not affect downstream decoding.

ii.
```python
import math
...
SESSION_SECONDS = 40 * 60
...
input_trials.append(env_vec.copy())
...
if args.show_processing:
    print('show-processing requested; plotting not yet implemented')
```

iii. The notes do not justify these extra pieces; they are simply present in the final script.
