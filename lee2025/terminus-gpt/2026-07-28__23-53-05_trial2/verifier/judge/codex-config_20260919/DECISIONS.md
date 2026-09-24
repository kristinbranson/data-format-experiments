# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses a hard-coded list of seven animals and loads each animal's extensionless, repository-preconverted joblib file from `data/`. It extracts the animal dictionary's `envs`, `trace`, and `position` arrays, then iterates through every environment/session. Unlike the reference, it does not load the original `.mat` files or the raw `blocked` field.

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

iii. The notes say these joblib files are the converted files used by the reference repository and contain all seven animals and 207 sessions. The agent chose them as Python-friendly versions of the MATLAB data.

## 1-b. How are the data split into subjects?

i. Each entry in the fixed `ANIMALS` list is one subject. Its list position defines `subject_idx`; in full mode all seven IDs are retained.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[animal])
```

iii. The notes identify the seven per-animal files and confirm their session counts sum to the paper's 207 sessions.

## 1-c. How are the data split into sessions?

i. The first dimension of each animal's `trace` and `position` arrays and each flattened `envs` entry represents a recording session. Each accepted source session becomes one output session.

ii.
```python
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
    if len(nt) < 2:
        continue
    neural.append(nt)
```

iii. The notes describe the source arrays as `(n_sessions, ...)` and report that conversion retained all 207 source sessions.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, nonoverlapping 60-second trials at 30 Hz (1,800 frames). Only complete trials before the inferred valid endpoint are retained, so the available files produce 39 trials per session.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
...
n_trials = valid_frames // FRAMES_PER_TRIAL
n_keep = n_trials * FRAMES_PER_TRIAL
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
```

iii. The requested artificial trial length is one minute. The notes acknowledge that padded arrays are slightly shorter than a nominal 40 minutes after valid-frame trimming and therefore yield 39 complete trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no individual-trial behavioral quality filter. Incomplete trailing data are dropped, and an entire session is rejected if it has fewer than two complete trials or no finite neurons.

ii.
```python
if n_trials < min_trials:
    return [], [], []
...
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The two-trial minimum implements the target-format requirement. No additional trial QC was documented because the source experiment has continuous sessions rather than native trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each session of the joblib animal record's `trace` array.

ii.
```python
traces = rec['trace']
...
session_to_trials(traces[s], positions[s], env_name)
```

iii. The agent inspected the values and concluded that stored traces are already sparse binary rising-phase calcium-event vectors described by the methods.

## 2-b. How is the `neural` data processed?

i. The stored neuron-by-time traces are truncated to complete trials, converted to `float32`, filtered to finite neurons, and sliced without rebinning. The agent does not recompute fluorescence derivatives, smoothing, z scores, or event thresholds.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
...
neural_trials.append(trace_sxn[:, sl])
```

iii. The notes state that sampled traces contain only binary values and that reference analysis code uses `trace` directly, indicating that the paper's rising-phase preprocessing had already occurred.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained only if every value over the retained complete-trial interval is finite. A session with no retained neuron is dropped. No place-cell reliability filtering is applied.

ii.
```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The agent reasoned that released traces already reflect artifact curation and that place-cell filtering is analysis-specific and inappropriate for a general neural decoder. Finite filtering removes cross-session NaN padding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Time zero is the beginning of each consecutive one-minute session chunk.

ii.
```python
'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The experiment is continuous and has no trial event; the notes say native simultaneous acquisition is preserved.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data remain at the native 30 Hz resolution, or 33.333 ms per sample. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The methods and notes report simultaneous behavioral and imaging acquisition at 30 Hz, so keeping native samples preserves alignment.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the session's string label in `rec['envs']`, not directly from `rec['blocked']` as in the human reference.

ii.
```python
envs = rec['envs'].ravel().tolist()
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
```

iii. The agent says the named-environment mapping follows the repository's `get_env_mat`; it planned to use `blocked` only as a cross-check, but the final code does not perform that check.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `get_env_mat` maps each of ten geometry names to a 3×3 binary matrix, flattens it to nine values, and copies the same static vector into every trial. Here `1` means accessible/open and `0` means blocked, the inverse semantics of the reference's blocked indicator.

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

iii. The notes justify a nine-dimensional mask as a direct representation of the conceptual 3×3 arena geometry and report following the repository's environment mapping.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the per-session x/y trajectories in `rec['position']`.

ii.
```python
positions = rec['position']
...
session_to_trials(traces[s], positions[s], env_name)
```

iii. The notes identify these as DeepLabCut x/y trajectories aligned to the neural recordings.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are clipped to the 0–75 cm arena, converted to x and y thirds, combined into one row-major categorical label, and given a singleton output dimension.

ii.
```python
x = np.clip(position_xy[0], 0, arena_size - 1e-6)
y = np.clip(position_xy[1], 0, arena_size - 1e-6)
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
return (yb * 3 + xb).astype(np.int64)
...
output_trials.append(pos_bins[sl][None, :])
```

iii. The task explicitly requires a time-varying 3×3 position category, and the paper describes a 75×75 cm arena partitioned conceptually into a 3×3 grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is assigned to `[0,25)`, `[25,50)`, or `[50,75]` cm after clipping. The final class is `3*y_bin + x_bin`, producing labels 0–8.

ii.
```python
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
xb = np.clip(xb, 0, 2)
yb = np.clip(yb, 0, 2)
return (yb * 3 + xb).astype(np.int64)
```

iii. Equal-width thirds implement the requested 3×3 discretization; clipping guarantees valid edge labels.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural arrays share native frame indices. Both are truncated to the same `n_keep`, and the same slice is used for each trial.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
...
neural_trials.append(trace_sxn[:, sl])
output_trials.append(pos_bins[sl][None, :])
```

iii. The methods state that imaging and behavior were simultaneously acquired at 30 Hz; the agent also spot-checked converted positions against direct source discretization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The last index where both position is fully finite and at least one neural value is finite defines the usable endpoint. Partial trailing trials are discarded. Neurons with any nonfinite value in retained time are removed, and empty/too-short sessions are omitted. Coordinates outside the arena are clipped.

ii.
```python
pos_valid = np.all(np.isfinite(position), axis=0)
neural_valid = np.any(np.isfinite(trace), axis=0)
valid = pos_valid & neural_valid
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
...
x = np.clip(position_xy[0], 0, arena_size - 1e-6)
```

iii. The agent treated NaNs as array padding or unavailable neurons and chose complete one-minute trials. The notes explicitly flag the loss of the nominal final minute for review but do not subsequently resolve it.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant operations are loading the large compressed joblib animal records and serializing the roughly 20 GB output pickle. Repeated full-array finite scans and materializing/copying every trial also consume time and memory.

ii.
```python
rec = joblib.load(Path('data') / animal)[animal]
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes do not provide measured step timings. They identify large-data loading as part of conversion and report the final pickle size, but leave the runtime table blank.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop could be replaced with reshaping/view construction for neural and output arrays; input vectors could be shared rather than copied. Animal and variable-length session loops are naturally retained.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The agent did not document any vectorization decision; its “code inefficiencies” and “speed-ups” sections are blank.

## 6-c. What processing does the code repeat multiple times?

i. For every session it rescans trace and position arrays for validity, recreates the geometry matrix from a repeated string label, discretizes all positions, and creates a separate copy of the same geometry vector for every trial.

ii.
```python
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
input_trials.append(env_vec.copy())
```

iii. No justification or explicit discussion of repeated processing appears in the notes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is little computed scientific content that is later discarded. The validity masks and intermediate clipped/discretized coordinate arrays are temporary but necessary. `math` is imported but unused, and repeated input copies unnecessarily inflate work and storage; `SESSION_SECONDS` is used only as metadata.

ii.
```python
import math
...
input_trials.append(env_vec.copy())
...
'session_duration_s': SESSION_SECONDS,
```

iii. The agent did not identify discarded processing in its notes.
