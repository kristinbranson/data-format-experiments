# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the original `.mat` files. Instead, it hardcodes the seven animal IDs and loads one extensionless per-animal joblib file from `data/<animal>` for each subject. Within each loaded record it reads the `envs`, `trace`, and `position` arrays, then later splits each session into trials.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']

for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    envs = rec['envs'].ravel().tolist()
    traces = rec['trace']
    positions = rec['position']
```

iii. In `CONVERSION_NOTES.md`, the AI says the extensionless files are “joblib-converted files used by the reference code.” In the trajectory it explicitly framed this as following the paper utilities (`load_dat`, `mat2joblib`) rather than reading the original `.mat` files directly.

## 1-b. How are the data split into subjects?

i. Subjects are the hardcoded animal names in `ANIMALS`. The script copies that list into `subjects` and builds a `subject_to_idx` map from it.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
```

iii. The notes list exactly seven animals and describe each extensionless file as a per-animal record keyed by animal ID, so the AI treated those animal IDs as the subject identifiers.

## 1-c. How are the data split into sessions?

i. Each animal’s `envs`, `trace`, and `position` arrays are treated as session-major arrays. The code iterates over `enumerate(envs)` and uses the same session index `s` to pull one session from `trace` and `position`.

ii.
```python
envs = rec['envs'].ravel().tolist()
traces = rec['trace']
positions = rec['position']
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the AI documented `envs` as shape `(n_sessions, 1)`, `position` as `(n_sessions, 2, n_frames_max)`, and `trace` as `(n_sessions, n_neurons, n_frames_max)`, and used that as the basis for session splitting.

## 1-d. How are the data split into trials?

i. Trials are 60-second non-overlapping chunks at 30 Hz, so 1800 frames per trial. For each session, the AI first infers how many valid frames exist, computes `n_trials = valid_frames // 1800`, truncates to a whole number of trials, and slices consecutive 1-minute windows.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS

def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    if n_trials < min_trials:
        return [], [], []
    n_keep = n_trials * FRAMES_PER_TRIAL
    ...
    for t in range(n_trials):
        sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
        neural_trials.append(trace_sxn[:, sl])
        input_trials.append(env_vec.copy())
        output_trials.append(pos_bins[sl][None, :])
```

iii. The notes and trajectory repeatedly refer to the instructed “1-minute trials.” The trajectory also shows the AI checking examples like `71866 // 1800 = 39` and treating the leftover frames as discarded remainder.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control rule beyond truncating each session to the last jointly valid frame and dropping any leftover partial trial. Sessions with fewer than two full trials are skipped entirely.

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
if n_trials < min_trials:
    return [], [], []
```

iii. The trajectory says this logic was introduced to handle padded session arrays and to satisfy the decoder requirement that each session contain at least two trials. No additional trial-level QC is recorded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-session slices of `rec['trace']`.

ii.
```python
rec = joblib.load(Path('data') / animal)[animal]
traces = rec['trace']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. Step 2 of `CONVERSION_NOTES.md` identifies `trace` as the calcium-trace array, and Step 5 maps `rec['trace'][session]` directly to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The AI assumes the stored traces are already the final neural representation. It does not recompute dF/F, derivatives, smoothing, or thresholding. It clips each session to the inferred valid frames, casts to `float32`, removes non-finite neurons, and then slices trials.

ii.
```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
...
trace_sxn = trace_sxn[finite_neurons]
...
neural_trials.append(trace_sxn[:, sl])
```

iii. In Steps 3-5 of `CONVERSION_NOTES.md`, the AI argues that the released `trace` arrays are already “binary rising-phase vectors treated as firing rates,” consistent with the paper methods, so the code should use them directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. After clipping to the kept portion of a session, the AI removes any neuron that has any non-finite value anywhere in that kept span.

ii.
```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The trajectory shows this was added only after verification failed with “NaN or Inf values” in some sessions. The AI justified it as removing neurons that were “not present/registered in some sessions.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The AI treats session time as the alignment axis and creates consecutive 1-minute windows from the start of the valid recording.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
    'off_start': 0.0,
    'off_end': float(TRIAL_SECONDS),
    ...
}
```

iii. The notes repeatedly say there is no native trial/event structure and that the trials are artificial chunks of a continuous session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz frame rate, so each time bin is `1000 / 30` ms. No temporal rebinning is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes cite the paper’s 30 Hz acquisition and say the stored traces and behavior are already aligned at that resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. In the final code, environment geometry is derived from `rec['envs']` only. The code does not read `rec['blocked']`.

ii.
```python
envs = rec['envs'].ravel().tolist()
...
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. In Step 5 and the trajectory, the AI says geometry is derived “primarily from `envs` via the reference `get_env_mat` mapping,” with `blocked` mentioned only as a possible cross-check. That cross-check never appears in the final script.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code maps each named environment to a hardcoded 3x3 binary matrix of open/blocked arena cells, flattens it to length 9, and reuses that static vector for every trial in the session.

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

iii. The AI’s notes say the decoder input should represent “which arena partitions are blocked” and that the reference environment mapping was the right way to encode this. The README in the trajectory also calls it a “9-dimensional static geometry vector derived from the reference environment mapping.”

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the per-session slices of `rec['position']`.

ii.
```python
positions = rec['position']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. Step 2 in the notes identifies `position` as the x/y trajectory array, and Step 5 maps it directly to the decoder output after discretization.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips x and y into the arena bounds, scales each coordinate into three equal bins across a 75 cm arena, and combines the x-bin and y-bin into a single label from 0 to 8.

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

iii. The notes cite the paper’s conceptual 3x3 partition of the 75 x 75 cm square and use that as the justification for the discretization.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each spatial axis is thresholded into thirds of the arena. The final category is `y_bin * 3 + x_bin`, giving 9 categories `0..8`.

ii.
```python
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
xb = np.clip(xb, 0, 2)
yb = np.clip(yb, 0, 2)
return (yb * 3 + xb).astype(np.int64)
```

iii. The AI’s recorded justification is the same as in 4-b: the target task asked for a 3x3 discretization and the notes tie that to the paper’s 3x3 conceptual partition.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are both clipped to the same `n_keep` frames and then sliced with the same per-trial indices, so alignment is frame-for-frame inside each session.

ii.
```python
n_keep = n_trials * FRAMES_PER_TRIAL
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_bins = discretize_position_3x3(pos_sxn)
...
sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
neural_trials.append(trace_sxn[:, sl])
output_trials.append(pos_bins[sl][None, :])
```

iii. The notes describe the raw `trace` and `position` streams as already synchronized at 30 Hz, and the trajectory includes explicit raw-vs-converted spot checks that `np.allclose` held after using the same slices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing/padded data by inferring the last frame where both position and at least one neural value are finite, truncating the session there, removing neurons with any non-finite values in the kept span, discarding leftover partial trials, and skipping sessions that become empty or have fewer than two full trials.

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

iii. The trajectory shows this was motivated by two observations: the joblib arrays looked padded to common lengths, and verification complained about NaN/Inf values until non-finite neurons were removed.

## 6-a. What are the most time-consuming steps of the code?

i. The code suggests the most time-consuming work is loading the large per-animal joblib files and iterating through every session/trial to build the Python lists. The script times the whole conversion and the full run took about 291 seconds.

ii.
```python
t0 = time.time()
data = convert_dataset(sample=sample)
...
print('elapsed_sec', round(time.time() - t0, 3))
```

iii. The trajectory discusses sample conversion taking about 66 seconds and the full conversion taking about 291 seconds, and it treats reruns as expensive but manageable. No more detailed profiling was implemented.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI left two obvious Python loops unvectorized: the nested subject/session loops in `convert_dataset`, and the per-trial loop in `session_to_trials` that appends slices one at a time.

ii.
```python
for animal in animals:
    ...
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. There is no explicit recorded justification for leaving these loops in place. The trajectory mentions the possibility of vectorization in the abstract, but the final code never applies it.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly reconstructs static per-session geometry vectors, copies that same vector once per trial, and repeatedly slices Python lists trial-by-trial for every session. In the trajectory, the AI also reran the sample and full conversion multiple times after verification failures.

ii.
```python
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
for t in range(n_trials):
    ...
    input_trials.append(env_vec.copy())
```

iii. No explicit efficiency justification was recorded. The repeated trial copying appears to be a straightforward implementation choice rather than a deliberate optimization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes a few pieces of work that do not help downstream decoding: it imports `math` but never uses it, exposes `--show-processing` but does not implement any plotting, and copies identical static geometry vectors into every trial instead of reusing one object.

ii.
```python
import math
...
ap.add_argument('--show-processing', action='store_true')
...
if args.show_processing:
    print('show-processing requested; plotting not yet implemented')
```

```python
input_trials.append(env_vec.copy())
```

iii. The only explicit justification in the trajectory is that plotting for `--show-processing` was “not yet implemented.” Otherwise these appear to be leftover or convenience choices rather than purposeful downstream processing.
