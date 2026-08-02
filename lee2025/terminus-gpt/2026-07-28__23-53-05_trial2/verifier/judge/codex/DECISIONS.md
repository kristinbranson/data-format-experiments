# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, then loads one extensionless per-animal joblib file from `data/` for each animal. From each loaded record it reads `envs`, `trace`, and `position`, and iterates over the session axis to build output sessions and trials.

ii. ```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    envs = rec['envs'].ravel().tolist()
    traces = rec['trace']
    positions = rec['position']
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. In `CONVERSION_NOTES.md`, the agent concluded that the reference repository operates on joblib-converted per-animal files, and the trajectory shows it treating those files as the preferred loading path after exploring `code/georepca1/src/utils.py`. It justified this as matching the reference code’s working format rather than reading the original `.mat` files directly.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by animal ID. Each loaded per-animal file corresponds to one mouse, and `subjects` is just the copied `ANIMALS` list.

ii. ```python
animals = ANIMALS[:2] if sample else ANIMALS
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
...
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
```

iii. The notes record seven per-animal files and seven mice, and the trajectory repeatedly refers to each extensionless file as one animal. The agent therefore mapped one file to one subject.

## 1-c. How are the data split into sessions?

i. Sessions are the first axis of the per-animal arrays. The agent iterates over `envs`, and for each session index `s` it takes `traces[s]` and `positions[s]` as one recording session.

ii. ```python
envs = rec['envs'].ravel().tolist()
traces = rec['trace']
positions = rec['position']
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. `CONVERSION_NOTES.md` says the explored joblib arrays have shapes like `(n_sessions, n_neurons, n_frames)` for `trace` and `(n_sessions, 2, n_frames)` for `position`, so the agent inferred that axis 0 indexes recording sessions.

## 1-d. How are the data split into trials?

i. Each session is split into consecutive non-overlapping 60-second trials at 30 Hz. The agent keeps only complete 1800-frame chunks and discards leftover frames at the end.

ii. ```python
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

iii. The instructions explicitly required 1-minute trials. In the trajectory, the agent investigated why many sessions yielded 39 trials and concluded that this came from real session lengths or valid frame spans rather than a bug, so it kept the full-minute chunking rule.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality score. Instead, the agent limits trialization to a contiguous valid frame span, discards incomplete final chunks, and skips sessions with fewer than two complete trials.

ii. ```python
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

iii. The trajectory shows the agent added this after verification exposed NaN/Inf issues. It treated valid-frame truncation and the minimum-trial requirement as pragmatic quality control for padded joblib arrays and decoder compatibility.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural data comes from `rec['trace']` in the per-animal joblib record.

ii. ```python
traces = rec['trace']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The notes and trajectory both identify `trace` as the neural array used throughout the reference analyses. The agent also checked sample values and concluded these stored traces were already binary rising-phase events.

## 2-b. How is the `neural` data processed?

i. The agent treats the stored `trace` as already-preprocessed binary rising-phase activity, truncates it to the kept frame span, casts it to `float32`, and then slices it into trials. It does not recompute calcium-event extraction.

ii. ```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
```

iii. `CONVERSION_NOTES.md` states that the methods describe binary rising-phase vectors and that inspected stored traces were already binary `{0,1}`. The trajectory explicitly records the agent resolving this consistency check before deciding to use `trace` directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Within each kept session, the agent removes any neuron whose trace contains a non-finite value anywhere in the retained frame span. If no neurons remain, it drops the session.

ii. ```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The trajectory shows this was added after `train_decoder.py --verify-only` reported NaN/Inf neural values. The agent justified it as removing neurons not valid in a given session from the padded joblib arrays.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. Neural trials are aligned only to session time and are defined as consecutive 1-minute chunks of the continuous recording.

ii. ```python
'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
```

iii. The agent repeatedly notes in the trajectory that the dataset has continuous 40-minute sessions and no natural trial onset to align to, so it used artificial 1-minute segmentation as required by the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data stays at the native 30 Hz sampling rate, so the time bin is about 33.33 ms and no temporal rebinning is applied.

ii. ```python
FPS = 30
...
'metadata': {
    'time_bin_size': 1000.0 / FPS,
    'fps': FPS,
}
```

iii. The notes extract 30 Hz synchronized acquisition from `methods.txt`, and nothing in the code changes the time axis apart from slicing into trials.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the session environment labels in `rec['envs']`, not from the raw `blocked` lists.

ii. ```python
envs = rec['envs'].ravel().tolist()
...
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. In Step 5 of the notes and in trajectory steps around 329-330, the agent decided that `envs` plus the reference `get_env_mat` function was the cleanest way to recover arena geometry, with `blocked` treated mainly as a sanity check during exploration.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent maps each environment name to a 3x3 occupancy matrix with `get_env_mat`, flattens that matrix to length 9, and uses 1 for open cells and 0 for blocked cells.

ii. ```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    ...
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
```

iii. The trajectory explicitly says the agent copied the reference `get_env_mat` semantics after comparing `envs` and `blocked`. The notes describe this as matching the reference geometry representation and satisfying the decoder-input requirement.

## 3-c. How is `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is treated as static within a session, so the same 9-dimensional vector is copied once per trial and paired with each neural trial from that session.

ii. ```python
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The notes describe geometry as session-level context that does not vary within a recording, so the agent aligned it at the trial level rather than frame-by-frame.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from `rec['position']`, which stores x/y position time series per session.

ii. ```python
positions = rec['position']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The explored data structure in the notes identifies `position` as a `(n_sessions, 2, n_frames)` array, and the trajectory ties it directly to the decoder output variable.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent clips x and y to the arena bounds, divides each axis into three equal bins across a 75 cm arena, and converts each frame to one of 9 grid labels.

ii. ```python
def discretize_position_3x3(position_xy, arena_size=75.0):
    x = np.clip(position_xy[0], 0, arena_size - 1e-6)
    y = np.clip(position_xy[1], 0, arena_size - 1e-6)
    xb = np.floor((x / arena_size) * 3).astype(int)
    yb = np.floor((y / arena_size) * 3).astype(int)
    xb = np.clip(xb, 0, 2)
    yb = np.clip(yb, 0, 2)
    return (yb * 3 + xb).astype(np.int64)
```

iii. The notes cite the paper’s conceptual 3x3 partition of the 75 x 75 cm arena. The trajectory also records a spot-check that the discretized output matched raw data for one session/trial.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each frame is categorized by x-bin and y-bin, then combined as `y_bin * 3 + x_bin` to produce category labels 0 through 8.

ii. ```python
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
...
return (yb * 3 + xb).astype(np.int64)
```

iii. This follows the paper/task requirement of a 3x3 position decoder and is the explicit encoding the agent chose in code.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned by using the same retained frame span and the same per-trial slices as neural data. The output for each trial is stored as shape `(1, time)`.

ii. ```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_bins = discretize_position_3x3(pos_sxn)
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    output_trials.append(pos_bins[sl][None, :])
```

iii. The agent’s notes emphasize the 30 Hz synchronized acquisition of imaging and behavior, and the trajectory records a raw-vs-converted sanity check confirming that trial 0 neural and output slices matched the raw session data.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All converted streams stay at the native 30 Hz sampling rate, so each time bin is about 33.33 ms and there is no temporal rebinning.

ii. ```python
FPS = 30
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
...
'metadata': {
    'time_bin_size': 1000.0 / FPS,
    'fps': FPS,
}
```

iii. The notes extract 30 Hz from the methods text, and the code performs slicing only, not resampling.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position data are aligned frame-for-frame by sharing the same valid span and the same 1-minute slice boundaries. The input geometry is static and replicated per trial rather than per frame.

ii. ```python
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
...
pos_bins = discretize_position_3x3(pos_sxn)
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The notes and trajectory both frame this as continuous-session alignment: synchronized 30 Hz neural/behavior streams plus a constant contextual geometry vector.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The agent handles missing or malformed data by inferring the last valid frame where position is finite and at least one neuron is finite, dropping neurons with any non-finite value in the retained span, discarding leftover partial trials, and skipping sessions with too little usable data. It does not impute or repair values.

ii. ```python
def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
    valid = pos_valid & neural_valid
    ...
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The trajectory shows these checks were added in response to verifier complaints about NaN/Inf neural values in sample conversion. The agent chose dropping over filling because the data are session-wise padded and the decoder expects clean arrays.

## 7-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading the large per-animal joblib files and then scanning/copying large session arrays during valid-frame detection, casting, neuron filtering, and trial slicing. The recorded full conversion took about 291 seconds.

ii. ```python
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    ...
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```
```python
def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
```

iii. The conversion log shows `elapsed_sec 291.051` for the full run, and the trajectory repeatedly discusses conversion runtime and treats data loading plus array passes as the conversion bottleneck.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop in `session_to_trials` could have been replaced by a reshape/split-based approach, and the per-session copying of identical static inputs could also have been reduced. The outer per-animal/per-session loops are structurally necessary.

ii. ```python
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The agent did not actually vectorize this section; its trajectory only notes that sample conversion was slow but manageable after correctness fixes. This assessment therefore comes mainly from the structure of the final code.

## 7-c. What processing does the code repeat multiple times?

i. For every session, the code recomputes valid-frame masks, neuron-finiteness masks, position discretization, and environment-vector construction, then copies the same static environment vector for every trial in that session.

ii. ```python
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
...
pos_bins = discretize_position_3x3(pos_sxn)
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
input_trials.append(env_vec.copy())
```

iii. This repetition is visible in the final code. The agent’s own notes focus on correctness rather than removing repeated work.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code copies the same static geometry vector once per trial even though it is identical within a session, and it casts continuous position to `float32` before immediately discretizing it. It also computes intermediate valid-frame logic whose only purpose is to discard trailing invalid data.

ii. ```python
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
input_trials.append(env_vec.copy())
```

iii. The agent prioritized producing decoder-compatible output after verifier failures rather than minimizing redundant work, so these small inefficiencies remain in the final script.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is the same as in section 6: the code trims to a valid frame span, removes neurons with non-finite values in that span, drops incomplete trailing chunks, and skips unusable sessions rather than attempting repair.

ii. ```python
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. This duplicate question is answered by the same verifier-driven fixes documented in the trajectory after NaN/Inf errors were discovered.

## 9-a. What are the most time-consuming steps of the code?

i. As in 7-a, the main cost is loading large animal files and repeatedly scanning/copying large session arrays during conversion.

ii. ```python
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. The full conversion runtime and the code structure support the same conclusion here.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. As in 7-b, the per-trial slicing/appending loop is the clearest candidate for vectorization.

ii. ```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. This repeats the same final-code observation as 7-b.

## 9-c. What processing does the code repeat multiple times?

i. As in 7-c, valid-frame detection, finite-neuron masking, geometry-vector construction, and static-input copying are repeated session by session and trial by trial.

ii. ```python
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
pos_bins = discretize_position_3x3(pos_sxn)
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
input_trials.append(env_vec.copy())
```

iii. This is the same code-structure observation as 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. As in 7-d, the main unnecessary work is repeated copying of identical static inputs and some transient casting/truncation steps whose only role is to get to the final categorical output.

ii. ```python
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
input_trials.append(env_vec.copy())
```

iii. This duplicates the final-code assessment from 7-d.
