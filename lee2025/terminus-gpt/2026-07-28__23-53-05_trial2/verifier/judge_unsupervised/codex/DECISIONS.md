# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes the seven animal IDs, loads one per-animal joblib file from `data/`, and only reads `envs`, `trace`, and `position`. It does not use the `.mat` files, `behav_dict`, or precomputed maps/results. Trials are then derived from each loaded session by `session_to_trials`.

ii. ```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']

for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    envs = rec['envs'].ravel().tolist()
    traces = rec['trace']
    positions = rec['position']
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. `CONVERSION_NOTES.md` Step 2 says the extensionless per-animal files are the joblib-converted files used by the reference code, with keys including `envs`, `position`, and `trace`. The trajectory shows the agent decided to use those joblib files rather than the `.mat` originals.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is treated as one subject. `subjects` is just the animal list, and `subject_idx` is assigned per retained session.

ii. ```python
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[animal])
```

iii. The notes record 7 animals / subjects and per-animal files; the trajectory repeatedly describes the dataset as one file per mouse.

## 1-c. How are the data split into sessions?

i. Sessions are the first dimension of the per-animal arrays. The code flattens `envs` and iterates `enumerate(envs)`, using the same index into `trace[s]` and `position[s]`.

ii. ```python
envs = rec['envs'].ravel().tolist()
traces = rec['trace']
positions = rec['position']
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. `CONVERSION_NOTES.md` Step 2 documents `envs` shape `(n_sessions, 1)`, `position` shape `(n_sessions, 2, n_frames_max)`, and `trace` shape `(n_sessions, n_neurons, n_frames_max)`, which is the basis for this split.

## 1-d. How are the data split into trials?

i. Sessions are cut into consecutive non-overlapping 1-minute chunks. The number of chunks is `valid_frames // 1800`, so only whole-minute trials are kept. Despite the notes initially planning 40 trials per 40-minute session, the implemented code yields 39 or 40 depending the session’s valid frame count.

ii. ```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS

valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
n_trials = valid_frames // FRAMES_PER_TRIAL
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. Step 5 in the notes says the plan was to split each 40-minute session into 1-minute trials. Later trajectory steps show the agent realized some animals produce 39 full trials because some sessions have 71,866 frames, while others produce 40.

## 1-e. How are trials filtered based on quality controls?

i. Trial QC is formatter-driven, not paper-driven. The code keeps only the contiguous prefix up to the last frame where position is finite and at least one neuron is finite, keeps only whole-minute chunks, skips sessions with fewer than 2 resulting trials, and skips sessions with zero fully finite neurons after neuron filtering.

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
n_trials = valid_frames // FRAMES_PER_TRIAL
if n_trials < min_trials:
    return [], [], []
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The trajectory shows this was added after verification reported `NaN`/`Inf` neural values. The agent justified dropping non-finite neurons per session as a practical fix for NaN-padded session arrays, not as a replication of the paper’s formal curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken directly from `rec['trace'][session]`.

ii. ```python
traces = rec['trace']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. Step 5 of the notes maps `rec['trace'][session]` to `neural`. The trajectory says the agent concluded the stored `trace` arrays were already the binary rising-phase representation described in the methods.

## 2-b. How is the `neural` data processed?

i. The code does not recompute the paper’s calcium preprocessing. It casts the stored trace slice to `float32`, truncates it to the kept full-minute prefix, drops NaN-padded neurons, and then slices it into trials. The raw session slice is already `(n_neurons, n_timepoints)`, so no transpose is applied in code.

ii. ```python
n_keep = n_trials * FRAMES_PER_TRIAL
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
...
neural_trials.append(trace_sxn[:, sl])
```

iii. The notes explicitly say the agent chose to use the stored binary `trace` directly rather than rerunning derivative / Gaussian / threshold preprocessing, because it believed the released `trace` already matched the paper’s binarized rising-phase vectors.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC consists only of dropping neurons that contain any non-finite values within the kept session prefix. No place-cell filter, split-half reliability filter, or other neuroscience-specific curation is applied in `convert_data.py`.

ii. ```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. The trajectory shows this decision was made after decoder verification failed on NaN neurons. The notes justify not applying place-cell filtering by treating that as analysis-specific rather than necessary for a general decoder dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no behavioral event alignment. Trials are aligned to trial start in session time: the script preserves the native framewise trace-position alignment and then cuts both streams with the same 1-minute slices from the start of the session.

ii. ```python
sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
neural_trials.append(trace_sxn[:, sl])
...
'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. Step 5 in the notes says the agent decided to preserve native 30 Hz alignment within each session and use the 1-minute trialization requested by the task as the effective alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data remain at the acquisition frame rate: 30 Hz, i.e. `1000 / 30 = 33.33 ms` per bin. No temporal rebinning is applied in `convert_data.py`.

ii. ```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes tie this to the methods statement that behavioral and imaging streams were acquired simultaneously at 30 Hz, and the code never aggregates multiple frames into coarser bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. In code, environment geometry is derived from the session label in `rec['envs']`. The notes say `blocked` would be used as a cross-check, but `convert_data.py` itself never reads `rec['blocked']`.

ii. ```python
envs = rec['envs'].ravel().tolist()
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
```

iii. The notes first considered `blocked` plus `envs`, then the trajectory records that the agent preferred using the reference `get_env_mat` mapping from environment names and keeping `blocked` only as a sanity check.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code maps each environment name to a 3x3 open/blocked matrix, flattens it to a 9-element vector, and stores that same static vector for every trial from that session.

ii. ```python
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

iii. The trajectory says the agent inspected the reference `get_env_mat` and decided to encode geometry as a 9-dimensional binary mask with `1` for open cells and `0` for blocked cells.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived directly from the per-session `rec['position'][session]` x/y trajectory.

ii. ```python
positions = rec['position']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. Step 5 in the notes maps `rec['position'][session]` to the decoder output and describes it as the time-varying variable to be discretized.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The script clips x and y into `[0, 75)`, rescales them into 3 bins per axis, floors to integer bin indices, clips those to `[0, 2]`, and converts the result into a single row-major class index.

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

iii. The notes justify this with the task requirement to decode 3x3 spatial bins and the paper’s conceptual 3x3 partition of the 75 x 75 cm square.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Categories are the 9 values `0..8`, computed as `yb * 3 + xb` after thresholding x and y into 3 bins each.

ii. ```python
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
xb = np.clip(xb, 0, 2)
yb = np.clip(yb, 0, 2)
return (yb * 3 + xb).astype(np.int64)
...
'output_values': [[f'bin_{i}' for i in range(9)]],
```

iii. The notes describe this as the agent’s chosen categorical 9-way output representation for the decoder.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position are aligned by shared session-time indexing. The same `sl` slice is applied to `trace_sxn` and `pos_bins`, so each output trial has the same 1800 timepoints as the corresponding neural trial.

ii. ```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_bins = discretize_position_3x3(pos_sxn)
...
sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
neural_trials.append(trace_sxn[:, sl])
output_trials.append(pos_bins[sl][None, :])
```

iii. The notes say the agent preserved the native 30 Hz post-hoc alignment between trace and position, then trialized both identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing-data handling is heuristic: trailing invalid frames are dropped by `infer_valid_frames`, NaN-padded neurons are removed per session, sessions with too little usable data are skipped, and out-of-range positions are clipped into the arena instead of being discarded or interpolated. There is no imputation.

ii. ```python
pos_valid = np.all(np.isfinite(position), axis=0)
neural_valid = np.any(np.isfinite(trace), axis=0)
valid = pos_valid & neural_valid
...
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
...
x = np.clip(position_xy[0], 0, arena_size - 1e-6)
y = np.clip(position_xy[1], 0, arena_size - 1e-6)
```

iii. The trajectory shows the non-finite neuron filter was added reactively after verifier failures. The notes present this as sensible handling of padded arrays rather than as a reference-paper procedure.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading the large per-animal joblib files, copying/casting large session arrays before slicing, iterating through every session and trial in Python, and finally serializing a very large pickle (`converted_data.pkl` is about 20 GB).

ii. ```python
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    ...
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes record the huge full output size and the agent mentioned sample conversion runtimes in the trajectory when discussing efficiency.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `session_to_trials` is the clearest vectorization opportunity: the code could reshape full-minute prefixes into `(n_trials, ..., 1800)` blocks instead of slicing/appending in Python. The repeated per-trial `env_vec.copy()` is also avoidable.

ii. ```python
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. The task instructions explicitly asked for vectorization where possible, but the final implementation kept explicit Python trial loops.

## 6-c. What processing does the code repeat multiple times?

i. It repeatedly copies the same static environment vector once per trial, repeatedly constructs slice objects and appends trial lists for each session, and repeatedly casts session arrays to `float32` after determining `n_keep`.

ii. ```python
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
...
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    ...
    input_trials.append(env_vec.copy())
```

iii. The code is simple and explicit, but not especially optimized; the notes left the “Code inefficiencies identified” section effectively unfinished.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It duplicates static session geometry into every trial, keeps all 33 ms bins even though downstream decoders may later aggregate or flatten them, and exposes a `--show-processing` path that performs no plotting work. It also computes metadata such as `SESSION_SECONDS` that are not used to drive conversion logic.

ii. ```python
SESSION_SECONDS = 40 * 60
...
input_trials.append(env_vec.copy())
...
if args.show_processing:
    print('show-processing requested; plotting not yet implemented')
```

iii. The trajectory shows the agent was focused on satisfying the verifier, not on a lean representation; the final code stores a large amount of repeated per-trial static data and leaves the requested plotting mode stubbed out.
