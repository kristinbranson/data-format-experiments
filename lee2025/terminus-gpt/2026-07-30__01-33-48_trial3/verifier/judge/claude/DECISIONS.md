# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (not `.mat` files) using `joblib.load()`. Each animal's joblib file is loaded by name from the `data/` directory. The code iterates over a hardcoded list of 7 animal names (`ANIMALS`), loads each one, then iterates over recording days within each animal to extract `trace`, `position`, `blocked`, and `envs`.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]

# In convert_dataset:
for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    envs = np.array(rec['envs']).reshape(-1)
    trace = np.asarray(rec['trace'])
    position = np.asarray(rec['position'])
    blocked = rec['blocked']
    n_days = len(envs)
```

iii. The AI chose joblib files because the reference code's `load_dat` function directly consumes joblib files. The AI documented in CONVERSION_NOTES.md that "Data are organized per animal and can be stored as joblib or MATLAB .mat files" and decided to use the native joblib format following the reference code pattern.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to one joblib file. The AI uses a hardcoded list of 7 animal names (`ANIMALS`) and iterates over them. Each animal becomes a separate subject.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]
animals = ANIMALS[:2] if sample else ANIMALS
data['subjects'] = animals.copy()
```

iii. The AI identified 7 mice from the data directory and hardcoded their names, matching the reference code's `load_dat` approach.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal file becomes a separate session. The AI iterates over `n_days = len(envs)` for each animal, creating one session per day.

ii.
```python
n_days = len(envs)
for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
    # ... process and append session
    data['neural'].append(neural_trials)
```

iii. Documented in CONVERSION_NOTES.md: "Sessions correspond to recording days / environments within each animal file."

## 1-d. How are the data split into trials?

i. Each ~40-minute continuous session is split into non-overlapping 60-second (1800-frame) trials. The number of full trials is `n_frames // FRAMES_PER_TRIAL`, and remainder frames are discarded. The AI takes `min(trace_day.shape[1], pos_day.shape[1])` as the number of frames.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS  # 1800

def split_session_into_trials(trace_day, pos_day, blocked_vec):
    n_frames = min(trace_day.shape[1], pos_day.shape[1])
    n_trials = n_frames // FRAMES_PER_TRIAL
    used = n_trials * FRAMES_PER_TRIAL
    trace_day = trace_day[:, :used]
    pos_day = pos_day[:, :used]
    # ...
    for i in range(n_trials):
        s = i * FRAMES_PER_TRIAL
        e = (i + 1) * FRAMES_PER_TRIAL
        neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The AI documented: "Native data have no trial structure; task specification requires at least two trials per session, so each day/session will become ~40 contiguous 1-minute trials."

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full 1-minute trials are skipped entirely. No per-trial quality filtering is applied.

ii.
```python
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI implemented this as a session-level filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in each animal's data, which contains binarized rising-phase calcium event vectors (0/1 values) at 30 Hz frame rate.

ii.
```python
trace = np.asarray(rec['trace'])
# ...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. Documented in CONVERSION_NOTES.md: "Native neural data are rise-extracted calcium event traces aligned to position frames" and "README states raw neural signal used by authors is rise-extracted calcium trace; value 1 indicates significant event."

## 2-b. How is the `neural` data processed?

i. The trace data is loaded as float32, non-finite neurons are removed, remaining NaN/inf values are replaced with 0 via `nan_to_num`, and then the data is cast to **uint8** before storing. No additional processing (smoothing, dF/F, etc.) is applied.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
# ...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The AI noted: "Reference methods/code treat the binarized rising-phase vector as firing rate for all analyses" so no dF/F computation was needed. The uint8 cast was used because the data is binary (0/1).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons where ANY timepoint is non-finite (NaN or inf) are removed from that session. This is done via `np.all(np.isfinite(trace_day), axis=1)`.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
```

iii. The AI documented: "Unregistered neurons with non-finite values are dropped per session/day before trial splitting." This is stricter than the reference approach (which only removes all-NaN neurons).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous and trials are artificial 60-second segments starting from the beginning of the session. Neural and position data are frame-aligned at 30 Hz and split using the same indices.

ii.
```python
# Both trace and position are split using same frame indices
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The AI noted in metadata: "Continuous session split into contiguous 1-minute windows from session start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30
# ...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI documented: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" and chose to preserve this native resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Input is derived from the `blocked` variable in each animal's data, which contains indices of blocked reward locations for each recording day.

ii.
```python
blocked = rec['blocked']
# ...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. The AI documented: "`blocked` is a per-day nested list/array of blocked partition indices, with `-1` meaning no blocked partition."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary one-hot vector. The function handles nested structures from the joblib format, filters NaN values, and treats `[-1]` as "no blocked positions" (all zeros). The vector is constant per session (same for all trials).

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
# ...
input_trials.append(blocked_vec.copy())
```

iii. The AI documented: "One-hot encoding allows the decoder to treat each blocked position independently. The encoding is per-session (constant across trials)."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena at each timepoint.

ii.
```python
position = np.asarray(rec['position'])
# ...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The AI documented: "Position was obtained from DeepLabCut head tracking."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes) using floor-based binning. Each axis is divided into 3 equal 25 cm bins. NaN values in position are replaced with 0 before discretization.

ii.
```python
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

iii. The AI documented: "Discretize x-y position into 3x3 spatial bins per frame; flatten 2D bin to 9-class categorical output."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). The arena (75 cm) is divided into 3 equal bins of 25 cm each on both x and y axes. The final bin label is `ybin * 3 + xbin`, giving values 0-8. The output is cast to uint8.

ii.
```python
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
return (ybin * N_POS_BINS + xbin).astype(np.int64)
# ...
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. Uses `np.floor` with `eps` clipping to keep boundary values in valid bins, producing 9 discrete categories.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are both at 30 Hz and aligned frame-for-frame. Both are sliced using the same frame indices when splitting into trials. The AI takes `min(trace_day.shape[1], pos_day.shape[1])` to handle any length mismatch.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
# Both split using same indices s:e
```

iii. Documented in metadata: "Continuous session split into contiguous 1-minute windows from session start."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Non-finite neurons are removed entirely from each session. (2) Remaining NaN/inf values in neural traces are replaced with 0 via `nan_to_num`. (3) NaN/inf values in position are also replaced with 0. (4) Sessions with fewer than 2 trials are skipped. (5) If trace and position have different lengths, the minimum is used.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
# ...
n_frames = min(trace_day.shape[1], pos_day.shape[1])
if n_trials < 2:
    continue
```

iii. The AI documented handling of non-finite neural values as a key issue found during development, and the `nan_to_num` on position data as defensive handling.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files is the most time-consuming step. The full conversion takes ~199 seconds (from the output log: "Converted 207 sessions in 198.78s"). The loading dominates since processing (neuron filtering, discretization, trial splitting) is vectorized and fast.

ii. N/A (timing is inherent to I/O operations)

iii. The AI tracked runtime in the conversion output but did not profile individual steps.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `blocked_to_vec` function uses nested Python loops to parse blocked entries, which could potentially be vectorized. The trial-splitting loop creates list slices sequentially, but this is standard and hard to vectorize meaningfully.

ii.
```python
def blocked_to_vec(blocked_entry):
    # ... nested Python loops
    for item in flat:
        arr = np.array(item).reshape(-1)
        for v in arr:
            if np.isnan(v):
                continue
            vals.append(int(v))
```

iii. The AI did not explicitly document vectorization opportunities.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing is evident. Each session is processed once. The `blocked_to_vec` call and `position_to_bins_3x3` call are each made once per session.

ii. N/A

iii. Not documented.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores detailed `session_info` metadata (session_id, environment name, neuron drop counts, frame counts, blocked vectors) that is not used by the downstream decoder. The `nan_to_num` on position data is likely unnecessary if position data doesn't actually contain NaN values. The `flatten_env_name` function processes environment names for session IDs that are only used in metadata.

ii.
```python
session_info.append({
    'session_id': session_id,
    'animal': animal,
    'day_index': int(day),
    'environment': flatten_env_name(envs[day]),
    'n_neurons': int(trace_day.shape[0]),
    'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
    'n_frames_raw': int(n_frames),
    'n_frames_used': int(used),
    'n_trials': int(n_trials),
    'blocked_vector': blocked_vec.astype(int).tolist(),
})
```

iii. Not explicitly documented as unnecessary, but the session_info is stored in metadata for documentation purposes.
