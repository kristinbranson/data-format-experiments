# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (one per animal) using `joblib.load()`. Each animal file contains a dictionary with keys including `trace`, `position`, `blocked`, and `envs`. The reference code's `load_dat` function is referenced conceptually but a simpler direct load is used.

ii.
```python
def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]

# In convert_dataset:
rec = load_animal(data_dir, animal)
trace = np.asarray(rec['trace'])
position = np.asarray(rec['position'])
blocked = rec['blocked']
```

iii. The AI noted that `load_dat` in the reference code directly consumes joblib files, so it chose joblib as the primary data source rather than `.mat` files. The CONVERSION_NOTES.md states: "Use native joblib files as source: `load_dat` in reference code directly consumes the joblib files; no need to recompute from MATLAB."

## 1-b. How are the data split into subjects?

i. The AI hardcodes a list of 7 animal names. Each animal corresponds to a separate joblib file. All sessions from one animal are assigned the same subject index.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

# In convert_dataset:
for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
```

iii. The AI identified 7 animals from the data directory and hardcoded them. This matches the paper's dataset of 7 mice.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal becomes a separate session. The AI iterates over the `envs` array to determine the number of days per animal, then processes each day independently.

ii.
```python
envs = np.array(rec['envs']).reshape(-1)
n_days = len(envs)
for day in range(n_days):
    trace_day = np.asarray(trace[day], dtype=np.float32)
    pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The AI followed the data structure where each animal has multiple recording days. CONVERSION_NOTES.md states: "Sessions correspond to recording days / environments within each animal file."

## 1-d. How are the data split into trials?

i. Each session (40-minute recording) is split into non-overlapping 1-minute trials (1800 frames at 30 Hz). The remainder frames that don't fill a complete trial are discarded. Sessions with fewer than 2 full trials are skipped.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS  # 30 * 60 = 1800

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
```

iii. The AI followed the instruction that "the experiment consists of long recording sessions, which will be split into 1-minute trials." The 2-trial minimum is based on the instruction that "there needs to be at least two trials within each session."

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full 1-minute trials are skipped entirely. No per-trial quality filtering is applied.

ii.
```python
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The instruction states "there needs to be at least two trials within each session in order to evaluate the decoder performance." The AI implemented this check. No additional trial-level quality filtering was applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib files, which contains binarized rising-phase calcium event vectors (0/1 values) at 30 Hz.

ii.
```python
trace = np.asarray(rec['trace'])
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md: "Native neural data are rise-extracted calcium event traces aligned to position frames." and "README/methods say binary rising-phase vector treated as firing rate."

## 2-b. How is the `neural` data processed?

i. Neural traces are loaded per day, filtered to remove non-finite neurons, NaN values are replaced with 0, and the data is cast to uint8. The data is indexed as `trace[day]` giving shape `(n_neurons, n_timepoints)`, then filtered and split into trials.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
# ...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The AI's trajectory notes explain: "drop non-finite neurons per session/day" was added after verification revealed NaN/Inf values from unregistered cells. The uint8 conversion was motivated by file size reduction (the data is binary 0/1).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with ANY non-finite value (NaN or Inf) across all timepoints are removed for that session. This is a stricter criterion than the reference, which only removes neurons that are ALL NaN.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
```

iii. The AI's trajectory (step 203-204) explains this was added to fix verification failures: "neural data contain NaN or Inf values for all listed trials... unregistered cells represented as NaN on some days." The AI chose `np.all(np.isfinite(...))` rather than `~np.all(np.isnan(...))`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous, and trials are artificial 1-minute segments starting from the beginning of the session. Neural and position data are already frame-aligned at 30 Hz.

ii.
```python
# Both trace_day and pos_day are indexed by the same frame indices
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The AI noted: "Preserve native 30 Hz frame alignment in converted data: Position and traces are already aligned framewise."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30
# ...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI noted: "Behavioral and cellular imaging streams were simultaneously acquired at 30 Hz."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in each animal's data, which contains indices of blocked reward locations for each recording session.

ii.
```python
blocked = rec['blocked']
blocked_vec = blocked_to_vec(blocked[day])
```

iii. The AI noted: "`blocked` is a per-day nested list/array of blocked partition indices, with `-1` meaning no blocked partition."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element one-hot binary vector. Each element corresponds to one of 9 arena partitions. A value of 1 indicates the partition is blocked. The special value `-1` means no partitions are blocked (all zeros). NaN values in the blocked data are skipped. The vector is static per session (same for all trials within a session).

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

# Applied per session:
input_trials.append(blocked_vec.copy())
```

iii. The AI designed a more robust parsing function that handles nested arrays and NaN values in the blocked data.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position output is derived from the `position` variable in each animal's data, which contains 2D (x, y) coordinates of the mouse at each timepoint.

ii.
```python
position = np.asarray(rec['position'])
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The AI noted: "Position was obtained from DeepLabCut head tracking."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal bins of 25 cm (arena is 75x75 cm). The bin label is computed as `ybin * 3 + xbin`. NaN values in position are replaced with 0 before binning.

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

iii. The 3x3 grid is specified by the instructions ("Mouse position discretized into 3 x 3 = 9 spatial bins").

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using `np.floor(clip(x, 0, 75-eps) / 25)` for each axis. This creates 3 bins per axis: [0, 25), [25, 50), [50, 75). The combined label is `ybin * 3 + xbin`, giving 9 categories (0-8). Output is cast to uint8.

ii.
```python
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
return (ybin * N_POS_BINS + xbin).astype(np.int64)
# ...
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The AI uses `np.floor` with clipping for discretization, which is functionally equivalent to the reference's `np.digitize` approach.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are already aligned at 30 Hz in the source data. Both are indexed by the same frame indices when splitting into trials.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
# ...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The AI uses `min(trace_day.shape[1], pos_day.shape[1])` to handle any discrepancy in lengths, then both arrays are sliced with the same indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Neurons with any non-finite value are dropped per session. (2) NaN values in position data are replaced with 0. (3) NaN values in blocked indices are skipped. (4) Remaining frames that don't fill a complete trial are discarded. (5) Sessions with < 2 trials are skipped. (6) Length mismatches between trace and position are handled by using the minimum length.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
n_frames = min(trace_day.shape[1], pos_day.shape[1])
```

iii. The AI addressed these after encountering verification failures with non-finite values.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files for each animal is the most time-consuming step. The full conversion completes in ~199 seconds for 207 sessions.

ii.
```python
t0 = time.time()
for subj_idx, animal in enumerate(animals):
    print(f'Loading {animal}...')
    rec = load_animal(data_dir, animal)
```

iii. The conversion output shows: "Converted 207 sessions in 198.78s". The I/O of loading large joblib files dominates.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `blocked_to_vec` function uses Python loops to iterate over blocked entries. The trial-splitting loop could potentially be vectorized using `np.split` or array reshaping.

ii.
```python
def blocked_to_vec(blocked_entry):
    # ...
    for item in flat:
        arr = np.array(item).reshape(-1)
        for v in arr:
            if np.isnan(v):
                continue
            vals.append(int(v))
```

iii. The AI did not identify or address these loops, though they are relatively small and not performance-critical.

## 6-c. What processing does the code repeat multiple times?

i. The code does not appear to repeat significant processing. Each animal is loaded once, each session processed once.

ii. N/A

iii. No redundant processing was identified.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `session_info` metadata for each session (environment name, neuron counts, frame counts, etc.) which is stored in metadata but not used by the decoder. The `nan_to_num` call on neural data after filtering all-non-finite neurons is redundant since the remaining neurons should already be finite.

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

iii. The session_info is useful for debugging but not used by the decoder. The `nan_to_num` after `isfinite` filtering is a safety net.
