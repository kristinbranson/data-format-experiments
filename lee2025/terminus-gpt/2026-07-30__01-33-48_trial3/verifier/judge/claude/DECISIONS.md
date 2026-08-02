# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (not .mat files) using `joblib.load`, accessing per-animal dictionaries with keys `trace`, `position`, `blocked`, and `envs`. Each animal file contains all sessions (days) for that subject.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]
```

iii. The AI noted that the reference code's `load_dat` function directly consumes joblib files, so it chose to use joblib as the data source rather than the `.mat` files. This matches the reference code's loading approach.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list `ANIMALS` of 7 animal names. Each animal name maps to a separate joblib file.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]
animals = ANIMALS[:2] if sample else ANIMALS
data['subjects'] = animals.copy()
```

iii. The AI identified the 7 subjects from the data directory files and hardcoded them.

## 1-c. How are the data split into sessions?

i. Each day within an animal's recording is treated as a separate session. The AI iterates over the `envs` array to determine the number of days and processes each day independently.

ii.
```python
n_days = len(envs)
for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
    # ... process session ...
    data['neural'].append(neural_trials)
```

iii. Each recording day corresponds to a different environment geometry and is treated as a separate session, consistent with the reference data structure.

## 1-d. How are the data split into trials?

i. Each session (day) is split into contiguous 1-minute (1800-frame at 30 Hz) non-overlapping segments. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS  # 1800

def split_session_into_trials(trace_day, pos_day, blocked_vec):
    n_frames = min(trace_day.shape[1], pos_day.shape[1])
    n_trials = n_frames // FRAMES_PER_TRIAL
    used = n_trials * FRAMES_PER_TRIAL
    # ...
    for i in range(n_trials):
        s = i * FRAMES_PER_TRIAL
        e = (i + 1) * FRAMES_PER_TRIAL
        neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. Per the task instructions, trials are defined as 60-second non-overlapping segments. The AI takes the minimum of trace and position frame counts to avoid misalignment.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 complete 1-minute trials are skipped entirely. No other trial-level quality filtering is applied.

ii.
```python
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The instructions require at least two trials per session for decoder evaluation. The AI enforces this constraint.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field of each animal's joblib file, which contains binarized rising-phase calcium event vectors.

ii.
```python
rec = load_animal(data_dir, animal)
trace = np.asarray(rec['trace'])
# ...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. The AI identified that `trace` contains the pre-processed binarized calcium event data, consistent with the reference code and paper description.

## 2-b. How is the `neural` data processed?

i. The trace data is cast to float32, then non-finite neurons are filtered out (see 2-c). Remaining NaN/inf values are replaced with 0. The data is then cast to uint8 when split into trials.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
# ...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The AI noted that the trace data represents binarized events (0/1 values), so no dF/F computation is needed. The cast to uint8 preserves the binary nature.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons where any value is non-finite (NaN or Inf) across the entire session are dropped. This removes neurons not recorded in that session.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
```

iii. The AI chose `np.isfinite` filtering rather than all-NaN filtering. This is stricter: it drops neurons that have ANY non-finite value, not just neurons that are entirely NaN.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous and trials are contiguous 1-minute segments from session start.

ii. N/A - no alignment code; trials are simply sequential 1-minute windows.

iii. There is no stimulus onset or behavioral event to align to. The AI documents this as "Continuous session split into contiguous 1-minute windows from session start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz frame rate (~33.33 ms per bin) is preserved. No temporal rebinning is applied.

ii.
```python
FPS = 30
# ...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The data is already at a consistent 30 Hz frame rate; no resampling is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Input is derived from the `blocked` field of the animal's data, which contains indices of blocked reward positions for each session/day.

ii.
```python
blocked = rec['blocked']
# ...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. The `blocked` field stores which of the 9 possible arena partitions are blocked in each session's geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-dimensional binary (one-hot) vector. Values of `-1` or `NaN` indicate no blocked partitions (all zeros). Valid indices 0-8 are set to 1.0.

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
```

iii. The AI's implementation is more defensive than the reference, handling nested arrays and NaN values in the blocked data structure from the joblib files.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The blocked vector is static per session. It is replicated for each trial within the session.

ii.
```python
input_trials.append(blocked_vec.copy())
```

iii. Blocked positions don't change within a session, so the same vector is used for every trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position output is derived from the `position` field, which contains 2D (x, y) coordinates of the mouse in the arena at each timepoint.

ii.
```python
position = np.asarray(rec['position'])
# ...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The position data records the animal's location in the 75x75 cm arena at each frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. 2D position is discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal bins of 25 cm. The grid label is `ybin * 3 + xbin`.

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

iii. The AI uses `floor(clip(x, 0, 75-eps) / 25)` to bin positions. This differs from the reference's `np.digitize` approach but should produce equivalent results.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized into 9 bins (0-8) using a 3x3 grid over the 75 cm arena. Bin edges are at 25 cm and 50 cm on each axis. Values are clipped to valid range.

ii.
```python
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
xbin = np.clip(xbin, 0, N_POS_BINS - 1)
ybin = np.clip(ybin, 0, N_POS_BINS - 1)
return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. The AI clips to [0, 75-eps) and uses floor division to get bin indices 0, 1, or 2.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are at the same native 30 Hz frame rate and are split into trials using the same frame indices.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
# ...
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. Both arrays are sliced with identical indices, ensuring frame-for-frame alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e: the native 30 Hz frame rate (~33.33 ms per bin) is preserved. No rebinning is applied.

ii.
```python
FPS = 30
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. No temporal rebinning is needed as the data is already at a consistent frame rate.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) data are frame-aligned at 30 Hz and split with the same indices. Input (blocked geometry) is static per session, so no temporal alignment is needed.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
# both neural and output sliced with same s:e indices
```

iii. The AI takes the minimum frame count between trace and position to handle any minor discrepancies, then splits both identically.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Non-finite neurons are dropped per session. Remaining NaN/inf values in neural data are replaced with 0. Position NaN/inf values are also replaced with 0. Sessions with < 2 trials are skipped.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The AI handles missing data defensively by first filtering non-finite neurons and then zeroing any remaining non-finite values.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the large joblib files from disk (I/O bound). The full conversion takes ~199 seconds for all 207 sessions.

ii. N/A

iii. From `conversion_full_out.txt`: "Converted 207 sessions in 198.78s"

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The `blocked_to_vec` function uses nested Python loops to parse blocked entries, which could potentially be simplified. The trial splitting loop is straightforward and not easily vectorized further.

ii.
```python
def blocked_to_vec(blocked_entry):
    # nested loops over blocked entries
    for item in flat:
        arr = np.array(item).reshape(-1)
        for v in arr:
            ...
```

iii. The blocked vector computation is called once per session and is not a bottleneck.

## 7-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session is processed once.

ii. N/A

iii. The code processes each animal/day pair exactly once in a single pass.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `session_info` metadata dictionaries for each session, which may not be used by the downstream decoder. The `nan_to_num` call on position data may be unnecessary if position data is always valid.

ii.
```python
session_info.append({
    'session_id': session_id,
    'animal': animal,
    'day_index': int(day),
    'environment': flatten_env_name(envs[day]),
    'n_neurons': int(trace_day.shape[0]),
    'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
    # ...
})
```

iii. This metadata is informational and stored in the output pickle for documentation purposes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. Non-finite neurons are dropped, remaining non-finite values are zeroed, sessions with < 2 trials are skipped.

ii. See question 6 code snippets.

iii. See question 6 justification.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Loading joblib files is the bottleneck at ~199 seconds total.

ii. N/A

iii. See 7-a.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The `blocked_to_vec` nested loops and trial splitting loop.

ii. See 7-b.

iii. See 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. No significant repeated processing.

ii. N/A

iii. See 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Session info metadata computation.

ii. See 7-d.

iii. See 7-d.
