# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files in the `data/` directory. For each animal, it calls `joblib.load()` to get a dictionary keyed by the animal ID, then extracts the `envs`, `trace`, `position`, and `blocked` fields. It iterates over all 7 animals (or 2 in sample mode) and all days within each animal.

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
```

iii. The AI noted that the reference code's `load_dat` function directly loads joblib files and extracts the same fields. The AI followed the same approach, loading the native joblib files rather than the MATLAB `.mat` versions.

## 1-b. How are the data split into subjects (mice)?

i. Each animal has its own data file. The outer loop iterates over the `ANIMALS` list, which defines 7 mice. Each mouse's data is loaded independently, and a `subj_idx` is tracked to map sessions to subjects.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    # ... process all sessions for this animal ...
    data['subject_idx'].append(subj_idx)
```

iii. The AI identified 7 animals from the data files, consistent with the reference code's structure where each animal's data is stored in a separate joblib file keyed by the animal ID.

## 1-c. How are the data split into sessions?

i. Within each animal, sessions correspond to recording days. The `trace`, `position`, `envs`, and `blocked` fields are indexed by day index. The code iterates over all days for each animal (typically 31 days per animal, 21 for QLAK-CA1-51, totaling 207 sessions).

ii.
```python
n_days = len(envs)
for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
    blocked_vec = blocked_to_vec(blocked[day])
    trace_day = np.asarray(trace[day], dtype=np.float32)
    pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The AI noted that the native data structure organizes recordings by day within each animal, with one session per day. This matches the paper's description: "one session was recorded per day."

## 1-d. How are the data split into trials?

i. Each 40-minute continuous session is split into contiguous 1-minute trials of 1800 frames (30 Hz x 60 seconds). The trailing partial minute (if any) is discarded. This yields approximately 39-40 trials per session.

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

iii. The AI justified this by noting the instructions require "1-minute trials within each session" and that native data has no trial structure. Sessions are ~71.9k-72.2k frames at 30 Hz (~40 minutes), yielding ~39-40 complete 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full 1-minute trials are skipped. No other trial-level quality filtering is applied (e.g., no velocity filtering, no activity threshold filtering).

ii.
```python
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The AI noted the instructions require "at least two trials within each session." No sessions were actually skipped since all 207 sessions are ~40 minutes long. The reference code's `decode_position_within` applies velocity thresholding and cell activity filtering per-timepoint within sessions, but this was not replicated in the conversion (it was left for downstream analysis).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field of each animal's dataset, which contains binarized rising-phase calcium event vectors (0/1 values indicating significant calcium transients).

ii.
```python
trace = np.asarray(rec['trace'])
# ...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. The AI noted from the README and methods: "trace: rise-extracted calcium traces, where '1' indicates a significant event." The methods text confirms: "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5, and 0 otherwise. This binary vector was treated as the firing rate in all further analyses."

## 2-b. How is the `neural` data processed?

i. Minimal processing: the trace data is loaded as float32, non-finite neurons are dropped per session, NaN values are replaced with 0, and the data is cast to uint8 (since values are binary 0/1). No additional smoothing, temporal binning, or normalization is applied.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
# ...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The AI's rationale was that the trace data is already preprocessed (binarized rising-phase events) and should be preserved as-is at native 30 Hz resolution. The uint8 casting was done to reduce file size since values are binary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with any non-finite values (NaN, inf) in a given session/day are dropped entirely for that session. This removes unregistered cells (cells not tracked on that specific day, which have NaN values in the trace). No place-cell filtering is applied.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
```

iii. The AI noted that place-cell filtering (split-half reliability) is used in the reference code for representational similarity analyses, NOT for position decoding. The `decode_position_within` function uses velocity and activity thresholds instead. The AI chose not to apply place-cell filtering, reasoning it was not required for the decoder task. The NaN-based filtering removes cells not registered on a given day.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No explicit alignment to a specific event is performed. The data is treated as continuous recordings split into contiguous 1-minute windows starting from the beginning of each session (frame 0). Neural and position data are already aligned framewise at 30 Hz.

ii.
```python
'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),  # 60.0
```

iii. The AI justified this by noting the experiment is continuous free exploration with no discrete trial events. The DAQ simultaneously acquires behavioral and cellular imaging streams at 30 Hz, so position and trace are already framewise aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz frame rate (~33.3 ms per frame) is preserved. No temporal rebinning is applied.

ii.
```python
FPS = 30
# ...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI noted the reference code's decoder uses temporal binning (bin_size=3, meaning 3 frames averaged together) internally, but decided to preserve the native resolution and let the downstream decoder handle its own temporal processing.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field of each animal's dataset, which encodes which partitions of the 3x3 grid arena are blocked (occluded) for each session/day.

ii.
```python
blocked = rec['blocked']
# ...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. The AI noted the README describes `blocked` as "location of blocked (occluded) partitions in 3x3 design of environment," organized as `[[0,1,2],[3,4,5],[6,7,8]]`, with -1 meaning no partitions blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The `blocked` field (nested lists/arrays of partition indices) is converted to a 9-dimensional binary vector where each element corresponds to one of the 9 grid positions. A partition index present in the blocked list sets that position to 1.0; -1 (no blocked partitions) results in all zeros. This is static per trial.

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

# Static per trial:
input_trials.append(blocked_vec.copy())
```

iii. The AI reasoned that each trial within a session has the same geometry (the arena doesn't change mid-session), so the blocked vector is static per trial. The 9-d binary encoding matches the 3x3 partition grid.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field, which contains x-y head position data tracked by DeepLabCut, with shape (2, n_frames) per session.

ii.
```python
position = np.asarray(rec['position'])
# ...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The AI noted the methods: "Position data were generated from tracking the head with DeepLabCut pose-estimation software."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x-y position is discretized into a 3x3 grid (9 spatial bins) by dividing the 75cm arena into 3 equal bins along each axis. The 2D bin index is flattened to a single integer (0-8) using row-major ordering (ybin * 3 + xbin).

ii.
```python
ARENA_SIZE_CM = 75.0
N_POS_BINS = 3

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

iii. The AI noted the instructions require "Mouse position discretized into 3 x 3 = 9 spatial bins" and that the arena is 75 x 75 cm.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is binned into 9 categories (bins 0-8) by dividing each axis into 3 equal 25cm bins. The binning uses floor division with position clipped to [0, 75-eps]. Positions are clipped to the arena boundaries before binning.

ii.
```python
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
xbin = np.clip(xbin, 0, N_POS_BINS - 1)
ybin = np.clip(ybin, 0, N_POS_BINS - 1)
return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. The AI uses even-width binning (25cm per bin) with clipping to handle edge cases.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame indices (both acquired at 30 Hz by the DAQ). No additional alignment is needed. When splitting into trials, the same frame range is used for both:

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

iii. The AI noted: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz." Using `min(trace_day.shape[1], pos_day.shape[1])` handles any minor frame count differences between streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data scenarios are handled:
- Unregistered neurons (NaN values in trace for a given day) are dropped per session.
- Remaining NaN/inf values in trace and position are replaced with 0.
- Frame count mismatches between trace and position are handled by taking the minimum.
- Sessions with fewer than 2 full trials are skipped.
- NaN values in the `blocked` field are skipped during parsing.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
# ...
n_frames = min(trace_day.shape[1], pos_day.shape[1])
# ...
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The AI documented finding non-finite neural values initially causing verification failures, which was resolved by dropping neurons with any non-finite values per session/day.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the per-animal joblib files (each ~1-2 GB). The full conversion runs in ~199 seconds. The inner loop splitting sessions into trials is lightweight since it's just array slicing. No explicit timing breakdowns were printed despite the instructions requesting them.

ii.
```python
t0 = time.time()
for subj_idx, animal in enumerate(animals):
    print(f'Loading {animal}...')
    rec = load_animal(data_dir, animal)
    # ...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The CONVERSION_NOTES.md did not document per-step timing information or bottleneck analysis, despite the instructions requesting it in Step 7.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `blocked_to_vec` function uses a Python loop to parse the blocked partition entries. The trial-splitting loop in `split_session_into_trials` iterates over trials to append slices. Both could potentially be vectorized, though the trial loop would be difficult to fully vectorize since it creates a list of variably-referenced arrays.

ii.
```python
# blocked_to_vec inner loop:
for item in flat:
    arr = np.array(item).reshape(-1)
    for v in arr:
        if np.isnan(v):
            continue
        vals.append(int(v))

# Trial splitting loop:
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The AI did not document vectorization opportunities. These loops are not major bottlenecks since `blocked_to_vec` runs only 207 times and the trial loop runs ~40 times per session.

## 6-c. What processing does the code repeat multiple times?

i. The `position_to_bins_3x3` function is called once in `split_session_into_trials` for the full session, and again in `make_processing_plot` for plotting sessions. This duplicates the position binning computation for plotted sessions.

ii.
```python
# In split_session_into_trials:
pos_bins = position_to_bins_3x3(pos_day)

# In convert_dataset (for plotting):
if show_processing and plotted < 2:
    pos_bins = position_to_bins_3x3(pos_day[:, :used])
```

iii. No justification was given for this duplication. It's a minor inefficiency only affecting up to 2 sessions when `--show-processing` is enabled.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed `session_info` metadata for every session (environment name, neuron counts, frame counts, blocked vectors, etc.) which is not used by the downstream decoder. The `nan_to_num` call on `trace_day` after filtering non-finite neurons is redundant since those neurons were already dropped. Position NaN replacement with 0 could produce incorrect bin assignments for missing position data (placing it at bin 0) rather than flagging it.

ii.
```python
# Redundant nan_to_num after filtering:
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)  # redundant for trace

# Session info metadata:
session_info.append({
    'session_id': session_id,
    'animal': animal,
    'day_index': int(day),
    'environment': flatten_env_name(envs[day]),
    # ...
})
```

iii. The session_info metadata is useful for documentation and debugging but is not consumed by the decoder. The redundant `nan_to_num` is a safety measure that has negligible performance impact.
