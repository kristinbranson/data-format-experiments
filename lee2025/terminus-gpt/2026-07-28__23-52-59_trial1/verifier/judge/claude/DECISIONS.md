# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` from per-animal joblib files in the data directory (e.g., `data/QLAK-CA1-08`). Each file is a dictionary keyed by animal ID, containing nested fields: `trace`, `position`, `envs`, `blocked`, etc. A hardcoded list `ANIMAL_IDS` specifies which animals to process.

ii.
```python
ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]

for animal_id in animal_ids:
    animal = load_animal(animal_id, data_dir)
    envs = np.array(animal['envs']).squeeze()
    traces = np.asarray(animal['trace'])
    positions = np.asarray(animal['position'])
    blocked = animal['blocked']
```

iii. The AI identified that the data was available in joblib format (used by the reference code's `load_dat` function) as well as `.mat` format. The AI chose joblib because the reference code's demo notebooks use `load_dat(animal, p, format="joblib")`. The CONVERSION_NOTES.md documents the joblib structure with keys `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, and `trace`.

## 1-b. How are the data split into subjects?

i. Each animal is identified by a hardcoded list `ANIMAL_IDS` containing 7 animal IDs. Each animal's data is loaded from a separate joblib file.

ii.
```python
ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI extracted animal IDs from inspecting the data directory and the reference code's `behav_dict`. Each joblib file corresponds to one subject.

## 1-c. How are the data split into sessions?

i. Each day within an animal's data becomes a separate session. The AI iterates over `n_days = traces.shape[0]` for each animal, processing each day independently.

ii.
```python
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx], ...
    )
```

iii. The CONVERSION_NOTES.md documents: "one 40 min session per day" and "Treat each day as a session in target format, then split each session into 1-minute trials as required by decoder task."

## 1-d. How are the data split into trials?

i. Each session (day) is split into contiguous 1-minute (1800-frame at 30 Hz) non-overlapping segments. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)  # 1800
n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The instructions specify "1-minute trials within each session." The AI follows this by computing `n_full_trials = n_frames // FRAMES_PER_TRIAL` and dropping any trailing partial-minute frames.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full trials are skipped entirely. No other trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI implements this as a minimum trial count check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the per-animal joblib data, which contains binary rising-phase calcium trace events (1 = significant calcium event).

ii.
```python
traces = np.asarray(animal['trace'])
# In process_day:
day_trace  # passed as traces[day_idx]
```

iii. The CONVERSION_NOTES.md documents: "Neural data are rise-extracted calcium traces where 1 indicates a significant event" and "Use provided trace directly as neural activity; do not recompute dF/F or deconvolution."

## 2-b. How is the `neural` data processed?

i. The AI filters out all-NaN neurons, replaces remaining NaN values with 0 using `nan_to_num`, casts to float32, and truncates to the minimum of trace and position frame counts.

ii.
```python
def process_day(day_trace, day_pos, ...):
    valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
    trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
    pos = np.asarray(day_pos, dtype=np.float32)
    n_frames = min(trace.shape[1], pos.shape[1])
    trace = trace[:, :n_frames]
```

iii. The CONVERSION_NOTES.md states: "Current implementation drops trailing partial-minute frames and filters neurons that are all-NaN within a day."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN (not recorded in that session/day) are removed. No other quality filtering (e.g., place cell selection, spatial reliability thresholds) is applied.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The AI noted that the `trace` array contains NaN entries for neurons not recorded in a given session. Only neurons with at least one valid value are kept.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous and trials are artificial 60-second segments starting from the beginning of each session.

ii. N/A (no alignment code; trials are sequential segments from session start)

iii. The metadata documents: `'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials'` and `'off_start': 0.0, 'off_end': TRIAL_SECONDS`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz imaging frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
# metadata:
'time_bin_size': 1000.0 / FRAME_RATE_HZ,  # ~33.33 ms
```

iii. The CONVERSION_NOTES.md notes the 30 Hz native sampling rate and that no rebinning is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from TWO sources: the `envs` field (environment name strings like 'square', 'o', 't', 'u', etc.) AND the `blocked` field (indices of blocked positions).

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
# In process_day:
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The AI identified from the reference code that environment geometry is defined by both the arena shape (from `envs`) and which positions are blocked (from `blocked`). The CONVERSION_NOTES.md states: "Build decoder input from 3x3 geometry/block mask per trial, static within each 1-minute trial."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts environment names to 3x3 binary masks (1=accessible, 0=wall) via `env_to_mask()`, converts blocked indices to 3x3 masks (1=accessible, 0=blocked) via `blocked_to_mask()`, then multiplies both masks element-wise to produce a combined 9-element accessibility mask.

ii.
```python
def env_to_mask(env_name):
    env = str(env_name)
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    # ... (10 environment types)

def blocked_to_mask(entry):
    # ... normalize entry
    mask = np.ones((3, 3), dtype=np.float32)
    for v in vals:
        if v == -1:
            continue
        r, c = divmod(int(v), 3)
        mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
```

iii. The AI reasoned that the decoder input should capture the full arena accessibility, combining both structural walls (environment geometry) and dynamic blocked positions.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry mask is static per-session (constant across all trials within a session). Each trial receives a copy of the same mask.

ii.
```python
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
# ...
input_trials.append(mask.copy())
```

iii. Environment geometry and blocked positions don't change within a session, so the mask is constant for all trials.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in the per-animal data, which contains 2D (x, y) coordinates of the animal tracked at 30 Hz.

ii.
```python
positions = np.asarray(animal['position'])
# In process_day:
pos = np.asarray(day_pos, dtype=np.float32)
```

iii. The CONVERSION_NOTES.md documents: "Position data were obtained from DeepLabCut head tracking."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes) using PER-SESSION min/max normalization. Each axis is independently scaled based on the observed range within each session, then divided into 3 equal bins.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    x = pos_xy[0]
    y = pos_xy[1]
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x[valid]
    yv = y[valid]
    xmin, xmax = np.min(xv), np.max(xv)
    ymin, ymax = np.min(yv), np.max(yv)
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]
```

iii. The AI used per-session min/max normalization rather than fixed arena bounds. This means bin boundaries vary across sessions depending on the mouse's actual spatial extent.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized into 9 bins (0-8) using `y_bin * 3 + x_bin`. Each axis is divided into 3 equal bins based on the per-session observed range. Invalid (non-finite) positions are assigned label -1.

ii.
```python
xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
```

iii. The 3x3 grid produces 9 position classes as required by the instructions ("3 x 3 = 9 spatial bins").

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame-level time base. Both are truncated to the minimum of their lengths, then split into identical 1800-frame trial segments.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
# Then both split by same indices in the trial loop
```

iii. The AI ensures alignment by truncating to the shorter array and then using the same frame indices for both neural and position data.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native 30 Hz frame rate (~33.33 ms bins). No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
'time_bin_size': 1000.0 / FRAME_RATE_HZ,  # ~33.33 ms
```

iii. Same as 2-e. The native frame rate is preserved without rebinning.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural (trace) and output (position) data share the same 30 Hz frame-level time base and are truncated to matching lengths. Input (environment geometry) is static per-trial and does not require temporal alignment.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
```

iii. Both streams are sampled at the same rate and aligned by index. The truncation to `min(trace, pos)` length handles any minor frame count mismatches.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. All-NaN neurons are removed. Remaining NaN values in active neurons are replaced with 0. Non-finite position values are marked with label -1. Frame count mismatches between trace and position are handled by truncation. Partial trials are dropped. Sessions with fewer than 2 trials are skipped.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
# Position: labels[~valid] = -1
# Trials: if len(neural_trials) < 2: continue
```

iii. The AI handles several edge cases: NaN neurons, remaining NaN values, position validity, frame count mismatches, and minimum trial counts.

## 7-a. What are the most time-consuming steps of the code?

i. Data loading via `joblib.load()` is the most I/O-intensive operation, as each animal's file contains large neural trace arrays across all sessions.

ii.
```python
def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]
```

iii. The CONVERSION_NOTES.md does not include detailed timing analysis. The agent trajectory shows the conversion ran without major performance issues.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials within each day and could be replaced with array reshaping (e.g., `trace.reshape(n_neurons, n_full_trials, FRAMES_PER_TRIAL)`), though the current list comprehension approach is straightforward.

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. Position discretization is called per-trial but could be vectorized across the entire session before splitting.

## 7-c. What processing does the code repeat multiple times?

i. The `discretize_position_to_3x3` function is called once per trial within each session. It could be called once per session on the full position array and then split into trials.

ii.
```python
for ti in range(n_full_trials):
    # ...
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. No explicit documentation of repeated processing in the CONVERSION_NOTES.md.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `mask.copy()` is called for each trial even though the mask is identical across all trials in a session. However, this is a minor cost. The `show_processing` plotting code runs only when requested and doesn't affect normal conversion.

ii.
```python
input_trials.append(mask.copy())
```

iii. No significant unnecessary processing was identified.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. All-NaN neurons are removed, remaining NaNs are replaced with 0, non-finite positions get label -1, frame count mismatches are resolved by truncation, partial trials are dropped, and sessions with <2 trials are skipped.

ii. See code snippets in question 6.

iii. The AI applies defensive handling for several edge cases found in the data.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Data loading via `joblib.load()` dominates, as each file contains large arrays for all sessions/neurons/frames.

ii. See code snippet in 7-a.

iii. Same as 7-a.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The per-trial loop in `process_day` could be vectorized via array reshaping, and position discretization could be done once per session.

ii. See code snippet in 7-b.

iii. Same as 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. Position discretization is called per-trial rather than once per session.

ii. See code snippet in 7-c.

iii. Same as 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. No significant unnecessary processing beyond per-trial mask copying.

ii. See code snippet in 7-d.

iii. Same as 7-d.
