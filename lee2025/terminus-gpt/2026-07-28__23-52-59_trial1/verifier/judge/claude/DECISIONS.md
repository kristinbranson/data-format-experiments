# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` from files named by animal ID (e.g., `QLAK-CA1-08`) in the `data/` directory. Each file is a dictionary keyed by the animal ID, containing fields `trace`, `position`, `envs`, `blocked`, etc. The AI hardcodes a list of 7 animal IDs (`ANIMAL_IDS`) and iterates over them.

ii.
```python
ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]
```

iii. The AI discovered the joblib format from the reference code's demo notebook (`load_dat(animal, p, format='joblib')`). The data directory contains both joblib files (no extension) and `.mat` files; the AI chose joblib based on the reference code pattern.

## 1-b. How are the data split into subjects?

i. Each animal ID in the hardcoded `ANIMAL_IDS` list corresponds to one subject. The subject name is the animal ID string.

ii.
```python
animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI identified 7 animals from the data directory and reference code. Each animal's data file contains all recording sessions for that subject.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal becomes a separate session. The AI iterates over days using `traces.shape[0]` (number of days per animal).

ii.
```python
traces = np.asarray(animal['trace'])
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx], ...)
```

iii. The paper states "one session was recorded per day," so each day maps to one session. The AI's CONVERSION_NOTES.md confirms this understanding.

## 1-d. How are the data split into trials?

i. Each session is split into contiguous 1-minute (1800-frame at 30 Hz) non-overlapping segments. Trailing partial-minute frames are discarded.

ii.
```python
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)  # 1800
n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
```

iii. The instructions specify "1-minute trials within each session." The AI correctly implements 60-second non-overlapping segments.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full trials are excluded entirely. No individual trial-level filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI implemented this as a session-level filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the per-animal data dictionary. This contains binary rising-phase calcium event traces.

ii.
```python
traces = np.asarray(animal['trace'])
day_trace = traces[day_idx]  # (n_neurons, n_timepoints)
```

iii. The AI's CONVERSION_NOTES.md states: "Neural data are rise-extracted calcium traces where 1 indicates a significant event; this suggests the decoder neural input should likely use these event traces rather than recomputing dF/F."

## 2-b. How is the `neural` data processed?

i. All-NaN neurons are filtered out, then remaining NaN values are replaced with 0.0 using `np.nan_to_num`. The data is cast to float32.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The AI recognized that some neurons have all-NaN traces for certain days (not recorded), and filtered those out. The remaining NaN replacement with 0 is an additional processing step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons that are entirely NaN (not recorded on that day) are removed. No other quality filtering (e.g., by spatial reliability, firing rate, or signal quality) is applied.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The AI's approach removes only completely absent neurons per day. No place-cell filtering or split-half reliability filtering is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials are contiguous 1-minute segments starting from the beginning of each recording session/day.

ii.
```python
# No alignment code - trials start from frame 0 of each day
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
```

iii. The recording is continuous with no stimulus events. The AI correctly treats the session start as the alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz frame rate is preserved. Time bin size is 1000/30 = 33.33 ms. No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
# metadata:
'time_bin_size': 1000.0 / FRAME_RATE_HZ,  # ~33.33 ms
```

iii. Both position and neural data are recorded at 30 Hz, so no resampling is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from TWO sources: the `envs` field (environment shape name, e.g., 'square', 'o', 't') AND the `blocked` field (indices of blocked reward positions). These are combined into a single accessibility mask.

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
# ...
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The AI identified both `envs` and `blocked` from the reference code and data, and combined them to create a comprehensive environment geometry representation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI creates two 3x3 binary masks: one from the environment shape name (`env_to_mask`) using hardcoded definitions, and one from blocked position indices (`blocked_to_mask`). These are multiplied element-wise to produce a combined accessibility mask (1=accessible, 0=blocked/walled), then flattened to a 9-element vector. This mask is static per session.

ii.
```python
def env_to_mask(env_name):
    env = str(env_name)
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    # ... (10 environment types defined)

def blocked_to_mask(entry):
    # ... normalizes entry
    mask = np.ones((3, 3), dtype=np.float32)
    for v in vals:
        r, c = divmod(int(v), 3)
        mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
```

iii. The AI's CONVERSION_NOTES.md states: "Build decoder input from 3x3 geometry/block mask per trial, static within each 1-minute trial." The combined mask captures both wall structure and blocked reward positions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` field, which contains 2D (x, y) coordinates of the mouse tracked at 30 Hz.

ii.
```python
positions = np.asarray(animal['position'])
pos = np.asarray(day_pos, dtype=np.float32)
```

iii. The paper mentions position tracking via DeepLabCut at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 categories). The AI uses per-trial min/max normalization: for each trial segment, it computes the min and max of x and y coordinates, then divides the range into 3 equal bins. Invalid (non-finite) positions get label -1.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    x = pos_xy[0]
    y = pos_xy[1]
    valid = np.isfinite(x) & np.isfinite(y)
    xmin, xmax = np.min(xv), np.max(xv)
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
    # similar for y
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]
```

iii. The AI's metadata notes: "Position discretized independently within each session/day into 3x3 bins using observed x/y range." However, the function is actually called per-trial (on `pos[:, s:e]`), making it per-trial normalization.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into 3 equal bins based on the observed position range within each trial. The bin label is computed as `y_bin * 3 + x_bin`, giving 9 categories (0-8).

ii.
```python
xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
```

iii. The per-trial normalization means bin boundaries vary across trials, unlike a fixed-arena approach.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are at the same 30 Hz frame rate. Both are truncated to the minimum shared length (`n_frames = min(trace.shape[1], pos.shape[1])`), then split into trials using the same indices.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
# Both split using same s:e indices in trial loop
```

iii. Frame-for-frame alignment ensures temporal consistency.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons are filtered out per day. Remaining NaN values in neural traces are replaced with 0.0 via `np.nan_to_num`. Non-finite position values result in output label -1. If trace and position have different lengths, both are truncated to the shorter one. Trailing frames that don't fill a complete 1-minute trial are discarded.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
# Invalid positions:
labels[~valid] = -1
```

iii. The AI handles multiple edge cases: absent neurons, mismatched array lengths, and non-finite positions.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the per-animal joblib files is the primary I/O bottleneck. Each file contains large neural trace arrays for all days. The vectorized processing (NaN filtering, position discretization, trial splitting) is fast.

ii. N/A (no explicit timing code beyond `time.time()` wrapper)

iii. The AI's CONVERSION_NOTES.md does not provide detailed timing breakdowns.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `process_day` creates trial slices one at a time in a Python loop. This could be replaced with `np.split` or array reshaping. The `blocked_to_mask` function loops over blocked indices individually.

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. No justification provided; the AI did not document efficiency considerations.

## 6-c. What processing does the code repeat multiple times?

i. The `discretize_position_to_3x3` function is called per trial, recomputing min/max statistics each time. If the intent were per-session normalization, this could be computed once per day. The `mask.copy()` is called per trial but the mask is identical for all trials in a session.

ii.
```python
# Called per trial in the loop:
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
input_trials.append(mask.copy())
```

iii. No justification provided.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `show_processing` plotting code generates matplotlib figures that are only used for visual inspection during development. The `valid_neurons` mask is returned but only used for `brain_region_idx` sizing, not for any downstream analysis.

ii.
```python
if show_processing and plt is not None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # ... plotting code ...
```

iii. No justification provided; the plotting is a development/debugging feature.
