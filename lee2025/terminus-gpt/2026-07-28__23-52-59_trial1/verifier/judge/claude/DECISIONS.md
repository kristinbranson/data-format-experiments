# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` on extensionless per-animal files in the data directory (e.g., `data/QLAK-CA1-08`). Each file contains a dict keyed by animal ID, with sub-keys for `trace`, `position`, `envs`, `blocked`, etc. A hardcoded list of 7 animal IDs (`ANIMAL_IDS`) is iterated over to load all subjects.

ii.
```python
ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]
```

iii. The AI chose `joblib` because the reference code's demo notebook uses `load_dat(animal, p, format="joblib")`, and the data directory contains both extensionless joblib files and `.mat` files. The AI followed the reference code's loading pattern.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of 7 animal IDs (`ANIMAL_IDS`). Each animal ID corresponds to one subject. In sample mode, only the first 2 animals are processed.

ii.
```python
animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The agent identified the 7 animal IDs from the data directory and the reference code's `behav_dict`.

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple days (recording sessions). The number of days is determined from the trace array shape (`traces.shape[0]`). Each day becomes a separate session in the output, provided it yields at least 2 full trials.

ii.
```python
traces = np.asarray(animal['trace'])
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(...)
    if len(neural_trials) < 2:
        continue
    data['neural'].append(neural_trials)
```

iii. The reference code's demo notebook operates per session/day, and the paper states "one session was recorded per day." Sessions with fewer than 2 trials are excluded to allow decoder train/test splitting.

## 1-d. How are the data split into trials?

i. Each session (day) is split into non-overlapping 1-minute segments (1800 frames at 30 Hz). Trailing partial-minute frames are discarded. Trial splitting is done inside `process_day()`.

ii.
```python
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)  # 1800
n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
```

iii. The instructions specify "1-minute trials within each session." The agent splits the continuous 40-minute recordings into 60-second non-overlapping segments.

## 1-e. How are trials filtered based on quality controls?

i. No per-trial quality filtering is applied beyond the session-level requirement of at least 2 trials. There is no filtering based on trial-level neural quality, behavioral quality, or other criteria.

ii. N/A (no trial filtering code)

iii. The agent did not identify any trial-level filtering criteria from the reference code or paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the per-animal joblib files. These are binary rising-phase calcium traces where 1 indicates a significant calcium event.

ii.
```python
traces = np.asarray(animal['trace'])
# In process_day:
day_trace  # shape: (n_neurons, n_frames)
```

iii. The agent determined from the code README and methods that `trace` contains rise-extracted calcium traces that are treated as the firing rate signal in all analyses.

## 2-b. How is the `neural` data processed?

i. Processing involves: (1) removing neurons that are all-NaN for that day (not registered), (2) replacing any remaining NaN values with 0.0 via `nan_to_num`, (3) casting to float32, and (4) truncating to match the minimum frame count between trace and position arrays.

ii.
```python
def process_day(day_trace, day_pos, ...):
    valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
    trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
    pos = np.asarray(day_pos, dtype=np.float32)
    n_frames = min(trace.shape[1], pos.shape[1])
    trace = trace[:, :n_frames]
```

iii. The agent reasoned that cells not registered on a given day appear as all-NaN (per the README), so they should be excluded. Any remaining sporadic NaN values are replaced with 0.0, treating them as no event detected.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filtering is removing all-NaN neurons (neurons not registered on that day). No additional quality filtering (e.g., firing rate thresholds, signal-to-noise, spatial reliability) is applied.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The agent noted that the `trace` arrays are already "analysis-ready binary event traces" pre-processed through the authors' pipeline, so further quality filtering beyond removing unregistered neurons was not deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of each session. Neural and position data share the same frame rate and are aligned frame-by-frame.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
```

iii. There is no stimulus onset or behavioral event to align to; both neural and behavioral data are sampled at the same 30 Hz frame rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per time bin). No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
# metadata:
'time_bin_size': 1000.0 / FRAME_RATE_HZ,  # ~33.33 ms
```

iii. Both neural and position data are natively at 30 Hz. The agent preserved this resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from TWO variables: `envs` (environment shape name, e.g., 'square', 'o', 't') AND `blocked` (indices of blocked partitions in a 3x3 grid). These are combined via element-wise multiplication of their respective 3x3 masks.

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
# In process_day:
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The agent identified both `envs` and `blocked` as relevant from the code README and Step 4 consistency analysis, reasoning that the decoder input should encode "which spatial sectors are available/blocked."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Processing involves: (1) converting environment shape names to hardcoded 3x3 binary masks via `env_to_mask()` (e.g., 'square' = all 1s, 'o' = center blocked), (2) converting blocked partition indices to 3x3 masks via `blocked_to_mask()` (1=accessible, 0=blocked), and (3) element-wise multiplying the two masks. The result is flattened to a 9-element float32 vector, static per trial.

ii.
```python
def env_to_mask(env_name):
    env = str(env_name)
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    # ... etc for t, u, rectangle, +, i, l, bit donut, glenn

def blocked_to_mask(entry):
    # ...
    mask = np.ones((3, 3), dtype=np.float32)
    for v in vals:
        r, c = divmod(int(v), 3)
        mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
```

iii. The agent reasoned that both the environment geometry (from `envs`) and blocked partitions should be combined to give a complete picture of which arena sectors are accessible.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field in the per-animal data, which contains 2D (x, y) coordinates of the mouse at each frame.

ii.
```python
positions = np.asarray(animal['position'])
# In process_day:
pos = np.asarray(day_pos, dtype=np.float32)
```

iii. The `position` variable records the animal's x-y location tracked via DeepLabCut at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes) using per-session min/max normalization. For each session, the x and y ranges are determined from the observed data, and positions are binned into 3 equal divisions of each axis. The grid label is `y_bin * 3 + x_bin`. Invalid (non-finite) positions are labeled -1.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    x = pos_xy[0]
    y = pos_xy[1]
    valid = np.isfinite(x) & np.isfinite(y)
    xmin, xmax = np.min(xv), np.max(xv)
    ymin, ymax = np.min(yv), np.max(yv)
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]
```

iii. The agent's metadata notes: "Position discretized independently within each session/day into 3x3 bins using observed x/y range."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). Each axis is divided into 3 equal bins based on the observed min/max range for that session. The category label is `y_bin * 3 + x_bin`, giving values 0-8.

ii.
```python
xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
```

iii. Per-session normalization ensures the bins cover the actual occupied area of each session, regardless of absolute coordinates.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same frame rate (30 Hz) and are truncated to the minimum frame count between the two. Both are then split into trials using the same frame indices.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
# Both split using same trial indices in the for loop
```

iii. The data are natively aligned since they share the same time base.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of missing data are handled: (1) Neurons not registered on a given day (all-NaN) are removed from that session. (2) Remaining sporadic NaN values in neural traces are replaced with 0.0. (3) Mismatched frame counts between trace and position are resolved by truncating to the minimum. (4) Non-finite position values are labeled -1. (5) Trailing partial-minute frames are discarded.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
# In discretize_position_to_3x3:
valid = np.isfinite(x) & np.isfinite(y)
labels[~valid] = -1
```

iii. The agent treated unregistered neurons as missing data per the README documentation. The `nan_to_num` approach treats sporadic NaN values as "no event detected."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the per-animal joblib files, which contain large neural trace arrays. Processing (NaN filtering, discretization, trial splitting) is fast by comparison. The agent tracked runtime via `time.time()` in the metadata.

ii.
```python
t0 = time.time()
for animal_id in animal_ids:
    animal = load_animal(animal_id, data_dir)
    # ...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. No detailed timing breakdown per step was documented in the CONVERSION_NOTES.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop (iterating over `n_full_trials` and slicing arrays) could potentially be vectorized using `np.reshape`, but the current loop-based approach is straightforward and not a bottleneck given the small number of trials per session (~39).

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The agent did not explicitly identify vectorization opportunities.

## 6-c. What processing does the code repeat multiple times?

i. The `discretize_position_to_3x3` function is called once per trial inside the trial-splitting loop, but it could have been called once on the full session's position data and then split into trials. This repeats the min/max computation for each trial slice (though in practice the function receives the full day's position data pre-sliced).

ii.
```python
for ti in range(n_full_trials):
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The agent did not document repeated processing concerns.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `env_to_mask()` function contains hardcoded masks for 10 different environment shapes, but the actual arena geometry information goes beyond what is strictly needed. Additionally, the `show_processing` plotting code generates visualizations that are not used in the final conversion.

ii.
```python
def env_to_mask(env_name):
    # 10 hardcoded environment shapes
    # ...

if show_processing and plt is not None:
    # plotting code
```

iii. The agent did not explicitly discuss unnecessary processing.
