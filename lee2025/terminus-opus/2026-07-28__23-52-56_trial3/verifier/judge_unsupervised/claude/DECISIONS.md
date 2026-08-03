# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded using `joblib.load()` from individual animal files in the `data/` directory. Each animal has a separate joblib file (e.g., `data/QLAK-CA1-08`). The 7 animals are hardcoded in the `ANIMALS` list. Each file contains a dictionary keyed by the animal ID, with sub-keys `trace`, `position`, `envs`, `blocked`, `maps`, `SFPs`, and `centroids`.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
# ...
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)
```

iii. The agent identified the `load_dat` function in the reference code (`utils.py`) which uses `joblib.load()` to load animal data. The agent replicated this approach directly, loading the same data format and extracting the same fields (`trace`, `position`, `envs`).

## 1-b. How are the data split into subjects (mice)?

i. Each animal file corresponds to one subject. The code iterates over the 7 animals in the `ANIMALS` list. A subject index is tracked per session, mapping sessions to their animal.

ii.
```python
for a_idx, animal in enumerate(animals_to_process):
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
    for sess in sessions:
        all_subject_idx.append(a_idx)
# ...
'subjects': [a for a in animals_to_process],
'subject_idx': np.array(all_subject_idx, dtype=int),
```

iii. The agent noted that the data has 7 animals with IDs matching the paper's description (QLAK-CA1-08, -30, -50, -51, -56, -74, -75). Each animal file is processed separately and the subject index is tracked for the output format.

## 1-c. How are the data split into sessions?

i. Each recording day is treated as one session. The `trace` array has shape `(n_sessions, n_neurons, n_timepoints)`, where the first dimension indexes days/sessions. Sessions per animal range from 21 to 31, totaling 207 sessions.

ii.
```python
n_sessions = trace.shape[0]
for day in range(n_sessions):
    env_name = envs[day, 0]
    tr = trace[day]  # (n_neurons, n_timepoints)
    pos = position[day]  # (2, n_timepoints)
    # ... process session
    sessions.append({...})
```

iii. The agent verified that 207 total sessions matched the paper's reported count: 31 sessions each for 6 animals and 21 sessions for QLAK-CA1-51.

## 1-d. How are the data split into trials?

i. Each ~40-minute recording session is split into 1-minute trials of exactly 1800 frames (30 fps x 60 seconds). The number of trials per session is `n_timepoints // 1800`, which yields 39-40 trials per session depending on session length. Remaining frames at the end of the session (less than 1800) are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60  # 1 minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
# ...
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The agent followed the instruction "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session." There is no trial structure in the original data (continuous 40-min recordings), so this splitting approach was invented per the task instructions.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are kept. Only the leftover frames at the end of a session (less than 1800 frames) are discarded.

ii. No specific filtering code exists beyond the truncation implicit in `n_trials = n_timepoints // FRAMES_PER_TRIAL`.

iii. The agent noted that the reference code has no explicit trial structure or trial-level filtering, since it works on continuous sessions. The task instructions do not specify trial-level quality control beyond what the reference code does.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field of the raw data, which contains binary (0/1) rising-phase calcium transient vectors. These are already preprocessed in the raw data.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
# ...
tr = trace[day]  # (n_neurons, n_timepoints)
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
```

iii. The agent identified from the paper and reference code that `trace` contains "binarized rising-phase vectors" where the derivative of the calcium trace is smoothed (sigma=5 frames), noise is estimated from negative values, z-scored, and thresholded at z > 2.5. This preprocessing is already applied in the provided data files.

## 2-b. How is the `neural` data processed?

i. Minimal processing is applied: (1) neurons that are all-NaN for a session are excluded via a mask, (2) any remaining NaN values in active neurons are replaced with 0, and (3) the data is cast to float32. No additional smoothing, temporal binning, or filtering is applied at conversion time.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]  # (n_active, n_timepoints)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
# ...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The agent decided to use the raw binary trace data without additional processing, reasoning that the decoder would handle temporal binning. The reference code's `fit_decoder` function applies temporal binning (3-frame average pooling) and Gaussian smoothing internally, but the agent chose not to replicate this at the conversion stage.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only NaN-based filtering is applied: neurons with all-NaN traces for a given session are excluded (these are neurons not tracked/detected on that day). No velocity-based or activity-based cell filtering is applied.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]  # (n_active, n_timepoints)
```

iii. The agent noted that the reference code (`decode_position_within`) applies cell filtering with `cell_threshold=5` (cells whose summed activity during movement exceeds 5) and velocity filtering (`v_thresh=5`). However, the agent chose not to apply these at conversion, stating "we do not apply these at conversion since our decoder handles this differently." This is a significant departure from the reference processing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the start of the recording session. Each trial starts at `t * 1800` frames from the beginning of the recording. There is no alignment to any specific behavioral event.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. The agent noted that the original data has no trial structure or specific alignment events. Since sessions are continuous recordings, the start of the recording is the natural alignment point. This is consistent with the task instruction to split sessions into 1-minute trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 1 frame at 30 fps, i.e., ~33.33 ms per time bin. No temporal rebinning is applied. Each trial has exactly 1800 time points.

ii.
```python
FPS = 30  # frames per second
# ...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The agent decided to keep the original 30 Hz resolution. The reference code's `fit_decoder` applies 3-frame temporal binning via AvgPool1d, but the agent chose not to replicate this, leaving temporal binning to the downstream decoder. This means the converted data has a different temporal resolution than what the reference code uses for decoding.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `envs` field in the raw data, which contains string environment names (e.g., 'square', 'o', 't', 'u', 'rectangle', '+', 'i', 'l', 'bit donut', 'glenn') for each session.

ii.
```python
envs = d['envs']         # (n_sessions, 1)
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The agent identified the `get_env_mat` function from the reference code `utils.py` that maps environment names to 3x3 binary matrices, and replicated it directly.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix using `get_env_mat()` (copied from reference code), where 1 indicates accessible regions and 0 indicates blocked regions. The matrix is flattened to a 9-element vector. This is static per trial (same value for all trials in a session).

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        # ... (10 environments)
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
# ...
env_mat = get_env_mat(env_name).flatten()  # (9,)
trial_input.append(env_mat.astype(np.float32))
```

iii. The agent directly copied the `get_env_mat` function from the reference code. The input is static per trial as specified in the instructions ("Static per-trial").

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position output is derived from the `position` field in the raw data, which contains x,y coordinates in cm (range 0-75) tracked at 30 Hz.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
pos = position[day]  # (2, n_timepoints)
```

iii. The agent identified that position data was tracked using DeepLabCut, as noted in the paper and methods.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x,y position coordinates (0-75 cm) are discretized into a 3x3 grid of spatial bins, producing a single categorical variable with 9 classes (0-8). Each bin covers 25 cm x 25 cm.

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. The agent followed the task instruction to discretize position into "3 x 3 = 9 spatial bins". The binning uses `floor(position / 25)` clipped to [0,2], then combines row and column indices into a single bin index.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded by dividing each coordinate by the bin size (25 cm) and taking the floor. Values are clipped to [0, 2] to handle edge cases at boundaries (e.g., position exactly at 75 cm). The combined index `x_bin * 3 + y_bin` produces 9 categories (0-8).

ii.
```python
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. The task specifies "3 x 3 = 9 spatial bins" and the 75 cm arena naturally divides into 25 cm bins. The agent verified that output values are in [0, 8] and the distribution across bins is reasonable.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by frame index. Both come from the same recording at 30 Hz, so they share the same time axis. For each trial, both neural and position data are extracted using the same start/end frame indices.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The agent noted that neural traces and position data are recorded synchronously at 30 Hz and share the same time dimension in the raw data, so no additional alignment is needed.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of missing data are handled: (1) Neurons with all-NaN traces (not tracked on that day) are excluded from the session entirely. (2) Any remaining NaN values in active neurons are replaced with 0 using `np.nan_to_num`. Leftover frames at the end of sessions that don't form a complete trial are silently discarded.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. The agent documented that NaN neurons are expected (CellReg tracking across days means not all neurons are detected every day). The `nan_to_num` call is described as a safety measure that "shouldn't happen but safety."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading each animal's data file with `joblib.load()` (12-23 seconds per animal), (2) Saving the final pickle file (40.7 seconds for 19.98 GB), and (3) Processing sessions (8-15 seconds per animal for all sessions). Total conversion time is ~4.2 minutes.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))  # 12-23s per animal
# ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)  # 40.7s
```

iii. The agent documented timing information and estimated total conversion time at ~4 minutes for all 7 animals, which is within the 15-minute target.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop could potentially be vectorized using `np.split` or array reshaping instead of iterating over trials. However, since the number of frames may not be evenly divisible by 1800, this is slightly non-trivial.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The agent did not specifically address vectorization opportunities in the notes. The trial-splitting loop is relatively fast (39-40 iterations with numpy slicing), so the gain from vectorization would be minimal compared to the I/O bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The `env_mat.astype(np.float32)` conversion is repeated for every trial within a session, even though the environment geometry is the same for all trials in a session. Similarly, `position_to_bin()` computes bins for the entire session, but only the per-trial slices are used.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # computed once per session
for t in range(n_trials):
    trial_input.append(env_mat.astype(np.float32))  # .astype repeated per trial
```

iii. These are minor inefficiencies. The position binning is actually efficient (computed once per session then sliced), and the `.astype` call on the small 9-element array is negligible.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores the full 30 Hz binary trace data (1800 timepoints per trial) even though the reference decoder applies 3-frame temporal binning (reducing to 600 timepoints). This results in a much larger output file (19.98 GB) than necessary. The `nan_to_num` call on active neurons is likely unnecessary since active neurons (non-all-NaN) typically don't have individual NaN values, but it's a safety measure.

ii.
```python
# Full resolution stored - 1800 frames per trial
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
# Safety NaN replacement - likely unnecessary
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. The agent chose to preserve the full temporal resolution to allow the downstream decoder flexibility, but this comes at a significant cost in file size (19.98 GB). If 3-frame binning were applied, the file would be roughly 3x smaller.
