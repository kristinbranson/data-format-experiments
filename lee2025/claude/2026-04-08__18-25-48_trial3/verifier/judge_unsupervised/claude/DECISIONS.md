# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads each animal's data file from the `data/` directory using `joblib.load()`. Each file is a joblib-serialized dictionary keyed by animal name, containing arrays for `trace`, `position`, `envs`, etc. The 7 animals are hardcoded in a list. All sessions (days) for each animal are processed in a loop. Within each session, the data is split into 1-minute trials of 1800 frames.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]

trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()  # (n_days,) string array
```

iii. The AI identified that each animal file is stored as a joblib file (extensionless) matching the reference code's `load_dat()` function. This is consistent with the reference code which uses `joblib.load()` to load animal data.

## 1-b. How are the data split into subjects (mice)?

i. Each of the 7 animals corresponds to one subject. The subjects list is the full list of 7 animal names. A `subject_idx` array maps each session to its corresponding subject by looking up the animal's index in the list.

ii.
```python
subjects = animals  # All 7 animals are subjects

for animal in animals_to_process:
    subject_id = animals.index(animal)
    # ... for each session:
    subject_idx_list.append(subject_id)
```

iii. The AI noted 7 subjects from both the paper ("5,413 unique neurons across 207 sessions" from 7 animals) and the data files. Each animal file contains all sessions for that animal.

## 1-c. How are the data split into sessions?

i. Each "day" (recording day) within an animal's data is treated as one session. The `trace` array has shape `(n_days, n_cells, n_frames)`, where the first dimension indexes days. The loop iterates `for day in range(n_days)`, treating each day as a separate session. This yields 207 total sessions (6 animals x 31 days + 1 animal x 21 days).

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape

for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
```

iii. The AI documented that each animal has 31 days of recording (except QLAK-CA1-51 with 21 days), matching the paper's description of "up to three total repetitions (31 days)."

## 1-d. How are the data split into trials?

i. Each session (~40 min at 30 Hz) is split into 1-minute trials of 1800 frames each. The number of trials per session is computed as `n_frames_total // trial_duration_frames`, discarding any remaining partial trial at the end.

ii.
```python
fps = 30
trial_duration_sec = 60
trial_duration_frames = fps * trial_duration_sec  # 1800

n_trials = n_frames_total // trial_duration_frames

for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The instructions explicitly state "1-minute trials within each session." The AI notes that sessions are ~40 min, yielding 39-40 trials per session, with ~1666 leftover frames discarded per session.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are included. Only the final partial trial (less than 1800 frames) is discarded.

ii. No filtering code exists; all trials from `range(n_trials)` are included.

iii. The AI noted "Trial curation: None in original (1 session = 1 day). We split into 1-min trials." The original paper did not have trial-level filtering since each session was one continuous recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from the `trace` variable in each animal's data file. This is a binary calcium event trace (0 or 1) indicating significant rising-phase calcium transients, recorded at 30 Hz via miniscope calcium imaging (GCaMP6f).

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
trace_day = trace[day]   # (n_cells, n_frames)
```

iii. The AI documented: "Neural data: Binary trace vector (1=significant rising-phase event, 0=no event). Already preprocessed." and "No delta F/F needed: The trace data is already binarized."

## 2-b. How is the `neural` data processed?

i. Minimal processing is applied:
1. Only cells registered (non-NaN) on a given day are included.
2. Any remaining NaN values are replaced with 0.
3. Data is cast to float32.
No additional smoothing, binning, or normalization is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
active_trace = np.nan_to_num(active_trace, nan=0.0)
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI noted: "Use raw binary trace (0/1 events) at native 30 Hz -- NO additional processing needed. Trace is already binarized by the original authors."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filtering is excluding cells that are not registered on a given day (all-NaN trace rows). There is no place cell filtering, no minimum firing rate threshold, and no velocity filtering applied to the neural data.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
```

iii. The AI decided to include all registered cells, citing the paper: "motivated the inclusion of all cells in subsequent analyses." The AI noted that the reference code has a velocity filter (v_thresh=5 cm/s) and cell threshold (>5 events) for decoding, but chose not to apply these, reasoning they were part of the reference paper's own decoding pipeline rather than a general data curation step. The AI also noted place cell filtering (p < 0.05 or p < 0.01) exists in the reference code but was not applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 1-minute segment within the recording session. The alignment is simply temporal segmentation: trial 0 starts at frame 0, trial 1 at frame 1800, etc. Neural and behavioral data are inherently synchronized since they share the same 30 Hz frame rate.

ii.
```python
start = trial_idx * trial_duration_frames
end = start + trial_duration_frames
trial_neural = active_trace[:, start:end]
```

iii. The AI set `temporal_alignment_event` to "Start of 1-minute trial segment within recording session" with `off_start=0.0` and `off_end=60.0`. Since the original experiment has continuous recordings without trial structure, the alignment is simply to the start of each 1-minute window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native recording frame rate of 30 Hz, corresponding to ~33.33 ms per time bin. No temporal rebinning is applied.

ii.
```python
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The AI noted that the reference code uses `temporal_bin_size=3` for its own Gaussian Naive Bayes decoding (which bins 3 frames together), but chose to keep the native resolution for the converted data. The instructions did not specify rebinning.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `input` is derived from the `envs` variable in each animal's data file, which contains the string name of the environment geometry for each day (e.g., 'square', 'o', 't', 'u', etc.).

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
env_name = str(envs[day])
```

iii. The AI documented that there are 10 environment geometries corresponding to different partition configurations of the 3x3 arena.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is converted to a 3x3 binary matrix using the `get_env_mat()` function (adapted from the reference code), where 1=accessible partition and 0=blocked partition. This matrix is then flattened to a 9-element vector. The input is static per trial (same value for all trials within a session).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
        # ... (10 environments total)
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)

env_mat = get_env_mat(env_name).flatten()  # (9,)
trial_input = env_mat.astype(np.float32)
```

iii. The AI copied the `get_env_mat()` function directly from the reference code. The input shape is (9,) -- static per trial with no time dimension.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `output` is derived from the `position` variable in each animal's data file, which contains continuous x-y coordinates (in cm) at 30 Hz.

ii.
```python
position = d['position']  # (n_days, 2, n_frames)
pos_day = position[day]   # (2, n_frames)
```

iii. The AI noted that position is tracked using DeepLabCut at 30 Hz, with coordinates in the range [0, 75] cm.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x-y position is discretized into a 3x3 spatial grid (9 bins), where each bin is 25x25 cm. The x and y coordinates are each divided into 3 bins using `np.floor(coord / bin_size)`, clipped to [0, 2], and combined into a single bin ID using `x_bin * 3 + y_bin`.

ii.
```python
def discretize_position_3x3(position, env_size=75.0):
    bin_size = env_size / 3.0
    x = np.clip(position[0], 0, env_size - 1e-10)
    y = np.clip(position[1], 0, env_size - 1e-10)
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    x_bin = np.clip(x_bin, 0, 2)
    y_bin = np.clip(y_bin, 0, 2)
    bin_ids = x_bin * 3 + y_bin
    return bin_ids
```

iii. The task specifies "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying." The AI's approach divides the 75 cm arena into three 25 cm bins in each dimension.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized into 9 spatial bins (0-8) using floor division of x and y coordinates by the bin size (25 cm). Values at or above 75 cm are clipped to stay within the valid range. The output is stored as int64 values.

ii.
```python
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
bin_ids = x_bin * 3 + y_bin

trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The 9 bins are named as `x[0-25]_y[0-25]`, `x[0-25]_y[25-50]`, ..., `x[50-75]_y[50-75]`. The output distribution shows bin 8 (x[50-75]_y[50-75]) is most frequent at ~20%.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are inherently aligned because they share the same 30 Hz frame rate. The same frame indices (`start:end`) are used to extract both neural and position data for each trial, ensuring perfect temporal alignment.

ii.
```python
trial_neural = active_trace[:, start:end].astype(np.float32)
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI noted: "All streams synchronous at 30 Hz" -- position is tracked at the same rate as the calcium imaging frames, so no interpolation or resampling is needed.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of missing data are handled:
1. Unregistered cells (NaN trace rows): excluded from that day's session via `active_mask`.
2. Any remaining NaN values in active cell traces: replaced with 0 using `np.nan_to_num()`.
3. Partial trials at the end of sessions: discarded (only complete 1-minute segments are kept).

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)

n_trials = n_frames_total // trial_duration_frames  # discards remainder
```

iii. The AI noted the safety measure of `nan_to_num` "shouldn't happen for active cells, but safety." The edge case handling was verified: ~1666 frames are discarded per session at the end.

## 6-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, loading each animal's joblib file is the most time-consuming step (8.9-22.6 seconds per animal). Processing each day within an animal takes 0.15-0.69 seconds. Total conversion time was ~215 seconds for all 7 animals.

ii. From conversion output:
```
Loading QLAK-CA1-08... Loaded in 12.8s
Loading QLAK-CA1-75... Loaded in 22.6s
```

iii. The AI documented loading time per animal and estimated 3-4 minutes total for full conversion, which was confirmed at ~3.6 minutes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop (`for trial_idx in range(n_trials)`) iterates over each trial and performs array slicing. This could potentially be vectorized using `np.reshape` to split the entire session's data into trials at once, though the current approach using array slicing is already efficient.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI noted that position discretization is already vectorized. The trial loop performs simple array slicing which is fast. The `env_mat.astype(np.float32)` is recomputed each trial but could be computed once outside the loop.

## 6-c. What processing does the code repeat multiple times?

i. The environment geometry input (`env_mat.astype(np.float32)`) is recomputed identically for every trial within a session, when it only needs to be computed once per session. The `get_env_mat()` function is called once per session, but the `.astype(np.float32)` conversion happens in the trial loop.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # computed once per session

for trial_idx in range(n_trials):
    trial_input = env_mat.astype(np.float32)  # repeated per trial
```

iii. This is a minor inefficiency. The float32 conversion could be moved outside the loop.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No significant unnecessary processing is performed. The code is relatively lean -- it loads data, filters active cells, discretizes position, and splits into trials. The `plot_processing()` function generates visualizations when `--show-processing` is used, but this is optional and only runs when explicitly requested. The environment geometry input could arguably be considered somewhat redundant since it's static per session (not time-varying), meaning it provides the same information across all trials within a session.

ii. The code is straightforward without obvious wasteful processing.

iii. The AI focused on efficiency from the start, noting "Direct numpy array slicing for trial splitting (no loops over frames)" and "Memory freed after each animal with `del`."
