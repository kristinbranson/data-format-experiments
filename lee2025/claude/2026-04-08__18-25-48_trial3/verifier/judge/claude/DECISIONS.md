# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each animal's data is stored as a joblib file (extensionless) in the `data/` directory. The AI loads each file using `joblib.load()`, which returns a dictionary keyed by animal name containing `trace`, `position`, `envs`, and other fields. The trace array has shape `(n_days, n_cells, n_frames)`, position has shape `(n_days, 2, n_frames)`, and envs has shape `(n_days, 1)`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()  # (n_days,) string array
```

iii. The AI identified in Step 2 of CONVERSION_NOTES.md that the data files are stored in joblib format (extensionless files). This matches the reference code's `load_dat()` function which supports joblib loading. The AI hardcodes the list of 7 animal names rather than discovering them via glob.

## 1-b. How are the data split into subjects?

i. Each joblib file in the data directory corresponds to one subject (mouse). The AI hardcodes the list of 7 animal names: `['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']`.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
subjects = animals  # All 7 animals are subjects
```

iii. The AI identified 7 subjects from the data directory and paper, matching the expected count. Note that `subjects = animals` assigns the full list of all 7 animals to `subjects` even in sample mode when only 1 animal is processed, though this only matters for sample mode and not the full conversion.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal becomes a separate session. The AI iterates over the day dimension of the trace array (`trace[day]`), producing one session per day per animal.

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape
for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
```

iii. The AI documented 207 total sessions (31 per animal for 6 animals, 21 for QLAK-CA1-51), consistent with the paper's reported 207 sessions.

## 1-d. How are the data split into trials?

i. Each session (~40 minutes at 30 Hz) is split into non-overlapping 1-minute trials of 1800 frames each. The last partial segment (fewer than 1800 frames) is discarded.

ii.
```python
trial_duration_frames = fps * trial_duration_sec  # 30 * 60 = 1800
n_trials = n_frames_total // trial_duration_frames
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The instructions specify "1-minute trials within each session." The AI produces 39-40 trials per session depending on whether the session has slightly under or over 72000 frames. Note: the AI uses `n_frames_total` from the trace array shape dimension, which is the same for all days within an animal. This means all days of the same animal get the same trial count.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial filtering is applied. All complete 1-minute segments are kept. Only the trailing partial segment is discarded.

ii. N/A — no filtering code.

iii. The AI noted in CONVERSION_NOTES.md that the original paper does not filter trials (one session = one day). Since trials are artificial 1-minute segments created for the decoder task, there are no quality-based trial exclusions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium events (0 = no event, 1 = significant rising-phase transient) at 30 Hz. The data was already preprocessed by the original authors from raw calcium imaging.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
trace_day = trace[day]   # (n_cells, n_frames)
```

iii. The AI identified in Step 1 of CONVERSION_NOTES.md that the trace data is "Binary trace vector (1=significant rising-phase event, 0=no event). Already preprocessed." and that "No delta F/F needed."

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) identify active (registered) cells for each day by checking for all-NaN rows, (2) replace any remaining NaN values with 0 using `np.nan_to_num`, and (3) cast to float32. No additional temporal processing, smoothing, or rate computation is applied.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI documented that the trace data is already binarized and no further processing is needed. The `nan_to_num` step is described as a safety measure for edge cases. The CONVERSION_NOTES state: "Use raw binary trace (0/1 events) at native 30 Hz — NO additional processing needed."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons whose trace is all-NaN for a given day (unregistered via CellReg cross-session tracking) are excluded. All other registered cells are included regardless of firing rate or place field properties.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
n_active = active_mask.sum()
```

iii. The AI noted in CONVERSION_NOTES Step 5: "All registered cells included (no place cell filtering, matching paper's approach)." The paper states it was "motivated the inclusion of all cells in subsequent analyses."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are created as sequential 1-minute segments from the start of each session. The temporal alignment event is simply the start of each trial segment.

ii. N/A — no alignment code beyond sequential trial splitting.

iii. The AI's metadata sets `'temporal_alignment_event': 'Start of 1-minute trial segment within recording session'` and `'off_start': 0.0`, reflecting that there is no stimulus event to align to.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per time bin). No temporal rebinning or downsampling is applied.

ii.
```python
fps = 30  # recording frame rate
...
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The AI documented: "Time bin size = 1/30 s ≈ 33.33 ms (raw frame rate)" in CONVERSION_NOTES Step 5.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable (environment names like 'square', 'o', 't', etc.), which is mapped through the `get_env_mat()` function from the reference code. This function converts environment names to 3×3 binary matrices where 1=accessible and 0=blocked.

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI adapted `get_env_mat()` directly from the reference code (`georepca1/src/utils.py`). CONVERSION_NOTES Step 5 states: "`get_env_mat()` returns 3x3 binary matrix: 1=accessible partition, 0=blocked. This will be our decoder input."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is looked up in a dictionary mapping names to 3×3 binary matrices. The matrix is flattened to a 9-element vector (1=accessible, 0=blocked) and cast to float32. This is static per session (all trials within a session share the same input).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
trial_input = env_mat.astype(np.float32)
```

iii. The AI chose this approach because it directly represents "which part of the arena is blocked" using the reference code's own environment geometry function. The encoding uses 1=accessible, 0=blocked.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per session — the same 9-element vector is used for every trial within a session. No temporal alignment is needed.

ii.
```python
trial_input = env_mat.astype(np.float32)  # same for all trials in session
```

iii. Environment geometry does not change within a session, so it is repeated identically for all trials.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse tracked with DeepLabCut, stored as shape `(n_days, 2, n_frames)` with values in [0, 75] cm.

ii.
```python
position = d['position']  # (n_days, 2, n_frames)
pos_day = position[day]   # (2, n_frames)
```

iii. The AI identified position tracking via DeepLabCut at 30 Hz in a 75×75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x,y position is discretized into a 3×3 grid of 9 spatial bins. Each axis is divided into 3 equal bins of 25 cm each. The bin edges are at 0, 25, 50, 75 cm. The formula `x_bin * 3 + y_bin` combines x and y bins into a single integer label (0-8).

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

iii. CONVERSION_NOTES Step 5: "Position (x,y) in [0, 75] cm → 3x3 grid of 25 cm bins. Bin edges: [0, 25, 50, 75] for both x and y. Combine x_bin and y_bin into single label: `bin_id = x_bin * 3 + y_bin` (0-8)."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized by dividing each axis into 3 bins of 25 cm each: [0,25), [25,50), [50,75]. Values at the upper boundary (75 cm) are clipped to the last bin. The 2D bins are combined into 9 categories (0-8).

ii.
```python
x = np.clip(position[0], 0, env_size - 1e-10)  # clips 75.0 → 74.999...
y = np.clip(position[1], 0, env_size - 1e-10)
x_bin = np.floor(x / bin_size).astype(int)  # bin_size = 25.0
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, 2)
y_bin = np.clip(y_bin, 0, 2)
```

iii. The task specification states "3 x 3 = 9 spatial bins." The AI uses floor division and clipping to handle edge cases at boundaries.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and are stored in arrays with the same frame dimension. Both are split into trials using the same frame indices, ensuring frame-for-frame alignment.

ii.
```python
# Both use the same start:end indexing
trial_neural = active_trace[:, start:end].astype(np.float32)
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. CONVERSION_NOTES Step 4 verifies: "All streams synchronous at 30 Hz."

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~33.33 ms (1/30 Hz). No temporal rebinning is applied — data is kept at the native recording frame rate.

ii.
```python
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. Same as 2-e. The AI preserves the native 30 Hz sampling rate.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural (trace) and output (position) are sampled synchronously at 30 Hz and share the same frame indices. Input (environment geometry) is static per session and has no temporal dimension. All three are split into trials using the same frame boundaries.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)  # static
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI verified in CONVERSION_NOTES Step 10 that "Trial boundaries verified: no overlap, no gaps" and checked edge cases at trial boundaries.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Three forms of data issues are handled: (1) Unregistered neurons (all-NaN traces per day) are excluded. (2) Any remaining isolated NaN values in active neurons are replaced with 0. (3) Partial trials at the end of sessions (fewer than 1800 frames) are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
n_trials = n_frames_total // trial_duration_frames  # discards remainder
```

iii. CONVERSION_NOTES Step 10 documents sanity checks confirming data integrity. The AI noted that NaN indicates unregistered cells from the CellReg cross-session registration.

## 7-a. What are the most time-consuming steps of the code?

i. Loading data files via `joblib.load()` is the dominant bottleneck. Each animal takes 9-23 seconds to load, while processing all sessions for an animal takes only a few seconds. Total conversion time is ~215 seconds, with loading accounting for the majority.

ii. N/A (observation about runtime, not a code decision).

iii. From conversion_full_out.txt: "Loaded in 12.8s" (QLAK-CA1-08) to "Loaded in 22.6s" (QLAK-CA1-75). Processing per session is 0.15-0.69s.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trial indices to slice neural, input, and output arrays. This could be vectorized using `np.array_split` or reshape operations. Additionally, `active_trace[:, start:end].astype(np.float32)` converts dtype per trial rather than once on the full array.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI noted in CONVERSION_NOTES Step 6: "Direct numpy array slicing for trial splitting (no loops over frames)" — the loop is over trials, not frames, which is reasonable since the output format requires a list of per-trial arrays. The loop is not a significant bottleneck compared to I/O.

## 7-c. What processing does the code repeat multiple times?

i. `env_mat.astype(np.float32)` is repeated inside the trial loop but the environment geometry doesn't change across trials within a session. This could be computed once outside the loop. Similarly, the `astype(np.float32)` cast on neural data is done per trial rather than once on the full session array.

ii.
```python
for trial_idx in range(n_trials):
    trial_input = env_mat.astype(np.float32)  # repeated per trial
    trial_neural = active_trace[:, start:end].astype(np.float32)  # cast per trial
```

iii. The AI's CONVERSION_NOTES did not explicitly identify these repeated operations, noting only "Vectorized position discretization" as an efficiency measure.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No obviously unnecessary processing is performed. The AI processes only the data needed for the target format (neural traces, position, environment geometry). The `np.nan_to_num` step is arguably unnecessary since active cells should not contain NaN values, but it is a safety measure rather than wasted computation.

ii. N/A

iii. The AI kept processing minimal, as documented in CONVERSION_NOTES Step 6.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. Unregistered neurons (all-NaN) are excluded per day. Isolated NaN values are replaced with 0. Incomplete trials are discarded.

ii. (Same code as 6)

iii. The AI verified in CONVERSION_NOTES Step 10 through spot-checks that data integrity is maintained across the conversion.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Data loading via joblib dominates at ~120 seconds out of ~215 total.

ii. N/A

iii. Same as 7-a.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The trial-splitting loop and per-trial dtype casts could be vectorized.

ii. (Same code as 7-b)

iii. Same as 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. `env_mat.astype(np.float32)` and per-trial `astype` calls are repeated unnecessarily.

ii. (Same code as 7-c)

iii. Same as 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. No significant unnecessary processing identified.

ii. N/A

iii. Same as 7-d.
