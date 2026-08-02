# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over a hardcoded list of 7 animal names. For each animal, it loads a joblib file from the `data/` directory using `joblib.load()`. Each file contains a dictionary keyed by the animal name, with sub-keys `trace`, `position`, and `envs`. The `trace` array has shape `(n_days, n_cells, n_frames)`, `position` has shape `(n_days, 2, n_frames)`, and `envs` has shape `(n_days, 1)`. The code then iterates over days (sessions) and splits each session into 1-minute trials (1800 frames at 30 Hz).

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

iii. The AI identified the data structure through exploration of the data files and reference code (CONVERSION_NOTES Steps 1-2). The reference code's `load_dat()` function uses the same joblib loading approach, and the AI documented that each animal file contains trace, position, envs, and other variables.

## 1-b. How are the data split into subjects?

i. Each of the 7 animal files corresponds to one subject (mouse). The AI maintains a `subjects` list of all animal names and tracks which sessions belong to which subject via `subject_idx`.

ii.
```python
subjects = animals  # All 7 animals are subjects
for animal in animals_to_process:
    subject_id = animals.index(animal)
    # ... process animal ...
    for s in range(n_sessions):
        subject_idx_list.append(subject_id)
```

iii. The AI documented 7 subjects matching the paper's description of 7 mice (QLAK-CA1-{08,30,50,51,56,74,75}). This matches the paper statement of "5,413 unique neurons across 207 sessions" from 7 animals.

## 1-c. How are the data split into sessions?

i. Each recording day for each animal is treated as one session. The `trace` array's first dimension indexes days, so the code iterates `for day in range(n_days)`. Six animals have 31 days and one (QLAK-CA1-51) has 21 days, totaling 207 sessions.

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape

for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
```

iii. The AI verified this matches the paper's statement of 207 total sessions (6 animals x 31 days + 1 animal x 21 days = 207).

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute (1800 frame) trials by integer division of total frames. The last partial trial (fewer than 1800 frames) is discarded. This yields 39-40 trials per session.

ii.
```python
trial_duration_frames = fps * trial_duration_sec  # 1800 frames (30 Hz * 60 s)
n_trials = n_frames_total // trial_duration_frames

for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The instructions specify "1-minute trials within each session." The AI used 1800 frames at 30 Hz = 60 seconds per trial. Sessions are ~39.9-40.1 minutes long, yielding 39 or 40 complete trials and discarding the last ~1666 frames (~55.5 seconds).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute trials are included. The only "filtering" is that the last partial trial (less than 1800 frames) is discarded.

ii.
```python
n_trials = n_frames_total // trial_duration_frames
# No filtering logic applied to trials
```

iii. The AI noted in CONVERSION_NOTES that the original paper does not filter trials: "Trial curation: None in original (1 session = 1 day). We split into 1-min trials." The reference code processes entire sessions without trial-level filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` variable in the raw data, which contains binary calcium event traces. The shape is `(n_days, n_cells, n_frames)` with values {0, 1, NaN}.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
trace_day = trace[day]   # (n_cells, n_frames)
```

iii. The AI identified the trace data as "Binary calcium events (binarized rising-phase transients)" from the reference code and paper. The CONVERSION_NOTES document: "Neural data: Binary trace vector (1=significant rising-phase event, 0=no event). Already preprocessed."

## 2-b. How is the `neural` data processed?

i. Minimal processing is applied. The binary trace is used directly at the native 30 Hz frame rate. Active (registered) cells are selected by filtering out rows that are all-NaN. Any remaining NaN values are replaced with 0. The data is cast to float32.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
active_trace = np.nan_to_num(active_trace, nan=0.0)
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI justified this by noting: "Use raw binary trace (0/1 events) at native 30 Hz -- NO additional processing needed. Trace is already binarized by the original authors." The CONVERSION_NOTES confirm the trace is already preprocessed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only filtering applied is exclusion of unregistered cells (cells with all-NaN traces on a given day, due to CellReg cross-day registration). No additional quality controls are applied -- specifically, no minimum event count threshold and no velocity-based filtering are used.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
```

iii. The AI noted in CONVERSION_NOTES: "All registered cells included (no place cell filtering, matching paper's approach)" and "Paper says 'motivated the inclusion of all cells in subsequent analyses'." However, the reference code's decoding pipeline uses `cell_threshold=5` (cells with >5 events when velocity > 5 cm/s), which the AI did not apply.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 1-minute trial segment. Since sessions are continuous recordings at 30 Hz, trials are simply consecutive non-overlapping 1800-frame windows starting from the beginning of the session. There is no event-based alignment (no stimulus onset or behavioral event to align to).

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI set `temporal_alignment_event` to "Start of 1-minute trial segment within recording session" and `off_start` to 0.0, `off_end` to 60.0. This reflects that there's no natural behavioral event to align to in this continuous foraging task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 1/30 Hz = 33.33 ms (the native camera frame rate). No temporal rebinning is applied. Each trial has 1800 timepoints.

ii.
```python
fps = 30  # recording frame rate
# In metadata:
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The AI noted that the native frame rate is 30 Hz and chose not to rebin. However, the reference code uses `temporal_bin_size=3` for its decoding pipeline (binning 3 frames together), but the AI treated this as internal to the decoder rather than a data preprocessing step.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable in the raw data, which contains string names of the environment geometry for each day (e.g., 'square', 'o', 't', 'u', etc.).

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI identified 10 unique environment geometries from the reference code and paper. The `get_env_mat()` function maps environment names to 3x3 binary matrices indicating which partitions are accessible.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via the `get_env_mat()` function (adapted from the reference code). The matrix is then flattened to a 9-element vector. Each element is 1 (partition accessible) or 0 (partition blocked).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        # ... 10 total environments
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)

env_mat = get_env_mat(env_name).flatten()  # (9,)
trial_input = env_mat.astype(np.float32)
```

iii. The AI directly adapted the `get_env_mat()` function from the reference code (georepca1/src/utils.py). The function is copied verbatim with the same environment-to-matrix mappings.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per trial (the same geometry applies to all timepoints within a trial, and indeed to all trials within a session/day). The input has shape `(9,)` per trial (no time dimension), meaning it is a per-trial constant.

ii.
```python
trial_input = env_mat.astype(np.float32)  # shape (9,), same for all trials in a session
```

iii. The instructions specify "Environment geometry to represent which part of the arena is blocked. Static per-trial." The AI correctly implemented this as a static (non-time-varying) input.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable in the raw data, which has shape `(n_days, 2, n_frames)` containing x,y coordinates in cm within the [0, 75] cm arena.

ii.
```python
position = d['position']  # (n_days, 2, n_frames)
pos_day = position[day]   # (2, n_frames)
```

iii. The AI documented that position data was tracked using DeepLabCut at 30 Hz, with coordinates in [0, 75] cm.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous x,y position is discretized into a 3x3 grid (9 spatial bins) using 25 cm bin edges. Each position coordinate is divided by the bin size (25 cm), floored, and clipped to [0, 2]. The x and y bin indices are combined into a single bin ID.

ii.
```python
def discretize_position_3x3(position, env_size=75.0):
    bin_size = env_size / 3.0  # 25.0 cm
    x = np.clip(position[0], 0, env_size - 1e-10)
    y = np.clip(position[1], 0, env_size - 1e-10)
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    x_bin = np.clip(x_bin, 0, 2)
    y_bin = np.clip(y_bin, 0, 2)
    bin_ids = x_bin * 3 + y_bin
    return bin_ids
```

iii. The instructions specify "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI implemented 25 cm bins matching the 75 cm arena size divided into 3 equal parts.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized into 9 bins (0-8) by dividing the 75 cm arena into a 3x3 grid with 25 cm bin edges. The bin ID is computed as `x_bin * 3 + y_bin` where x_bin and y_bin are each in {0, 1, 2}. This produces integer categories directly usable as class labels.

ii.
```python
bin_ids = x_bin * 3 + y_bin  # 0-8
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The output distribution shows all 9 bins are represented, with bin 8 (upper-right corner) most frequent at ~20% and bin 4 (center) least frequent at ~5.7%, which makes sense because the center partition is blocked in several environments.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and share the same frame indices. The same frame range `[start:end]` is used to slice both neural and position data for each trial, so they are inherently aligned.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. Both data streams come from the same recording system at 30 Hz, so frame-level alignment is guaranteed by using the same indices.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms (1/30 Hz), the native camera frame rate. No temporal rebinning is applied. The reference code uses `temporal_bin_size=3` for its decoder, but the AI did not apply this rebinning to the converted data.

ii.
```python
fps = 30
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The AI treated the temporal binning in the reference code as internal to the reference's decoder pipeline rather than a data preprocessing step.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural (trace) and output (position) are both recorded at 30 Hz on the same frame clock. They are sliced using identical frame indices for each trial. The input (environment geometry) is static per trial and does not require temporal alignment.

ii.
```python
trial_neural = active_trace[:, start:end]
trial_output = bin_ids[start:end].reshape(1, -1)
trial_input = env_mat  # static, no time dimension
```

iii. Since trace and position data come from the same recording system and share the same frame indices, temporal alignment is inherent in the data structure.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Two types of missing data are handled: (1) Unregistered cells (NaN traces) are excluded per day by checking if all values in a cell's trace are NaN. (2) Any remaining NaN values in active cell traces are replaced with 0. The last partial trial (fewer than 1800 frames) is silently discarded.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
n_trials = n_frames_total // trial_duration_frames  # discards last partial trial
```

iii. The AI documented these decisions in CONVERSION_NOTES: "Cross-day registration: Via CellReg; NaN for unregistered cells." The NaN-to-0 replacement is described as a safety measure for active cells.

## 7-a. What are the most time-consuming steps of the code?

i. Loading the joblib data files is the most time-consuming step, taking ~13 seconds per animal. The total conversion for all 7 animals takes ~3.6 minutes, with data loading dominating the runtime.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal_name))
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s", flush=True)
```

iii. The AI documented timing in CONVERSION_NOTES Step 7: "Load 1 animal: ~13s, Process 31 sessions: ~9s, Total per animal: ~22s, Estimated full (7 animals): ~3-4 min."

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials to slice the neural, input, and output arrays. This could be replaced with a single `np.array_split()` or `reshape()` operation on the full-session arrays, avoiding per-trial Python loop overhead.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
    trials_neural.append(trial_neural)
```

iii. With ~39-40 trials per session and 207 sessions, the loop overhead is minimal, and the AI chose not to optimize this further given the total runtime was well under 15 minutes.

## 7-c. What processing does the code repeat multiple times?

i. The environment matrix computation `env_mat.astype(np.float32)` is repeated for every trial within a session, though the environment is the same for all trials in a session. This is a trivially wasteful repetition. Also, `discretize_position_3x3` could be called once and the bin_ids sliced, which is what the code actually does (discretization is outside the trial loop).

ii.
```python
# Outside trial loop (efficient):
bin_ids = discretize_position_3x3(pos_day)
env_mat = get_env_mat(env_name).flatten()

# Inside trial loop (minor redundancy):
trial_input = env_mat.astype(np.float32)  # repeated cast
```

iii. The code is actually well-structured with most processing outside the trial loop. The repeated `astype(np.float32)` on the same env_mat is trivially wasteful but has negligible performance impact.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the entire data dictionary for each animal (including `SFPs`, `centroids`, `maps`, `blocked` fields) but only uses `trace`, `position`, and `envs`. The unused fields consume memory during loading. The `del dat, d` at the end of processing each animal releases this memory.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
# Only uses: d['trace'], d['position'], d['envs']
# Ignores: d['SFPs'], d['centroids'], d['maps'], d['blocked']
del dat, d  # Free memory
```

iii. Since the data is stored in monolithic joblib files, selective loading is not straightforward. The AI appropriately frees memory after processing each animal.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6: Unregistered cells (all-NaN rows in trace) are excluded per day. Remaining NaN values are replaced with 0. Incomplete trials at the end of sessions are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
```

iii. The AI handled NaN values as a natural consequence of cross-day cell registration (CellReg), where cells not detected on a given day have NaN traces.

## 9-a. What are the most time-consuming steps of the code?

i. Same as question 7-a: Loading joblib data files (~13s per animal, ~90s total for all 7 animals) is the dominant cost. Array slicing and processing are fast by comparison.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
```

iii. The total conversion time of ~3.6 minutes is well under the 15-minute threshold specified in the instructions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as question 7-b: The per-trial loop that slices arrays could be replaced with a reshape or array_split operation.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    # ... slice arrays ...
```

iii. The loop performs simple array slicing and appending, which is already efficient due to numpy's view semantics.

## 9-c. What processing does the code repeat multiple times?

i. Same as question 7-c: The `env_mat.astype(np.float32)` cast is repeated per trial within a session. The discretization of position and identification of active cells are correctly done once per session (not repeated).

ii.
```python
for trial_idx in range(n_trials):
    trial_input = env_mat.astype(np.float32)  # repeated per trial
```

iii. This repetition has negligible performance impact.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as question 7-d: Loading entire animal data dictionaries (including SFPs, centroids, maps, blocked) when only trace, position, and envs are needed. Additionally, the processing plots generated with `--show-processing` are not used in the final conversion.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
# Loads but doesn't use: d['SFPs'], d['centroids'], d['maps'], d['blocked']
```

iii. The monolithic file format makes selective loading impractical. The AI appropriately frees memory after processing each animal.
