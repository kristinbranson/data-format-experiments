# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, then loads one extensionless `joblib` file per animal from `data/`. Inside each file it expects a top-level dict keyed by animal name, and then reads `trace`, `position`, and `envs`. Trials are not loaded directly; they are created later by splitting each day/session.

ii.
```python
data_dir = 'data'
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
...
trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()  # (n_days,) string array
```

iii. In `CONVERSION_NOTES.md`, the AI says the reference `load_dat()` supports joblib, and it explicitly claims its loading matches `load_dat(animal, p, format='joblib')`. It also documents the extensionless files as the main per-animal data store.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hard-coded animal IDs. The full `subjects` list is set from that hard-coded list, and each processed session gets a subject index from `animals.index(animal)`.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
subjects = animals  # All 7 animals are subjects
...
for animal in animals_to_process:
    subject_id = animals.index(animal)
    ...
    subject_idx_list.append(subject_id)
```

iii. The notes list those seven mice repeatedly as the complete subject set and describe each extensionless file as one animal.

## 1-c. How are the data split into sessions?

i. The AI treats each day within an animal file as one session. It iterates over the first dimension of `trace`, `position`, and `envs`, and each `day` becomes one output session.

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape
...
for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
    ...
    sessions_neural.append(trials_neural)
    sessions_input.append(trials_input)
    sessions_output.append(trials_output)
```

iii. `CONVERSION_NOTES.md` says “Each recording day is a separate session in our output” and documents the raw arrays as `(n_days, ...)`.

## 1-d. How are the data split into trials?

i. Each session/day is split into contiguous, non-overlapping 1-minute trials at 30 Hz, so each trial is 1800 frames. The number of trials is `n_frames_total // 1800`, and the trailing remainder is dropped.

ii.
```python
fps = 30  # recording frame rate
trial_duration_sec = 60  # 1 minute trials
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
n_trials = n_frames_total // trial_duration_frames
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say “Trial duration = 1 minute: Instructions say ‘1-minute trials within each session’” and “last partial trial discarded if < 1800 frames.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality filtering. Once a day/session is split into 1-minute chunks, every chunk is kept.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    ...
    trials_neural.append(trial_neural)
    trials_input.append(trial_input)
    trials_output.append(trial_output)
```

iii. The notes explicitly say “Trial curation: None in original (1 session = 1 day). We split into 1-min trials.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the raw `trace` array inside each joblib animal file.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]       # (n_cells, n_frames)
```

iii. The notes describe `trace` as the already-binarized calcium-event matrix and state that this is the neural source variable.

## 2-b. How is the `neural` data processed?

i. The AI assumes the joblib `trace` is already arranged as `(cells, frames)` per day, so it does not transpose. It filters to active rows, replaces any remaining NaNs with zero, and casts each trial to `float32`. It does not do deconvolution, dF/F computation, smoothing, or temporal rebinning.

ii.
```python
trace_day = trace[day]       # (n_cells, n_frames)
...
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The notes say the traces are “raw binary trace (0/1 events) at native 30 Hz” and “already binarized by the original authors,” so no extra neural preprocessing was considered necessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural quality filter is removing cells whose entire day/session trace is NaN, treating those as unregistered for that day.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
n_active = active_mask.sum()
```

iii. The notes justify this by saying cross-day registration uses NaNs for unregistered cells and that “all registered cells” should be included.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align neural data to any biological or task event. It effectively treats the start of each artificial 1-minute chunk as the alignment point and records that in metadata.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
...
'metadata': {
    ...
    'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
    'off_start': 0.0,
    'off_end': float(trial_duration_sec),
```

iii. The notes explain that the recordings are continuous and that the 1-minute trials are imposed for the decoder task rather than coming from the original experiment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz frame rate, corresponding to `1000/30 ≈ 33.33 ms` per bin. No temporal rebinning is applied.

ii.
```python
fps = 30  # recording frame rate
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
'metadata': {
    ...
    'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The notes explicitly say “Use raw binary trace directly” and “No additional binning of neural data.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the raw `envs` session labels, not from the raw `blocked` variable.

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
...
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The notes say the decoder input should be environment geometry and explicitly select `envs[day] -> get_env_mat(env)` as the mapping.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps each environment name to a hand-coded 3x3 binary accessibility matrix with `get_env_mat()`, flattens it to length 9, casts it to `float32`, and reuses the same static vector for every trial in that session.

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
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
trial_input = env_mat.astype(np.float32)
```

iii. The notes say this function was adapted from the reference code and that “Environment geometry as input” was chosen because the task asked for which partitions are blocked.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives the decoder output from the raw `position` array.

ii.
```python
position = d['position'] # (n_days, 2, n_frames)
...
pos_day = position[day]      # (2, n_frames)
```

iii. The notes describe `position` as the DeepLabCut-tracked `(x, y)` mouse coordinates in centimeters.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips each coordinate to the 75 cm arena bounds, divides each axis into three 25 cm bins, floors to integer bin indices, clips to `[0, 2]`, and combines the two per-axis bins into a single class label.

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

iii. The notes justify this as matching the requested 3x3 spatial output and the 75x75 cm arena partition structure.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholding is implicit from the 25 cm bin size: `0-25`, `25-50`, and `50-75` cm on each axis. The final categories are the 9 combinations of those x/y bins, encoded as integers `0-8`.

ii.
```python
bin_size = env_size / 3.0
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, 2)
y_bin = np.clip(y_bin, 0, 2)
bin_ids = x_bin * 3 + y_bin
...
for i in range(3):
    for j in range(3):
        x_start = i * 25
        x_end = (i + 1) * 25
        y_start = j * 25
        y_end = (j + 1) * 25
        output_bin_names.append(f"x[{x_start}-{x_end}]_y[{y_start}-{y_end}]")
```

iii. The notes state “Position (x,y) in [0, 75] cm -> 3x3 grid of 25 cm bins” and “Combine x_bin and y_bin into single label.”

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned to neural data by using the same per-day frame axis and the same `start:end` indices when constructing each trial.

ii.
```python
trace_day = trace[day]       # (n_cells, n_frames)
pos_day = position[day]      # (2, n_frames)
...
bin_ids = discretize_position_3x3(pos_day)  # (n_frames,)
...
start = trial_idx * trial_duration_frames
end = start + trial_duration_frames
trial_neural = active_trace[:, start:end].astype(np.float32)
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say all streams are synchronous at 30 Hz and that trial boundaries were checked for correct alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes neurons that are all-NaN for a session, fills any remaining NaNs in retained neural traces with zero, and drops trailing frames that do not make up a full 1-minute trial. It does not do additional correction for position samples.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
n_trials = n_frames_total // trial_duration_frames
```

iii. The notes justify all-NaN removal via cross-day registration and describe `nan_to_num(..., nan=0.0)` as a “safety” step that “shouldn't happen for active cells.”

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify loading each large per-animal joblib file as the main cost, followed by processing all sessions/trials and writing the very large pickle. Optional plotting is extra work when `--show-processing` is enabled.

ii.
```python
t0 = time.time()
print(f"Loading {animal_name}...", flush=True)
dat = joblib.load(os.path.join(data_dir, animal_name))
...
t_total = time.time() - t_total_start
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
...
if show_processing:
    plot_processing(...)
```

iii. `CONVERSION_NOTES.md` includes runtime estimates such as “Load 1 animal ~13s,” “Process 31 sessions ~9s,” and “Total per animal ~22s.”

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the inner `for trial_idx in range(n_trials)` loop, which slices and appends one trial at a time for neural, input, and output arrays. That could have been replaced by a reshape/split-based approach. The nested loop that builds `output_bin_names` is also minor avoidable scalar work.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
    trials_neural.append(trial_neural)
    trials_input.append(trial_input)
    trials_output.append(trial_output)
...
for i in range(3):
    for j in range(3):
        ...
        output_bin_names.append(...)
```

iii. The notes do not explicitly discuss vectorization opportunities; this is inferred from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts the same session-level environment vector to `float32` for every trial, repeatedly appends the same static input once per trial, and repeatedly computes slice boundaries in the per-trial loop. It also does repeated per-animal bookkeeping such as `animals.index(animal)`.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    ...
    trial_input = env_mat.astype(np.float32)
    trials_input.append(trial_input)
...
for animal in animals_to_process:
    subject_id = animals.index(animal)
```

iii. There is no explicit justification for these repeated operations in the notes or trajectory.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main extra work is optional visualization and plotting support. The code also collects `sessions_env` and returns it from `process_animal`, but the final saved dataset never uses it. Timing/summary reporting is also purely diagnostic.

ii.
```python
sessions_env = []
...
sessions_env.append(env_name)
...
if show_processing:
    plot_processing(animal_name, d, sessions_neural, sessions_input,
                   sessions_output, sessions_env, active_cells_per_session,
                   trial_duration_frames)
...
return {
    'neural': sessions_neural,
    'input': sessions_input,
    'output': sessions_output,
    'envs': sessions_env,
    'active_cells': active_cells_per_session,
}
```

iii. The notes justify the plotting path as a sanity-check/visual inspection aid, not as part of the downstream decoder input pipeline.
