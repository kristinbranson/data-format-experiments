# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 7 animal IDs, then loads each animal from the extensionless joblib file in `data/` with `joblib.load`. Within each loaded animal dict, it reads `trace`, `position`, and `envs`. It processes one animal at a time and appends each day/session into the final top-level lists. This mirrors the reference `load_dat(..., format="joblib")` behavior, but calls `joblib.load` directly instead of importing `load_dat`.

ii.
```python
def process_animal(animal_name, data_dir, trial_duration_frames=1800, show_processing=False):
    ...
    dat = joblib.load(os.path.join(data_dir, animal_name))
    d = dat[animal_name]
    ...
    trace = d['trace']       # (n_days, n_cells, n_frames)
    position = d['position'] # (n_days, 2, n_frames)
    envs = d['envs'].flatten()  # (n_days,) string array
```

```python
data_dir = 'data'
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

for animal in animals_to_process:
    result = process_animal(animal, data_dir, trial_duration_frames,
                           show_processing=show)
```

iii. In `CONVERSION_NOTES.md` Step 10, the agent justifies this as matching reference loading: `joblib.load(f'data/{animal}')` versus reference `load_dat(animal, p, format='joblib')`. The trajectory also shows it inspected the reference `load_dat` function and noted that the reference joblib path is `os.path.join(p_data, f"{animal}")`.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded animal list. The agent sets `subjects = animals` and assigns one subject index per session based on the animal’s position in that list.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
subjects = animals  # All 7 animals are subjects
subject_idx_list = []
...
for animal in animals_to_process:
    subject_id = animals.index(animal)
    ...
    for s in range(n_sessions):
        ...
        subject_idx_list.append(subject_id)
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent documented the dataset as 7 animals with known IDs, and in Step 5 it explicitly decided that “one session = one decoder session” and animals are the subjects.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day as one session. For each animal it loops over `range(n_days)` and creates one session entry per day in `all_neural`, `all_input`, and `all_output`.

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

iii. `CONVERSION_NOTES.md` Step 5 says “One session = one decoder session: Each recording day is a separate session in our output,” based on the paper/methods statement that sessions are daily 40-minute recordings.

## 1-d. How are the data split into trials?

i. Each day/session is cut into contiguous 1-minute trials at 30 Hz, so each trial is 1800 frames. Trial boundaries are purely temporal segments within the session, not behavioral events.

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

iii. In `CONVERSION_NOTES.md` Step 5, the agent justifies this from the task instruction “1-minute trials within each session” and notes that the original study did not have trialized sessions, so this split is task-driven.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter. The code keeps every full 1-minute chunk and only discards the last partial chunk of each session because it does not fill a complete 1800-frame trial.

ii.
```python
n_trials = n_frames_total // trial_duration_frames
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    ...
```

iii. In `CONVERSION_NOTES.md` Step 3 and Step 5, the agent states that there was no trial-level filtering in the original study and that it would split sessions into 1-minute trials. In Step 10 it also records that “~1666 frames discarded per session (last partial minute) — acceptable.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the raw `trace` array for each day. The agent interprets `trace` as already-preprocessed binary calcium event data.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]       # (n_cells, n_frames)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 3, the agent states that `trace` is “binarized calcium trace / binary calcium events” and that no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The agent uses the binary trace almost unchanged. It removes rows that are all NaN, replaces any remaining NaNs with 0, slices trials, and casts to `float32`. It does not reproduce the reference decoder’s additional trace smoothing and 3-frame temporal binning.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Use raw binary trace (0/1 events) at native 30 Hz — NO additional processing needed” and “No velocity filtering: the paper's velocity filter was for decoding within their pipeline; we let the decoder handle this.” The trajectory shows the agent knew the reference decoder used `gaussian_filter1d(..., sigma=temporal_bin_size)` and `AvgPool1d(... temporal_bin_size=3)`, but chose not to port that processing into the converted dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural filtering in the conversion is dropping cells whose entire day trace is NaN, which the agent interprets as “unregistered on that day.” It does not apply the reference decoder’s velocity-based sample filter or the `cell_threshold > 5 events` filter.

ii.
```python
# A cell is registered if its trace is not all NaN
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent decided “All registered cells included” and “No velocity filtering.” In Step 3 it acknowledged that the reference decoding code used `>5 events when v>5cm/s`, but treated that as not relevant to conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological or task event alignment. Neural data are aligned to the start of each synthetic 1-minute segment, with the same `start:end` indices used for neural and output position. The metadata names the alignment event as the start of the 1-minute trial segment.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    ...
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
    'off_start': 0.0,
    'off_end': float(trial_duration_sec),
    ...
}
```

iii. `CONVERSION_NOTES.md` Step 3 says all streams are synchronous at 30 Hz, and Step 5 says sessions are split into 1-minute trials. So the justification is that no within-session stream alignment is needed beyond shared frame indexing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native recording frame rate: 30 Hz, or about 33.33 ms per sample. No temporal rebinning is applied in the conversion script.

ii.
```python
fps = 30  # recording frame rate
trial_duration_sec = 60  # 1 minute trials
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “Time bin size = 1/30 s ≈ 33.33 ms (raw frame rate)” and “NO additional processing needed.” This was a deliberate departure from the reference decoder code, which the trajectory shows used `temporal_bin_size=3`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from `envs[day]`, not from the raw `blocked` field. The day’s environment-name string is converted into a 3x3 accessibility matrix with `get_env_mat`.

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
...
env_name = str(envs[day])
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 5, the agent says it reused the reference `get_env_mat()` logic and mapped `envs[day] -> get_env_mat(env)`. The trajectory shows it inspected the reference `get_env_mat` implementation in `utils.py`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The processing is a lookup from environment name to a binary 3x3 matrix where 1 means accessible and 0 means blocked, followed by flattening to a 9-element vector. The same static vector is repeated as the per-trial input for every trial in that session.

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
```

```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
trial_input = env_mat.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says the decoder input should be “which partitions are accessible” and that `get_env_mat()` from the reference code provides exactly that representation.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position is derived directly from the raw `position` array for each day.

ii.
```python
position = d['position'] # (n_days, 2, n_frames)
...
pos_day = position[day]      # (2, n_frames)
...
bin_ids = discretize_position_3x3(pos_day)  # (n_frames,)
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the agent documents `position` as the raw x-y behavior stream and maps it directly to the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent clips raw x and y to the valid arena range `[0, 75)`, divides each axis into three equal 25 cm bins, computes integer bin indices with `floor`, then combines the x and y bin indices into one categorical bin ID per frame.

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

iii. `CONVERSION_NOTES.md` Step 5 justifies this from the decoder task requirement “3 x 3 = 9 spatial bins” and from the environment’s 75 cm size, giving 25 cm bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous position is categorized with x and y edges at 0, 25, 50, and 75 cm. Category IDs are assigned in row-major order as `x_bin * 3 + y_bin`, yielding 9 classes labeled 0 through 8.

ii.
```python
bin_size = env_size / 3.0
...
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
...
bin_ids = x_bin * 3 + y_bin
```

```python
for i in range(3):
    for j in range(3):
        x_start = i * 25
        x_end = (i + 1) * 25
        y_start = j * 25
        y_end = (j + 1) * 25
        output_bin_names.append(f"x[{x_start}-{x_end}]_y[{y_start}-{y_end}]")
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent writes out the same edges and category formula as its justification.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural activity are aligned by taking the same frame slices from synchronous 30 Hz streams. For each trial, `trial_output` uses exactly the same `start:end` frame indices as `trial_neural`.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    ...
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 3 says the DAQ acquired behavioral and cellular streams simultaneously at 30 Hz, and Step 10 says trial-boundary spot checks matched exactly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or awkward values with simple defensive defaults: unregistered cells are dropped if their whole row is NaN, any remaining NaNs in kept cells are zero-filled, positions are clipped to the valid arena range, unknown environments return a 3x3 NaN matrix, and incomplete trailing session fragments are dropped because they do not make a full 1-minute trial.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
```

```python
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
...
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
n_trials = n_frames_total // trial_duration_frames
```

iii. The justification in `CONVERSION_NOTES.md` is mostly pragmatic rather than reference-derived: Step 5 says active cells are the non-NaN rows and Step 10 says discarding the last partial minute is “acceptable.” The trajectory and notes do not show a deeper reference-based justification for zero-filling residual NaNs or clipping positions.

## 6-a. What are the most time-consuming steps of the code?

i. The main expensive steps are loading each large joblib animal file from disk, iterating through all sessions/trials to slice and append arrays into nested Python lists, and finally serializing the ~20 GB pickle. Plot generation is also expensive when `--show-processing` is enabled.

ii.
```python
t0 = time.time()
print(f"Loading {animal_name}...", flush=True)
dat = joblib.load(os.path.join(data_dir, animal_name))
...
for day in range(n_days):
    ...
    for trial_idx in range(n_trials):
        ...
        trials_neural.append(trial_neural)
        trials_input.append(trial_input)
        trials_output.append(trial_output)
```

```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. `CONVERSION_NOTES.md` Step 7 reports load/process timing per animal, and Step 9 reports a full output size near 20 GB and total conversion time around 215 seconds. That is the agent’s own justification for where the runtime goes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest candidate is the inner `for trial_idx in range(n_trials)` loop, which repeatedly slices arrays and appends Python objects. Since all trials in a session are fixed-length and contiguous, the session arrays could have been truncated once and reshaped instead of sliced trial-by-trial. The nested loop that builds `output_bin_names` is trivial but also unnecessary.

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
```

```python
output_bin_names = []
for i in range(3):
    for j in range(3):
        ...
        output_bin_names.append(f"x[{x_start}-{x_end}]_y[{y_start}-{y_end}]")
```

iii. In `CONVERSION_NOTES.md` Step 6, the agent highlights only that it avoided looping over frames, but the remaining trial loop is still a clear vectorization opportunity visible in the code.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats per-trial slicing and type-casting for each session, repeats the same static `env_mat.astype(np.float32)` conversion for every trial in a session, and repeats similar append/list-building work across all modalities. In sample mode it also processes the first animal’s full session set before truncating to two sessions.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
for trial_idx in range(n_trials):
    ...
    trial_input = env_mat.astype(np.float32)
```

```python
if args.sample:
    animals_to_process = [animals[0]]
    max_sessions = 2
...
result = process_animal(animal, data_dir, trial_duration_frames,
                       show_processing=show)
...
if max_sessions is not None:
    n_sessions = min(n_sessions, max_sessions)
```

iii. The trajectory and notes show the agent was focused on frame-level vectorization, but the code still repeats work at the trial/session level. The sample-mode truncation after full processing is an especially direct example.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In sample mode, it processes all sessions for the first animal and only keeps two of them. In full mode, it builds and returns `envs` per animal inside `process_animal`, but the final dataset never stores that field. It also computes `n_cells_total` and imports several unused modules, though those are minor.

ii.
```python
if args.sample:
    # Process first 2 sessions (days) from first animal only
    animals_to_process = [animals[0]]
    max_sessions = 2
...
result = process_animal(animal, data_dir, trial_duration_frames,
                       show_processing=show)
...
if max_sessions is not None:
    n_sessions = min(n_sessions, max_sessions)
```

```python
sessions_env = []
...
sessions_env.append(env_name)
...
return {
    'neural': sessions_neural,
    'input': sessions_input,
    'output': sessions_output,
    'envs': sessions_env,
    'active_cells': active_cells_per_session,
}
```

iii. `CONVERSION_NOTES.md` does not call these out explicitly, but the code shows them directly. The sample-mode over-processing is especially clear because truncation happens only after `process_animal` finishes.
