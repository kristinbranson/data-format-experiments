# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the `.mat` files. It hardcodes the 7 animal IDs, then loads each extensionless `joblib` file from `data/` with `joblib.load`. It processes one animal at a time and then iterates over all days in that animal as sessions, later splitting each day into trials.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
def process_animal(animal_name, data_dir, trial_duration_frames=1800, show_processing=False):
    ...
    dat = joblib.load(os.path.join(data_dir, animal_name))
    d = dat[animal_name]
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI justified this by concluding that “Each animal is stored as a joblib file (extensionless) in `data/`.” In trajectory notes it also stated that the dataset contains “Python joblib files or MATLAB .mat files,” and chose the joblib path.

## 1-b. How are the data split into subjects?

i. The AI treats each hardcoded animal ID as one subject. The output `subjects` field is the full hardcoded list of 7 animal names, and `subject_idx` is assigned from the position of each animal in that list.

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

iii. `CONVERSION_NOTES.md` Step 2 identifies the dataset as 7 animals with IDs `QLAK-CA1-{08,30,50,51,56,74,75}`, and Step 5 says “One session = one decoder session,” implying the animal IDs are the subject split.

## 1-c. How are the data split into sessions?

i. Within each loaded animal, the AI treats each recording day as a separate session. It iterates over the first axis of `trace`, `position`, and `envs`, and each `day` becomes one session in the exported dataset.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()  # (n_days,) string array

n_days, n_cells_total, n_frames_total = trace.shape
...
for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly records the decision: “One session = one decoder session: Each recording day is a separate session in our output.”

## 1-d. How are the data split into trials?

i. Each session/day is split into non-overlapping 1-minute trials of 1800 frames at 30 Hz. The AI computes `n_trials` by integer division and discards any remainder at the end of the session.

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

iii. In Step 5 notes, the AI justified this from the task instructions: “Trial duration = 1 minute: Instructions say ‘1-minute trials within each session’.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-level quality filtering. Trials are kept as long as they fit within the 1800-frame segmentation; only the incomplete tail is dropped.

ii.
```python
n_trials = n_frames_total // trial_duration_frames

for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    ...
```

iii. `CONVERSION_NOTES.md` Step 3 says “No trial-level filtering in original study,” and Step 5 repeats that trial creation is only the imposed 1-minute splitting from the task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the `trace` variable in the joblib data structure. For each day/session it uses `trace[day]`.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]       # (n_cells, n_frames)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI maps ``trace[day]`` directly to the target `neural` field and describes it as “raw binary trace (0/1 events) at native 30 Hz.”

## 2-b. How is the `neural` data processed?

i. The AI assumes the traces are already binarized calcium-event data and does not do deconvolution, smoothing, or temporal rebinning. It filters to active cells, optionally fills any remaining NaNs with 0, and slices the result into per-trial `(n_neurons, 1800)` float32 arrays.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI’s written justification appears in Step 1 and Step 5 notes: it states the neural data is a “Binary trace vector” and “already preprocessed,” so “NO additional processing [is] needed.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are retained if they are not all-NaN on that day. After that, the AI replaces any remaining NaNs in retained cells with zero as a safety step.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
```

iii. `CONVERSION_NOTES.md` says “Include only cells registered on that specific day (non-NaN rows)” and “All registered cells included.” The code comment adds the zero-fill rationale: “shouldn’t happen for active cells, but safety.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use a behavioral or stimulus event. It treats the start of each artificial 1-minute segment as the alignment event and slices neural and behavioral streams with the same `start:end` indices.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    ...
```

and in metadata:
```python
'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
'off_start': 0.0,
'off_end': float(trial_duration_sec),
```

iii. The notes justify this implicitly rather than explicitly: Step 3 says “All streams synchronous at 30 Hz,” and Step 5 frames trials as artificial 1-minute chunks rather than event-centered trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native 30 Hz frame rate, so the time bin is about 33.33 ms. It does not apply temporal rebinning.

ii.
```python
fps = 30  # recording frame rate
trial_duration_sec = 60  # 1 minute trials
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. In Step 5 notes, the AI states “Time bin size = 1/30 s ≈ 33.33 ms (raw frame rate)” and “NO additional processing needed.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI does not use the raw `blocked` variable. It derives the decoder input from the per-day environment name stored in `envs`, then maps that name to a 3x3 geometry matrix with `get_env_mat`.

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
...
env_name = str(envs[day])
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. Step 5 in `CONVERSION_NOTES.md` explicitly records this mapping as ``envs[day] -> get_env_mat(env)`` and says this will be the decoder input. The trajectory shows the AI chose this because `get_env_mat()` existed in the reference code and directly encoded the 3x3 environment shape.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts each environment name into a 3x3 binary accessibility matrix, flattens it to length 9, and reuses the same vector for every trial in that session. The vector encodes accessible partitions (`1`) rather than blocked-partition indices.

ii.
```python
def get_env_mat(env):
    ...
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
trial_input = env_mat.astype(np.float32)
```

iii. In Step 5 notes, the AI justifies this as “Environment geometry as flattened 3x3 binary matrix (9 values)” because the task said the decoder input should be environment geometry. Step 1 also notes that `get_env_mat()` comes from the reference code and returns a 3x3 binary matrix.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives the decoded output from the raw `position` variable for each day/session.

ii.
```python
position = d['position'] # (n_days, 2, n_frames)
...
pos_day = position[day]      # (2, n_frames)
```

iii. `CONVERSION_NOTES.md` Step 5 maps ``position[day]`` directly to the target `output` field and describes it as x-y position in `[0, 75]` cm.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips x and y coordinates into `[0, 75)`, divides each axis into 25 cm bins, floors to integer bin indices, clamps those indices to `[0, 2]`, and then combines them into a single class label with `bin_ids = x_bin * 3 + y_bin`.

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

iii. Step 5 notes justify the 3x3 discretization from the task requirement “3 x 3 = 9 spatial bins,” and explicitly state the chosen label rule: ``bin_id = x_bin * 3 + y_bin``.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI thresholds continuous position into 9 discrete categories by splitting each axis at 25 cm and 50 cm, yielding a 3x3 grid. Category IDs are then assigned using the `x_bin * 3 + y_bin` numbering scheme.

ii.
```python
bin_size = env_size / 3.0
...
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
...
bin_ids = x_bin * 3 + y_bin
```

iii. In Step 5 notes, the AI explains this as “Position (x,y) in [0, 75] cm → 3x3 grid of 25 cm bins” and says this “gives 9 spatial bins matching the 3x3 partition structure.”

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes position and neural activity are frame-synchronous and aligns them by slicing both with the same `start:end` trial boundaries. The per-trial output keeps the same 1800 timepoints as the neural trial.

ii.
```python
trial_neural = active_trace[:, start:end].astype(np.float32)
...
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. Step 3 notes state that “All streams synchronous at 30 Hz,” which is the AI’s justification for using identical frame indices for both outputs and neural data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes cells that are entirely NaN on a given day, fills any remaining NaNs in retained traces with zero, clips position values into the valid arena range, and drops any incomplete trial remainder at the end of a session.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
...
n_trials = n_frames_total // trial_duration_frames
```

iii. The written rationale appears in Step 5 notes (“Include only cells registered on that specific day”) plus the in-code comment that zero-filling remaining NaNs is a safety measure. The notes also say the final short chunk is discarded when making 1-minute trials.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by per-animal data loading with `joblib.load`; optional plotting is also expensive when `--show-processing` is enabled. The code prints load and total runtimes, reflecting that I/O was treated as the main cost.

ii.
```python
t0 = time.time()
print(f"Loading {animal_name}...", flush=True)
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s", flush=True)
...
if show_processing:
    plot_processing(...)
```

iii. `CONVERSION_NOTES.md` Step 6 highlights loading one animal at a time and freeing memory after each animal, which implies the AI recognized data loading as the expensive step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code keeps an explicit Python loop over trials inside every session. Those trial slices could have been reshaped or split in a more vectorized way. The nested loops that build output bin names are also trivial overhead but unnecessary.

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
        output_bin_names.append(f"x[{x_start}-{x_end}]_y[{y_start}-{y_end}]")
```

iii. The AI did not explicitly justify keeping these loops. Step 6 only claims “Direct numpy array slicing for trial splitting (no loops over frames),” so the remaining trial loop seems to have been accepted as simple enough.

## 6-c. What processing does the code repeat multiple times?

i. Within the per-trial loop, the AI repeatedly casts the same static `env_mat` to float32 for every trial and repeatedly slices/casts neural and output arrays trial by trial. In sample mode it also processes all sessions for the first animal and only later truncates to 2 sessions.

ii.
```python
for trial_idx in range(n_trials):
    ...
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
...
result = process_animal(animal, data_dir, trial_duration_frames,
                       show_processing=show)

n_sessions = len(result['neural'])
if max_sessions is not None:
    n_sessions = min(n_sessions, max_sessions)
```

iii. The only explicit efficiency justification is Step 6’s note about using numpy slicing and `del` for memory. There is no explicit justification for the repeated per-trial casts or for processing all sessions before truncating sample mode output.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes optional plotting and plotting-only bookkeeping (`sessions_env`, `active_cells_per_session`) that are not part of the exported dataset. In sample mode it fully processes all sessions for one animal even though only the first 2 are kept. It also returns `envs` and `active_cells` from `process_animal` even though only `active_cells` is used later.

ii.
```python
sessions_env = []
active_cells_per_session = []
...
sessions_env.append(env_name)
active_cells_per_session.append(n_active)
...
if show_processing:
    plot_processing(...)
...
return {
    'neural': sessions_neural,
    'input': sessions_input,
    'output': sessions_output,
    'envs': sessions_env,
    'active_cells': active_cells_per_session,
}
...
if max_sessions is not None:
    n_sessions = min(n_sessions, max_sessions)
for s in range(n_sessions):
    all_neural.append(result['neural'][s])
```

iii. Step 6 justifies plotting as “show processing” visualization, but there is no downstream decoder need for it. There is also no written justification for processing extra sessions in sample mode and then discarding them.
