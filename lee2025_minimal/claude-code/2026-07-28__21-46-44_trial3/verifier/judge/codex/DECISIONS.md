# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the raw `.mat` files used by the human reference. Instead, it hard-coded the seven animal IDs, loaded one preprocessed per-animal file per mouse with `joblib.load`, extracted the animal-specific dictionary entry, and then iterated through all days in that dictionary. Trials were created later by slicing each day into 1-minute chunks.

ii. 
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal_idx, animal in enumerate(animals):
    print(f"Loading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]

    n_days = d['trace'].shape[0]
    ...
    for day in range(n_days):
        trace_day = d['trace'][day]
        pos_day = d['position'][day]
        env_name = str(d['envs'][day, 0])
```

iii. The justification in `CONVERSION_NOTES.md` is implicit rather than explicit: it describes the dataset as “7 animals” and “207 sessions,” matching the per-animal joblib files. The trajectory also shows the agent inspected the non-`.mat` animal files and the paper’s repo code path that uses `load_dat(..., format="joblib")`, then based its loader on that representation rather than the `.mat` files.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hard-coded animal IDs. `subjects` is simply `list(animals)`, and each loaded animal file is treated as one mouse.

ii. 
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
subjects = list(animals)
...
for animal_idx, animal in enumerate(animals):
    ...
    subject_idx_list.append(animal_idx)
```

iii. `CONVERSION_NOTES.md` explicitly lists the seven animals and treats them as the subject set. The AI appears to have justified this from the paper statistics and the available per-animal files.

## 1-c. How are the data split into sessions?

i. Each day within an animal file is treated as one session. The AI equates “recording day” with “session.”

ii. 
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
    ...
    neural_all.append(neural_trials)
    input_all.append(input_trials)
    output_all.append(output_trials)
```

iii. `CONVERSION_NOTES.md` states, “Each session = one recording day.” That is the AI’s explicit rationale.

## 1-d. How are the data split into trials?

i. Each session/day is split into non-overlapping 1-minute trials at 30 Hz, so each trial is 1800 frames. The number of trials is `n_timepoints // 1800`; any remainder is dropped.

ii. 
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
...
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL

    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. `CONVERSION_NOTES.md` explicitly says sessions are split into “1-minute trials (1800 frames at 30 Hz)” and that remainder frames are discarded.

## 1-e. How are trials filtered based on quality controls?

i. The AI did not apply per-trial quality filtering. It only imposed a session-level guard: if a session had fewer than 2 possible 1-minute trials, it would skip that session entirely.

ii. 
```python
n_trials = n_timepoints // FRAMES_PER_TRIAL
if n_trials < 2:
    print(f"  Day {day}: skipping, only {n_trials} possible trials")
    continue
```

iii. `CONVERSION_NOTES.md` frames this as satisfying the decoder requirement that each session have at least two trials: “Minimum 2 trials per session guaranteed.” No additional trial-quality rationale is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the `trace` array inside each animal dictionary, specifically `d['trace'][day]`.

ii. 
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. `CONVERSION_NOTES.md` says the neural data are “binary calcium transient events,” and the script header documents `trace` as the neural source variable.

## 2-b. How is the `neural` data processed?

i. The AI assumes `trace_day` is already a binary calcium-event matrix with shape `(n_cells, n_timepoints)`. It keeps only “registered” cells, replaces any remaining NaNs with 0, and stores each trial as `float32`. It does not do additional temporal processing or rebinning.

ii. 
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
...
traces = trace_day[registered]  # (n_registered, n_timepoints)
traces = np.nan_to_num(traces, nan=0.0)
...
trial_traces = traces[:, start:end]
neural_trials.append(trial_traces.astype(np.float32))
```

iii. `CONVERSION_NOTES.md` justifies this by claiming the traces are already “binary calcium transient events from rise-phase extraction (z-score > 2.5 threshold)” and that no additional filtering is needed beyond keeping registered cells.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removes cells whose entire trace is NaN for that day, treats those as unregistered cells, and adds a session-level rule that skips a day if fewer than 5 registered cells remain. It also fills any remaining NaNs with 0.

ii. 
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()

if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue

traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the all-NaN removal as keeping only “registered cells (non-NaN traces).” The notes do not justify the `<5`-cell session skip or the NaN-to-zero replacement beyond calling the latter a safety measure in code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats session start as the alignment point and defines each trial relative to the start of the recording session. There is no stimulus/event alignment; alignment is by common slicing boundaries within the continuous recording.

ii. 
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_S,
    ...
}
...
start = t * FRAMES_PER_TRIAL
end = (t + 1) * FRAMES_PER_TRIAL
trial_traces = traces[:, start:end]
```

iii. `CONVERSION_NOTES.md` says, “Trials aligned to start of each 1-minute segment from session start,” and also notes that neural and behavioral streams were simultaneously acquired at 30 Hz.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native 30 Hz sampling, corresponding to 33.33 ms bins, with no rebinning.

ii. 
```python
FPS = 30  # recording frame rate
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. `CONVERSION_NOTES.md` explicitly states “Time bin size: 33.33 ms (1000/30)” and says the streams are “natively aligned at 30 Hz.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment input from the environment label `d['envs'][day, 0]`, not from the `blocked` variable. It uses that string label to select a hard-coded 3x3 geometry.

ii. 
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name)
```

iii. `CONVERSION_NOTES.md` says the input is “Derived from `get_env_mat()` function in reference code.” The trajectory shows the agent relied on the repo’s joblib-style `envs` representation rather than the raw `blocked` indices.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps the environment name to a hard-coded 3x3 binary matrix where 1 means open and 0 means blocked, then flattens that matrix into a 9-element vector and reuses the same vector for every trial in the session.

ii. 
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    if env in env_mats:
        return np.array(env_mats[env], dtype=float)
...
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
...
input_trials.append(env_input.astype(np.float32))
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as a “3x3 binary matrix representing open (1) vs blocked (0) partitions,” flattened in row-major order and kept static per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives the output from the per-day position array `d['position'][day]`.

ii. 
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. `CONVERSION_NOTES.md` explicitly describes the output source as “Continuous x-y position (0-75 cm).”

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips `x` and `y` to arena bounds, divides each axis into three equal 25 cm bins, converts each coordinate pair to a single class index, and stores the result as a time-varying categorical sequence.

ii. 
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS, arena_size=ARENA_SIZE):
    x = np.clip(position[0], 0, arena_size - 1e-10)
    y = np.clip(position[1], 0, arena_size - 1e-10)

    bin_size = arena_size / n_bins
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)

    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)

    bin_idx = x_bin * n_bins + y_bin
    return bin_idx
```

iii. `CONVERSION_NOTES.md` justifies this as matching the physical 3x3 partitioning of the 75 cm arena, with one bin index per time step.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into 9 categories by flooring each coordinate into one of three bins per axis. The categories are named `row0_col0` through `row2_col2`, and the code combines bins as `x_bin * 3 + y_bin`.

ii. 
```python
bin_size = arena_size / n_bins
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
...
bin_idx = x_bin * n_bins + y_bin
...
for r in range(N_SPATIAL_BINS):
    for c in range(N_SPATIAL_BINS):
        position_bin_names.append(f"row{r}_col{c}")
```

iii. `CONVERSION_NOTES.md` states that position is discretized into “3x3 = 9 spatial bins,” each 25 cm by 25 cm, and describes the categories as row-major bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes neural and position streams are already synchronized frame-by-frame at 30 Hz, then slices both with the same trial boundaries so each neural trial and position trial cover the same frames.

ii. 
```python
trial_traces = traces[:, start:end]  # (n_registered, FRAMES_PER_TRIAL)
trial_pos = pos_bins[start:end]  # (FRAMES_PER_TRIAL,)

neural_trials.append(trial_traces.astype(np.float32))
output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
```

iii. `CONVERSION_NOTES.md` explicitly says “All time series (neural, position) are natively aligned at 30 Hz recording rate” and references simultaneous DAQ acquisition.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing whole-cell traces are handled by removing all-NaN cells. Any remaining NaNs are replaced with 0. Incomplete trailing frames at the end of a session are discarded because only full 1-minute trials are kept. The code also contains session-level guards that skip days with fewer than 5 registered cells or fewer than 2 trials.

ii. 
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
...
if n_registered < 5:
    ...
    continue
...
traces = np.nan_to_num(traces, nan=0.0)
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
if n_trials < 2:
    ...
    continue
```

iii. The documented justification is partial: `CONVERSION_NOTES.md` explicitly defends using only non-NaN registered cells and dropping sub-minute remainders. It does not explicitly defend the remaining-NaN-to-zero replacement or the `<5`-cell skip.

## 6-a. What are the most time-consuming steps of the code?

i. The AI did not document this directly, but from the implementation the most time-consuming work is loading each large per-animal file with `joblib.load` and then iterating through every day and every trial to slice large arrays.

ii. 
```python
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    for day in range(n_days):
        ...
        for t in range(n_trials):
            start = t * FRAMES_PER_TRIAL
            end = (t + 1) * FRAMES_PER_TRIAL
```

iii. There is no explicit justification in `CONVERSION_NOTES.md`. This is inferred from the trajectory and the code structure.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop could have been vectorized or replaced with reshaping because it slices contiguous 1800-frame blocks from arrays that are already uniformly sampled. The name-building loops for inputs and outputs are also simple comprehensions that could be collapsed.

ii. 
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL

    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]

    neural_trials.append(trial_traces.astype(np.float32))
    input_trials.append(env_input.astype(np.float32))
    output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
...
for r in range(N_SPATIAL_BINS):
    for c in range(N_SPATIAL_BINS):
        position_bin_names.append(f"row{r}_col{c}")
```

iii. The AI did not discuss vectorization explicitly. This section is inferred from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts the same static environment vector to `float32` once per trial, and repeatedly computes start/end indices inside the trial loop. It also rebuilds the input/output label lists with loops every run.

ii. 
```python
input_trials.append(env_input.astype(np.float32))  # static per trial, shape (9,)
...
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
...
for r in range(N_SPATIAL_BINS):
    for c in range(N_SPATIAL_BINS):
        input_names_list.append(f"grid_{r}_{c}")
```

iii. There is no explicit justification for these repeated operations in the notes or trajectory.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script spends time on sanity-check printing and summary statistics that are not used by the downstream decoder, including unique-value checks, per-subject neuron summaries, and aggregate counts. It also stores extensive metadata fields that are not needed for decoder training itself.

ii. 
```python
sample_neural = data['neural'][0][0]
unique_vals = np.unique(sample_neural)
print(f"Sample neural unique values: {unique_vals}")
...
total_neurons = sum(data['neural'][s][0].shape[0] for s in range(n_sessions))
print(f"Total neuron-sessions: {total_neurons}")
...
for subj_id, subj_name in enumerate(data['subjects']):
    ...
    print(f"  {subj_name}: {len(session_indices)} sessions, neurons/session: "
          f"min={min(n_neurons_list)}, max={max(n_neurons_list)}, mean={np.mean(n_neurons_list):.0f}")
```

iii. The AI justified these as “Sanity Checks” in `CONVERSION_NOTES.md`, mainly to compare paper-level summary statistics and verify internal consistency, not because the decoder requires them.
