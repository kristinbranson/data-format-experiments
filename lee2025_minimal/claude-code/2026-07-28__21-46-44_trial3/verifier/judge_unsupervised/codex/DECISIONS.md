# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads one preprocessed joblib file per hard-coded animal ID from `/app/data`. It reads the nested dict entry for that animal, then uses the `trace` array shape to determine the number of days/sessions and frames available for later trial splitting.

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
    n_cells_total = d['trace'].shape[1]
    n_timepoints = d['trace'].shape[2]
```

iii. In the trajectory, the agent noted that the data directory contained both `.mat` files and extensionless joblib files, and it chose the joblib files as the convenient Python-native representation. This matches its own summary in `CONVERSION_NOTES.md`, which treats each animal file as the source dataset.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. The output `subjects` field is just `list(animals)`, and each kept session appends the corresponding integer subject index.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

subjects = list(animals)
subject_idx_list = []

...

subject_idx_list.append(animal_idx)
```

iii. In the trajectory, the agent explicitly adopted the 7 animal IDs it found in the dataset directory. `CONVERSION_NOTES.md` repeats the same 7-animal list as the subject set.

## 1-c. How are the data split into sessions?

i. Each recording day is treated as one session. The code iterates over the first dimension of `trace` and pulls day-matched `position` and `envs` entries.

ii.
```python
n_days = d['trace'].shape[0]

for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. The agent stated in the trajectory that “Sessions = days,” and `CONVERSION_NOTES.md` says “Each session = one recording day.”

## 1-d. How are the data split into trials?

i. Within each session/day, the recording is divided into non-overlapping 1-minute trials at 30 Hz, i.e. 1800 frames per trial. Any leftover frames that do not fill a full minute are discarded.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial

...

n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL

    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. This comes directly from the user instructions, and the agent repeated the same rule in both the trajectory and `CONVERSION_NOTES.md` (“Sessions split into 1-minute trials”).

## 1-e. How are trials filtered based on quality controls?

i. There is no real per-trial quality-control filter. Instead, the code drops trailing incomplete trial fragments by integer division, skips whole sessions with fewer than 5 registered neurons, and skips whole sessions with fewer than 2 complete trials.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()

if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue

...

n_trials = n_timepoints // FRAMES_PER_TRIAL
if n_trials < 2:
    print(f"  Day {day}: skipping, only {n_trials} possible trials")
    continue
```

iii. The trajectory does not show a paper-derived rationale for the `<5 neurons` or `<2 trials` thresholds; they appear to be ad hoc safety checks. `CONVERSION_NOTES.md` documents only that remainder frames are discarded and that the 2-trial minimum is satisfied in practice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` array for each animal/day.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The notes and trajectory both describe `trace` as the binary calcium transient representation used for neural activity.

## 2-b. How is the `neural` data processed?

i. The code keeps only registered cells for the session, replaces any remaining NaNs with 0, slices the retained trace matrix into 1-minute trials, and stores each trial as `float32`.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]  # (n_registered, n_timepoints)
traces = np.nan_to_num(traces, nan=0.0)

...

trial_traces = traces[:, start:end]
neural_trials.append(trial_traces.astype(np.float32))
```

iii. In the trajectory, the agent decided to use “binary calcium traces, only registered cells (non-NaN) per session.” `CONVERSION_NOTES.md` likewise says the neural data are binary transient events and that only registered cells are included.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural filtering is removal of cells whose entire day trace is NaN, i.e. cells not registered on that day. No place-cell, split-half reliability, running-speed, or per-trial neural filter is applied during conversion.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()

if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue

traces = trace_day[registered]
```

iii. The agent’s notes justify this by saying only registered cells are kept and that “No additional filtering applied.” In the trajectory, it explicitly chose “only registered cells (non-NaN) per session.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-triggered alignment. Neural trials are aligned to the start of each artificial 1-minute segment measured from session start, and the metadata names the session start as the alignment event.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL

    trial_traces = traces[:, start:end]

...

'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_S,
    ...
}
```

iii. `CONVERSION_NOTES.md` says “Trials aligned to start of each 1-minute segment from session start,” and the trajectory describes the recordings as continuous 40-minute sessions rather than event-locked trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native 30 Hz sampling rate, corresponding to 33.33 ms bins. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30  # recording frame rate

...

'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
'recording_fps': FPS,
```

iii. The agent relied on the methods text stating that behavioral and imaging streams were acquired at 30 Hz, and repeated in the trajectory/notes that it was keeping the native 30 Hz resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input environment geometry is derived from the per-day `envs` string label, not from the raw `blocked` field.

ii.
```python
env_name = str(d['envs'][day, 0])

...

env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
```

iii. The trajectory shows that the agent read the reference `get_env_mat()` helper from the paper code and then chose to use environment-name lookup to build the geometry input. `CONVERSION_NOTES.md` says the input is “Derived from `get_env_mat()` function in reference code.”

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a hard-coded 3x3 binary open/blocked template, converted to a NumPy array, flattened to length 9, cast to `float32`, and duplicated for every trial in the session as a static input vector.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        'u':         [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+':         [[0,1,0],[1,1,1],[0,1,0]],
        'i':         [[1,1,1],[0,1,0],[1,1,1]],
        'l':         [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn':     [[1,1,0],[1,1,1],[0,1,1]],
    }
    ...

env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()

...

input_trials.append(env_input.astype(np.float32))
```

iii. The agent justified this by pointing to the reference `get_env_mat()` helper and the task requirement that environment geometry be static per trial. That is reflected in both the trajectory and `CONVERSION_NOTES.md`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output mouse position is derived from the raw per-day `position` array.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The notes and trajectory both describe `position` as the animal’s x-y coordinates in the 75 cm square arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code clips x and y coordinates to the `[0, 75)` arena range, bins each coordinate into 3 equal-width bins, combines the two 1D bins into one categorical index, and then slices that categorical time series into 1-minute trials.

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

...

pos_bins = discretize_position(pos_day)
```

iii. In the trajectory, the agent decided that the decoder output should be mouse position discretized into 3x3 bins, and `CONVERSION_NOTES.md` says each axis is divided into 25 cm bins to match the 3x3 environment partitioning.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Category thresholds are 0-25 cm, 25-50 cm, and 50-75 cm along each axis. The code uses `floor(coord / 25)` after clipping, then combines the two axis bins into 9 classes.

ii.
```python
bin_size = arena_size / n_bins
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)

...

bin_idx = x_bin * n_bins + y_bin
```

iii. The agent’s notes explicitly justify the 25 cm thresholds by the 75 cm arena size and the 3x3 partition design described in the paper/task.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned to neural data frame-for-frame by using the same per-session frame indices and the same trial boundaries. The code slices `traces` and `pos_bins` with identical `[start:end]` ranges.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL

    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]

    neural_trials.append(trial_traces.astype(np.float32))
    output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
```

iii. The methods excerpt says the DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz, and `CONVERSION_NOTES.md` states that neural and position time series are natively aligned at 30 Hz.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing whole-day cell traces are treated as unregistered cells and dropped. Any remaining NaNs inside retained traces are replaced with 0 as a safety measure. Partial trailing session fragments shorter than 1 minute are discarded by the integer trial split.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)

...

n_trials = n_timepoints // FRAMES_PER_TRIAL
```

iii. The trajectory shows the agent was concerned about NaN traces from cells that were not registered on some days. `CONVERSION_NOTES.md` documents registered-cell filtering; the `nan_to_num` step appears to be the agent’s own defensive handling rather than a paper-derived rule.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the very large per-animal joblib datasets from disk, iterating through every day/trial to materialize session/trial lists, and writing the huge final pickle. The actual arithmetic is minimal compared with I/O and data copying.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))

...

for day in range(n_days):
    ...
    for t in range(n_trials):
        ...
        neural_trials.append(trial_traces.astype(np.float32))
        input_trials.append(env_input.astype(np.float32))
        output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))

...

with open(output_path, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. This is not explicitly discussed in the notes, but it follows from the code structure and the reported multi-gigabyte output files.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial-construction loop could have been replaced by reshaping/splitting the already contiguous arrays into trial blocks. The repeated per-trial append/cast path for static inputs is also easily vectorizable.

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
```

iii. The code does not justify these loops; this is an efficiency evaluation of what the agent wrote.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts and appends the same static environment vector once per trial, even though it does not change within a session. It also repeatedly computes slice boundaries in Python for every trial.

ii.
```python
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()

...

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    ...
    input_trials.append(env_input.astype(np.float32))
```

iii. This repetition is implicit in the implementation; there is no separate justification in the notes beyond the choice to make the input static per trial.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion script performs diagnostic work that is not part of the saved decoder dataset: console summary statistics, `sanity_checks`, `unique`/count summaries, and metadata counters used only for reporting. It also computes `env_input.astype(np.float32)` repeatedly per trial rather than once per session.

ii.
```python
total_sessions = 0
total_trials = 0
total_neurons_per_session = []

...

print(f"\n=== Conversion Summary ===")
print(f"Animals: {len(animals)}")
print(f"Sessions: {total_sessions}")
print(f"Total trials: {total_trials}")

...

def sanity_checks(data):
    ...
    sample_neural = data['neural'][0][0]
    unique_vals = np.unique(sample_neural)
    print(f"Sample neural unique values: {unique_vals}")
```

iii. The notes emphasize these sanity checks as validation, but they are not used by downstream analyses once the pickle is written.
