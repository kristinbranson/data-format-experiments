# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not load the provided `.mat` files. Instead, it assumes there are per-animal serialized files in `data/` whose names exactly match a hard-coded animal list, and it loads each one with `joblib.load`. Within each loaded object it expects a top-level animal key containing arrays such as `trace`, `position`, and `envs`.

ii.
```python
DATA_DIR = "data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for a_idx, animal in enumerate(animals):
    print(f"Processing {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. In trajectory step 31 the agent inspected one of the joblib files and found fields `['SFPs', 'blocked', 'centroids', 'envs', 'maps', 'position', 'trace']`. In step 36 it justified the loading strategy by saying the data structure was per animal with `trace (n_days, n_cells, n_frames)`, `position (n_days, 2, n_frames)`, and `envs (n_days, 1)`.

## 1-b. How are the data split into subjects?

i. Subjects are the entries of the hard-coded `ANIMALS` list. The output `subjects` field is just a copy of that list, and sessions inherit the subject index from the outer animal loop.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = animals.copy()
...
for a_idx, animal in enumerate(animals):
    ...
    subject_idx_all.append(a_idx)
```

iii. The trajectory does not contain a separate explicit justification beyond the file inspection. The agent treated each named animal file as one mouse and carried that assumption into the output.

## 1-c. How are the data split into sessions?

i. Each day within an animal file is treated as one session. The agent reads the number of days from `d['envs'].shape[0]` and loops over `day` to build one output session per day.

ii.
```python
n_days = d['envs'].shape[0]
...
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]  # (n_cells, n_frames)
    position_day = d['position'][day]  # (2, n_frames)
    ...
    neural_all.append(session_neural)
    input_all.append(session_input)
    output_all.append(session_output)
```

iii. In trajectory step 36 the agent explicitly stated: “Each animal has multiple recording sessions (days)” and “Each session = 1 day = 1 environment.”

## 1-d. How are the data split into trials?

i. Within each session/day, trials are non-overlapping 60-second chunks. The code computes how many full one-minute chunks fit into `n_frames`, then loops over those trial windows and drops any remainder.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
TRIAL_FRAMES = FPS * TRIAL_DURATION_S  # 1800 frames per trial
...
n_trials = n_frames // TRIAL_FRAMES
...
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES

    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
```

iii. In trajectory step 36 the agent justified this by saying each “~40 minute session breaks into ~40 one-minute trials,” and by using the recorded 30 Hz frame rate to convert 1 minute into 1800 frames.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality-control filter. The agent only drops partial trailing chunks implicitly via integer division, and it skips entire sessions if fewer than 2 full trials exist.

ii.
```python
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
...
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
```

iii. The trajectory does not show a separate QC rationale for trials. The closest justification is that the agent was trying to satisfy the decoder-format requirement that a session contain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `d['trace'][day]`, which the agent interprets as binary rise-extracted calcium transient traces arranged as `(n_cells, n_frames)` for each session/day.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_frames)
```

iii. In trajectory steps 33 and 35 the agent checked the unique values of `trace` and saw `[0., 1.]`. In step 36 it therefore justified the decision as “Trace is binary (0/1) - rise-extracted calcium transients.”

## 2-b. How is the `neural` data processed?

i. The agent keeps the per-day traces in cell-by-frame layout, removes unregistered cells, replaces any remaining NaNs with zero, and then rebins the neural data from 30 Hz frames into 1-second bins by summing 30 consecutive binary frames.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]  # (n_registered, n_frames)
...
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
...
def bin_neural_data(trace, time_bin_frames):
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    truncated = trace[:, :n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)
...
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
```

iii. In trajectory step 36 the agent argued that the 30 Hz signal was “quite dense” and that it should “consider downsampling to a more manageable time bin size like 1-second intervals to reduce dimensionality,” producing 60 timepoints per trial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are kept only if they are not all-NaN within that day/session. After that, any remaining NaNs are zero-filled “to be safe.”

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]  # (n_registered, n_frames)
...
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
...
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. In trajectory step 36 the agent said it would “only include registered cells that don’t have NaN values in each session.” The zero-fill step is not separately justified in the trajectory; it appears only in the code comment “shouldn’t happen but be safe.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not align neural activity to a behavioral event. It slices continuous recordings into contiguous minute-long windows. However, the metadata describes the alignment event as `"Start of recording session"` with `off_start = 0.0` and `off_end = 60.0`.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
...
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
```

iii. The trajectory does not give a full alignment rationale. Step 36 frames the data as continuous sessions split into one-minute trials, and the later metadata labels those trials as aligned to session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted trials are at 1-second resolution. The code rebins the original 30 Hz data by grouping 30 frames at a time, summing neural events within each second, and taking one output label per second.

ii.
```python
FPS = 30
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
...
'time_bin_size': TIME_BIN_MS,
```

iii. In trajectory step 36 the agent explicitly justified 1-second bins as a way to reduce dimensionality while keeping a usable temporal structure for decoding.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the per-session environment label `d['envs'][day, 0]`, not from the raw `blocked` variable.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
```

iii. In trajectory step 36 the agent said it would use “environment geometry as a static 3x3 binary matrix per trial,” based on the session’s environment identity.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment string is converted by `get_env_mat` into a hand-coded 3x3 binary accessibility matrix, then flattened to a 9-element vector and copied into every trial in the session.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    if env in envs:
        return np.array(envs[env], dtype=float)
...
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
...
session_input.append(env_flat)  # (9,) static per trial
```

iii. In trajectory step 36 the agent justified this as representing “environment geometry as a static 3x3 binary matrix per trial,” and in the file docstring it claimed that this matched the reference helper `get_env_mat`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output position is derived from `d['position'][day]`, interpreted as 2D `x, y` position by frame.

ii.
```python
position_day = d['position'][day]  # (2, n_frames)
```

iii. In trajectory step 36 the agent justified this with the observation that position was stored as `x-y coordinates in cm (0-75 range)`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first discretizes each frame’s continuous `(x, y)` coordinates into one of 9 bins using per-session maxima rather than fixed arena edges. It then rebins each one-minute trial into 1-second windows and uses the modal class within each second as the output label.

ii.
```python
def discretize_position(position, n_bins=3):
    x = position[0]
    y = position[1]

    x_max = np.nanmax(x) + POSITION_BUFFER
    y_max = np.nanmax(y) + POSITION_BUFFER

    x_bin = np.floor(x / (x_max / n_bins)).astype(int)
    y_bin = np.floor(y / (y_max / n_bins)).astype(int)
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx
...
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
trial_output = trial_output.reshape(1, -1)  # (1, 60)
```

iii. In trajectory step 36 the agent justified the coarse 3x3 discretization by saying it wanted to “predict discretized mouse position into 9 spatial bins,” and paired that with the 1-second downsampling choice.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into three equal-width bins using the observed session-specific maximum coordinate. The final class index is `x_bin * 3 + y_bin`, yielding labels `0` through `8`.

ii.
```python
x_max = np.nanmax(x) + POSITION_BUFFER
y_max = np.nanmax(y) + POSITION_BUFFER

x_bin = np.floor(x / (x_max / n_bins)).astype(int)
y_bin = np.floor(y / (y_max / n_bins)).astype(int)
x_bin = np.clip(x_bin, 0, n_bins - 1)
y_bin = np.clip(y_bin, 0, n_bins - 1)

bin_idx = x_bin * n_bins + y_bin
```

iii. The trajectory does not separately defend the thresholding formula. The only explicit rationale is the step 36 statement that the 75 cm arena would be divided into a 3x3 grid.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by slicing the same frame ranges for each one-minute trial. After that, both streams are rebinned into 1-second windows using the same 30-frame bin size, with neural data summed and position reduced by mode.

ii.
```python
trial_trace = trace_registered[:, t_start:t_end]  # (n_registered, 1800)
trial_pos_bins = pos_bins[t_start:t_end]  # (1800,)

trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
trial_output = trial_output.reshape(1, -1)  # (1, 60)
```

iii. The trajectory does not spell this out, but step 36 couples the 1-minute trial split with the 1-second binning for both neural and positional data, implying shared windows for alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent removes all-NaN cells, skips sessions with zero registered cells, zero-fills any remaining NaNs in neural traces, drops incomplete trailing frames by integer division, and skips sessions with fewer than two full trials.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
n_registered = registered.sum()

if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue

trace_registered = np.nan_to_num(trace_registered, nan=0.0)
...
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The trajectory explicitly justifies only the registered-cell filter (step 36). The remaining safeguards are code-level defensive choices rather than separately argued decisions.

## 6-a. What are the most time-consuming steps of the code?

i. By code inspection, the dominant work is loading each per-animal file and then looping through every day and every 1-minute trial to rebin neural data and compute per-second positional modes. The repeated `scipy.stats.mode` call inside `bin_position` is especially expensive relative to the very simple reference pipeline.

ii.
```python
for a_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    for day in range(n_days):
        ...
        for trial in range(n_trials):
            ...
            trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
            trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. The trajectory does not explicitly analyze runtime. The closest hint is step 36, where the agent chose 1-second bins to reduce dimensionality, implicitly accepting extra preprocessing work.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop could have been vectorized by reshaping whole-session neural and position arrays once, instead of slicing and reprocessing every trial separately. `bin_position` also recomputes the mode trial-by-trial rather than operating on a larger batched array.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES

    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]

    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. There is no explicit trajectory justification for leaving these loops unvectorized.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly slices per-trial windows and repeatedly calls `bin_neural_data` and `bin_position` for each trial, even though both could be done once at the session level. It also appends the same static `env_flat` vector separately for every trial in a session.

ii.
```python
for trial in range(n_trials):
    ...
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)

    session_neural.append(trial_neural)
    session_input.append(env_flat)  # (9,) static per trial
    session_output.append(trial_output)
```

iii. The trajectory does not explicitly justify this repeated processing; it follows from the agent’s chosen trial-by-trial implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded detail is the original 30 Hz temporal structure: the agent sums 30 frames of neural activity into one value and collapses 30 frames of position into one modal class, so 29/30 of the framewise timing detail is thrown away. It also computes and stores metadata aggregates such as `total_unique_neurons`, `total_sessions`, and `total_trials` that are not used by the decoder.

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
...
return reshaped.sum(axis=2).astype(float)
...
from scipy.stats import mode
binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
...
'metadata': {
    ...
    'total_unique_neurons': total_neurons_unique,
    'total_sessions': total_sessions,
    'total_trials': total_trials,
}
```

iii. The trajectory explicitly justified the temporal collapse only as a dimensionality reduction choice (step 36). It did not justify the extra metadata as necessary for downstream use.
