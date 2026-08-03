# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the `.mat` files. It hard-codes the seven animal IDs, then loads one `joblib` file per animal from `data/`. Each loaded file is a dictionary keyed by animal ID, and the per-animal arrays inside that dictionary are then used to build sessions and trials.

ii.
```python
DATA_DIR = "data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals):
    print(f"Processing {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. The trajectory shows that the AI inspected the unextended animal files with `joblib.load` and decided they directly exposed `envs`, `trace`, `position`, and `blocked` in convenient NumPy structures. The provided code README and example code also mention joblib-format animal files, which the AI appears to have taken as justification for loading those instead of the `.mat` files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the hard-coded `ANIMALS` list. The output `subjects` field is just a copy of that list, and `subject_idx` is assigned from the loop index over that list.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

subjects = animals.copy()

for a_idx, animal in enumerate(animals):
    ...
    subject_idx_all.append(a_idx)

data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array(subject_idx_all),
}
```

iii. In the trajectory, the AI verified that the dataset contained seven named animals and then treated those IDs as the canonical subject list. The notes also cite the seven named mice as a sanity check.

## 1-c. How are the data split into sessions?

i. Each recording day inside an animal file becomes one session. The AI reads `n_days = d['envs'].shape[0]` and iterates `for day in range(n_days)`, treating each day/environment exposure as a separate output session.

ii.
```python
n_days = d['envs'].shape[0]

for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]  # (n_cells, n_frames)
    position_day = d['position'][day]  # (2, n_frames)
    ...
    neural_all.append(session_neural)
    input_all.append(session_input)
    output_all.append(session_output)
```

iii. The script docstring says "Each day = one session = one environment geometry," and the trajectory explicitly states "Each session = 1 day = 1 environment."

## 1-d. How are the data split into trials?

i. Yes, according to the AI's chosen format. Within each day/session, trials are contiguous non-overlapping 1-minute chunks. The AI uses integer division by `TRIAL_FRAMES = 1800` and slices the session into `trial` windows of 1800 frames each.

ii.
```python
FPS = 30  # frames per second
TRIAL_DURATION_S = 60  # trial duration in seconds
TRIAL_FRAMES = FPS * TRIAL_DURATION_S  # 1800 frames per trial

n_trials = n_frames // TRIAL_FRAMES

for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES

    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
```

iii. The trajectory and conversion notes both say that each approximately 40-minute recording session is split into 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filter. All full 1-minute trial chunks are kept. The code does, however, skip entire sessions if they have zero registered cells or fewer than two full trials, and it drops any leftover frames that do not fit into a complete 1-minute trial.

ii.
```python
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue

n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue

for trial in range(n_trials):
    ...
```

iii. The notes say "All sessions included (no filtering by quality)," so the AI's intended QC story was basically "no trial filtering." The extra `n_registered == 0` and `n_trials < 2` guards appear to be defensive checks added for decoder-format compatibility rather than paper-derived curation rules.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the per-day slice of the `trace` array inside each joblib animal file.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_frames)

registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
```

iii. The notes state that the neural source is the `trace` field and describe it as binary calcium transient traces.

## 2-b. How is the `neural` data processed?

i. The AI filters out all-NaN cell rows, fills any remaining NaNs with zeros, then rebins each 1800-frame trial into 60 one-second bins by summing every 30 binary frames. It does not keep the native 30 Hz time series.

ii.
```python
def bin_neural_data(trace, time_bin_frames):
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    truncated = trace[:, :n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)

registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
trace_registered = np.nan_to_num(trace_registered, nan=0.0)

trial_trace = trace_registered[:, t_start:t_end]
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
```

iii. The trajectory explicitly shows the AI deciding that 1800 frames per trial was "quite dense" and that it should "consider downsampling to a more manageable time bin size like 1-second intervals to reduce dimensionality." The notes then justify the binning as summing 30 binary frames into one-second event counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered by whether the entire row is NaN on that day. Rows that are all NaN are removed as unregistered cells. After that, the AI also replaces any remaining NaNs with zeros.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]  # (n_registered, n_frames)
...
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. The notes justify the main filter as "Only cells registered (non-NaN) on each day are included per session." The extra `np.nan_to_num` step is justified only by the inline comment "shouldn't happen but be safe"; there is no paper-based justification for that step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus/event alignment step. The AI slices the continuous daily recording into consecutive 1-minute trials. In metadata, however, it labels the alignment event as the "Start of recording session" and records trial offsets from that.

ii.
```python
n_trials = n_frames // TRIAL_FRAMES
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    ...

'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
    ...
}
```

iii. The notes do not describe a real experimental alignment event. The only explicit justification is the metadata choice in the script; operationally, the AI treated trial start within a continuous recording as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the 30 Hz neural data into one-second bins. Each one-second bin is the sum of 30 original binary frames, and the output metadata reports a `time_bin_size` of 1000 ms.

ii.
```python
FPS = 30
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms

trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)

'metadata': {
    ...
    'time_bin_size': TIME_BIN_MS,
    ...
}
```

iii. The trajectory gives the direct rationale: the AI thought the native 1800 frames per trial were too dense and wanted "a more manageable time bin size like 1-second intervals." The notes repeat that this produced 60 time bins per trial.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` string label for each day, not from the `blocked` variable. It reads the daily environment name and passes it to `get_env_mat`.

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

env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)
```

iii. The trajectory shows the AI reading the reference `get_env_mat` helper and then deciding to use a "3x3 binary accessibility matrix" as decoder input. The notes explicitly say the environment geometry is "Derived from `get_env_mat()` function in reference code."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the daily environment name into a 3x3 binary matrix of accessible versus blocked tiles, flattens that matrix into a 9-element vector, and reuses the same static vector for every trial in the session.

ii.
```python
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
...
session_input.append(env_flat)  # (9,) static per trial
```

iii. The notes justify this as a representation of "accessible (1) vs blocked (0) partitions," flattened to 9 values and kept static per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position is derived from the per-day slice of the `position` array.

ii.
```python
position_day = d['position'][day]  # (2, n_frames)
...
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)
```

iii. The trajectory shows the AI inspecting `position` directly and noting that it contained x-y coordinates in centimeters spanning the 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first discretizes the framewise x-y coordinates into 3 x 3 spatial bins using the session-specific maximum x and y values, then rebins time by taking the modal spatial bin within each one-second window.

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

def bin_position(bin_indices, time_bin_frames):
    ...
    reshaped = truncated.reshape(n_bins, time_bin_frames)
    from scipy.stats import mode
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
    return binned
```

iii. The notes describe the intended logic as "floor(position / (max + epsilon) * 3)" followed by taking the mode over 30 frames per one-second bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each x and y coordinate is thresholded into three bins using equal divisions of the observed session maximum, not fixed 0-25-50-75 cm edges. The code then clips the bin indices to 0-2 and combines them as `x_bin * 3 + y_bin`. The inline comment says "row * n_cols + col," but the actual code uses x first, then y.

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

iii. The notes say the arena is split into approximately 25 cm bins and describe the combined index as "row * 3 + col," but the code itself implements `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns position and neural data by using the same trial boundaries and then the same one-second temporal bins within each trial. Neural activity is summed over each 30-frame window, while position is summarized by the modal category over that same 30-frame window.

ii.
```python
trial_trace = trace_registered[:, t_start:t_end]  # (n_registered, 1800)
trial_pos_bins = pos_bins[t_start:t_end]  # (1800,)

trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
trial_output = trial_output.reshape(1, -1)  # (1, 60)
```

iii. The notes explicitly justify the output as "Time-varying: mode of 30 frames per 1-second time bin," which implies alignment to the same 30-frame windows used for the neural sums.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness mainly by removing all-NaN cell rows, converting any remaining NaNs to zeros, skipping sessions with no usable cells, and silently dropping remainder frames that do not fill a complete 1-minute trial or 1-second bin.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]

if n_registered == 0:
    ...

trace_registered = np.nan_to_num(trace_registered, nan=0.0)

n_trials = n_frames // TRIAL_FRAMES

truncated = trace[:, :n_bins * time_bin_frames]
...
truncated = bin_indices[:n_bins * time_bin_frames]
```

iii. The notes justify the registered-cell filter and dropped remainder frames. The remaining-NaN-to-zero behavior is only justified by the code comment "shouldn't happen but be safe," not by the paper or reference code.

## 6-a. What are the most time-consuming steps of the code?

i. The heaviest work is likely loading each large animal file from disk and then repeatedly rebinding neural and position data for every trial. The most expensive added processing step is the per-trial `scipy.stats.mode` call used in `bin_position`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))

for trial in range(n_trials):
    ...
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)

def bin_position(bin_indices, time_bin_frames):
    ...
    from scipy.stats import mode
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
```

iii. There is no explicit efficiency justification in the notes. The only related justification in the trajectory is the AI's desire to reduce dimensionality by downsampling to one-second bins, which in turn created this extra processing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner `for trial in range(n_trials)` loop could have been replaced by session-level reshaping and batching for both neural and position data. The two small loops that build `position_labels` and `input_labels` could also be compacted, though they are not performance-critical.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
    session_neural.append(trial_neural)
    session_input.append(env_flat)
    session_output.append(trial_output)

for row in range(N_SPATIAL_BINS):
    for col in range(N_SPATIAL_BINS):
        position_labels.append(f"row{row}_col{col}")
```

iii. The AI did not justify these loops. They appear to be straightforward implementation choices rather than deliberate performance decisions.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same truncate/reshape/bin logic for every trial, repeats importing `mode` inside `bin_position`, and repeatedly appends the same static environment vector once per trial instead of broadcasting or reusing a shared immutable object at session construction time.

ii.
```python
def bin_position(bin_indices, time_bin_frames):
    ...
    from scipy.stats import mode
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
    return binned

for trial in range(n_trials):
    ...
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
    session_input.append(env_flat)
```

iii. There is no explicit justification for these repeated operations. The trajectory suggests the AI prioritized getting a manageable rebinned dataset over keeping the processing minimal.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest unnecessary processing is the one-second aggregation itself: it collapses 30 native frames into one neural count and one modal position label, discarding within-second timing and within-second path information. It also computes bookkeeping metadata such as `total_unique_neurons`, `total_sessions`, and `total_trials` that are not used by the downstream decoder.

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms

trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)

'metadata': {
    ...
    'total_unique_neurons': total_neurons_unique,
    'total_sessions': total_sessions,
    'total_trials': total_trials,
}
```

iii. The trajectory explicitly justifies the temporal aggregation as a convenience choice to reduce dimensionality, not as a reference-matching requirement. There is no explicit downstream justification for the extra summary metadata.
