# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, then loads one joblib file per animal from `data/<animal>`. For each animal file it reads the nested dictionary `dat[animal]`, then uses `envs`, `trace`, and `position` as the core streams. Trials are not loaded separately from disk; they are created later by slicing each day's time series into 1-minute chunks.

ii. ```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals):
    print(f"Processing {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]

    n_days = d['envs'].shape[0]
    n_cells_total = d['trace'].shape[1]
    n_frames = d['trace'].shape[2]
```

iii. In the trajectory, the agent summarized the raw structure as `trace (n_days, n_cells, n_frames), position (n_days, 2, n_frames), envs (n_days, 1)` and decided to treat those as the primary inputs. `CONVERSION_NOTES.md` also states that the dataset contains 7 mice, 207 sessions, and 10 geometries, which the script aims to reproduce.

## 1-b. How are the data split into subjects?

i. The agent splits data by subject using the outer loop over the hard-coded `ANIMALS` list. It copies that list directly into `subjects` and appends the current animal index to `subject_idx_all` once per session.

ii. ```python
subjects = animals.copy()
subject_idx_all = []

for a_idx, animal in enumerate(animals):
    ...
    for day in range(n_days):
        ...
        subject_idx_all.append(a_idx)
```

iii. The trajectory says "Each animal has multiple recording sessions (days)" and the notes list the seven mouse IDs explicitly. There is no more elaborate subject parsing; the agent assumes one file per mouse and one fixed ordering of mice.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day as one session. It iterates over `range(n_days)` where `n_days = d['envs'].shape[0]`, and for each `day` it creates one session entry in `neural`, `input`, and `output`.

ii. ```python
n_days = d['envs'].shape[0]

for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]
    position_day = d['position'][day]
    ...
    neural_all.append(session_neural)
    input_all.append(session_input)
    output_all.append(session_output)
```

iii. The trajectory explicitly states "Each session = 1 day = 1 environment," and `CONVERSION_NOTES.md` repeats "Each day for each animal = one session." This mirrors the paper/methods statement that one 40-minute session was recorded per day.

## 1-d. How are the data split into trials?

i. After choosing one day/session, the agent divides the full time series into contiguous 1-minute trials. It assumes 30 Hz sampling, defines a trial as 1800 frames, computes `n_trials = n_frames // 1800`, and drops any remainder frames at the end of the session.

ii. ```python
FPS = 30
TRIAL_DURATION_S = 60
TRIAL_FRAMES = FPS * TRIAL_DURATION_S

n_trials = n_frames // TRIAL_FRAMES

for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES

    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
```

iii. The trajectory says the decoder task involves long sessions that "break into ~40 one-minute trials." `CONVERSION_NOTES.md` gives the per-animal frame counts and explicitly documents that leftover frames are dropped when they do not make a full 60-second trial.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no trial-level quality control. The agent only skips a session if it would yield fewer than two 1-minute trials. Otherwise all derived trials from that session are kept, with no filtering for running speed, occupancy, behavior quality, or label validity.

ii. ```python
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The justification in `CONVERSION_NOTES.md` says "All sessions included (no filtering by quality)." The only explicit rationale comes from the task requirement that there must be at least two trials per session for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data comes directly from the raw dataset field `trace`, after selecting one day and one subset of registered cells. The agent does not derive neural activity from movies, calcium traces, or maps; it trusts the stored `trace` field as the preprocessed neural signal.

ii. ```python
trace_day = d['trace'][day]  # (n_cells, n_frames)
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
```

iii. `CONVERSION_NOTES.md` says the source is "Binary calcium transient traces (`trace` field)." The trajectory also records the agent's summary that the trace is already binary `0/1` rise-extracted calcium activity.

## 2-b. How is the `neural` data processed?

i. The agent treats the binary `trace` values as event indicators, splits them into 1-minute trials, and then rebins each trial into 1-second bins by summing 30 frames per bin. The output `neural` matrix for each trial is therefore event counts per second, with shape `(n_registered_neurons, 60)`.

ii. ```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second

def bin_neural_data(trace, time_bin_frames):
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    truncated = trace[:, :n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)

trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
```

iii. The trajectory shows this was an explicit invention by the agent: it said 30 Hz was "quite dense" and chose to "downsampl[e] to a more manageable time bin size like 1-second intervals." `CONVERSION_NOTES.md` repeats that choice as "Summed over 30 frames = 1-second bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural filtering is limited to dropping cells that are all-NaN on a given day, which the agent interprets as unregistered cells. After that, any remaining NaNs are replaced with zero. The agent does not apply movement-based filtering, activity thresholds, split-half reliability filters, or place-cell selection.

ii. ```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
...
if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue

trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. `CONVERSION_NOTES.md` justifies this as "Only cells registered (non-NaN) on each day are included per session" and "No additional filtering ... applied." The trajectory similarly says it will "only include registered cells that don't have NaN values in each session."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align neural activity to a behavioral event such as reward, stimulus onset, or movement onset. Instead it makes contiguous 60-second trial windows and indexes neural data relative to each window. In metadata it describes the alignment event as "Start of recording session," even though later trials are actually indexed from their own `t_start:t_end` slice.

ii. ```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    ...

'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
}
```

iii. The justification is implicit rather than explicit: the trajectory frames the task as turning long sessions into 1-minute trials, not as event-locked analysis. The notes do not add a stronger justification beyond describing 1-minute trial structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 1-second bins, created by summing 30 original frames recorded at 30 Hz. Yes, the script performs temporal rebinning on both neural and output position data.

ii. ```python
FPS = 30
TIME_BIN_FRAMES = FPS
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms

'metadata': {
    'time_bin_size': TIME_BIN_MS,
    ...
}
```

iii. The trajectory explicitly says the agent chose 1-second bins to make the data "more manageable." `CONVERSION_NOTES.md` and `README.md` both present 1-second bins as a deliberate design choice.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The agent derives environment geometry from the per-day string label in `envs`, not from the raw `blocked` field. It converts the string name for each day into a 3x3 geometry matrix using a copied version of `get_env_mat()`.

ii. ```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
```

iii. The notes say the geometry input is "Derived from `get_env_mat()` function in reference code." The trajectory also shows the agent reading the reference `get_env_mat` implementation from `utils.py` and deciding to reuse that mapping.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent maps each environment name to a hard-coded 3x3 binary matrix with `1` for accessible bins and `0` for blocked bins, then flattens that matrix into a length-9 static vector. The same vector is repeated for every trial from the same session.

ii. ```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(envs[env], dtype=float)

env_flat = env_mat.flatten()
...
session_input.append(env_flat)  # (9,) static per trial
```

iii. `CONVERSION_NOTES.md` says the input is a "3x3 binary matrix representing accessible (1) vs blocked (0) partitions" and that it is flattened to 9 values. The trajectory specifically mentions using a flattened 3x3 binary accessibility matrix for the decoder input.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The decoded output comes directly from the raw `position` field for each day. The script uses the two recorded coordinates for that day and converts them into discrete spatial-bin labels.

ii. ```python
position_day = d['position'][day]  # (2, n_frames)
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)
```

iii. `CONVERSION_NOTES.md` says "Position (x, y) discretized into 3x3 = 9 spatial bins." The methods excerpt and repository README identify `position` as the tracked head position stream.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first discretizes each frame's x and y coordinates into 3 bins per dimension using the per-session maxima of x and y separately. It then combines the per-axis bins into a single 0-8 category and reduces each 30-frame second to the modal spatial bin.

ii. ```python
def discretize_position(position, n_bins=3):
    x = position[0]
    y = position[1]

    x_max = np.nanmax(x) + POSITION_BUFFER
    y_max = np.nanmax(y) + POSITION_BUFFER

    x_bin = np.floor(x / (x_max / n_bins)).astype(int)
    y_bin = np.floor(y / (y_max / n_bins)).astype(int)
    ...
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx

trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
trial_output = trial_output.reshape(1, -1)
```

iii. The trajectory states that the agent was "dividing the 75×75 cm arena into a 3×3 grid" and planned to output one categorical variable for the 9 bins. The notes then justify the details as `floor(position / (max + epsilon) * 3)` followed by per-second mode pooling.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Category thresholds are created by dividing the observed x range of that session into three equal-width x bins and the observed y range into three equal-width y bins. After clipping to `[0, 2]`, the script combines them as `x_bin * 3 + y_bin` to produce 9 categories labeled `row0_col0` through `row2_col2`.

ii. ```python
x_bin = np.floor(x / (x_max / n_bins)).astype(int)
y_bin = np.floor(y / (y_max / n_bins)).astype(int)
x_bin = np.clip(x_bin, 0, n_bins - 1)
y_bin = np.clip(y_bin, 0, n_bins - 1)
bin_idx = x_bin * n_bins + y_bin

position_labels = []
for row in range(N_SPATIAL_BINS):
    for col in range(N_SPATIAL_BINS):
        position_labels.append(f"row{row}_col{col}")
```

iii. `CONVERSION_NOTES.md` explicitly describes the thresholding as "floor(position / (max + epsilon) * 3)" and category combination as "row * 3 + col (0-8)." The trajectory's justification is that the task requested 3x3 spatial bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned by using the same frame slices for each trial and the same 30-frame binning boundaries within each trial. Neural data are summed within each second, while position is reduced to the modal category for those same 30 frames.

ii. ```python
trial_trace = trace_registered[:, t_start:t_end]  # (n_registered, 1800)
trial_pos_bins = pos_bins[t_start:t_end]  # (1800,)

trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
```

iii. The methods excerpt says behavioral and cellular imaging streams were acquired simultaneously and timestamped for post-hoc alignment. The agent's justification is implicit: it bins both streams from the same frame windows, which preserves synchrony in the converted format.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing neural registration by dropping all-NaN cells on a given day. It then zero-fills any remaining NaNs in the retained neural matrix "to be safe." It also drops leftover frames at the end of a session that do not fill a complete 1-minute trial, and raises an error on unknown environment names. It does not implement special handling for missing position samples.

ii. ```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]

if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue

trace_registered = np.nan_to_num(trace_registered, nan=0.0)
...
n_trials = n_frames // TRIAL_FRAMES
...
else:
    raise ValueError(f"Unknown environment: {env}")
```

iii. `CONVERSION_NOTES.md` justifies dropping non-registered cells and says the replacement of residual NaNs is only a safety measure. The trajectory also shows the agent thinking about "how to handle cells that weren't recorded during a trial," which led to this design.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading each large animal joblib file and then iterating through all animals, all days, and all trials to rebin neural and position data. Within those loops, the repeated per-trial reshape/sum and especially `scipy.stats.mode` for every 1-second position bin are likely the dominant processing costs.

ii. ```python
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

iii. The agent does not discuss runtime in detail in its notes, but the trajectory shows it was concerned about decoder dimensionality and therefore introduced 1-second binning. That concern implies it expected trialwise processing volume to matter.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop could have been vectorized by reshaping a whole day at once into `(n_trials, ...)` arrays and binning all trials together. The nested loops that build `position_labels` and `input_labels` are also unnecessary Python loops, though minor. The biggest avoidable scalar work is repeated trial slicing and repeated calls to `mode`.

ii. ```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)

for row in range(N_SPATIAL_BINS):
    for col in range(N_SPATIAL_BINS):
        position_labels.append(f"row{row}_col{col}")
```

iii. No explicit justification is given in the notes because the agent prioritized a simple implementation. The trajectory suggests the main design priority was manageability and passing validation, not maximizing efficiency.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly appends the same static environment vector once per trial within a session, even though it is constant for the whole day. It also repeatedly recomputes per-trial neural binning and per-trial position mode pooling using the same trial length and same bin size across the entire dataset.

ii. ```python
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()
...
for trial in range(n_trials):
    ...
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
    ...
    session_input.append(env_flat)  # (9,) static per trial
```

iii. `CONVERSION_NOTES.md` frames the geometry as "static per trial," which is why the same session-level value is duplicated across all trial entries. The trajectory gives no alternative plan for sharing or caching repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script keeps metadata counters (`total_unique_neurons`, `total_sessions`, `total_trials`) and prints progress summaries that are not needed for the downstream decoder itself. It also imports unused modules such as `sys` and `deepcopy`. In data processing terms, the major extra step is the invented 1-second rebinning and per-second position mode reduction, which permanently discards within-second structure before downstream analysis.

ii. ```python
import sys
from copy import deepcopy
...
total_sessions = 0
total_trials = 0
total_neurons_unique = 0
...
print(f"\nConversion complete:")
print(f"  Total sessions: {total_sessions}")
print(f"  Total trials: {total_trials}")
print(f"  Total unique neurons: {total_neurons_unique}")
```

iii. The notes justify the 1-second aggregation as a pragmatic simplification for the decoder rather than as something taken from the reference code. The trajectory makes that explicit by saying the original 30 Hz data felt "quite dense" and that the agent wanted a "more manageable" representation.
