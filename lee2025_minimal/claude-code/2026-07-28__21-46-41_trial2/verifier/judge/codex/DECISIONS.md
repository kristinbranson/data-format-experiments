# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the reference `.mat` files with `h5py`. Instead, it uses a hard-coded list of animal IDs, loads one joblib-serialized file per animal from `data/<animal>`, pulls `envs`, `trace`, and `position` from the loaded dictionary, and then iterates through days and 1-minute trial slices.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for a_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
    n_days = d['envs'].shape[0]
    ...
    for day in range(n_days):
        trace_day = d['trace'][day]
        position_day = d['position'][day]
```

iii. The trajectory says the AI inferred a per-animal structure of `trace (n_days, n_cells, n_frames)`, `position (n_days, 2, n_frames)`, and `envs (n_days, 1)` and treated that as the primary dataset representation. The notes justify the choice by claiming the resulting totals match the paper statistics exactly.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the hard-coded `ANIMALS` list. `subjects` is copied directly from that list, and `subject_idx_all` stores the enumeration index of the current animal for each session.

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

iii. The notes describe the dataset as having 7 mice with those exact IDs, and the trajectory summary repeats that the converted dataset contains those 7 subjects.

## 1-c. How are the data split into sessions?

i. The AI treats each day for each animal as one session. The number of sessions for an animal is `d['envs'].shape[0]`, and the loop variable `day` indexes sessions.

ii.
```python
n_days = d['envs'].shape[0]
...
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]
    position_day = d['position'][day]
```

iii. The trajectory explicitly states, "Each session = 1 day = 1 environment," and the notes repeat "Each day for each animal = one session."

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 1-minute trials. The code assumes 30 Hz sampling, defines 1800 frames per trial, computes `n_trials = n_frames // 1800`, and slices each session with `t_start` and `t_end`. Any trailing remainder is dropped because only full trials are iterated.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
TRIAL_FRAMES = FPS * TRIAL_DURATION_S
...
n_frames = d['trace'].shape[2]
...
n_trials = n_frames // TRIAL_FRAMES
...
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
```

iii. The notes say each 40-minute session is split into 1-minute trials and list per-animal frame remainders that are dropped. The trajectory also says each 40-minute session breaks into about forty 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control filter. The code only enforces session-level checks: it skips sessions with no registered cells and skips sessions with fewer than 2 full 1-minute trials. Incomplete trailing frames are implicitly discarded when `n_trials` is floored.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
n_registered = registered.sum()

if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue
...
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
    continue
```

iii. The trajectory and notes do not describe any trial-level QC beyond satisfying the target format. The `<2`-trial guard is best explained by the instructions requiring at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `trace` array inside each animal dictionary, specifically `d['trace'][day]`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_frames)
```

iii. The notes identify the source as the `trace` field and describe it as binary calcium transient traces.

## 2-b. How is the `neural` data processed?

i. The AI assumes the session trace is already arranged as `(cells, frames)`. It removes all-NaN cells, replaces any remaining NaNs with zero, splits the session into 1-minute chunks, and rebins each trial from 30 Hz to 1-second bins by summing every 30 frames.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
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
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
```

iii. The trajectory says the traces are binary rise-extracted events and that 1800 frames per minute was "quite dense," so the AI chose 1-second bins "to reduce dimensionality while preserving temporal structure." The notes repeat that neural data is summed over 30 frames into event counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered at the session level by removing rows that are all NaN across time. If a day has no remaining registered cells, the entire session is skipped. Any residual NaNs in kept cells are zero-filled.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
n_registered = registered.sum()

if n_registered == 0:
    print(f"  Skipping {animal} day {day}: no registered cells")
    continue

trace_registered = np.nan_to_num(trace_registered, nan=0.0)
```

iii. The notes justify this as keeping only cells "registered (non-NaN) on each day." The zero-fill step is justified there only as a safety measure: "shouldn't happen but be safe."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use a paper-defined external event. Instead, it treats each artificial 1-minute segment as aligned to the start of the recording/trial window. This is reflected operationally by slicing contiguous trial windows and in metadata by labeling the alignment event as `"Start of recording session"`.

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

iii. The notes say sessions are continuous 40-minute recordings split into 1-minute trials. The explicit "start of recording session" alignment label appears to be the AI's own metadata choice rather than a paper-derived justification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 1-second bins. The AI rebins the original 30 Hz data by summing every 30 frames into one time bin, producing 60 bins per 1-minute trial.

ii.
```python
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
...
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
```

iii. The trajectory explicitly says the AI chose 1-second bins to make the decoder input more manageable, and the notes repeat that each 1-minute trial becomes 60 time bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment input from `d['envs'][day, 0]`, which is a session-level environment name string, not from the `blocked` variable.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name)
```

iii. The trajectory says the environment geometry comes in as a static 3x3 binary matrix per trial, and the notes say it is derived from `get_env_mat()` from the reference code.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a hand-written 3x3 binary template where `1` means accessible and `0` means blocked. The matrix is then flattened into a 9-element static vector per trial.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(envs[env], dtype=float)
...
env_mat = get_env_mat(env_name)  # 3x3 binary
env_flat = env_mat.flatten()  # (9,)
```

iii. The notes justify this as representing environment geometry directly as a 3x3 accessible-versus-blocked map, and the trajectory says the decoder input will be a flattened 3x3 binary matrix.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The input is treated as static within a session. The same flattened environment vector is attached to every trial in that session, so alignment is by trial membership rather than by timepoint.

ii.
```python
session_input = []
...
for trial in range(n_trials):
    ...
    session_input.append(env_flat)  # (9,) static per trial
```

iii. The notes explicitly say the geometry input is static per trial, and the trajectory describes it as a session-level 3x3 matrix flattened per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` array for each day, `d['position'][day]`, which stores x-y coordinates over time.

ii.
```python
position_day = d['position'][day]  # (2, n_frames)
```

iii. The trajectory identifies `position` as x-y coordinates in centimeters, and the notes describe the output as derived from `(x, y)` position.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first discretizes every frame's x-y position into one of 9 bins using per-session maxima along x and y, not fixed arena edges. It then slices trials and reduces each 30-frame block to the modal position class, yielding a `(1, 60)` output per trial.

ii.
```python
def discretize_position(position, n_bins=3):
    x = position[0]
    y = position[1]
    x_max = np.nanmax(x) + POSITION_BUFFER
    y_max = np.nanmax(y) + POSITION_BUFFER
    x_bin = np.floor(x / (x_max / n_bins)).astype(int)
    y_bin = np.floor(y / (y_max / n_bins)).astype(int)
    ...
    return bin_idx
...
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)
...
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
trial_output = trial_output.reshape(1, -1)
```

iii. The notes justify the coarse output as 3x3 spatial bins and describe the 1-second output as the mode of 30 frames. The trajectory frames the choice as reducing temporal density while preserving structure for the decoder.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into three equal ranges based on that session's observed maximum (`x_max / 3`, `y_max / 3`). The code floors the normalized coordinates, clips them to `0..2`, and combines them as `x_bin * 3 + y_bin`, producing integer categories `0..8`.

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

iii. The notes justify the use of 9 coarse spatial bins in a 75x75 arena, although the note text summarizes the combined index as "row * 3 + col" while the code itself uses `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by the same trial boundaries and then by the same 30-frame time bins inside each trial. Neural activity is summed over each 1-second window, and position is reduced to the modal class over that same 1-second window.

ii.
```python
trial_trace = trace_registered[:, t_start:t_end]  # (n_registered, 1800)
trial_pos_bins = pos_bins[t_start:t_end]  # (1800,)

trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
trial_output = trial_output.reshape(1, -1)  # (1, 60)
```

iii. The trajectory says the AI wanted neural and output sequences at a coarser, "more manageable" 1-second resolution, and the notes describe the output as time-varying at 60 bins per trial.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted time-varying data is rebinned to 1-second resolution. Each 1-minute trial contains 60 time bins. Neural data is summed within each second, and output position is reduced to the modal category within each second.

ii.
```python
TIME_BIN_FRAMES = FPS
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES
...
trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. The trajectory explicitly motivates this as a dimensionality-reduction step, and the notes repeat "Each trial: 1800 frames (60 seconds) -> 60 time bins (1 second each)."

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output data use identical 1-minute trial boundaries and identical 30-frame windows inside each trial. The input geometry is static per trial, so it is aligned at the trial level rather than per timepoint.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    trial_trace = trace_registered[:, t_start:t_end]
    trial_pos_bins = pos_bins[t_start:t_end]
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
    session_input.append(env_flat)
```

iii. The trajectory explains the 1-second windowing as a unified representation for the decoder. The notes describe the input as static and the output/neural data as 60-bin trial sequences.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing cells are handled by removing all-NaN traces. If a session has no registered cells, it is skipped. Any remaining NaNs in kept cells are silently converted to zero. Partial end-of-session remainders are dropped through integer division. Sessions with fewer than 2 complete trials are skipped.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
...
if n_registered == 0:
    ...
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
...
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    ...
```

iii. The notes explicitly justify all-NaN removal and frame remainders. The zero-fill step is described only as a defensive safeguard, and the minimum-trial check is implied by the decoder format requirements rather than by the paper.

## 7-a. What are the most time-consuming steps of the code?

i. The most expensive steps are likely loading each full animal joblib file, iterating over every day and trial, summing neural data into 1-second bins for every trial, and repeatedly computing the modal position class for every 30-frame window.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_days):
    ...
    for trial in range(n_trials):
        ...
        trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
        trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. The AI does not explicitly discuss runtime hotspots. The closest justification is the trajectory's statement that 1800 frames per trial were "quite dense," which motivated extra 1-second binning work to make the decoder inputs more manageable.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop could be vectorized by reshaping an entire session once instead of slicing and rebinnig each trial separately. The output label construction loops could also be replaced with comprehensions or array operations, but those are minor compared with the day/trial processing loop.

ii.
```python
for day in range(n_days):
    ...
    for trial in range(n_trials):
        t_start = trial * TRIAL_FRAMES
        t_end = (trial + 1) * TRIAL_FRAMES
        trial_trace = trace_registered[:, t_start:t_end]
        trial_pos_bins = pos_bins[t_start:t_end]
        trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
        trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
...
for row in range(N_SPATIAL_BINS):
    for col in range(N_SPATIAL_BINS):
        position_labels.append(f"row{row}_col{col}")
```

iii. There is no explicit efficiency justification in the notes. The trajectory suggests the AI prioritized dimensionality reduction and decoder convenience over preserving a simpler vectorized pipeline.

## 7-c. What processing does the code repeat multiple times?

i. The code repeatedly slices trial boundaries, repeatedly rebins each trial's neural data, repeatedly computes 1-second output modes, and repeatedly appends the same static `env_flat` vector once per trial in a session.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    ...
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
    session_input.append(env_flat)
```

iii. The notes and trajectory do not present this repetition as a deliberate design goal. It follows from the AI's choice to turn each 1-minute trial into 60 aggregated time bins.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads full per-animal joblib objects even though it only uses `envs`, `trace`, and `position`. It also computes per-frame position classes and then discards that finer detail by taking a 1-second mode. Any leftover frames at the end of a session are dropped.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
...
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)  # (n_frames,)
...
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
...
n_trials = n_frames // TRIAL_FRAMES
```

iii. The trajectory justifies the 1-second reduction as simplifying the decoder input, not as preserving all information. The notes likewise emphasize decoder performance and paper-level counts rather than minimizing discarded detail.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is the same as in section 6: all-NaN cells are removed, empty-cell sessions are skipped, any remaining NaNs are zero-filled, incomplete trailing frames are dropped, and sessions with fewer than 2 complete trials are skipped.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
trace_registered = trace_day[registered]
...
if n_registered == 0:
    ...
trace_registered = np.nan_to_num(trace_registered, nan=0.0)
...
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    ...
```

iii. The notes justify NaN-based cell exclusion and dropped remainders; the other safeguards are inferred from the trajectory and the target format constraints.

## 9-a. What are the most time-consuming steps of the code?

i. The same hotspots described in 7-a apply here: per-animal `joblib.load`, nested day/trial loops, neural summation into 1-second bins, and repeated modal aggregation of position.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_days):
    ...
    for trial in range(n_trials):
        ...
        trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
        trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
```

iii. No explicit runtime discussion appears in the notes; the trajectory only explains the choice of coarser time bins to reduce dimensionality.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The same vectorization opportunities described in 7-b apply here: session-wide reshaping could replace the per-trial slicing/rebinning loop, and the small label-construction loops are also trivially vectorizable.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    ...
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
...
for row in range(N_SPATIAL_BINS):
    for col in range(N_SPATIAL_BINS):
        input_labels.append(f"geometry_row{row}_col{col}")
```

iii. The AI never explicitly defends the loop-heavy structure. The trajectory suggests it was optimizing for decoder-manageable bins, not for conversion efficiency.

## 9-c. What processing does the code repeat multiple times?

i. The same repeated work described in 7-c applies here: repeated trial slicing, repeated neural summation, repeated output mode calculation, and repeated insertion of the same static environment vector for every trial in a session.

ii.
```python
for trial in range(n_trials):
    t_start = trial * TRIAL_FRAMES
    t_end = (trial + 1) * TRIAL_FRAMES
    ...
    trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)
    trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
    session_input.append(env_flat)
```

iii. The repetition follows from the 1-second-per-trial representation described in the trajectory and notes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The same unnecessary work described in 7-d applies here: loading large joblib objects that include unused fields, computing frame-level position bins before collapsing them to a 1-second mode, and dropping leftover frames that do not fill a full trial.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
...
pos_bins = discretize_position(position_day, N_SPATIAL_BINS)
...
trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)
...
n_trials = n_frames // TRIAL_FRAMES
```

iii. The AI's written justification focuses on a more manageable decoder representation and strong decoder accuracy, not on preserving all raw temporal detail or minimizing wasted work.
