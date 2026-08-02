# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI chose to load a hard-coded list of seven animal files from `data/` using `joblib`, not by scanning `.mat` files. For each animal file it opens the top-level dict, then iterates through `trace`, `position`, and `envs` day by day. Trials are not loaded directly; they are created later by splitting each day after temporal binning.

ii.
```python
DATA_DIR = 'data'
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal in animals:
    print(f"\nLoading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. In `CONVERSION_NOTES.md`, the AI says the reference workflow uses `load_dat(..., format='joblib')` and that each animal joblib file contains `trace`, `position`, `envs`, `maps`, and related fields. In the trajectory it explicitly planned to "load data from joblib files for each animal."

## 1-b. How are the data split into subjects?

i. The AI treated each hard-coded animal ID as one subject. Subject names are the animal strings themselves, and `subject_idx` is the index of that animal in the sorted unique subject list.

ii.
```python
all_subjects = sorted(set(animals))
...
subject_id = animal
if subject_id not in all_subjects:
    all_subjects.append(subject_id)
subj_idx = all_subjects.index(subject_id)
...
subject_idx_list.append(subj_idx)
```

iii. The notes say "Subject identification: Use animal ID strings," and the trajectory summarizes the dataset as seven animals processed independently.

## 1-c. How are the data split into sessions?

i. The AI treated each day within an animal file as one session. It uses the first dimension of `d['trace']` as the session/day count and iterates over `range(n_days)`.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]
    pos_day = d['position'][day]
```

iii. `CONVERSION_NOTES.md` states "Each 'session' = one day of recording" and marks "Session = Day" as confirmed during consistency checking.

## 1-d. How are the data split into trials?

i. The AI split each session into fixed 1-minute non-overlapping trials after first rebinnig the continuous streams to 1-second bins. With 60-second trials and 1-second bins, each trial is 60 bins long. Any remainder shorter than a full trial is dropped by floor division.

ii.
```python
TIME_BIN_SEC = 1.0
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)
...
def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_timebins = len(data)
        n_trials = n_timebins // trial_length
        ...
    else:
        n_timebins = data.shape[-1]
        n_trials = n_timebins // trial_length
        ...
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. The notes say "Trial definition: Split each ~40 min session into 1-minute trials (60 seconds = 60 time bins at 1s resolution)." The trajectory repeats that each session should produce about 39-40 one-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control filter. The only gate is at the session level: if a day yields fewer than two trials after splitting, the whole session is skipped.

ii.
```python
n_trials = len(neural_trials)

if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The likely justification is the task-format requirement that each session contain at least two trials. The notes also say "No explicit trial curation in reference code."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from the per-day slice of `d['trace']` in the joblib dataset.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The notes identify `trace` as the binary calcium-event matrix and explicitly map `trace (binary events)` to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The AI used the binary trace directly, filtered to valid cells, then temporally rebinned it from 30 Hz to 1 Hz by averaging every 30 frames. It did not compute dF/F or apply additional smoothing inside the conversion script.

ii.
```python
def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)
...
valid_trace = trace_day[valid_mask]
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. `CONVERSION_NOTES.md` says "Use the binary trace data directly. Average within each 1-second time bin to get firing rates" and argues that no further dF/F-like preprocessing is needed because the traces are already preprocessed binary event vectors.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removed cells it considered unregistered on that day by checking whether the first timepoint is `NaN`. It kept all remaining registered cells and explicitly chose not to apply activity-threshold or place-cell filtering in the conversion.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
n_valid = valid_mask.sum()
```

iii. The notes say "Only include cells that are registered (non-NaN) on that day" and separately record the decision "No cell activity filtering" and "No velocity filtering" for the conversion output, even though the paper's decoder applies such filters later.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI did not align trials to any behavioral event. It treated the start of the recording session as the alignment point and then cut the continuous session into contiguous 60-second chunks.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
    ...
}
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
```

iii. The notes describe the recordings as continuous exploration data and state that sessions are simply split into one-minute trials for the decoder format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI set the converted temporal resolution to 1 second and rebinned the 30 Hz data by averaging 30 frames per bin.

ii.
```python
FPS = 30
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)
...
'time_bin_size': TIME_BIN_SEC * 1000,
```

iii. The notes say "Time bin size: Use 1 second (30 frames) bins" because it was judged more practical for the decoder format, even while acknowledging the reference decoder uses much finer temporal bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derived the environment input from the environment-name field `d['envs'][day, 0]`, not from the raw `blocked` variable.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name).flatten()
```

iii. In both the notes and the trajectory, the AI says it chose `env_mat` from the environment name because it believed the raw `blocked` indices did not match the geometry convention it wanted to expose.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI mapped each environment name to a hard-coded 3x3 binary occupancy matrix, flattened that matrix to length 9, and stored the result as a static float vector for each trial.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square': [[1,1,1],[1,1,1],[1,1,1]],
        'o': [[1,1,1],[1,0,1],[1,1,1]],
        't': [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(env_mats[env], dtype=float)
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
session_input.append(env_mat.astype(np.float32))
```

iii. The notes say "Use get_env_mat(env_name) for input, not blocked field directly" and justify that choice as a better match to the intended geometry of the arena.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is treated as static within a session and duplicated once per trial. It is not time-varying within a trial.

ii.
```python
for trial_idx in range(n_trials):
    session_input.append(env_mat.astype(np.float32))
```

iii. The notes explicitly describe the input as "Static per trial (same for all timepoints in trial)."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derived the output position from the per-day slice of `d['position']`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The notes map `position (x,y)` directly to the target `output` field.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first converted each frame's 2D position to a 3x3 spatial bin using a session-specific bin size based on that session's maximum position value. It then rebinned time to 1-second bins by taking the mode spatial bin within each 30-frame window.

ii.
```python
def bin_position_to_grid(position, n_spatial_bins=3):
    buffer = 1e-5
    pos_max = np.nanmax(position) + buffer
    bin_size = pos_max / n_spatial_bins
    pos_binned = np.floor(position / bin_size).astype(int)
    pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
    bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
    return bin_idx

def bin_position_temporal(position, time_bin_frames, n_spatial_bins=3):
    bin_idx = bin_position_to_grid(position, n_spatial_bins)
    ...
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result
```

iii. The notes say "Bin position into 3x3 grid" and "Use the mode (most frequent) position bin within each 1-second window." The trajectory says the AI validated this approach by checking that the center bin is unoccupied in the `'o'` environment.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI used three equal-width bins per axis, defined by `bin_size = pos_max / 3` for that session, clipped coordinates into `[0, 2]`, and combined the axis bins as `x_bin * 3 + y_bin` to obtain categories `0..8`.

ii.
```python
bin_size = pos_max / n_spatial_bins
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. The notes call this "the same binning approach as the reference code," although the implemented rule is actually based on a per-session maximum and on `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligned output to neural data by applying the same 1-second temporal binning window size to both streams and then splitting both on the same 60-bin trial boundaries. Each trial output is reshaped to `(1, time)`.

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
...
session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. The notes say the decoder inputs and outputs should use the same 1-second bins and the same 1-minute trial splits.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 1-second bins for both neural and position streams. Yes, temporal rebinning is applied from 30 Hz to 1 Hz.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)
...
'time_bin_size': TIME_BIN_SEC * 1000,
```

iii. The notes present this as an explicit design choice for practicality and reduced data size.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are both converted into 1-second bins using the same 30-frame windows and then split into the same 60-second trial boundaries. Input is static per trial, so the same 9-value geometry vector is attached to every trial from that session.

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
...
session_input.append(env_mat.astype(np.float32))
```

iii. The notes say the task should use a static environment input together with time-varying neural and position streams, all organized into the same 1-minute trial structure.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI handled missing/unregistered cells by removing rows whose first trace sample is `NaN`. It also implicitly discards leftover samples that do not fill a complete 1-second bin or 1-minute trial, and it would skip any session that yields fewer than two trials.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
...
n_bins = n_timepoints // time_bin_frames
trace_truncated = trace[:, :n_bins * time_bin_frames]
...
n_trials = n_timebins // trial_length
...
if n_trials < 2:
    ...
    continue
```

iii. The notes repeatedly describe NaN rows as cells not registered on that day and treat truncation of incomplete tails as acceptable for fixed-length trial construction.

## 7-a. What are the most time-consuming steps of the code?

i. The AI's own documentation says the expensive part is per-animal load-plus-process time. From the code, the main costs are sequential `joblib.load`, per-session temporal binning, per-bin position-mode computation, and optional plotting.

ii.
```python
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
    pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. `CONVERSION_NOTES.md` estimates roughly 23 seconds per animal for load and processing and separately notes that data loading is sequential and that the mode computation loop could be a bottleneck.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the explicit loop over time bins in `bin_position_temporal`. The trial-construction loops in `split_into_trials` and the per-trial append loop in `convert_data` could also be reduced or batched.

ii.
```python
result = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
...
for t in range(n_trials):
    start = t * trial_length
    end = start + trial_length
    trials.append(data[..., start:end])
...
for trial_idx in range(n_trials):
    session_neural.append(neural_trials[trial_idx])
    session_input.append(env_mat.astype(np.float32))
    session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. The notes explicitly identify the mode computation loop as something that "could vectorize with `scipy.stats.mode`."

## 7-c. What processing does the code repeat multiple times?

i. The code repeats `env_mat.astype(np.float32)` for every trial even though the vector is constant within a session. If `--show-processing` is enabled, it also recomputes valid-cell masks, binned neural traces, and binned position traces for the first three days even though analogous computations were already done during conversion.

ii.
```python
for trial_idx in range(n_trials):
    session_input.append(env_mat.astype(np.float32))
...
if show_processing:
    plot_processing(d, animal, data_dir)
...
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
...
pos_binned = bin_position_temporal(d['position'][day], TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. There is no strong explicit written justification beyond the notes' emphasis on generating processing plots for sanity checking.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional visualization work in `plot_processing` is not used in the saved dataset or by downstream decoder training. The script also tracks `total_neurons_sum` but never uses it in the output.

ii.
```python
total_neurons_sum = 0
...
total_neurons_sum += n_valid
...
if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. The notes justify plots as validation aids, not as part of the converted dataset itself.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6: the AI drops cells marked missing by a `NaN` at the first trace sample, truncates incomplete tails when binning or splitting trials, and skips sessions only if they would end up with fewer than two trials.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
...
n_bins = n_timepoints // time_bin_frames
trace_truncated = trace[:, :n_bins * time_bin_frames]
...
n_trials = n_timebins // trial_length
...
if n_trials < 2:
    continue
```

iii. The notes frame NaN handling as removal of unregistered cells and do not describe any additional repair for malformed entries.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: sequential file loading, temporal binning, per-bin mode calculation for position, and optional plot generation are the dominant costs.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. The notes' runtime estimates and identified inefficiencies are the only explicit justification.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-time-bin loop in `bin_position_temporal` is the clearest target, with additional opportunities in trial splitting and per-trial list building.

ii.
```python
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
...
for t in range(n_trials):
    ...
...
for trial_idx in range(n_trials):
    ...
```

iii. The notes explicitly call out the mode computation loop as vectorizable.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: it repeatedly casts the static environment vector for every trial and optionally recomputes neural/position binning inside the plotting helper.

ii.
```python
session_input.append(env_mat.astype(np.float32))
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
...
pos_binned = bin_position_temporal(d['position'][day], TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. The repeated work seems to be accepted as part of simple implementation and visualization support.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: optional plotting work and the unused `total_neurons_sum` accumulator do not contribute to the saved dataset used downstream.

ii.
```python
total_neurons_sum = 0
...
total_neurons_sum += n_valid
...
if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. The notes present the plotting as a sanity-check aid rather than required conversion output.
