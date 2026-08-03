# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a hard-coded list of seven animal files from `data/` using `joblib`, not the `.mat` files used in the human reference. It processes each animal sequentially, then each day/session within each animal, and only later splits the continuous session into trials.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal in animals:
    print(f"\nLoading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
...
    for day in range(n_days):
        ...
        neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
        output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. In `CONVERSION_NOTES.md`, the agent says the paper code loads data via `load_dat(..., format="joblib")` and explicitly planned to "load data from joblib files for each animal." The trajectory shows it chose the joblib representation after exploring the reference code and dataset structure.

## 1-b. How are the data split into subjects?

i. Subjects are the hard-coded animal IDs. Each loaded joblib file corresponds to one mouse, and the subject ID is the animal name string.

ii.
```python
all_subjects = sorted(set(animals))
...
subject_id = animal
if subject_id not in all_subjects:
    all_subjects.append(subject_id)
subj_idx = all_subjects.index(subject_id)
```

iii. The notes describe the dataset as seven named animals and state "Subject identification: Use animal ID strings."

## 1-c. How are the data split into sessions?

i. Each `day` within an animal is treated as one recording session. The output session lists are appended once per day.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    ...
    neural_sessions.append(session_neural)
    input_sessions.append(session_input)
    output_sessions.append(session_output)
    subject_idx_list.append(subj_idx)
```

iii. The notes say "Each 'session' = one day of recording" and "1 session per day."

## 1-d. How are the data split into trials?

i. The agent temporally rebins each full session into 1-second bins first, then divides each session into non-overlapping 60-bin chunks, so each trial is 60 seconds long at 1 Hz. Any leftover partial chunk is dropped.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)  # 60 time bins per trial
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. In the notes, the agent chose "1 second bins" and "Split each ~40 min session into 1-minute trials (60 seconds = 60 time bins at 1s resolution)."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Incomplete trailing chunks are implicitly discarded by integer division, and sessions with fewer than two resulting trials are skipped entirely.

ii.
```python
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
n_trials = len(neural_trials)
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The notes say "No explicit trial curation in reference code" and separately note the decoder-format requirement that each session should have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data comes from the per-day `trace` array in the joblib data structure.

ii.
```python
n_days = d['trace'].shape[0]
...
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The notes identify `trace` as binary calcium-event data and map `trace` directly to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The agent treats the traces as binary event data, keeps valid cells for that day, and averages every 30 frames into 1-second bins, producing per-cell mean event rates for each 1-second bin.

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

iii. The notes say "Use the binary trace data directly" and "Average within each 1-second time bin to get firing rates." The trajectory also states the agent chose 1-second bins as "more practical."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural filter is to drop cells whose first timepoint is `NaN`, using that as a proxy for unregistered cells on that day. The code does not apply place-cell filtering, velocity filtering, or an activity-threshold filter during conversion.

ii.
```python
trace_day = d['trace'][day]
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
...
n_valid = valid_mask.sum()
```

iii. The notes say "Only include cells that are registered (non-NaN) on that day," and explicitly justify "No velocity filtering" and "No cell activity filtering" during conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not align neural data to an external behavioral or stimulus event. Instead, it treats the start of the recording session as the alignment point in metadata and creates fixed 60-second trial windows from that continuous recording.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
    ...
}
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
```

iii. The trajectory and notes repeatedly describe the recordings as continuous sessions split into 1-minute trials; the metadata string formalizes that as "Start of recording session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 1-second bins (30 frames at 30 Hz). Yes, temporal rebinning is applied to both neural and position data before trialization.

ii.
```python
FPS = 30
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)
...
'time_bin_size': TIME_BIN_SEC * 1000,
```

iii. The notes explicitly choose "1 second (30 frames) bins" and describe the reference decoder's smaller binning as less practical for this output format.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The agent derives environment geometry from the `envs` environment-name field, not from the raw `blocked` indices.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. In the trajectory, the agent states that `blocked` did not match the geometry it expected and concluded that `env_mat` from the environment name "correctly represents the environment shape," so it used that instead.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped through a hand-copied `get_env_mat` lookup table to a 3x3 binary matrix of open/closed partitions, then flattened to a 9-element vector and duplicated across all trials in that session as a static input.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square': [[1,1,1],[1,1,1],[1,1,1]],
        'o': [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(env_mats[env], dtype=float)
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
session_input.append(env_mat.astype(np.float32))
```

iii. The notes say "Use get_env_mat(env_name).flatten() -> 9 values" and the trajectory says this was chosen because it better matched the actual environment geometry than the `blocked` field.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the per-day `position` array.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The notes map `position (x,y)` directly to the target output field.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first bins each frame's continuous `(x, y)` position into a 3x3 spatial grid using a session-specific maximum position, then temporally rebins by taking the modal spatial bin within each 1-second window. Each trial stores the rebinned labels as shape `(1, 60)`.

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
...
def bin_position_temporal(position, time_bin_frames, n_spatial_bins=3):
    bin_idx = bin_position_to_grid(position, n_spatial_bins)
    ...
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result
...
session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. The notes say to "bin position into 3x3 grid" and to use the modal position bin within each 1-second window. The trajectory records that this design was chosen after checking occupancy in the blocked environments.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholds are not fixed at 25 cm intervals. Instead, the code computes `bin_size = (nanmax(position)+1e-5)/3`, floors `x` and `y` separately, clips each axis into `0..2`, and converts the pair to a single class index as `x_bin * 3 + y_bin`.

ii.
```python
buffer = 1e-5
pos_max = np.nanmax(position) + buffer
bin_size = pos_max / n_spatial_bins
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. The notes say "Use the same binning approach as the reference code: `floor(pos / bin_size)` where `bin_size = (max_pos + buffer) / 3`," although this is the agent's interpretation rather than the human reference implementation.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned by using the same session, the same 1-second temporal windows, and the same 60-second trial boundaries. Position is converted to one label per 1-second bin, while neural data is averaged over the same 30-frame windows.

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. The notes state that both streams are binned to 1-second resolution and then split into the same 1-minute trials.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing/unregistered neurons by excluding cells whose first sample is `NaN`. It also truncates leftover frames/time bins that do not fill a complete temporal bin or trial. It does not explicitly impute or repair missing position values.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
...
trace_truncated = trace[:, :n_bins * time_bin_frames]
...
n_trials = n_timebins // trial_length
...
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The notes emphasize registered/non-NaN cells and dropping partial trailing segments; they do not describe any more elaborate missing-data repair.

## 6-a. What are the most time-consuming steps of the code?

i. The agent identifies loading and per-session processing as the main costs, with the notes specifically calling out sequential data loading and the position-mode computation during temporal binning.

ii.
```python
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
...
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
```

iii. `CONVERSION_NOTES.md` says "Load + process" took about 23s/animal and notes "Mode computation in `bin_position_temporal` uses a loop" and "Data loading is sequential."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent explicitly identifies the loop in `bin_position_temporal` that computes the per-bin mode with `np.unique` as vectorizable.

ii.
```python
result = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
```

iii. The notes say this "could vectorize with `scipy.stats.mode`."

## 6-c. What processing does the code repeat multiple times?

i. The code repeats some computations outside the main conversion path: `plot_processing` recomputes valid-cell masks, rebins neural activity, and re-bins positions for visualization after those same steps were already done for conversion. It also repeatedly casts and appends the same static environment vector for every trial in a session.

ii.
```python
if show_processing:
    plot_processing(d, animal, data_dir)
...
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
...
pos_binned = bin_position_temporal(d['position'][day], TIME_BIN_FRAMES, N_SPATIAL_BINS)
...
for trial_idx in range(n_trials):
    session_input.append(env_mat.astype(np.float32))
```

iii. The notes only partially acknowledge repeated work by mentioning the position-mode loop and plotting option; they do not discuss the repeated per-trial duplication of the static input vector.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code maintains bookkeeping and optional visualization work that is not used by downstream decoder analyses, including the unused `total_neurons_sum` accumulator and the optional plotting path. It also duplicates the same static environment vector into every trial instead of storing a shared session-level object.

ii.
```python
total_neurons_sum = 0
...
total_neurons_sum += n_valid
...
if show_processing:
    plot_processing(d, animal, data_dir)
...
session_input.append(env_mat.astype(np.float32))
```

iii. The notes do not explicitly frame these as unnecessary downstream computations; this is mostly evident from the code structure itself.
