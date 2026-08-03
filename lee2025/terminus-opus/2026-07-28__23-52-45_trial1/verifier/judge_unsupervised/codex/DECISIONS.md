# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes the seven animal IDs, then sequentially loads one joblib file per animal from `data/<animal>`. After `joblib.load(...)`, it takes the nested dictionary entry `dat[animal]` and iterates through every recording day inside that object.

ii. 
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for animal in animals:
    print(f"\nLoading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]

    n_days = d['trace'].shape[0]
    ...
    for day in range(n_days):
```

iii. In Step 1 of `CONVERSION_NOTES.md`, the agent wrote that the reference loader can use `format="joblib"` and returns `{animal: dataset_dict}`. The notes treat the joblib files as acceptable preprocessed inputs and do not justify going back to the raw `.mat` files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified directly by animal ID string. The script builds `subjects` from the animal list and stores one `subject_idx` per session based on the current animal.

ii. 
```python
all_subjects = sorted(set(animals))

subject_id = animal
if subject_id not in all_subjects:
    all_subjects.append(subject_id)
subj_idx = all_subjects.index(subject_id)
...
subject_idx_list.append(subj_idx)
```

iii. Step 5 of `CONVERSION_NOTES.md` says: "Subject identification: Use animal ID strings."

## 1-c. How are the data split into sessions?

i. Each recording day inside an animal file is treated as one session. The session loop is `for day in range(n_days)`, where `n_days = d['trace'].shape[0]`.

ii. 
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]
    pos_day = d['position'][day]
```

iii. Step 4 of `CONVERSION_NOTES.md` explicitly says "Session = Day" and that this gives the expected 207 sessions.

## 1-d. How are the data split into trials?

i. The script first rebins the continuous session into 1-second bins, then splits those rebinned time series into fixed 1-minute trials. Because each trial is 60 seconds and each bin is 1 second, each trial has 60 time bins. Any leftover partial trial at the end is discarded by integer division.

ii. 
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)

neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)

neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. Step 5 of `CONVERSION_NOTES.md` justifies this with two linked decisions: use 1-second bins because they are "more practical," and define trials as 1-minute chunks to get about 39-40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no per-trial quality control. The code keeps every full-length trial produced by `split_into_trials`, drops only the final incomplete tail implicitly, and skips an entire session only if fewer than two trials remain.

ii. 
```python
def split_into_trials(data, trial_length):
    ...
    n_trials = n_timebins // trial_length
    trials = []
    for t in range(n_trials):
        start = t * trial_length
        end = start + trial_length
        trials.append(data[..., start:end])
    return trials

...
n_trials = len(neural_trials)

if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` say there is "No explicit trial curation in reference code," and the agent appears to have added only the "at least two trials per session" rule from the decoder format requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` variable for each day.

ii. 
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. The notes repeatedly identify `trace` as the neural source variable and describe it as binary calcium-event data.

## 2-b. How is the `neural` data processed?

i. For each day, the code removes unregistered cells, then averages the binary trace over non-overlapping 30-frame windows to produce a per-second firing-rate-like value. The rebinned matrix is stored as `float32`.

ii. 
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]

def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)
```

iii. Step 5 says: "Use the binary trace data directly. Average within each 1-second time bin to get firing rates." The notes argue this is a practical representation and explicitly choose it over the finer temporal scale in the paper code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is a registration/missing-data filter: neurons with `NaN` in the first frame are dropped for that day. The script does not apply place-cell filtering, velocity filtering, or low-activity filtering.

ii. 
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
...
brain_region_idx_sessions.append(np.zeros(n_valid, dtype=int))  # All CA1
```

iii. Step 5 says "Only include cells that are registered (non-NaN) on that day," "No velocity filtering," and "No cell activity filtering." The notes justify this by saying the downstream decoder can handle stationary periods and low-activity cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script aligns everything to the start of the recording session. It does not align to a within-session behavioral event; instead it starts from time 0 of each day, bins the continuous recording, and cuts contiguous 1-minute chunks.

ii. 
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
    ...
}
```

iii. Step 4 of `CONVERSION_NOTES.md` says "Temporal alignment: Start of recording," and the final metadata repeats that choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 1-second bins. Yes, temporal rebinning is applied: neural traces are averaged over 30 frames and position is reduced to one label per 30-frame window by taking the mode.

ii. 
```python
FPS = 30
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
...
'time_bin_size': TIME_BIN_SEC * 1000,
```

iii. Step 5 explicitly says: "Time bin size: Use 1 second (30 frames) bins." The notes acknowledge that the reference decoder uses a much finer temporal scale but defend 1-second bins as more practical.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from `envs`, not from `blocked`. For each day the code reads the environment name string and converts it to a 3x3 geometry matrix.

ii. 
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name).flatten()
```

iii. Step 4 of `CONVERSION_NOTES.md` says: "Use get_env_mat(env_name) for input, not blocked field directly." The rationale given is that `env_mat` "matches geometry."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is passed through a copied `get_env_mat` lookup table, producing a 3x3 matrix with `1` for open partitions and `0` for blocked partitions. That matrix is flattened to length 9 and reused as a static per-trial vector.

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

iii. Step 5 says: "Environment input: Use get_env_mat(env_name).flatten() -> 9 binary values. Static per trial."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw `position` variable for each day.

ii. 
```python
pos_day = d['position'][day]  # (2, n_timepoints)
...
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. Step 5 maps `position (x,y)` to the decoder output variable.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code computes a session-specific maximum position, divides that range into three equal bins per axis, floors each frame into an `x` bin and `y` bin, clips to `[0, 2]`, combines them into one class label, then takes the most frequent class within each 1-second window.

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
```

iii. Step 5 says this follows the same binning idea as the reference code, with `pos_binned = floor(pos / bin_size)` and "Use the mode (most frequent) position bin within each 1-second window."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into a 3x3 categorical grid using a dynamic bin width based on the maximum position observed in that session. Each coordinate is clipped into 3 bins, then converted to one of 9 class labels with `x_bin * 3 + y_bin`.

ii. 
```python
pos_max = np.nanmax(position) + buffer
bin_size = pos_max / n_spatial_bins
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. Step 5 says: "Bin position into 3x3 grid (9 bins). Use the same binning approach as the reference code ... Output = x_bin * 3 + y_bin."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The position output is aligned to neural data by applying the same 30-frame temporal windows and then the same 60-bin trial boundaries. Each neural 1-second bin is paired with one position class label from the same second.

ii. 
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)

neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
...
session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. Step 5 frames this as using a common 1-second time base for both streams before trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/unregistered neurons are dropped using a `NaN` mask. Partial end-of-session windows and partial end-of-session trials are discarded by truncation via floor division. Unknown environment labels raise an exception. Sessions with fewer than two trials are skipped entirely.

ii. 
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]

trace_truncated = trace[:, :n_bins * time_bin_frames]
bin_idx_truncated = bin_idx[:n_bins * time_bin_frames]
...
n_trials = n_timebins // trial_length
...
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. Step 10 of `CONVERSION_NOTES.md` lists these as edge-case checks: NaN cells are excluded, the final ~55 s remainder is discarded, and trial boundaries were checked for off-by-one issues.

## 6-a. What are the most time-consuming steps of the code?

i. The main runtime cost is sequential per-animal loading and per-day preprocessing: loading the joblib files, temporal binning of neural traces, temporal mode-reduction of positions, and construction of all per-trial lists. In `--show-processing` mode, the script also recomputes neural and position binning for plotting.

ii. 
```python
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    for day in range(n_days):
        neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
        pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
        ...

if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. Step 7 of `CONVERSION_NOTES.md` estimates about 23 s per animal for load-plus-process and Step 6 explicitly calls out the position-mode computation as a notable inefficiency.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization candidates are the per-bin mode loop in `bin_position_temporal`, the Python loops inside `split_into_trials`, and the per-trial append loop that rebuilds `session_neural`, `session_input`, and `session_output` one trial at a time.

ii. 
```python
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]

for t in range(n_trials):
    start = t * trial_length
    end = start + trial_length
    trials.append(data[..., start:end])

for trial_idx in range(n_trials):
    session_neural.append(neural_trials[trial_idx])
    session_input.append(env_mat.astype(np.float32))
    session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. Step 6 of `CONVERSION_NOTES.md` explicitly mentions the mode loop as something that "could vectorize with `scipy.stats.mode`." No fuller justification is given for leaving the other Python loops in place.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several small pieces of work: it casts the same `env_mat` to `float32` once per trial, it rechecks membership in `all_subjects` even though `all_subjects` was already built from `sorted(set(animals))`, and `plot_processing()` recomputes the valid-cell mask, binned neural activity, and binned position after the main conversion already computed them.

ii. 
```python
all_subjects = sorted(set(animals))
...
if subject_id not in all_subjects:
    all_subjects.append(subject_id)

for trial_idx in range(n_trials):
    session_input.append(env_mat.astype(np.float32))

def plot_processing(d, animal, data_dir):
    ...
    valid_mask = ~np.isnan(trace_day[:, 0])
    valid_trace = trace_day[valid_mask]
    neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
    ...
    pos_binned = bin_position_temporal(d['position'][day], TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. The notes describe the plotting pass as a processing visualization step and do not mention any attempt to reuse values computed during conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script tracks `subjects_list` and `total_neurons_sum` but never uses them in the saved output. In `--show-processing` mode it also generates plotting work that is purely diagnostic and not consumed by downstream analyses. The repeated per-trial cast of a static `env_mat` is likewise thrown away after list construction.

ii. 
```python
subjects_list = []
...
total_neurons_sum = 0
...
total_neurons_sum += n_valid

if show_processing:
    plot_processing(d, animal, data_dir)
```

iii. No explicit justification is given in the notes for `subjects_list` or `total_neurons_sum`; the plotting work is justified only as a visual sanity-check requirement from the task instructions.
