# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a hard-coded list of seven animal files from `/app/data` using `joblib.load`. For each animal file, it reads the nested dictionary entries `trace`, `position`, and `envs`, then iterates through all days/sessions and later splits them into trials.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    trace_all = dat[animal]['trace']
    position_all = dat[animal]['position']
    envs_all = dat[animal]['envs']
```

iii. In `CONVERSION_NOTES.md` Step 1 and trajectory steps 4-6, the AI justified this by concluding that the reference repository primarily uses Python joblib animal files via `load_dat(..., format='joblib')`, and treated those files as the canonical source to match the paper code.

## 1-b. How are the data split into subjects?

i. The AI treats each hard-coded animal ID as one subject, and uses the ordered `ANIMALS` list directly as the `subjects` field.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = list(animals)  # animal IDs as subject names
```

iii. In `CONVERSION_NOTES.md` Step 2 and trajectory steps 11-13, the AI noted that there are seven animal files and that each corresponds to one mouse, so it used those animal IDs as subject identifiers.

## 1-c. How are the data split into sessions?

i. The AI treats each recording day within an animal file as one session. It iterates over the first dimension of `trace`, `position`, and `envs`, processes that day, and appends one session to the output lists.

ii.
```python
n_days = trace_all.shape[0]
...
for day in range(n_days):
    env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
    ...
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 13, 16, and 17, the AI concluded that one day in the joblib arrays corresponds to one 40-minute recording session, matching the paper’s one-session-per-day structure.

## 1-d. How are the data split into trials?

i. Each session is split into consecutive, non-overlapping 1-minute trials. The AI first temporally bins the session to 100 ms bins, then divides the result into chunks of 600 bins per trial; any remainder is discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS  # 1800 frames
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins
...
n_timebins_total = neural_binned.shape[1]
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
...
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 17 and 29, the AI justified this as the required adaptation from continuous 40-minute sessions to the decoder task’s 1-minute trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply explicit trial-level quality-control filtering. It keeps all 1-minute trial segments from any session that has at least one registered cell.

ii.
```python
if n_registered == 0:
    return [], [], [], 0
...
for t in range(n_trials):
    ...
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory step 17, the AI explicitly decided against velocity-based or place-cell-based trial curation, arguing those filters were specific to the paper’s within-session Bayesian decoder rather than required by this task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the per-animal `trace` array in the joblib files.

ii.
```python
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
...
neural_trials, input_trials, output_trials, n_registered = process_session(
    trace_all[day], position_all[day], env_name
)
```

iii. In `CONVERSION_NOTES.md` Steps 1-3 and trajectory steps 9, 11, and 13, the AI states that `trace` already contains the paper’s preprocessed binarized calcium-event traces, so this is the appropriate neural source variable.

## 2-b. How is the `neural` data processed?

i. The AI keeps only registered cells, converts remaining NaNs to zero, applies Gaussian smoothing with sigma 3 frames, then temporally averages with `AvgPool1d` over non-overlapping 3-frame windows to produce continuous-valued neural signals at 100 ms resolution.

ii.
```python
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
...
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. In `CONVERSION_NOTES.md` Steps 1 and 5 and trajectory steps 9, 17, 23, 28, and 29, the AI justified this by trying to match the paper’s `fit_decoder` preprocessing and by reacting to the downstream validator’s complaint that purely binary 0/1 neural data would be rejected.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neural data by removing cells it considers unregistered on a session: cells whose trace is NaN at the first frame. It does not apply place-cell filtering or the decoder code’s `>5` event threshold, and it zero-fills any remaining NaNs after registration filtering.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
...
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
```

iii. In `CONVERSION_NOTES.md` Steps 2-5 and trajectory steps 13, 16, and 17, the AI justified this by observing that NaNs indicate cells not registered on that day, and by deciding that place-cell and movement/activity thresholds were specific to the reference decoder analysis rather than mandatory for the exported dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align neural data to an experimental event. Instead, it treats the start of each artificial 1-minute segment as the alignment point and records that in metadata.

ii.
```python
'metadata': {
    'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric environment exploration',
    'time_bin_size': TIME_BIN_MS,
    'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 17 and 29, the AI justified this as a consequence of the task-defined trialization of continuous sessions, since there is no natural stimulus onset event in the source data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data has 100 ms bins. The AI rebins the original 30 Hz data by smoothing and averaging every 3 frames.

ii.
```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
...
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 15, 17, 23, and 29, the AI justified this as matching the paper’s decoder preprocessing and as necessary to transform binary events into continuous-valued neural features accepted by the downstream decoder.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` session labels, not from the raw `blocked` arrays.

ii.
```python
envs_all = dat[animal]['envs']          # (n_days, 1)
...
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
...
env_mat = get_env_mat(env_name)
```

iii. In `CONVERSION_NOTES.md` Steps 1 and 5 and trajectory steps 9 and 10, the AI justified this by finding `get_env_mat` in the paper code and concluding that the named geometry itself was the most direct representation of arena structure for the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps each environment name to a 3×3 binary occupancy matrix with `get_env_mat`, flattens it to a 9-element vector, casts it to `float32`, and reuses the same static vector for every trial in that session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()
...
input_trial = env_flat.astype(np.float32)
input_trials.append(input_trial)
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 10, 17, and 29, the AI justified this as matching the paper’s geometry utilities and as a natural way to provide static contextual information for each 1-minute trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives mouse position from the `position` array in each animal file.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
...
neural_trials, input_trials, output_trials, n_registered = process_session(
    trace_all[day], position_all[day], env_name
)
```

iii. In `CONVERSION_NOTES.md` Steps 1-3 and trajectory steps 9, 11, and 13, the AI identified `position` as the raw x-y trajectory in centimeters and used it as the source for decoder targets.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI temporally averages x and y positions with the same 3-frame pooling used for neural data, then converts the pooled coordinates into 3×3 spatial bins.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
...
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 15, 17, and 29, the AI justified this as keeping position aligned with the 100 ms neural bins while adapting the paper’s position-decoding preprocessing to the required 3×3 output grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI divides both x and y position ranges into three equal 25 cm bins over a 75 cm arena, clips them to `[0, 2]`, and combines the two axis bins into one category with `x_bin * 3 + y_bin`.

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)

# Convert to single category: x_bin * 3 + y_bin
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 16, 17, and 33, the AI justified this as matching the arena’s 3×3 partition structure and giving the decoder nine coarse spatial classes.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns position to neural data by applying the same temporal pooling to both streams and then slicing both into trials using the same trial-bin indices.

ii.
```python
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory steps 17, 29, and 48, the AI justified this as preserving frame-to-frame correspondence after rebinning, and later reported that its sanity checks showed exact agreement between its processed neural and output trial slices and the raw inputs under that processing scheme.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI treats NaN traces as missing registration, removes those cells session-wise, replaces any remaining NaNs with zeros, skips sessions with zero registered cells, and discards leftover frames/time bins that do not fill a complete pooled trial.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
...
if n_registered == 0:
    return [], [], [], 0
...
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
...
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
```

iii. In `CONVERSION_NOTES.md` Steps 2-5 and trajectory steps 13, 16, 17, and 29, the AI justified this by interpreting NaNs as unregistered cells, assuming any residual NaNs were anomalous but safest to coerce to zero, and accepting truncation of incomplete tails when forming fixed-length trials.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify file loading as the dominant cost, with per-animal `joblib.load` taking tens of seconds, while per-session processing is much faster.

ii.
```python
for animal_idx, animal in enumerate(animals):
    ...
    t0 = time.time()
    dat = joblib.load(os.path.join(data_dir, animal))
    print(f"  Loaded in {time.time()-t0:.1f}s")
```

iii. In `CONVERSION_NOTES.md` Step 7 and trajectory steps 37, 41, 44, and 47, the AI explicitly reported timing estimates showing that loading each large animal file dominated total runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the explicit loop over trials inside `process_session`, which slices and appends each trial one by one rather than reshaping the binned arrays. The outer animal/day loops are structurally necessary, but the per-trial assembly could have been more vectorized.

ii.
```python
neural_trials = []
input_trials = []
output_trials = []

for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    input_trial = env_flat.astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The AI did not give an explicit justification for keeping this loop. This assessment is inferred from the code structure rather than from `CONVERSION_NOTES.md`.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly reconstructs the pooling layer for each session, repeatedly casts the same environment vector to `float32` once per trial, and repeatedly slices trial windows in Python for every session.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
...
env_flat = env_mat.flatten()
...
for t in range(n_trials):
    ...
    input_trial = env_flat.astype(np.float32)
```

iii. The AI did not explicitly discuss these repetitions in its notes or trajectory. This is inferred from direct inspection of `convert_data.py`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the default conversion path, there is not much discarded scientific processing, but the script includes optional visualization work (`save_processing_plot`) and logging/timing instrumentation that are not used by downstream decoder analyses. It also computes `n_cells` only for printing.

ii.
```python
n_cells = trace_all.shape[1]
print(f"  {n_days} days, {n_cells} cells, {trace_all.shape[2]} frames/day")
...
if show_processing and sessions_processed <= 2:
    save_processing_plot(...)
```

iii. The AI’s notes justify the optional plotting as a sanity-check tool in Step 7, but there is no explicit justification that this processing is needed for the final exported dataset.
