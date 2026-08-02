# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (one per animal) using `joblib.load()`. Each file contains a dictionary keyed by animal ID, with sub-fields `trace`, `position`, `envs`, and `blocked`. The data directory also contains `.mat` files (HDF5 format), which the AI did not use.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
envs_all = dat[animal]['envs']          # (n_days, 1)
```

iii. The AI chose joblib files because the reference code's `load_dat()` function uses joblib. The `.mat` files contain the same data in HDF5 format, but the AI followed the reference code's loading approach.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject. Subject names are hardcoded as a list of 7 animal IDs (`ANIMALS` constant).

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = list(animals)
```

iii. The AI hardcoded the animal list based on files found in the data directory during exploration.

## 1-c. How are the data split into sessions?

i. Each day within a subject's data becomes a separate session. The AI iterates over the first axis of the trace array (`n_days`), treating each day as a session.

ii.
```python
n_days = trace_all.shape[0]
for day in range(n_days):
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
```

iii. Each day is a separate recording session, consistent with the paper describing 207 total sessions across 7 animals.

## 1-d. How are the data split into trials?

i. Each session is split into 1-minute non-overlapping segments. The AI applies temporal binning first (3-frame bins), so trials are 600 time bins (not 1800 frames). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS  # 1800 frames
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins
...
n_timebins_total = neural_binned.shape[1]
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end]  # (n_registered, 600)
```

iii. The instruction specifies 1-minute trials. The AI splits after temporal binning, yielding 600 time bins per trial instead of 1800 frames.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are kept. Sessions with no registered cells are skipped.

ii.
```python
if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue
```

iii. The paper does not describe trial-level filtering for the continuous recording sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib files, which contains binarized rising-phase calcium events (0/1 values) with shape `(n_cells, n_frames)` per day.

ii.
```python
trace_all = dat[animal]['trace']  # (n_days, n_cells, n_frames)
...
registered_trace = trace_day[registered_mask]  # (n_registered, n_frames)
```

iii. The `trace` variable contains pre-processed binary calcium event data as described in the paper.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps: (1) Gaussian smoothing with sigma=3 frames, and (2) temporal binning via AvgPool1d with kernel_size=3 and stride=3. This converts binary traces to continuous firing rates at 100 ms resolution. NaN values are replaced with 0.

ii.
```python
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The AI followed the reference code's `fit_decoder` function, which applies `gaussian_filter1d(traces, sigma=temporal_bin_size)` followed by `AvgPool1d(kernel_size=temporal_bin_size)`. The AI reasoned that this processing should be applied during data conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with NaN in the first timepoint are considered unregistered and excluded from that session. No further quality filtering (e.g., place cell selection, activity thresholds, velocity filtering) is applied.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
```

iii. The AI checked only the first timepoint for NaN to determine registration status, assuming that unregistered cells have NaN at all timepoints.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous and trials are artificial 60-second segments starting from the beginning of the recording.

ii. N/A (trials are simply sequential segments of the continuous recording)

iii. There is no stimulus onset or behavioral event to align to in this continuous exploration paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 3-frame average pooling after Gaussian smoothing, producing 100 ms time bins (10 Hz). Each trial has 600 time bins.

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per temporal bin
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins
```

iii. The AI adopted the temporal binning from the reference code's `fit_decoder` function, which uses AvgPool1d with kernel_size=3.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field in the joblib files, which contains environment name strings (e.g., 'square', 'o', 't'). These are converted to 3x3 binary matrices using the `get_env_mat()` function copied from the reference code.

ii.
```python
envs_all = dat[animal]['envs']  # (n_days, 1)
...
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()  # (9,)
```

iii. The AI used the environment name to look up a pre-defined 3x3 binary accessibility matrix, where 1 = accessible and 0 = blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix via `get_env_mat()` (copied from reference code's `utils.py`), then flattened to a 9-element vector. The matrix indicates which spatial bins are accessible (1) vs blocked (0).

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
```

iii. This directly uses the reference code's approach to represent environment geometry.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The input is static per trial (and per session). The same 9-element vector is replicated for every trial within a session.

ii.
```python
input_trial = env_flat.astype(np.float32)  # (9,) static per trial
...
input_trials.append(input_trial)
```

iii. Environment geometry does not change within a session, so it is constant across all trials and timepoints.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` field in the joblib files, containing 2D (x, y) coordinates in cm with shape `(2, n_frames)` per day.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
...
position_day = position_all[day]  # (2, n_frames)
```

iii. The position variable records the animal's location tracked via DeepLabCut.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first temporally bins the position data using the same AvgPool1d as the neural data (kernel=3), then discretizes the binned position into a 3x3 grid. Each axis is divided into 3 equal bins of 25 cm.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
```

iii. The AI applied temporal binning to position to match the neural data's temporal resolution, then discretized into 3x3 bins as specified by the task.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by dividing each axis into 3 equal bins spanning 0-75 cm. The category is computed as `x_bin * 3 + y_bin`, giving 9 categories (0-8).

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS  # ~25.00 cm
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]  # x_bin * 3 + y_bin
```

iii. A small buffer (1e-5) is added to the arena size to handle edge cases at exactly 75 cm. The ordering is `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is temporally binned using the same AvgPool1d pooling as neural data, then split into trials with the same indices. This ensures frame-for-frame alignment.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
# Same pooling object used for neural data
```

iii. Both neural and position data go through the same temporal binning pipeline, maintaining alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning of 3 frames, producing 100 ms time bins. The native 30 Hz data (33.33 ms) is averaged into 10 Hz bins.

ii.
```python
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

iii. The AI adopted the temporal binning from the reference `fit_decoder` function.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output (position) are temporally binned using the same AvgPool1d pooling, then split into trials with the same indices. Input is static per trial, so no temporal alignment is needed.

ii.
```python
# Same pooling for both:
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
# Same trial splitting:
neural_trial = neural_binned[:, t_start:t_end]
output_trial = pos_category[t_start:t_end]
```

iii. Using the same pooling object and trial indices ensures temporal alignment between neural and output data.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Unregistered neurons (NaN in first timepoint) are excluded. Any remaining NaN values are replaced with 0 via `np.nan_to_num`. Remainder frames not filling a complete trial are discarded. Sessions with zero registered cells are skipped.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
```

iii. The AI handles NaN conservatively by filtering unregistered neurons and zeroing any remaining NaN values.

## 7-a. What are the most time-consuming steps of the code?

i. Loading the joblib files is the most time-consuming step (20-80 seconds per animal). Processing (smoothing, binning, discretization) is comparatively fast (~0.7-1.0s per session).

ii. N/A

iii. Documented in CONVERSION_NOTES.md: total conversion time was ~427 seconds (~7.1 minutes).

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials sequentially, but this is a simple slicing operation. The per-session processing loop cannot be easily vectorized due to variable session lengths and the sequential file I/O.

ii.
```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end]
```

iii. The main bottleneck is I/O, not computation, so vectorization would provide minimal benefit.

## 7-c. What processing does the code repeat multiple times?

i. The `AvgPool1d` pooling object is recreated inside `process_session` for every session call, though it could be created once and reused.

ii.
```python
# Inside process_session, called once per session:
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

iii. This is a minor inefficiency; creating the pooling object is fast.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The Gaussian smoothing and temporal binning applied by the AI are additional processing steps that the reference solution does not perform. These transform the data before it reaches the decoder.

ii.
```python
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32), sigma=GAUSS_SIGMA, axis=1)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The decoder training script (`train_decoder.py`) may apply its own preprocessing, making the AI's preprocessing potentially redundant or conflicting.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. Unregistered neurons are filtered by checking NaN at the first timepoint. Remaining NaN values are replaced with 0. Incomplete trials are discarded.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
```

iii. The approach is conservative and prevents NaN propagation through the smoothing and binning steps.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. File I/O (loading joblib files) dominates at 20-80 seconds per animal. Total conversion ~7 minutes.

ii. N/A

iii. The AI documented timing in CONVERSION_NOTES.md and confirmed the total was within estimates.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The trial splitting loop could theoretically use `np.split` or array reshaping instead of a Python loop, but the performance impact is negligible.

ii.
```python
for t in range(n_trials):
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
```

iii. The main bottleneck is I/O, not computation.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. The pooling object is recreated per session. Additionally, the `get_env_mat` dictionary is rebuilt on every call.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square': np.array([[1,1,1],[1,1,1],[1,1,1]]),
        ...
    }
    return env_mats.get(env, ...)
```

iii. Both are minor inefficiencies with negligible performance impact.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. The Gaussian smoothing and temporal binning are preprocessing steps that the reference solution does not apply. The `train_decoder.py` script may apply its own preprocessing, making this redundant.

ii. See 7-d code snippets.

iii. If the downstream decoder applies its own temporal binning, the AI's preprocessing would result in double-processing of the data.
