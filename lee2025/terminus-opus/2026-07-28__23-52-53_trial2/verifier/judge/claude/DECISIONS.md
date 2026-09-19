# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (not .mat files) using `joblib.load()`. Each animal's data is loaded from a file named by animal ID in the `data/` directory. The loaded object is a dictionary keyed by animal ID, containing arrays for `trace`, `position`, `envs`, and `blocked`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
envs_all = dat[animal]['envs']          # (n_days, 1)
```

iii. The AI followed the reference code's `load_dat()` function which uses `joblib.load()` for non-MATLAB files. The data directory contained both joblib files and .mat files; the AI chose joblib because the reference code used that format. The AI hardcodes the list of 7 animal IDs as a constant `ANIMALS` list.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to a separate joblib file. The animal ID is used as the subject name. The list of animals is hardcoded as `ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", ...]`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = list(animals)
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
```

iii. The 7 animal IDs were identified from the data directory contents and reference code. Hardcoding the list ensures consistent ordering.

## 1-c. How are the data split into sessions?

i. Each day of recording for each animal becomes a separate session. The loaded data has shape `(n_days, n_cells, n_frames)`, and the AI iterates over the first dimension (days).

ii.
```python
n_days = trace_all.shape[0]
for day in range(n_days):
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
```

iii. Each day represents a separate recording session in a potentially different environment. The total is 207 sessions across 7 animals, consistent with the paper.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute non-overlapping trials. After temporal binning (3 frames per bin), each trial has 600 time bins. Remainder frames that don't fill a complete trial are discarded.

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
```

iii. The instructions specify "1-minute trials within each session." After temporal binning, each trial has 600 time bins (1800 frames / 3 = 600 bins at 100ms resolution).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are included. Sessions with zero registered neurons are skipped entirely.

ii.
```python
if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue
```

iii. The paper describes no explicit trial curation. The reference code's velocity filtering (>5 cm/s) was specific to the Bayesian decoder and was not applied here.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binarized calcium event traces with shape `(n_cells, n_frames)` per day. Values are 0 or 1 (rising-phase calcium events), with NaN for unregistered cells.

ii.
```python
trace_all = dat[animal]['trace']  # (n_days, n_cells, n_frames)
...
registered_trace = trace_day[registered_mask]  # (n_registered, n_frames)
```

iii. The `trace` variable contains pre-processed binary calcium events from the original paper's extraction pipeline.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps from the reference `fit_decoder` function: (1) Gaussian smoothing with sigma=3 frames along the time axis, (2) temporal binning via AvgPool1d with kernel_size=3 and stride=3. This converts binary 0/1 traces into continuous firing rates at 100ms resolution. Remaining NaN values are replaced with 0.

ii.
```python
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The reference `fit_decoder` function applies `gaussian_filter1d(traces, sigma=temporal_bin_size)` then `AvgPool1d(kernel_size=temporal_bin_size)`. The AI replicated this processing to match the reference code's approach for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are NaN at the first timepoint are excluded (considered unregistered for that session). This is checked via `~np.isnan(trace_day[:, 0])`. No further neuron filtering (place cell filtering, activity thresholds) is applied.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
```

iii. The AI followed the reference code convention of checking the first frame for NaN to determine registration status. Place cell filtering was not applied because "use all registered cells to give the decoder maximum information."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is used. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of each session. The alignment event is described as "Start of 1-minute trial segment."

ii.
```python
'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),
```

iii. There is no stimulus onset or behavioral event to align to. Each trial is simply a consecutive segment of continuous recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 3 frames are averaged into 1 time bin via AvgPool1d. The resulting time bin size is 100ms (3 frames at 30 Hz). Each trial has 600 time bins instead of the native 1800 frames.

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per temporal bin
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins
...
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. This matches the reference `fit_decoder` function which uses `AvgPool1d(kernel_size=3)` for temporal binning.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable (environment name string per session), which is converted to a 3x3 binary matrix using the reference code's `get_env_mat()` function.

ii.
```python
envs_all = dat[animal]['envs']  # (n_days, 1)
env_name = envs_all[day, 0]
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()  # (9,)
```

iii. The `get_env_mat()` function was copied directly from the reference code (`utils.py`). It maps environment names (e.g., 'square', 'o', 't') to 3x3 binary matrices indicating which partitions of the arena are accessible (1) or blocked (0).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary to get a 3x3 binary matrix, then flattened to a 9-element vector. This is static (constant) per trial within a session. Values are 0 (blocked) or 1 (accessible).

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
input_trial = env_flat.astype(np.float32)  # (9,) static per trial
```

iii. This approach uses the structured environment representation from the reference code rather than the raw `blocked` indices. The 3x3 matrix captures the spatial layout of the arena geometry.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable, which contains 2D coordinates (x, y) of the mouse in the arena at each frame, with shape `(2, n_frames)` per day.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
```

iii. Position is recorded via DeepLabCut tracking in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position data is first temporally binned using the same AvgPool1d as neural data (kernel=3), then spatially discretized into a 3x3 grid. The continuous x,y values are divided by bin_size (75/3 = 25 cm + small buffer) and cast to integer, then combined as `x_bin * 3 + y_bin`.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. The temporal binning of position before discretization matches the reference `fit_decoder` approach which temporally bins behavioral data the same way as neural data.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D position is divided into a 3x3 grid with bin edges at 25 cm and 50 cm. Each axis is discretized independently by dividing by bin_size (~25.00 cm) and taking the integer part. Combined category = `x_bin * 3 + y_bin`, yielding 9 categories (0-8).

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS  # ~25.00003
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. The 3x3 grid is specified in the task instructions ("3 x 3 = 9 spatial bins"). The formula `x_bin * 3 + y_bin` treats x as rows and y as columns.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both neural and position data are temporally binned using the same AvgPool1d pooling (kernel=3, stride=3), then split into trials using the same time bin indices. This ensures frame-for-frame alignment at 100ms resolution.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
neural_trial = neural_binned[:, t_start:t_end]
output_trial = pos_category[t_start:t_end]
```

iii. Using the same temporal binning and trial splitting ensures exact alignment between neural activity and position labels.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of missing/problematic data are handled: (1) Unregistered neurons (NaN at first frame) are excluded per session. (2) Any remaining NaN values in registered traces are replaced with 0. (3) Remainder frames that don't complete a full 1-minute trial are discarded.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
...
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL  # remainder dropped
```

iii. The `nan_to_num` call handles edge cases where a registered neuron might have sporadic NaN values within its trace.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the animal data files via `joblib.load()`, which takes 20-80 seconds per animal. Processing per session is ~0.7-1.0 seconds. Total conversion time for all data is ~7 minutes.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
print(f"  Loaded in {time.time()-t0:.1f}s")
```

iii. The AI documented timing estimates: loading ~210s total, processing ~170s total, overall ~380-427s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop could potentially be vectorized using `np.split` or array reshaping instead of list comprehension. However, the AI's trial splitting is already efficient since it uses numpy slicing.

ii.
```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end]
```

iii. The AI did not identify specific vectorization opportunities since processing time per session was already fast (~1s).

## 6-c. What processing does the code repeat multiple times?

i. The AvgPool1d pooling layer is instantiated once per session call (`process_session`), but uses the same parameters each time. The `get_env_mat()` lookup is called per session but is trivial.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

iii. The repeated instantiation is minimal overhead. The code does not perform any computationally expensive redundant processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The Gaussian smoothing and temporal binning add processing that the reference solution does not perform. This transforms the native 30 Hz data into 10 Hz data, reducing temporal resolution. Additionally, the `save_processing_plot` function generates visualization plots that are not used in downstream analysis.

ii.
```python
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The AI justified this processing as matching the reference `fit_decoder` function, but the downstream decoder (`train_decoder.py`) operates on whatever temporal resolution is provided. The reference human solution does not apply this pre-processing, keeping data at native 30 Hz.
