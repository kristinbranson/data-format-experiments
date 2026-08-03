# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-serialized files (one per animal) using `joblib.load()`. Each file is a dictionary keyed by animal ID, containing arrays for `trace`, `position`, `envs`, and `blocked`. The animal IDs are hardcoded in a list (`ANIMALS`). This contrasts with the reference, which loads `.mat` files via `h5py` and discovers them by globbing the data directory.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(data_dir, animal))
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
envs_all = dat[animal]['envs']          # (n_days, 1)
```

iii. The AI chose joblib files because the reference code's `load_dat()` function also uses joblib. Both .mat and joblib files contain the same underlying data. The AI documented in CONVERSION_NOTES.md Step 1 that "load_dat" in `utils.py` loads animal data from joblib/MATLAB files.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to one joblib file. The subject names are the animal IDs from the hardcoded `ANIMALS` list.

ii.
```python
subjects = list(animals)  # animal IDs as subject names
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
```

iii. The AI identified 7 subjects from the data directory, matching the paper's description.

## 1-c. How are the data split into sessions?

i. Each day of recording for each animal becomes a separate session. Sessions are indexed by iterating over the `n_days` dimension of the trace/position arrays.

ii.
```python
n_days = trace_all.shape[0]
for day in range(n_days):
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
    all_neural.append(neural_trials)
    all_subject_idx.append(animal_idx)
```

iii. The CONVERSION_NOTES document 207 total sessions (31 per animal except one with 21), matching the paper.

## 1-d. How are the data split into trials?

i. Each continuous recording session (~40 min) is split into 1-minute non-overlapping trials. However, since the AI applies temporal binning (3 frames per bin), trials are 600 time bins long (not 1800 frames). Remainder frames/bins that don't fill a complete trial are discarded.

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
    neural_trial = neural_binned[:, t_start:t_end]
```

iii. Per the task instructions, "trials" are 1-minute segments. The AI splits after temporal binning, producing 600-bin trials instead of 1800-frame trials.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial filtering is applied. Sessions with zero registered neurons are skipped (but this doesn't happen in practice). Remainder frames that don't complete a full trial are discarded.

ii.
```python
if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue
```

iii. The CONVERSION_NOTES state: "No explicit trial-level curation in the paper (sessions are continuous recordings)." The AI decided not to apply velocity filtering, noting it was specific to the Bayesian decoder in the reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium event data (binarized rising-phase extraction) with shape `(n_cells, n_frames)` per day.

ii.
```python
trace_all = dat[animal]['trace']  # (n_days, n_cells, n_frames)
...
registered_trace = trace_day[registered_mask]  # (n_registered, n_frames)
```

iii. The CONVERSION_NOTES identify the trace as "binarized rising-phase calcium events (0/1)" from "z-score > 2.5" threshold.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps from the reference `fit_decoder` function:
1. Gaussian smoothing with `sigma=3` frames along the time axis
2. Temporal binning via `AvgPool1d` with `kernel_size=3, stride=3`

This converts binary spike trains to continuous firing rate estimates at 10 Hz (100ms bins).

The reference solution does NOT apply any processing — it only transposes and casts to float32, keeping data at native 30 Hz.

ii.
```python
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()  # (n_registered, n_timebins)
```

iii. The AI justified this by citing the reference code's `fit_decoder` function, which applies the same smoothing and binning before decoding. The CONVERSION_NOTES say: "Use reference code approach: Gaussian smooth trace with sigma=3, then AvgPool1d with kernel_size=3."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by checking if the first frame is NaN. Neurons with NaN at `trace_day[:, 0]` are considered unregistered. Any remaining NaNs are replaced with 0.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
```

iii. The AI checks only the first frame, unlike the reference which checks all frames (`~np.all(np.isnan(trace), axis=0)`). The AI noted that unregistered cells have NaN for the entire trace, so checking the first frame should be equivalent. No place cell or activity threshold filtering is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous and trials are artificial 60-second segments starting from the beginning of the session. Neural data is processed (smoothed + temporally binned) before being split into trials.

ii.
```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end]
```

iii. The CONVERSION_NOTES metadata says: "Start of 1-minute trial segment within 40-minute recording session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning from 30 Hz (~33.33ms bins) to ~10 Hz (100ms bins) using AvgPool1d with kernel_size=3. Each trial has 600 time bins instead of 1800 frames.

The reference solution keeps the native 30 Hz resolution (~33.33ms bins) with no rebinning. Each trial has 1800 time bins.

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per temporal bin
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins
```

iii. The AI justified this by citing the reference `fit_decoder` function which uses temporal_bin_size=3. CONVERSION_NOTES say: "Temporal binning: Use reference code approach: Gaussian smooth trace with sigma=3, then AvgPool1d with kernel_size=3."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the input from the `envs` variable (environment name strings like 'square', 'o', 't', etc.) using the `get_env_mat()` function from the reference code. This produces a 3x3 binary matrix indicating accessible positions.

The reference derives the input from the `blocked` variable (indices of blocked reward positions), encoding them as a one-hot vector of 9 positions.

ii.
```python
envs_all = dat[animal]['envs']  # (n_days, 1)
...
env_name = envs_all[day, 0]
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()  # (9,)
```

iii. The AI chose `get_env_mat()` because it was available in the reference code. The CONVERSION_NOTES say: "Input construction: get_env_mat() copied from reference." The environment geometry matrix (1=accessible, 0=blocked) is conceptually the complement of the blocked positions vector (1=blocked, 0=accessible).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a predefined 3x3 binary matrix via `get_env_mat()`, then flattened to a 9-element vector. This is static per session (same for all trials).

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
env_flat = env_mat.flatten()  # (9,)
input_trial = env_flat.astype(np.float32)  # (9,) static per trial
```

iii. The AI used `get_env_mat()` directly from the reference code's `utils.py`. The mapping is hardcoded for 10 known environment geometries.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, containing 2D (x, y) coordinates of the mouse in cm.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
...
position_day = position_all[day]  # (2, n_frames)
```

iii. The CONVERSION_NOTES identify position as "raw x,y in cm (0-75 range)" from DeepLabCut tracking.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first temporally bins position using the same AvgPool1d as neural data, then discretizes into a 3x3 grid. The reference discretizes position at native resolution without temporal binning.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()  # (2, n_timebins)
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS  # ~25.000003
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
```

iii. The AI followed the reference `fit_decoder` approach of temporally binning both neural and behavioral data before further processing.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into a 3x3 grid (9 categories). Each axis (x, y) is divided into 3 equal bins of ~25 cm each. The category is computed as `x_bin * 3 + y_bin`.

The reference uses `y_bin * 3 + x_bin` ordering.

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS  # ~25.000003
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]  # x_bin * 3 + y_bin
```

iii. The AI uses floor division for binning (with a small BUFFER to handle the boundary at 75 cm). The bin ordering (x * 3 + y vs y * 3 + x) differs from the reference but doesn't affect decoder performance.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both neural and position data are temporally binned using the same AvgPool1d before being split into trials at the same indices, ensuring frame-by-frame alignment.

ii.
```python
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
neural_trial = neural_binned[:, t_start:t_end]
output_trial = pos_category[t_start:t_end]
```

iii. Both streams are processed in the same function with the same temporal binning, then sliced with the same trial indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered neurons (NaN at first frame) are removed. Any remaining NaNs in registered neurons are replaced with 0. Remainder frames that don't fill a complete trial are discarded. Sessions with no registered neurons would be skipped.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
```

iii. The `nan_to_num` call is described as a safety measure ("shouldn't happen but be safe"). The AI also handles the edge case of varying frame counts across sessions by discarding remainder frames.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib files (~20-80s per animal). Processing per session takes ~0.7-1.0s. Total conversion time was ~7.1 minutes for the full dataset.

ii. N/A (timing is printed during execution)

iii. CONVERSION_NOTES Step 7 documents: Loading ~210s total, Processing ~170s total, Total ~380-427s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop (iterating over `n_trials` to slice arrays) could be replaced with `np.array_split` or reshape operations. The per-session loop could potentially be parallelized across sessions.

ii.
```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end]
```

iii. The AI did not identify these as issues. The code runs in ~7 minutes total, which is acceptable.

## 6-c. What processing does the code repeat multiple times?

i. The `AvgPool1d` pooling object is recreated for every session (inside `process_session`), though it could be created once and reused. The Gaussian smoothing parameters are recomputed each call.

ii.
```python
def process_session(...):
    ...
    pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
    neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
    ...
    pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
```

iii. This is a minor inefficiency; the pooling object is lightweight and the overhead is negligible.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The Gaussian smoothing and temporal binning applied to neural data may be considered unnecessary preprocessing, as the downstream decoder (`train_decoder.py`) may handle its own preprocessing. The reference solution does not apply these steps. Additionally, `save_processing_plot` computes various statistics and visualizations only when `--show-processing` is set.

ii.
```python
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The AI applied these processing steps based on the reference `fit_decoder` function, believing them to be part of the required processing pipeline.
