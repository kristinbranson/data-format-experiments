# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (not `.mat` files) using `joblib.load()`. Each animal has a joblib file in the `data/` directory. The file contains a dict keyed by animal name with arrays for `trace`, `position`, `envs`, and `blocked`. A hardcoded list of 7 animal names (`ANIMALS`) is used to iterate over subjects.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(data_dir, animal))
animal_data = dat[animal]
```

iii. The AI noted that the reference code's `load_dat` function supports both joblib and MATLAB formats, and chose joblib since the data was available in that format. Documented in CONVERSION_NOTES.md Step 1.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject. The subject names come from the hardcoded `ANIMALS` list.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = list(animals_to_process)
```

iii. The AI identified 7 animals from the data directory and hardcoded them. Each animal's data is in a separate file.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days. The `trace` array has shape `(n_days, n_cells, n_frames)`. Each day becomes a separate session by iterating over the first dimension.

ii.
```python
n_days = animal_data['trace'].shape[0]
for day in range(n_days):
    ...
    neural_trials, input_trials, output_trials, n_valid = process_session(
        animal_data, day, ...)
```

iii. The AI noted that each animal has 31 days (except QLAK-CA1-51 with 21), totaling 207 sessions matching the paper.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute non-overlapping segments. After temporal binning (3 frames -> 1 bin), each trial has 600 time bins (BINS_PER_TRIAL = 1800/3 = 600). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
...
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]  # (n_valid, 600)
```

iii. Per the instruction: "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session."

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped entirely. No per-trial quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. The instructions state "There needs to be at least two trials within each session." The AI enforces this by skipping sessions with fewer than 2 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium events (0/1) representing rising phases of calcium transients, z-scored > 2.5.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The AI identified that `trace` is already binarized preprocessed data, noting "No dF/F needed - data is already preprocessed binary events."

## 2-b. How is the `neural` data processed?

i. The AI applies Gaussian smoothing (sigma=3 frames) followed by temporal average pooling (bin size=3 frames), matching the reference `fit_decoder` function's preprocessing. This converts the data from 30 Hz (1800 frames/trial) to 10 Hz (600 bins/trial, 100ms bins). The result is cast to float32.

ii.
```python
def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)
```

iii. The AI explicitly followed the reference `fit_decoder` function: "Reference fit_decoder applies gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0) before binning. We should do the same (sigma=3 frames along time axis)." (CONVERSION_NOTES.md Step 5, decision 10)

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only registered (non-NaN) cells for each session are included. The AI checks whether the first frame of a cell is NaN to determine validity.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]  # (n_valid, n_frames)
```

iii. The AI noted: "Include only registered cells per session (not NaN). Reference decoder also uses all registered cells per session." No velocity-based or activity-threshold cell filtering is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials start from the beginning of the recording session and are segmented sequentially.

ii. N/A (alignment is implicit in the sequential trial splitting)

iii. The AI documented: "Trials start at beginning of session recording. Align to session start." The metadata sets `temporal_alignment_event: 'Start of recording session'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 3-frame bins at 30 Hz = 100 ms time bins. Each trial has 600 time bins instead of 1800 raw frames.

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per time bin (from reference fit_decoder)
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
...
binned_trace = temporal_bin_trace(valid_trace)  # (n_valid, n_total_bins)
```

iii. The AI justified this by referencing the `fit_decoder` function's `temporal_bin_size=3` parameter. Documented as "Decoder temporal bin: 3 frames (100ms)" from code analysis.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable (environment name strings like 'square', 'o', 't', etc.) which is then mapped to a 3x3 binary accessibility matrix using the `get_env_mat` function from the reference code.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() ...)
...
env_input = get_env_input(env_name)  # (9,)

ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    ...
}
```

iii. The AI copied the environment matrix definitions from the reference code's `get_env_mat` function, noting this represents "which partitions of 3x3 grid are blocked."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is looked up in a dictionary of 3x3 binary matrices, where 1 = accessible, 0 = blocked. The matrix is flattened to a 9-element vector. This is static per trial (same for all timepoints within a session).

ii.
```python
def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    return mat.flatten().astype(np.float32)
...
input_trials.append(env_input)  # (9,) static
```

iii. The AI used the reference code's `get_env_mat` approach rather than the raw `blocked` variable, arguing this directly represents environment geometry.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena at each frame.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The position data records the animal's location in a 75x75 cm arena at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first temporally binned (mean of each 3-frame bin), then discretized into a 3x3 spatial grid using floor division by 25 cm bin size, clipped to [0, 2]. The grid label is computed as `x_bin * 3 + y_bin`.

ii.
```python
def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin

binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
```

iii. The AI chose 25 cm bins (75/3) to create the 3x3 grid required by the instructions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using `floor(pos / 25)` clipped to [0, 2] for each axis. The combined bin index is `x_bin * 3 + y_bin`, producing 9 categories (0-8).

ii.
```python
x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
return x_bin * n_bins + y_bin
```

iii. The AI uses `floor` division rather than `np.digitize`, and combines bins as `x_bin * 3 + y_bin` (x-major ordering) rather than `y_bin * 3 + x_bin` (y-major).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both position and neural data are temporally binned using the same bin size (3 frames), ensuring frame-for-frame alignment. They are then split into trials using the same bin indices.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)  # (n_valid, n_total_bins)
binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
...
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1)
```

iii. Both arrays are processed to the same temporal resolution and split using identical indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered cells (NaN at first frame) are excluded per session. Sessions with fewer than 2 trials are skipped. Remainder frames/bins that don't fill a complete trial are discarded.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
...
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. The AI documented NaN filtering and noted that all sessions in practice had 39-40 trials, so the <2 trial filter never triggered.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files and applying Gaussian smoothing + temporal binning are the most time-consuming steps. The conversion log shows per-animal times of 17-44 seconds, with total conversion taking approximately 3-4 minutes.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
```

iii. The AI tracked timing per animal and estimated full conversion at ~4.3 minutes, well under the 15-minute limit.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials sequentially, but this is a minor overhead since slicing operations are fast. The per-day loop could potentially be parallelized across days.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
    neural_trials.append(trial_neural)
```

iii. No specific vectorization issues were noted by the AI. The code is reasonably efficient.

## 6-c. What processing does the code repeat multiple times?

i. The Gaussian smoothing and temporal binning are applied once per session, which is appropriate since each session has different valid cells. No redundant processing was identified.

ii. N/A

iii. The AI's code processes each session exactly once with no repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The Gaussian smoothing (sigma=3) applied to the neural trace is additional processing that the reference human solution does not apply. This processing is part of the reference decoder code's preprocessing, but the downstream `train_decoder.py` may apply its own preprocessing, potentially double-processing the data.

ii.
```python
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
```

iii. The AI justified this as matching the reference `fit_decoder` function, but it may be unnecessary if the downstream decoder script handles its own preprocessing.
