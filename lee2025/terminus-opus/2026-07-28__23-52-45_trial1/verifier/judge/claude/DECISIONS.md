# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (not `.mat` files) using `joblib.load()`. Each animal's data is stored in a joblib file in the `data/` directory. The AI hardcodes the list of 7 animal IDs and iterates over them, loading each with `joblib.load(os.path.join(data_dir, animal))`. The loaded dict is indexed by the animal name to get the dataset dictionary containing `trace`, `position`, `envs`, `blocked`, etc.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
```

iii. The AI identified that the data directory contains both `.mat` files and joblib files, and chose to use the joblib format because the reference code's `load_dat()` function uses `joblib.load()`. The AI noted this in CONVERSION_NOTES.md Step 1: "Data is loaded via `load_dat(animal, p, format='joblib')` which returns `{animal: dataset_dict}`".

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The AI hardcodes the list of 7 animal IDs in `ALL_ANIMALS` and iterates over them. Subject names are the animal ID strings.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
    subject_id = animal
```

iii. The AI identified 7 subjects from the data files and hardcoded them. This matches the reference paper's description.

## 1-c. How are the data split into sessions?

i. Within each subject's data, sessions correspond to recording days. The AI iterates over the first dimension of the `trace` array (`d['trace'].shape[0]` gives `n_days`), treating each day as a separate session.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. The AI confirmed that each day = one session, consistent with the paper's 207 total sessions across 7 animals.

## 1-d. How are the data split into trials?

i. Each ~40-minute recording session is split into 1-minute (60-second) non-overlapping trials. After temporal rebinning to 1-second bins, each trial has 60 time bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)  # 60 time bins per trial
...
def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_timebins = len(data)
        n_trials = n_timebins // trial_length
        trials = []
        for t in range(n_trials):
            start = t * trial_length
            end = start + trial_length
            trials.append(data[start:end])
        return trials
```

iii. The instructions specify "1-minute trials within each session." The AI splits after temporal rebinning, so trial_length=60 (bins) rather than 1800 (frames).

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped entirely. No per-trial quality filtering is applied.

ii.
```python
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The instructions state "There needs to be at least two trials within each session." The AI enforces this. In practice, all sessions are ~40 minutes, yielding 39-40 trials each, so no sessions are actually skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib data, which contains binary calcium event traces (0/1 values representing the rising phase of calcium transients).

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The AI identified that `trace` contains binary (0/1) values from the CONVERSION_NOTES.md: "The trace data is BINARY (0/1) - rising phase of calcium transients."

## 2-b. How is the `neural` data processed?

i. The AI applies temporal rebinning: the binary trace data is averaged within 1-second (30-frame) windows using reshape and mean, producing firing rate estimates. The data is also cast to float32.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
...
def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)
```

iii. The AI chose 1-second bins as documented in CONVERSION_NOTES.md Step 5: "Use 1 second (30 frames) bins. This provides reasonable temporal resolution while reducing data size."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons that are registered (non-NaN) on a given day are included. The AI checks the first timepoint for NaN to determine validity.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]  # (n_valid_cells, n_timepoints)
```

iii. The AI noted that cells not registered on a given day have NaN traces. Checking only the first timepoint is sufficient if NaN status is consistent across all timepoints for a given cell-day.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of the recording.

ii. N/A (alignment is implicit via slicing from recording start)

iii. The AI noted in metadata: `'temporal_alignment_event': 'Start of recording session'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning from the native 30 Hz (33.33 ms) to 1-second (1000 ms) bins. This is a 30x downsampling. The metadata reports `time_bin_size: 1000` ms.

ii.
```python
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
...
'time_bin_size': TIME_BIN_SEC * 1000,  # in ms
```

iii. CONVERSION_NOTES.md Step 5: "The reference code uses temporal_bin_size=3 (100ms) for decoding, but for our decoder format, 1-second bins are more practical and still capture spatial behavior well."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field in the data (environment name strings like 'square', 'o', 't', etc.), which are then mapped to 3x3 binary geometry matrices using the `get_env_mat()` function copied from the reference code.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI used the `get_env_mat()` function from the reference code's `utils.py`, which maps environment names to 3x3 binary matrices indicating accessible (1) vs blocked (0) regions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix using a lookup dictionary (from `get_env_mat()`), then flattened to a 9-element vector. This vector is static (same for all trials within a session).

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
env_mat = get_env_mat(env_name).flatten()
session_input.append(env_mat.astype(np.float32))
```

iii. The AI chose to use the environment geometry matrix rather than the blocked indices directly, reasoning that it captures the spatial structure of the environment.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` field in the data, which contains 2D (x,y) coordinates of the mouse at each timepoint.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The `position` data records the animal's location in the 75x75 cm arena at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is discretized into a 3x3 grid (9 classes). The AI uses a data-driven approach: it computes `pos_max = np.nanmax(position) + buffer` and divides by 3 to get bin sizes. Each coordinate is floor-divided to get bin indices. The grid label is `x_bin * 3 + y_bin`.

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
```

iii. The AI chose data-driven binning (using `nanmax`) rather than a fixed 75 cm arena size. Also uses `x_bin * 3 + y_bin` ordering.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by floor-dividing continuous coordinates by the bin size (`pos_max / 3`). Resulting values are clipped to [0, 2] per axis, then combined as `x_bin * 3 + y_bin` to produce 9 categories (0-8).

ii.
```python
pos_binned = np.floor(position / bin_size).astype(int)
pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. The 3x3 grid with 9 classes matches the instruction requirement of "3 x 3 = 9 spatial bins."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is first spatially binned at the native 30 Hz rate, then temporally binned into 1-second windows using the mode (most frequent position bin in each window). This temporal binning matches the neural data's 1-second bins.

ii.
```python
def bin_position_temporal(position, time_bin_frames, n_spatial_bins=3):
    bin_idx = bin_position_to_grid(position, n_spatial_bins)
    n_bins = n_timepoints // time_bin_frames
    bin_idx_reshaped = bin_idx_truncated.reshape(n_bins, time_bin_frames)
    result = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result
```

iii. By using the same temporal binning (30 frames per bin) for both neural and position data, they remain aligned in the time dimension.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with NaN at the first timepoint are excluded from that session. Remainder frames that don't fill a complete 1-second bin or 1-minute trial are discarded. Sessions with fewer than 2 trials are skipped.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
...
n_bins = n_timepoints // time_bin_frames  # remainder discarded
...
n_trials = n_timebins // trial_length  # remainder discarded
...
if n_trials < 2:
    continue
```

iii. The AI's approach handles the main data quality issue (unregistered neurons with NaN traces) and ensures structural validity (minimum 2 trials per session).

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files is the most time-consuming step. The AI reported ~15-27 seconds per animal, totaling ~153 seconds for all 7 animals.

ii. N/A (timing is from output logs)

iii. From conversion_full_out.txt, total time was 153.0s for 7 animals.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The mode computation in `bin_position_temporal` uses a Python loop over time bins with `np.unique` per bin.

ii.
```python
result = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
```

iii. The AI noted this inefficiency in CONVERSION_NOTES.md Step 6: "Mode computation in bin_position_temporal uses a loop (could vectorize with scipy.stats.mode)."

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each animal/day is processed once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No obviously unnecessary processing was identified. The code is relatively streamlined, processing only what is needed for the output format.

ii. N/A

iii. N/A
