# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from **joblib** files (not .mat files) using `joblib.load()`. Each animal has a corresponding joblib file in the `data/` directory. The loaded dict is keyed by the animal name and contains arrays for `trace`, `position`, `envs`, `blocked`, etc. The data is 3D: `trace` has shape `(n_days, n_cells, n_timepoints)` and `position` has shape `(n_days, 2, n_timepoints)`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]

n_days = d['trace'].shape[0]
n_cells_total = d['trace'].shape[1]
n_timepoints = d['trace'].shape[2]
```

iii. The AI noted in CONVERSION_NOTES.md that the reference code uses `load_dat(animal, p, format="joblib")` which loads joblib files. The AI chose to use the same data source format as the reference code's primary loading pathway.

## 1-b. How are the data split into subjects?

i. Each animal name in the hardcoded list `ALL_ANIMALS` corresponds to one subject. The list contains 7 animal IDs. Each animal's data is loaded from a separate joblib file.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. The AI identified 7 subjects from the data directory and hardcoded them into the script.

## 1-c. How are the data split into sessions?

i. Each "day" within a subject's data becomes a separate session. The AI iterates over the first dimension of the trace array (`n_days`), where each day is one recording session in one environment.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The AI confirmed that each day = one session, totaling 207 sessions across all animals, consistent with the paper.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute trials. After temporal binning to 1-second resolution (60 time bins per trial), the session is split into non-overlapping 60-bin segments. Remainder frames are discarded. Sessions with fewer than 2 trials are skipped.

ii.
```python
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)  # 60 time bins per trial
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)

n_trials = len(neural_trials)
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The instructions specify splitting sessions into 1-minute trials. The AI chose 60 time bins per trial at 1-second resolution.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped (though in practice all sessions have 39-40 trials). No other trial-level filtering is applied.

ii.
```python
if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. The instructions state there must be at least 2 trials per session. The AI implemented this check. No velocity or other quality filtering is applied at the trial level.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib data, which contains binary calcium event traces (0/1 values representing the rising phase of calcium transients).

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The AI identified that `trace` contains binary events (0/1) representing preprocessed calcium imaging data. The paper describes the rising phase extraction process.

## 2-b. How is the `neural` data processed?

i. The binary trace data is (1) filtered to remove unregistered neurons (NaN check on first timepoint), then (2) temporally binned into 1-second bins by averaging 30 frames, producing firing rates.

ii.
```python
def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)
...
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. The AI noted that averaging binary events over 30 frames produces firing rates. The 1-second bin size was chosen as a "practical" resolution for the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons not registered on a given day are filtered out by checking if the first timepoint is NaN. Only neurons with a non-NaN value at timepoint 0 are kept. No activity threshold or velocity filtering is applied.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
n_valid = valid_mask.sum()
```

iii. The AI noted that the reference code's decode function applies velocity and activity filters, but chose not to apply them, stating "we include all cells and let the decoder handle it."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. Trials are simply consecutive 1-minute segments starting from the beginning of the recording session. The alignment event is "Start of recording session."

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. There is no stimulus onset or behavioral event to align to in this continuous free-exploration task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 1-second temporal binning (averaging 30 frames at 30 Hz). This produces 60 time bins per 1-minute trial. The metadata reports `time_bin_size: 1000` ms.

ii.
```python
TIME_BIN_SEC = 1.0  # Time bin size in seconds
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. The AI justified this as "more practical" than the reference code's 3-frame bins, and stated 1-second bins "still capture spatial behavior well."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable in the joblib data, which contains environment name strings (e.g., 'square', 'o', 't') for each day. The AI uses a lookup function `get_env_mat()` copied from the reference code to convert environment names to 3x3 binary matrices.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI copied the `get_env_mat()` function from the reference code's `utils.py`, which maps environment names to binary 3x3 matrices indicating which areas are accessible.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary that maps each of the 10 environment types to a 3x3 binary matrix. The matrix is flattened to a 9-element vector. This is static per trial (same for all trials in a session).

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

iii. The environment geometry matrix represents which of the 9 spatial regions in the 3x3 grid are accessible (1) or blocked (0). This is consistent with the reference code's representation.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` variable in the joblib data, which contains 2D (x, y) coordinates of the animal at 30 Hz.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The position data records the animal's tracked location in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is discretized into a 3x3 grid (9 classes). Bin edges are computed from the maximum observed position value (plus a small buffer) divided by 3. Positions are assigned to bins using `floor(position / bin_size)` and clipped to valid range [0, 2].

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

iii. The AI uses the maximum observed position to define bin edges, rather than the known arena size (75 cm). The bin index is computed as `x_bin * 3 + y_bin`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into a 3x3 grid using floor division. The 2D position is converted to a single index: `bin_idx = x_bin * 3 + y_bin`, giving 9 categories (0-8). The bin edges are derived from the data's maximum position value.

ii.
```python
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
```

iii. The AI chose `x*3+y` ordering for the bin index. The reference uses `y*3+x` ordering.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is first binned into the 3x3 grid at the native 30 Hz, then temporally binned into 1-second bins using the mode (most frequent bin within each 1-second window). Both neural and position data undergo the same temporal binning and trial splitting.

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
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
```

iii. Using mode for position binning ensures the most representative position within each 1-second window is selected, matching the temporal resolution of the neural data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons not registered on a given day are identified by NaN at the first timepoint and excluded. Remainder frames at the end of a session that don't fill a complete trial are discarded. Sessions with fewer than 2 trials are skipped. No handling of NaN values within position data is mentioned (though `np.nanmax` is used in binning).

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
...
n_trials = n_timebins // trial_length  # remainder dropped
```

iii. The AI's NaN check only looks at the first timepoint, assuming that if a neuron is NaN at timepoint 0, it's NaN everywhere. This is based on the data structure where unregistered cells have NaN for the entire day.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib data files is the most time-consuming step (~15-27 seconds per animal). The total conversion takes ~153 seconds for all 7 animals. Processing (binning, splitting) is fast by comparison.

ii. N/A (timing is logged but not in a specific code block)

iii. The AI documented timing in the conversion output showing per-animal times.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The mode computation in `bin_position_temporal` uses a Python loop over time bins with `np.unique` for each bin. This could be vectorized using `scipy.stats.mode`.

ii.
```python
for i in range(n_bins):
    values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
    result[i] = values[np.argmax(counts)]
```

iii. The AI noted this in CONVERSION_NOTES.md as an identified inefficiency but did not fix it.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each animal is loaded once, and each day is processed once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporal binning (averaging 30 frames into 1-second bins) adds processing that discards temporal resolution. Additionally, the mode computation for position within each time bin adds complexity. With the reference approach (no rebinning), both of these steps would be unnecessary.

ii. N/A

iii. The AI's choice to temporally rebin introduces additional processing steps that the reference solution avoids.
