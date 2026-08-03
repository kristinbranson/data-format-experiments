# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` from non-`.mat` files in the `data/` directory. Each animal's file is a joblib-serialized dictionary keyed by the animal ID, containing subdictionaries with `trace`, `position`, `envs`, `blocked`, etc. The AI hardcodes the list of 7 animal names in the `ANIMALS` constant.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)
```

iii. The AI noted in CONVERSION_NOTES.md Step 1 that data is loaded with `joblib.load()` from non-`.mat` files, which matches how the reference code's `load_dat` utility works. The reference solution instead uses `h5py` to load `.mat` files. Both formats are available in the data directory.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to a separate file in the data directory. The AI hardcodes the 7 animal names in the `ANIMALS` list and iterates over them. Subject names are the animal IDs (e.g., "QLAK-CA1-08").

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for a_idx, animal in enumerate(animals_to_process):
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
```

iii. The AI identified 7 animals from the data directory, consistent with the paper. The reference solution uses `glob` to find `.mat` files and derives subject names from filenames, which is more flexible.

## 1-c. How are the data split into sessions?

i. Within each animal's data, the `trace` array has shape `(n_sessions, n_neurons, n_timepoints)`. The AI iterates over the first axis to extract each session.

ii.
```python
n_sessions = trace.shape[0]
for day in range(n_sessions):
    tr = trace[day]  # (n_neurons, n_timepoints)
    pos = position[day]  # (2, n_timepoints)
    env_name = envs[day, 0]
```

iii. Each recording day for an animal is treated as a separate session. This matches the reference approach, which also treats each recording session index as a separate session.

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute recording session is split into non-overlapping 60-second trials (1800 frames at 30 Hz). Remainder frames are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. Per the task instructions, trials are defined as 1-minute segments. This matches the reference solution's `split_into_trials` function.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are included. Incomplete final segments are discarded.

ii. N/A (no filtering code)

iii. Neither the reference solution nor the AI apply trial-level quality filtering beyond discarding remainder frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary rising-phase calcium transients with shape `(n_sessions, n_neurons, n_timepoints)`.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
...
tr = trace[day]  # (n_neurons, n_timepoints)
```

iii. The AI correctly identified `trace` as the neural data source, consistent with the reference solution which uses `trace` from the `.mat` files.

## 2-b. How is the `neural` data processed?

i. The AI filters out all-NaN neurons (not recorded in that session), replaces any remaining NaN values with 0, and casts to float32. No additional temporal processing (e.g., smoothing, rebinning) is applied.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]  # (n_active, n_timepoints)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The AI noted in CONVERSION_NOTES.md that traces are already preprocessed binary calcium transients. The `nan_to_num` step is a safety measure. The reference solution does not include the `nan_to_num` step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN (not recorded in that session) are removed. No additional filtering (e.g., cell activity threshold, velocity-based filtering) is applied at the conversion stage.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
```

iii. The AI decided not to apply velocity filtering or cell activity filtering at conversion time, noting that these are applied during decoding in the reference code. This matches the reference solution's approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are artificial 60-second segments starting from the beginning of the recording.

ii. N/A (no alignment code)

iii. There is no stimulus onset or discrete event to align to. The AI set `temporal_alignment_event` to "Start of recording session" in the metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The AI noted that the reference code applies 3-frame temporal binning during decoding (in `fit_decoder`), but chose not to apply it at conversion time.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the input from the `envs` variable (environment name strings like 'square', 'o', 't', etc.) and uses the `get_env_mat()` function copied from the reference code to convert environment names to 3x3 binary matrices representing accessible positions.

ii.
```python
env_name = envs[day, 0]
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI copied `get_env_mat` directly from the reference code's `utils.py`. The reference solution instead uses the `blocked` variable from `.mat` files, encoding which positions are blocked as a one-hot vector.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a predefined 3x3 binary matrix via `get_env_mat()`, then flattened to a 9-element vector. 1 = accessible position, 0 = blocked position. The input is static per trial (same for all trials within a session).

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
trial_input.append(env_mat.astype(np.float32))
```

iii. The AI used the canonical environment geometry representation from the reference code. The reference solution uses one-hot encoding of blocked positions from the `blocked` variable. These encode complementary information (accessible vs blocked positions).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D (x, y) coordinates of the animal in the arena at each timepoint.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
...
pos = position[day]  # (2, n_timepoints)
```

iii. The AI correctly identified the `position` variable as the source, consistent with the reference solution.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes) by dividing each axis into 25 cm bins (75 cm / 3). The bin index is computed as `x_bin * 3 + y_bin`.

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. The AI discretizes each axis into 3 equal bins. The binning per axis produces the same results as the reference (`np.digitize` with edges at [25, 50]). However, the combined bin index uses `x_bin * 3 + y_bin` whereas the reference uses `y_bin * 3 + x_bin`, resulting in a transposed grid mapping.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (0-8) using equal-width spatial bins. Each axis of the 75 cm arena is divided into 3 bins of 25 cm each. Values at boundaries are clipped to valid range [0, 2].

ii.
```python
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. The per-axis binning is equivalent to the reference solution. The combined index ordering differs (see 4-b).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and share the same timepoint indices. Both are split into trials using the same start/end indices, ensuring frame-by-frame alignment.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI correctly aligns neural and position data by using the same time indices for both arrays. This matches the reference approach.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons (not recorded in a given session) are removed. Any remaining NaN values in active neurons are replaced with 0 via `np.nan_to_num`. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. The reference solution only filters all-NaN neurons and does not apply `nan_to_num`. The AI added `nan_to_num` as a safety measure in case any active neurons have sporadic NaN values.

## 6-a. What are the most time-consuming steps of the code?

i. Loading data via `joblib.load()` is the most time-consuming step (9-23 seconds per animal). Processing sessions is relatively fast (~0.3-0.6s per session). Saving the pickle file takes ~40 seconds for the full dataset (19.98 GB).

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
```

iii. From the conversion output, loading times range from 9.2s (QLAK-CA1-51, smallest) to 23.0s (QLAK-CA1-75). Total conversion time was 253.7s (~4.2 min).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially, appending to lists. This could be vectorized using `np.split` or array reshaping. However, the loop is not a performance bottleneck.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The reference solution also uses a similar loop-based approach for trial splitting. The performance impact is minimal since the bottleneck is I/O.

## 6-c. What processing does the code repeat multiple times?

i. The `env_mat.astype(np.float32)` cast is repeated for every trial within a session, though `env_mat` is constant per session. This is wasteful but trivial in impact.

ii.
```python
trial_input.append(env_mat.astype(np.float32))  # repeated for each trial
```

iii. The environment geometry is static per session, so the cast could be done once outside the trial loop.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `session_info` metadata dictionaries for every session. The `nan_to_num` operation is unnecessary if no partial NaN values exist in active neurons. Otherwise, no significant unnecessary processing is performed.

ii.
```python
session_info.append({
    'animal': sess['animal'],
    'env_name': sess['env_name'],
    'n_active': sess['n_active'],
    'n_trials': len(sess['neural']),
})
```

iii. The session_info is stored in metadata and does not affect decoder performance. It adds minor overhead but provides useful documentation.
