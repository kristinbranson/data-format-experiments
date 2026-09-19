# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-processed joblib files (one per subject) in the `data/` directory. Each joblib file contains a dictionary keyed by animal name, with arrays for `trace`, `position`, `envs`, and `blocked`. The AI uses `joblib.load()` and accesses `dat[animal]` to get the per-animal data dictionary containing 3D arrays `(n_days, n_cells, n_frames)` for trace, `(n_days, 2, n_frames)` for position, etc.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(data_dir, animal))
animal_data = dat[animal]
```

iii. The AI identified that the data directory contains both `.mat` files and joblib files. The AI chose joblib format because the reference code's `load_dat` function supports both formats and joblib provides pre-structured 3D arrays, avoiding the complexity of dereferencing HDF5 object references required by the `.mat` files.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The AI hardcodes the list of 7 animal names and iterates over them. Each animal's data is loaded separately.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = list(animals_to_process)
for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
```

iii. Each file contains all recording sessions for one animal. The animal name is used as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days (sessions). The AI iterates over the day index dimension of the 3D arrays. Each day becomes a separate session in the output.

ii.
```python
n_days = animal_data['trace'].shape[0]
for day in range(n_days):
    trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
    position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The `trace` array has shape `(n_days, n_cells, n_frames)`, so indexing the first dimension gives one recording session per day.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute non-overlapping trials. After temporal binning (3 frames → 1 bin), each trial has 600 time bins (BINS_PER_TRIAL = 1800 frames / 3). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
...
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]  # (n_valid, 600)
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)  # (1, 600)
```

iii. Per the task instructions, trials are defined as 1-minute segments of the continuous recording.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped entirely. No individual trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI enforces this requirement by skipping sessions that produce fewer than 2 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium event traces (0/1 values indicating rising phase of calcium transients, z-scored > 2.5) with shape `(n_cells, n_frames)` per session.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The `trace` variable contains pre-processed binary neural events stored by the original authors.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps matching the reference code's `fit_decoder` function: (1) Gaussian smoothing along the time axis with sigma=3 frames using `scipy.ndimage.gaussian_filter1d`, and (2) average pooling with a kernel/stride of 3 frames, reducing temporal resolution from 30 Hz to 10 Hz (100 ms bins).

ii.
```python
def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    # Gaussian smooth along time axis
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    # Average pool: reshape and mean
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)
```

iii. The AI explicitly followed the reference code's `fit_decoder` function which applies `gaussian_filter1d(traces, sigma=temporal_bin_size)` followed by average pooling with `temporal_bin_size=3`. This was documented as a key decision in CONVERSION_NOTES.md Step 5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are NaN at the first timepoint (indicating they were not registered/recorded in that session) are excluded. Only neurons with valid data at frame 0 are kept.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]  # (n_valid, n_frames)
n_valid = valid_mask.sum()
```

iii. The `trace` array contains NaN for neurons not registered in a given session. Checking the first frame is sufficient because NaN status is consistent across all frames for a given neuron-session pair.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. Trials start at the beginning of the recording session and are contiguous 1-minute segments.

ii.
```python
# Temporal alignment: Trials start at beginning of recording session
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),
```

iii. There is no stimulus onset or behavioral event to align to — the recording is a continuous exploration session. Trials are aligned to the start of the recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning from the native 30 Hz (33.33 ms) to 10 Hz (100 ms), using 3-frame bins matching the reference code's `fit_decoder` function. Each trial has 600 time bins (instead of 1800 at native resolution).

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per time bin
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
```

iii. The AI matched the temporal binning used in the reference paper's decoder code (`fit_decoder` with `temporal_bin_size=3`), reasoning that the reference authors chose this bin size for optimal decoding performance.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable in the data (which stores the environment name string per session, e.g., 'square', 'o', 't'), combined with a hardcoded lookup table (`ENV_MATRICES`) that maps environment names to 3x3 binary matrices representing which grid cells are accessible.

ii.
```python
env_name = str(animal_data['envs'][day_idx].squeeze())
...
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
    ...
}
```

iii. The AI used the reference code's `get_env_mat` function which converts environment names to 3x3 binary accessibility matrices. This captures which positions the mouse can visit.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in the `ENV_MATRICES` dictionary to get a 3x3 binary matrix (1=accessible, 0=blocked). This matrix is flattened to a 9-element float32 vector. The input is static (same for all timepoints and trials within a session).

ii.
```python
def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    return mat.flatten().astype(np.float32)
...
env_input = get_env_input(env_name)  # (9,)
input_trials.append(env_input)  # static per trial
```

iii. The AI used the reference code's approach for representing environment geometry. The 9-element vector encodes the full spatial layout of the arena.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the 75x75 cm arena at each timepoint.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The `position` variable records the animal's tracked location at each frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first temporally binned by averaging within 3-frame windows (matching the neural temporal binning), then discretized into a 3x3 grid (9 classes). Each axis is divided into 3 bins of 25 cm each using `floor(pos / 25)`, clamped to [0, 2]. The combined bin index is `x_bin * 3 + y_bin`.

ii.
```python
def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned

def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. Position is temporally binned to match the neural data's temporal resolution before discretization, ensuring temporal alignment between neural and output data.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). Each axis of the 75 cm arena is divided into 3 equal bins of 25 cm using `floor(pos / 25)`, clipped to [0, 2]. The combined label is `x_bin * 3 + y_bin`, producing values 0-8.

ii.
```python
SPATIAL_BIN_SIZE = ARENA_SIZE_CM / N_SPATIAL_BINS  # 25 cm per bin
...
x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
return x_bin * n_bins + y_bin
```

iii. A 3x3 grid produces 9 categories as specified in the instructions. Clipping ensures positions at boundaries (e.g., exactly 75 cm) map to valid bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is temporally binned using the same 3-frame averaging window as the neural data, ensuring both have the same number of time bins. Both are then split into trials using the same bin indices.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)  # (n_valid, n_total_bins)
binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
# Both split with same indices
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1)
```

iii. Both neural and position data undergo the same temporal binning before trial splitting, maintaining frame-for-frame alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons not registered in a session (NaN at first frame) are excluded from that session. Sessions with fewer than 2 trials are skipped. Remainder frames/bins that don't fill a complete trial are discarded. Sessions with zero valid cells are skipped.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
if n_valid == 0:
    return [], [], [], 0
...
if len(neural_trials) < 2:
    continue
```

iii. NaN filtering removes unregistered neurons. The < 2 trials check ensures decoder training viability per the instructions. Zero-valid-cell sessions are gracefully handled.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib files (I/O bound) and the Gaussian smoothing + temporal binning of the trace data per session. The AI reported ~37 seconds per animal on average and ~4.3 minutes for all 7 animals.

ii. N/A (timing from conversion output)

iii. The AI timed its code and documented timing estimates. The total conversion time of ~4.3 minutes was well within the 15-minute limit.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial splitting loop (iterating over trials to slice arrays) could potentially be replaced with a reshape operation if all trials had the same size. The loop over days within each animal is inherently sequential due to variable session lengths.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
```

iii. The AI's code is reasonably efficient. The trial splitting loop is lightweight (just slicing), and the heavy computation (Gaussian smoothing, binning) is vectorized using numpy/scipy operations across all cells and timepoints.

## 6-c. What processing does the code repeat multiple times?

i. The environment name is extracted twice for each session — once in `convert_dataset` (for logging) and once inside `process_session` (for computing the environment input). This is minor redundancy.

ii.
```python
# In convert_dataset:
env_name = str(animal_data['envs'][day].squeeze())
# In process_session:
env_name = str(animal_data['envs'][day_idx].item() ...)
```

iii. This duplication has negligible performance impact as it's just a string lookup.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The position data is first temporally binned (averaged over 3-frame windows) before discretization. Since position is ultimately discretized into coarse 25 cm bins, the averaging step has minimal effect on the final categorical output — discretizing at native resolution would yield nearly identical results. The Gaussian smoothing of binary neural traces may also be considered unnecessary preprocessing if the downstream decoder handles raw binned counts effectively.

ii.
```python
binned_pos = bin_position(position)  # average position over 3-frame windows
pos_bins = discretize_position(binned_pos)  # then discretize
```

iii. The AI applied temporal binning to position to maintain consistency with the neural data processing, but the information loss from averaging before discretization is minimal given the coarse spatial bins.
