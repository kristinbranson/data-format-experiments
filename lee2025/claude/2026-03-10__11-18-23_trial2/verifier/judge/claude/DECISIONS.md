# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files (one per subject) using `joblib.load()`. Each joblib file contains a dictionary keyed by the animal name, with arrays for `trace`, `position`, `envs`, and `blocked`. A hardcoded list of 7 animal names (`ANIMALS`) is iterated over to process all subjects. This differs from the reference, which loads `.mat` files using `h5py`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(data_dir, animal))
animal_data = dat[animal]
```

iii. The AI noted (CONVERSION_NOTES.md Step 1) that the reference code's `load_dat` function supports both `.mat` (via h5py) and joblib formats. The AI chose joblib because the data directory contained both formats and joblib was directly supported by the reference code. The hardcoded animal list ensures consistent ordering.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The subject name is the animal identifier from the hardcoded `ANIMALS` list.

ii.
```python
for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
```

iii. The AI identified that each file contains all recording sessions for one animal, matching the data organization described in the paper (7 animals total).

## 1-c. How are the data split into sessions?

i. Each recording day within a subject becomes a separate session. The number of days is determined from the `trace` array shape (first dimension). Each day's data is extracted by indexing into the arrays.

ii.
```python
n_days = animal_data['trace'].shape[0]
for day in range(n_days):
    trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
    position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The AI confirmed 207 total sessions (31 days x 6 animals + 21 days for QLAK-CA1-51), matching the paper.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute non-overlapping trials. After temporal binning by 3 frames, each trial is 600 time bins (BINS_PER_TRIAL = 1800 / 3 = 600). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
...
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
```

iii. The instructions specify "1-minute trials within each session." The AI chose to split after temporal binning, yielding 600 bins per trial at 100ms resolution.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped. No other trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI implemented this check. In practice, all sessions have 39-40 trials, so no sessions are actually skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium event data (0/1 values from rising-phase detection of calcium transients, z-scored > 2.5). Shape is `(n_cells, n_frames)` per day.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The AI correctly identified (CONVERSION_NOTES.md Step 1) that the trace data is "already binarized (0/1) - rising phase of calcium transients, z-scored > 2.5" and that "No dF/F needed."

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps from the reference `fit_decoder` function: (1) Gaussian smoothing with sigma=3 frames along the time axis, and (2) average pooling with kernel=3, stride=3 (temporal binning by 3 frames). The result is cast to float32.

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

iii. The AI documented (CONVERSION_NOTES.md Step 5, Decision 10) that the reference `fit_decoder` applies `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` before binning, and replicated this processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are NaN at the first timepoint are excluded (indicating unregistered cells for that day). No velocity filtering or activity threshold is applied.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
```

iii. The AI noted (CONVERSION_NOTES.md Step 5) that cells tracked via CellReg have NaN traces for days they are not registered. It chose to include "only registered cells per session (not NaN)" and decided against velocity filtering ("that's decoder-specific").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is used. Trials begin at the start of the recording session and are defined as consecutive 1-minute segments.

ii. N/A (no explicit alignment code; trials are sequential segments from start of recording)

iii. The AI documented `temporal_alignment_event: 'Start of recording session'` in the metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning from 30Hz native (33.33ms) to 100ms bins (3 frames per bin). This matches the reference `fit_decoder` temporal binning.

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per time bin
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
```

iii. The AI noted (CONVERSION_NOTES.md Step 1) that the reference decoder uses "temporal_bin_size=3" and decided to apply this pre-processing.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` variable, which contains the environment name string for each session (e.g., 'square', 'o', 't'). This is converted to a 3x3 binary matrix using `get_env_mat()` from the reference code.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() ...)
env_input = get_env_input(env_name)  # (9,)
...
def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    return mat.flatten().astype(np.float32)
```

iii. The AI used the reference code's `get_env_mat` function to convert environment names to 3x3 binary matrices, where 1 = accessible and 0 = blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is looked up in a dictionary of 10 predefined 3x3 binary matrices (from the reference code). The matrix is flattened to a 9-element vector. Values are 1 (accessible) or 0 (blocked).

ii.
```python
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    ...
}
def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    return mat.flatten().astype(np.float32)
```

iii. The AI directly reused the `get_env_mat` environment definitions from the reference code.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The input is static per trial (not time-varying). The same 9-element environment vector is used for all trials within a session.

ii.
```python
input_trials.append(env_input)  # (9,) static
```

iii. Environment geometry does not change within a session, so no temporal alignment is needed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the 75x75 cm arena at each timepoint.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The position variable records the animal's location tracked via DeepLabCut.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is first temporally binned (mean of 3 consecutive frames), then discretized into a 3x3 grid. Each axis is divided into 3 equal 25-cm bins. The grid label is `x_bin * 3 + y_bin`.

ii.
```python
def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin

def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned
```

iii. Position is temporally binned to match the neural data's temporal resolution, then discretized into 3x3 grid (9 classes).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using `floor(pos / 25)`, clamped to [0, 2], giving 3 bins per axis and 9 total categories. The combined label is `x_bin * 3 + y_bin` (values 0-8).

ii.
```python
x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
return x_bin * n_bins + y_bin
```

iii. The 25 cm bin size creates a 3x3 grid over the 75 cm arena. `np.clip` handles positions at the boundary (75 cm).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is temporally binned using the same bin size (3 frames) as the neural data, then discretized. Both are split into trials using the same bin indices, ensuring alignment.

ii.
```python
binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
...
trial_output = pos_bins[start:end].reshape(1, -1)
```

iii. By binning position and neural data with the same temporal bin size and splitting at the same trial boundaries, temporal alignment is maintained.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100ms (3 frames at 30Hz). Temporal rebinning is applied: gaussian smoothing (sigma=3) followed by average pooling with stride=3. Each trial contains 600 time bins.

ii.
```python
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
```

iii. The AI replicated the temporal binning from the reference `fit_decoder` function.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural data and position data are both temporally binned from 30Hz to 10Hz using 3-frame bins, then split into 1-minute trials at the same indices. Input is static per trial (environment geometry).

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)  # gaussian smooth + avg pool
binned_pos = bin_position(position)  # mean of 3-frame bins
pos_bins = discretize_position(binned_pos)
# Both split at same trial boundaries
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1)
```

iii. The AI ensures alignment by applying the same temporal binning factor to both neural and position data before splitting into trials.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Unregistered neurons (NaN at first timepoint) are excluded per session. Sessions with fewer than 2 trials are skipped. Remainder frames/bins that don't fill a complete trial are discarded. The AI also noted and fixed an output dtype issue (changed from float32 to int64 for the decoder).

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
...
if len(neural_trials) < 2:
    continue
```

iii. The AI documented all issues and fixes in CONVERSION_NOTES.md Steps 10 and 12.

## 7-a. What are the most time-consuming steps of the code?

i. Loading joblib files is the most time-consuming I/O step. The gaussian smoothing and temporal binning are the most computation-heavy steps per session (~0.3-1.0s per session depending on cell count).

ii. N/A

iii. The AI timed each session and animal (total conversion ~258s for 7 animals, ~37s per animal average).

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop constructs trials via a Python for loop with array slicing. This could potentially be vectorized using reshape operations.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
```

iii. The AI did not identify this as a bottleneck since the loop is simple slicing with minimal overhead.

## 7-c. What processing does the code repeat multiple times?

i. The environment name is extracted twice: once in `convert_dataset` for logging and once in `process_session` for processing. This is minor redundancy.

ii.
```python
# In convert_dataset:
env_name = str(animal_data['envs'][day].squeeze())
# In process_session:
env_name = str(animal_data['envs'][day_idx].item() ...)
```

iii. This duplication has negligible performance impact.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The gaussian smoothing step (sigma=3 frames) is applied to the binary trace data before average pooling. Whether this smoothing is necessary or beneficial for the downstream decoder is debatable. Additionally, the temporal binning itself may be unnecessary if the downstream decoder could operate on native 30Hz data.

ii.
```python
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
```

iii. The AI included this to match the reference `fit_decoder` preprocessing, though the human reference chose not to pre-apply these transforms.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6 above. Unregistered neurons (NaN) are filtered out per session. Remainder frames are discarded. Sessions with < 2 trials are skipped. Output dtype was corrected from float32 to int64.

ii. See 6 above.

iii. See 6 above.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Loading joblib files and gaussian smoothing + temporal binning are the main bottlenecks. Total conversion time: ~258s for 7 animals.

ii. N/A

iii. The AI documented timing per animal in the conversion output.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The trial-splitting loop is the main candidate for vectorization.

ii. See 7-b.

iii. See 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. Environment name extraction is done twice per session.

ii. See 7-c.

iii. See 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. The gaussian smoothing and temporal binning may be unnecessary if the decoder operates on raw data. Additionally, the code generates processing visualization plots when `--show-processing` is used, which are not needed for the final output.

ii. See 7-d.

iii. See 7-d.
