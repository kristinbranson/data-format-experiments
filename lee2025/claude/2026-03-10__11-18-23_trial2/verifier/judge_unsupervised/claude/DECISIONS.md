# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from individual joblib files for each of the 7 animals. Each file is loaded with `joblib.load()`, which returns a dictionary keyed by the animal name. The data for each animal includes `trace` (neural), `position`, `envs`, and `blocked` fields. The 7 animal names are hardcoded in the `ANIMALS` list. The code iterates through each animal, loads its file, then iterates through all days (sessions) within that animal to process trials.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    n_days = animal_data['trace'].shape[0]
    for day in range(n_days):
        neural_trials, input_trials, output_trials, n_valid = process_session(
            animal_data, day, ...)
```

iii. The AI identified the `load_dat` function in the reference code (`utils.py:61`) which loads animal data from joblib format. The CONVERSION_NOTES confirm the data structure: `trace` is (n_days, n_cells, n_frames), `position` is (n_days, 2, n_frames), `envs` is (n_days, 1).

## 1-b. How are the data split into subjects?

i. Each animal (subject) corresponds to one joblib file. The AI iterates through the `ANIMALS` list, loading one file per subject. The subject index (`a_idx`) is tracked and appended to `subject_idx_list` for each session belonging to that animal.

ii.
```python
for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    ...
    subject_idx_list.append(a_idx)

data = {
    'subjects': subjects,
    'subject_idx': np.array(subject_idx_list, dtype=np.int64),
    ...
}
```

iii. The AI documented 7 subjects in the CONVERSION_NOTES, matching the paper's statement of 7 animals. Subject names are the animal identifiers (e.g., "QLAK-CA1-08").

## 1-c. How are the data split into sessions?

i. Each day of recording for each animal constitutes one session. The AI iterates through `n_days` (from `animal_data['trace'].shape[0]`) for each animal. Sessions with fewer than 2 valid trials are skipped. The total number of sessions is 207, matching the reference paper.

ii.
```python
n_days = animal_data['trace'].shape[0]
for day in range(n_days):
    ...
    if len(neural_trials) < 2:
        print(f"  Day {day}: skipped (< 2 trials)")
        continue
    all_neural.append(neural_trials)
    ...
    total_sessions += 1
```

iii. The AI noted that each animal has 31 sessions (6 animals) or 21 sessions (QLAK-CA1-51), totaling 207 sessions, consistent with the reference paper's "207 sessions."

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute trials. At 30 Hz with temporal bin size of 3 frames, each trial contains 600 time bins (1800 frames / 3 = 600 bins). The total number of bins is computed, and trials are extracted sequentially. Remainder frames at the end of a session that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS  # 1800 frames
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 bins

n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI documented that sessions are 39-40 trials each (71866 frames = 39 trials, 72219 frames = 40 trials), consistent with the 40-minute session duration stated in the paper. The instruction specified "1-minute trials."

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filtering is that sessions producing fewer than 2 trials are skipped entirely. There is no per-trial quality filtering based on behavioral criteria (e.g., velocity thresholds). The reference code applies velocity filtering at the timepoint level during decoding, but the AI chose not to apply this at the data conversion stage.

ii.
```python
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. The AI's CONVERSION_NOTES state: "No explicit trial filtering in the reference (entire sessions used)" and "Velocity filter: timepoints with smoothed speed > 5 cm/s included for decoding." The AI treated velocity filtering as decoder-specific rather than data curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `trace` field of each animal's data. This is a binary (0/1) array representing the rising phase of calcium transients, already preprocessed (z-scored > 2.5 threshold applied to the derivative of the calcium signal, Gaussian smoothed).

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The AI noted: "Neural data: `trace` is already binarized (0/1) - rising phase of calcium transients, z-scored > 2.5" and "No dF/F needed - data is already preprocessed binary events."

## 2-b. How is the `neural` data processed?

i. The neural data processing involves two steps matching the reference `fit_decoder` function: (1) Gaussian smoothing along the time axis with sigma=3 frames, and (2) average pooling with kernel size 3 and stride 3 (non-overlapping bins). This converts binary events to smoothed firing rate estimates at 100ms resolution.

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

iii. The AI documented that this matches the reference `fit_decoder` function which applies `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` before average pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons based on registration status only: cells with NaN values at the first time point are excluded (indicating the cell was not registered/detected on that day). No additional quality controls are applied (no velocity-based cell activity threshold, no place cell filtering).

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
n_valid = valid_mask.sum()
if n_valid == 0:
    return [], [], [], 0
```

iii. The AI noted: "For decoding: cells with >5 events during moving periods included (per-session)" as a reference code behavior but stated "No place cell filtering: Reference decoding uses all registered cells, not just place cells." The AI chose to include all registered cells without the activity threshold filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Trials are sequentially extracted from the beginning of each session's recording. The first trial starts at frame 0 (time bin 0) of the session.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
```

iii. The AI documented in metadata: `'temporal_alignment_event': 'Start of recording session'` and `'off_start': 0.0`. The CONVERSION_NOTES state: "Trials start at beginning of session recording. Reference processes full session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100ms (3 frames at 30 Hz). Temporal rebinning is applied: the original 30 Hz data (33.3ms per frame) is binned by a factor of 3 (Gaussian smooth then average pool), yielding 10 Hz (100ms bins).

ii.
```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
```

iii. The AI cited the reference `fit_decoder` code which uses `temporal_bin_size=3` and confirmed this yields 100ms time bins. This is recorded in metadata as `'time_bin_size': 100.0`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` field in the animal data, which contains a string name for each day's environment (e.g., "square", "o", "t", "u", etc.).

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
```

iii. The AI identified the `get_env_mat` function in the reference code (`utils.py:215`) which converts environment name strings to 3x3 binary matrices.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name string is mapped to a 3x3 binary matrix using a hardcoded dictionary (`ENV_MATRICES`), matching the reference code's `get_env_mat` function. The matrix is then flattened to a 9-element vector. Each element is 1 (accessible) or 0 (blocked).

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

iii. The AI stated: "Environment geometry as 3x3 binary matrix (from `get_env_mat`), flattened to 9 values. Static per trial."

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is static per trial (same geometry for the entire session). There is no temporal alignment needed since the input is not time-varying. The same 9-element vector is assigned to every trial within a session.

ii.
```python
env_input = get_env_input(env_name)  # (9,)
...
input_trials.append(env_input)  # (9,) static
```

iii. The AI documented the input shape as `(9,)` (static per trial), not `(9, n_timepoints)`, meaning it is constant across all time bins within a trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` field in the animal data, which contains x,y coordinates in centimeters at 30 Hz.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The AI noted that position data comes from "DeepLabCut head tracking" and ranges from 0 to 75 cm in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position processing involves: (1) temporal binning by averaging every 3 frames (matching neural temporal binning), then (2) discretization into a 3x3 spatial grid. The continuous x,y position is converted to bin indices.

ii.
```python
def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned

binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
```

iii. The AI documented this as matching the reference temporal binning approach for behavioral data.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Continuous x,y position (0-75 cm) is discretized into a 3x3 grid with 25 cm bins. X and y are independently binned using `floor(pos / 25)`, clamped to [0, 2]. The combined bin index is `x_bin * 3 + y_bin`, yielding 9 categories (0-8).

ii.
```python
def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. The instructions specified "Mouse position discretized into 3 x 3 = 9 spatial bins." The AI used 25 cm bins (75 cm / 3).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is temporally binned using the same bin size (3 frames) as neural data, then discretized. Both are split into the same trial boundaries, ensuring time-bin-by-time-bin alignment.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)  # (n_valid, n_total_bins)
binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
# Both split with same start:end indices
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI ensures temporal alignment by applying the same binning scheme and trial boundaries to both neural and position data.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Same as 2-e. The temporal resolution is 100ms. Rebinning is applied: 3 frames at 30Hz are averaged into one bin. This is documented in metadata as `time_bin_size: 100.0` ms.

ii.
```python
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
```

iii. The AI cited the reference code's `temporal_bin_size=3` parameter.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output data are both temporally binned from the same raw 30 Hz data using the same bin boundaries (every 3 frames). Input (environment geometry) is static per trial and requires no temporal alignment. All three are split into trials using the same time bin indices.

ii.
```python
# All computed from same session data, split with same indices:
trial_neural = binned_trace[:, start:end]   # (n_neurons, 600)
trial_output = pos_bins[start:end]          # (600,)
input_trials.append(env_input)              # (9,) static
```

iii. The AI verified alignment through processing plots (`--show-processing`) and spot-checks documented in CONVERSION_NOTES Step 10.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing data is handled in several ways: (1) Unregistered cells (NaN in trace) are excluded per session via `~np.isnan(trace[:, 0])`. (2) Sessions with 0 valid cells return empty lists. (3) Sessions with <2 trials are skipped. (4) Environment name extraction handles both scalar and array types. (5) Different frame counts across animals (71866-72219) are handled naturally by the binning and trial-splitting logic. (6) Position values at arena boundaries (exactly 75 cm) are clipped to the valid bin range.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
if n_valid == 0:
    return [], [], [], 0

if len(neural_trials) < 2:
    continue

env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])

x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
```

iii. The AI documented edge case handling in CONVERSION_NOTES Step 10 Check 5.

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib files for each animal (I/O bound). The second most expensive step is the Gaussian smoothing and temporal binning of neural traces, which operates on large (n_cells x n_frames) matrices. The full conversion takes ~258 seconds across all 7 animals.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))  # I/O heavy
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)  # compute heavy
```

iii. The AI logged per-animal timing: the largest animals (CA1-50, CA1-75) take ~45s each. The AI estimated full conversion at ~4.3 minutes, well under the 15-minute limit.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop (iterating `n_trials` times to slice arrays) could potentially be vectorized using `np.split` or array reshaping. However, this loop performs simple slicing operations that are already efficient. The main per-session processing is already vectorized (Gaussian smoothing and binning operate on full arrays).

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI noted in Step 6 that it aimed to "write efficient code: vectorize loops, avoid unnecessary file I/O." The core processing (smoothing, binning, discretization) is vectorized using numpy/scipy operations.

## 7-c. What processing does the code repeat multiple times?

i. The environment name extraction (`str(animal_data['envs'][day])`) is performed twice per session: once in `convert_dataset` (line 265) and once inside `process_session` (lines 130-133). This is redundant but trivial in terms of performance.

ii.
```python
# In convert_dataset:
env_name = str(animal_data['envs'][day].squeeze())

# In process_session:
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
```

iii. No explicit justification was given for this duplication. It appears to be an oversight rather than a deliberate decision.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does not perform significant unnecessary processing. The Gaussian smoothing is applied to the entire session before splitting into trials, which means a small amount of smoothing at trial boundaries uses data from adjacent trials. The remainder frames at the end of sessions (after the last complete trial) are processed through smoothing and binning but then discarded. However, these are minor and unavoidable.

ii.
```python
# Entire session is smoothed and binned first:
binned_trace = temporal_bin_trace(valid_trace)  # processes all frames
# Then only complete trials are used:
n_trials = n_total_bins // BINS_PER_TRIAL  # remainder discarded
```

iii. The AI did not discuss unnecessary processing explicitly. The approach of processing the entire session before splitting is standard and avoids edge effects at trial boundaries.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6. NaN-valued cells (unregistered) are excluded per session. Different frame counts per animal are handled by the floor-division trial splitting. Position boundary values are clipped. Environment name parsing handles multiple data types (scalar, array, numpy types).

ii. See code snippets in question 6.

iii. The AI documented in CONVERSION_NOTES Step 10 that edge cases were checked: "Sessions with different frame counts handled: 71866/3=23955 bins -> 39 trials; 72219/3=24073 bins -> 40 trials" and "Position at exactly 75cm -> clipped to bin 2."

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. Loading joblib files is the primary I/O bottleneck. Gaussian smoothing of large neural trace matrices is the primary compute bottleneck. Per-animal processing ranges from 17.8s (CA1-51, smallest) to 45.8s (CA1-75, largest).

ii. See code snippets in 7-a.

iii. The AI tracked timing per animal and estimated total conversion time. The actual full conversion took 258.8 seconds.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The trial-splitting loop is the main remaining loop that could theoretically be vectorized, but it performs simple array slicing that is already efficient. The per-day loop within each animal could potentially be parallelized (e.g., using joblib.Parallel), but the AI chose sequential processing.

ii. See code snippets in 7-b.

iii. The AI noted that the code runs well under the 15-minute limit and did not pursue further optimization.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. Environment name extraction is done twice per session. Additionally, the `del dat` at line 302 explicitly frees memory after each animal, which is good practice but means data cannot be reused if re-processing is needed.

ii.
```python
del dat  # Free memory after each animal
```

iii. No explicit justification given for the duplication.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. The smoothing of remainder frames (beyond the last complete trial) and the Gaussian smoothing across trial boundaries are the only processing that generates data later discarded. The code also loads the entire animal data structure including fields like `maps`, `SFPs`, and `centroids` that are not used in the conversion, which wastes memory during loading.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))  # loads ALL fields
animal_data = dat[animal]  # includes maps, SFPs, centroids - unused
```

iii. The AI did not discuss selective field loading, though joblib does not easily support partial loading of pickle-serialized objects.
