# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-format files (extensionless) in the `data/` directory using `joblib.load()`. Each file is a dictionary keyed by animal name, containing arrays for `trace`, `position`, `envs`, `blocked`, etc. The animal names are hardcoded as a list. This differs from the reference, which loads `.mat` files using `h5py`.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()  # (n_days,) string array
```

iii. The AI identified that the data directory contains both `.mat` files and extensionless joblib files with the same data. It chose the joblib format because it was simpler to load and already had the data in numpy-friendly shapes. The CONVERSION_NOTES.md documents the data structure exploration in Step 2.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to a separate file in the data directory and a separate entry in the hardcoded `animals` list. Each file contains all recording sessions for one subject.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
subjects = animals  # All 7 animals are subjects
...
for animal in animals_to_process:
    subject_id = animals.index(animal)
```

iii. The AI identified that each file corresponds to one animal. The subject names are the animal IDs.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject's data file becomes a separate session. The trace array has shape `(n_days, n_cells, n_frames)`, so iterating over the first axis yields individual sessions.

ii.
```python
n_days, n_cells_total, n_frames_total = trace.shape
for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
```

iii. The AI's notes state "Each session is one recording day from one animal" (Step 5). This matches the paper's description of 31 days per animal (21 for one animal).

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into 1-minute non-overlapping trials of 1800 frames (at 30 Hz). Remainder frames that don't fill a complete trial are discarded.

ii.
```python
n_trials = n_frames_total // trial_duration_frames
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The instructions specify "1-minute trials within each session." At 30 Hz, 1 minute = 1800 frames. This yields ~39-40 trials per session (8187 total).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 1-minute segments are included. Only partial trials at the end of sessions are discarded.

ii. N/A (no filtering code)

iii. The AI notes that the original paper had no trial-level curation since each session was treated as one continuous recording. The AI followed this convention.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium event data (0/1) with shape `(n_days, n_cells, n_frames)`.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]   # (n_cells, n_frames)
```

iii. The AI identified that the trace data is already binarized (rising-phase calcium transients), matching the paper's description.

## 2-b. How is the `neural` data processed?

i. The AI filters out unregistered neurons (all-NaN), then applies `np.nan_to_num()` to replace any remaining NaN values with 0, and casts to float32. The reference solution does NOT apply `nan_to_num()`.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
# Replace any remaining NaN with 0
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The AI justified the `nan_to_num` as a safety measure: "Replace any remaining NaN with 0 (shouldn't happen for active cells, but safety)". The reference does not include this step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons whose trace is all-NaN for that day (unregistered via CellReg) are excluded. All other registered neurons are included, with no place cell filtering or event count thresholds.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
n_active = active_mask.sum()
```

iii. The AI noted that the paper "motivated the inclusion of all cells in subsequent analyses" and that no place cell filtering was needed. This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are sequential 1-minute segments from the start of the session. Alignment is to the start of each trial segment.

ii. N/A (no alignment code beyond sequential splitting)

iii. The AI noted there is no stimulus event to align to — the sessions are continuous recordings in open-field environments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz frame rate is preserved (time bin size ~33.33 ms). No temporal rebinning is applied.

ii.
```python
fps = 30
# time_bin_size = 1000.0 / fps = ~33.33 ms
# No rebinning code exists
```

iii. The AI noted that data streams are all synchronous at 30 Hz and no binning is needed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the input from the `envs` variable (environment name strings like 'square', 'o', 't', etc.) combined with the `get_env_mat()` function from the reference code. This differs from the reference solution, which derives the input from the `blocked` variable containing indices of blocked positions.

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
...
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The AI adapted `get_env_mat()` directly from the reference code repository (`georepca1/src/utils.py`), which maps environment names to 3x3 binary matrices. The AI's notes document this decision in Step 5.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a 3x3 binary matrix via `get_env_mat()`, where 1=accessible partition and 0=blocked partition. This is flattened to a 9-element vector. The input is static per trial/session. This is the **inverse** of the reference encoding, which uses 1=blocked and 0=not blocked via `encode_blocked()`.

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
...
env_mat = get_env_mat(env_name).flatten()
trial_input = env_mat.astype(np.float32)
```

iii. The AI justified using `get_env_mat()` because it comes directly from the reference code and encodes "which partitions are accessible." The instructions say the input should represent "which parts of the arena are blocked," but the accessibility encoding conveys equivalent information (just inverted).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, containing 2D (x, y) coordinates of the mouse in the 75x75 cm arena at each frame.

ii.
```python
position = d['position']  # (n_days, 2, n_frames)
pos_day = position[day]   # (2, n_frames)
```

iii. Matches the reference approach.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI discretizes continuous (x, y) positions into a 3x3 grid by computing `floor(pos / bin_size)` with bin_size = 25 cm, clipped to [0, 2]. The reference uses `np.digitize()` with edges from `np.linspace()`.

ii.
```python
def discretize_position_3x3(position, env_size=75.0):
    bin_size = env_size / 3.0
    x = np.clip(position[0], 0, env_size - 1e-10)
    y = np.clip(position[1], 0, env_size - 1e-10)
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    x_bin = np.clip(x_bin, 0, 2)
    y_bin = np.clip(y_bin, 0, 2)
    bin_ids = x_bin * 3 + y_bin
    return bin_ids
```

iii. The AI's approach yields equivalent binning to the reference for interior points. Edge handling differs slightly (the AI clips to `env_size - 1e-10` vs. the reference using `np.digitize` + `np.clip`).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI combines x_bin and y_bin into a single label using `bin_id = x_bin * 3 + y_bin` (x-major / row-major with x as the row). The reference uses `bin_id = y_bin * 3 + x_bin` (y-major). This results in a **different mapping** of spatial locations to bin IDs, though both produce 9 categories (0-8).

ii.
```python
# AI code:
bin_ids = x_bin * 3 + y_bin

# Reference code:
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The AI chose x-major ordering while the reference uses y-major ordering. Both are valid conventions that produce 9 unique spatial bins. The decoder can learn either mapping.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate and are stored with matching time indices. Both are split into trials using the same frame indices, ensuring alignment.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. Frame-by-frame alignment is preserved because both streams share the same time axis.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered neurons (all-NaN traces) are excluded per session. Any remaining NaN values in active neuron traces are replaced with 0 via `np.nan_to_num()`. Partial trials at the end of sessions are discarded. The reference solution only filters all-NaN neurons and does not apply `nan_to_num()`.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
```

iii. The AI described the `nan_to_num` as a safety measure. The reference does not include this, implying active neurons should not have NaN values.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files is the most time-consuming step (~13-23 seconds per animal). Processing each day takes <1 second. Total conversion takes ~215 seconds (~3.6 minutes).

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
# Loading takes 8-23 seconds per animal
```

iii. The AI documented timing in CONVERSION_NOTES.md Step 7 and confirmed total time was well under the 15-minute threshold.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates per-trial with Python indexing rather than using array reshaping. This could be vectorized using `np.reshape` for the neural and output data.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI noted this loop but did not vectorize it, as the number of trials per session (~40) is small and performance was acceptable.

## 6-c. What processing does the code repeat multiple times?

i. The AI loads the entire animal data file for each animal, which includes loading all days' data at once. No redundant processing was identified — each day is processed once.

ii. N/A

iii. The code processes each animal/day/trial once without repetition.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes optional processing plot generation (`plot_processing()`) that is only triggered with `--show-processing`. No other unnecessary processing was identified. The code stores `sessions_env` environment name lists that are not included in the final output dictionary.

ii.
```python
sessions_env.append(env_name)
# env names are tracked but not saved to the output pickle
```

iii. The environment names are used for logging and plotting but not saved in the final data structure.
