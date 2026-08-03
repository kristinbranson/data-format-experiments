# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a fixed list of seven animal-specific `joblib` files from `data/`, not the per-subject `.mat` files used by the human reference. It then pulls session/day arrays out of each loaded animal dictionary and processes each day as a session.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    n_days = animal_data['trace'].shape[0]
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly says the data are organized as joblib files keyed by animal name and later claims this “matches reference `load_dat` with format=`joblib`.” The trajectory and notes show it chose the higher-level prepacked representation used in the paper code rather than the raw `.mat` files.

## 1-b. How are the data split into subjects (mice)?

i. Each hard-coded animal ID is treated as one subject. The output `subjects` list is just the list of processed animal names.

ii.
```python
subjects = list(animals_to_process)

for a_idx, animal in enumerate(animals_to_process):
    ...
    subject_idx_list.append(a_idx)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI writes “Session = day” and treats each animal as contributing multiple sessions. The subject identifiers are the animal names such as `QLAK-CA1-08`.

## 1-c. How are the data split into sessions?

i. Within each subject/animal, each day index in the `trace`, `position`, and `envs` arrays is treated as a separate session.

ii.
```python
n_days = animal_data['trace'].shape[0]

for day in range(n_days):
    neural_trials, input_trials, output_trials, n_valid = process_session(
        animal_data, day,
        show_processing=do_plot,
        animal_name=animal,
        fig_axes=fig_axes
    )
```

iii. `CONVERSION_NOTES.md` states “Session = day: Each day of recording is one session.” The AI justified this from the dataset structure, where each animal file contains day-indexed arrays.

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 1-minute trials after temporal binning. At 30 Hz with 3-frame bins, each trial becomes 600 time bins. Any remainder shorter than a full trial is dropped implicitly.

ii.
```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE

n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. In Step 5, the AI says “Trial = 1-minute segment” and explicitly computes 1800 frames per minute, then 600 bins per trial after 3-frame temporal binning.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only guard is at the session level: sessions with fewer than 2 generated trials are skipped entirely.

ii.
```python
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    if fig is not None:
        plt.close(fig)
    continue
```

iii. The target format required at least two trials per session, and the AI enforced that requirement directly. In `CONVERSION_NOTES.md` it also wrote that there was “No velocity filtering” and no explicit trial curation in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` output is derived from `animal_data['trace'][day_idx]`.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The notes describe `trace` as binary calcium events with NaNs for unregistered cells and state that no dF/F computation is needed because the data are already preprocessed event traces.

## 2-b. How is the `neural` data processed?

i. The AI filters to “valid” cells, applies Gaussian smoothing along time, then average-pools into 3-frame temporal bins, producing float32 `(n_cells, n_bins)` arrays.

ii.
```python
def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)

...

valid_trace = trace[valid_mask]
binned_trace = temporal_bin_trace(valid_trace)
```

iii. The AI explicitly justified this in Step 5 of `CONVERSION_NOTES.md`: it says temporal binning should match `fit_decoder`, with 3-frame bins and Gaussian smoothing before binning, and later claims this matches the reference decoder exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells whose first frame is not NaN, treating that as the marker for a registered cell on that day. It does not apply place-cell, velocity, or activity-threshold filtering in the conversion script.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]  # (n_valid, n_frames)
n_valid = valid_mask.sum()
```

iii. In `CONVERSION_NOTES.md`, the AI says “Cell filtering: Include only registered cells per session (not NaN). No velocity filtering” and “No place cell filtering.” The rationale was that those later filters were decoder-specific, not conversion-specific.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to an experimental event such as stimulus onset. Instead, it treats the start of the recording session as the alignment point and slices consecutive 1-minute windows from there.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
    ...
}
```

iii. In Step 5, the AI states “Temporal alignment: Trials start at beginning of session recording. Align to session start.” That was its interpretation of how to define an alignment event for artificial 1-minute trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. Yes: the AI rebins the native 30 Hz data by smoothing and then averaging every 3 frames.

ii.
```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms

'metadata': {
    'time_bin_size': TIME_BIN_MS,
    'temporal_bin_frames': TEMPORAL_BIN_SIZE,
}
```

iii. `CONVERSION_NOTES.md` repeatedly states that the bin size should be 3 frames / 100 ms to match the paper’s decoder code (`fit_decoder`).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from `animal_data['envs'][day_idx]`, not from the raw `blocked` field.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
env_input = get_env_input(env_name)  # (9,)
```

iii. In Step 5, the notes explicitly map ``envs[day] -> get_env_mat()`` to the input field and say the input should be a 3x3 geometry matrix derived from the reference `get_env_mat` logic.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps each environment name string to a hard-coded 3x3 binary matrix of accessible vs blocked partitions, flattens it to length 9, and repeats that static vector for every trial in the session.

ii.
```python
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    ...
}

def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    if mat is None:
        raise ValueError(f"Unknown environment: {env_name}")
    return mat.flatten().astype(np.float32)

...

input_trials.append(env_input)  # (9,) static
```

iii. The AI justified this as matching the reference `get_env_mat` environment-geometry representation. `CONVERSION_NOTES.md` says the matrix is flattened and kept static per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `output` position labels are derived from `animal_data['position'][day_idx]`.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The notes describe `position` as x-y position in centimeters over time and map it directly to the decoder output field.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first averages position within each 3-frame temporal bin, then discretizes the binned x-y coordinates into a 3x3 spatial grid.

ii.
```python
def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned

...

binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
```

iii. In Step 5, the AI says the output should be time-varying and use the same temporal binning as the neural data; it presents this as part of matching the reference decoder’s temporal handling while adapting the spatial bins to 3x3.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI divides both x and y by 25 cm, floors to integers, clamps each axis to `0..2`, and combines them as `x_bin * 3 + y_bin` to get labels `0..8`.

ii.
```python
def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. `CONVERSION_NOTES.md` states that position should be converted into 3x3 bins with `floor(pos / 25)` and combined into a single label 0-8.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns position to neural data by applying the same 3-frame temporal binning, then slicing both arrays with the same 600-bin trial boundaries.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)  # (n_valid, n_total_bins)
binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)

n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The AI’s notes say the same temporal bin size should be used for neural and position data, then trials start at session start and use shared indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing cells are handled by dropping cells marked NaN in the first frame. Incomplete trailing timepoints are dropped when forming temporal bins and full trials. Unknown environment names raise an error. Sessions with fewer than two trials are discarded.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])

n_bins = n_frames // bin_size
trimmed = smoothed[:, :n_bins * bin_size]

if len(neural_trials) < 2:
    ...
    continue

if mat is None:
    raise ValueError(f"Unknown environment: {env_name}")
```

iii. The notes say unregistered cells appear as NaN and should be excluded, that trial counts should satisfy the decoder requirement, and that remainder frames are acceptable to drop. The AI did not add more elaborate missing-data repair.

## 6-a. What are the most time-consuming steps of the code?

i. Implicitly, the expensive steps are loading each animal file, Gaussian smoothing / temporal binning for every session, and optional visualization. The script times work at the animal and day level, suggesting those were the performance concerns the AI tracked.

ii.
```python
for a_idx, animal in enumerate(animals_to_process):
    t0 = time.time()
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    for day in range(n_days):
        t1 = time.time()
        ...
        binned_trace = temporal_bin_trace(valid_trace)
        ...
        elapsed = time.time() - t1
```

iii. `CONVERSION_NOTES.md` includes runtime estimates per animal and full conversion, and the code contains explicit timing and optional plotting. That indicates the AI regarded file loading and repeated per-session smoothing/binning as the main costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loop is the per-trial Python loop that slices and appends each 1-minute segment. The nested loop that constructs `output_values_position` is also trivial overhead.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
    neural_trials.append(trial_neural)
    input_trials.append(env_input)
    output_trials.append(trial_output)

for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        output_values_position.append(f"row{i}_col{j}")
```

iii. The AI did not call these loops out explicitly in the notes, but they are direct consequences of its implementation strategy: generate lists of trials in Python after already having contiguous binned arrays.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly extracts and normalizes environment names, creates identical static input vectors for each trial, and repeatedly allocates plotting structures for the first few sessions when `--show-processing` is enabled.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
if isinstance(env_name, np.ndarray):
    env_name = str(env_name.squeeze())

...

input_trials.append(env_input)  # (9,) static

...

if do_plot:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
```

iii. The notes emphasize environment-geometry lookup and processing visualization, but they do not discuss avoiding repeated work. The repetition is visible in the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes optional plotting, extensive timing / summary bookkeeping, and manual duplication of the reference environment-matrix mapping inside the converter instead of directly using the already-available reference helper. None of that affects the final decoder inputs/outputs. It also computes summary counters only for logging and metadata.

ii.
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'georepca1', 'src'))

ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    ...
}

total_sessions = 0
total_trials = 0
total_valid_cell_days = 0
```

iii. The notes mention `--show-processing` figures, timing estimates, and consistency summaries. Those are useful for documentation, but they are not consumed by downstream analyses of the converted dataset.
