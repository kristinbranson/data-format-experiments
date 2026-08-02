# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, loads one `joblib` file per animal from `data/`, indexes into the returned dictionary by animal name, and then iterates over each day/session inside that animal record. Trials are not loaded directly; they are created later inside `process_session`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    n_days = animal_data['trace'].shape[0]
```

iii. In `CONVERSION_NOTES.md`, the agent says the reference loader supports joblib, that each animal has a joblib file keyed by animal name, and that dataset statistics from those files matched the paper exactly. The trajectory and notes repeatedly justify this as the most direct way to recover all 7 mice and 207 sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by animal file name. The subject list is just `ANIMALS`, and each accepted session gets `subject_idx = a_idx` for the current animal.

ii.
```python
subjects = list(animals_to_process)
...
for a_idx, animal in enumerate(animals_to_process):
    ...
    subject_idx_list.append(a_idx)
```

iii. The notes state that “each animal has a joblib file in `data/`” and list seven named mice. The justification is that one file corresponds to one mouse, so file/animal identity defines the subject split.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day within an animal as one session. It iterates over `range(n_days)` and calls `process_session(animal_data, day, ...)` once per day.

ii.
```python
n_days = animal_data['trace'].shape[0]
...
for day in range(n_days):
    neural_trials, input_trials, output_trials, n_valid = process_session(
        animal_data, day,
        show_processing=do_plot,
        animal_name=animal,
        fig_axes=fig_axes
    )
```

iii. `CONVERSION_NOTES.md` explicitly records “Session = day” as a key decision, justified by the data layout (`trace`, `position`, `envs` all indexed by day) and by the paper’s description of one recording session per day.

## 1-d. How are the data split into trials?

i. Sessions are split into non-overlapping 1-minute trials after temporal binning. At 30 Hz and 3-frame bins, each session becomes `n_total_bins`, and each trial is 600 bins (60 seconds). Any leftover partial trial is dropped.

ii.
```python
TEMPORAL_BIN_SIZE = 3
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE
...
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes justify 1-minute windows as required by the task, and explicitly state that a 40-minute session should yield about 39-40 trials after binning. The agent chose to split after 3-frame binning because it believed the decoder should run on 100 ms bins.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only guard is at the session level: sessions with fewer than 2 trials are skipped, and sessions with zero valid cells return empty trial lists.

ii.
```python
if n_valid == 0:
    return [], [], [], 0
...
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    ...
    continue
```

iii. The notes say there is “No explicit trial filtering” in the reference and that the format requirement only needs at least two trials per session. The trajectory reflects this as a decoder-format safeguard rather than a scientific QC rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data comes from `animal_data['trace'][day_idx]`.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The notes describe `trace` as binary calcium-event data, already preprocessed by the original pipeline, and therefore as the correct raw source for neural activity.

## 2-b. How is the `neural` data processed?

i. The agent filters to valid cells, Gaussian-smooths each cell’s trace along time with `sigma=3` frames, then average-pools in non-overlapping 3-frame windows to produce 100 ms bins.

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
binned_trace = temporal_bin_trace(valid_trace)
```

iii. `CONVERSION_NOTES.md` says this was chosen to match the paper’s decoder code (`fit_decoder`) and that the same 3-frame temporal binning should be reused here. The notes repeatedly frame this as matching the reference decoder rather than preserving the raw frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered by a “registered on this day” mask based on whether the first frame is not `NaN`. If no valid cells remain, the session is dropped.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
n_valid = valid_mask.sum()

if n_valid == 0:
    return [], [], [], 0
```

iii. The notes justify this with the claim that non-registered cells have `NaN` traces on days where they are absent. The agent decided to use the first frame as the registration check and not apply additional activity or place-cell filtering in conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align to a stimulus/event. It aligns everything to session start and then carves the recording into consecutive 1-minute windows.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
    ...
}
...
start = t * BINS_PER_TRIAL
end = (t + 1) * BINS_PER_TRIAL
trial_neural = binned_trace[:, start:end]
```

iii. The notes explicitly say “Temporal alignment: Trials start at beginning of session recording. Align to session start.” The justification is that exploration is continuous and there is no trial onset event in the source experiment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. Yes: 3 raw frames at 30 Hz are rebinned into one time bin after Gaussian smoothing.

ii.
```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes justify this as matching the temporal binning used by the reference decoder code and explicitly call out “3 frames at 30 Hz = 100 ms.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from `animal_data['envs'][day_idx]`, not from the raw `blocked` field.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
```

iii. The notes say the input should come from environment identity and cite `get_env_mat` from the reference utilities, arguing that the geometry matrix is the decoder-relevant contextual variable.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent maps each environment name to a hand-coded 3x3 binary occupancy matrix, flattens it to length 9, and reuses that same static vector for every trial in the session.

ii.
```python
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    ...
}
...
def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    if mat is None:
        raise ValueError(f"Unknown environment: {env_name}")
    return mat.flatten().astype(np.float32)
...
env_input = get_env_input(env_name)
input_trials.append(env_input)
```

iii. The notes explicitly state “Input: Environment geometry as 3x3 binary matrix (from `get_env_mat`), flattened to 9 values. Static per trial.” The justification is that this matches the arena partitions used in the paper’s geometric manipulation.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. It is treated as static within a session and repeated once per trial, with no within-trial temporal variation.

ii.
```python
env_input = get_env_input(env_name)  # (9,)
...
input_trials.append(env_input)  # (9,) static
```

iii. The notes say the environment geometry is session-level context that does not change within a recording, so only trial-level replication is needed.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from `animal_data['position'][day_idx]`.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The notes describe this field as the tracked `x, y` position in centimeters over time and therefore the correct raw source for the decoder target.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first averages position over 3-frame windows, then discretizes each binned `(x, y)` point into one of nine 3x3 spatial bins.

ii.
```python
def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned
...
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
```

iii. The notes justify this as matching the temporal binning used for neural activity so that the decoder sees synchronized 100 ms inputs and outputs.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is converted to a bin with `floor(position / 25 cm)` and clipped into `[0, 2]`; the final category is `x_bin * 3 + y_bin`, giving values `0..8`.

ii.
```python
def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. The notes justify the 3x3 grid as task-required coarse spatial decoding and explicitly record the formula `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is averaged into the same 3-frame bins as neural activity and then split into trials using the same start/end indices.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
...
start = t * BINS_PER_TRIAL
end = (t + 1) * BINS_PER_TRIAL
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say neural and behavioral streams should share the same 100 ms bins and then be cut into identical 1-minute windows.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins, produced by Gaussian smoothing plus 3-frame average pooling. Temporal rebinning is therefore central to the conversion.

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per time bin
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
...
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
```

iii. The notes repeatedly justify 100 ms bins as matching the reference decoder implementation.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position are aligned by applying the same 3-frame temporal binning and the same 1-minute trial boundaries. The environment input is static for the whole session and duplicated per trial.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
env_input = get_env_input(env_name)
...
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
input_trials.append(env_input)
```

iii. The notes explicitly say the data streams should share the same 100 ms timing, with geometry acting as trial-level/session-level context.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The code handles missing registrations by removing cells with `NaN` in the first frame, drops sessions with zero valid cells, silently truncates leftover partial bins/trials, skips sessions with fewer than two trials, and raises a hard error for unknown environment names.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
...
if n_valid == 0:
    return [], [], [], 0
...
trimmed = smoothed[:, :n_bins * bin_size]
...
if len(neural_trials) < 2:
    ...
    continue
...
if mat is None:
    raise ValueError(f"Unknown environment: {env_name}")
```

iii. The notes justify NaN-based filtering as inherited from cell registration, and they justify dropped remainder bins/trials as acceptable because sessions vary slightly in frame count. The trajectory shows the strict `ValueError` path was intended as a guard against unexpected environment labels.

## 7-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are loading each animal’s large joblib file, per-session Gaussian smoothing over all cells and frames, and optional Matplotlib plotting when `--show-processing` is enabled.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
...
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
...
fig.savefig(f'processing_{animal}_day{day}.png', dpi=100)
```

iii. The notes include per-animal runtime estimates (~37 s average) and describe the conversion as doing full-session smoothing/binning over large arrays. The trajectory also shows the agent created optional processing plots for early sessions.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` could have been replaced with reshaping/splitting the already binned arrays. The small loops that build `output_values_position` and repeatedly append the same static input are also vectorizable, though less important.

ii.
```python
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    ...
    neural_trials.append(trial_neural)
    input_trials.append(env_input)
    output_trials.append(trial_output)
...
output_values_position = []
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        output_values_position.append(f"row{i}_col{j}")
```

iii. The notes do not explicitly discuss vectorization, but the code structure makes the trial-building loop the clearest avoidable Python-level loop.

## 7-c. What processing does the code repeat multiple times?

i. It normalizes/extracts environment names more than once, reuses the same static environment vector by appending it trial-by-trial, recomputes trial boundaries in Python for every trial, and performs similar per-session bookkeeping/logging on every day.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
env_name = str(animal_data['envs'][day].squeeze())
...
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    input_trials.append(env_input)
```

iii. The notes emphasize detailed sanity checks and session-by-session reporting; the implementation reflects that by repeating conversion and logging steps for every day/session.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional visualization work (`matplotlib` imports, figure creation, saving PNGs) is not part of the serialized dataset. The code also computes progress/timing summaries used only for logging.

ii.
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
...
if do_plot:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
...
fig.savefig(f'processing_{animal}_day{day}.png', dpi=100)
...
elapsed = time.time() - t1
print(f"  Day {day}: {env_name}, {n_valid} cells, "
      f"{len(neural_trials)} trials, {elapsed:.2f}s")
```

iii. The trajectory shows the agent intentionally generated processing-overview figures and extensive logs for validation, but none of that work feeds into `converted_data.pkl`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is the same as in section 6: first-frame-NaN cell filtering, dropping zero-cell sessions, truncating partial bins/trials, skipping sessions with fewer than two trials, and raising an error for unknown environment labels.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
...
if n_valid == 0:
    return [], [], [], 0
...
trimmed = smoothed[:, :n_bins * bin_size]
...
if len(neural_trials) < 2:
    ...
    continue
...
if mat is None:
    raise ValueError(f"Unknown environment: {env_name}")
```

iii. The notes justify these as practical safeguards around registration gaps, variable recording lengths, and malformed metadata.

## 9-a. What are the most time-consuming steps of the code?

i. The same answer as 7-a applies: the expensive parts are large-file loading, Gaussian smoothing/temporal binning across all sessions, and optional plotting.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
...
fig.savefig(f'processing_{animal}_day{day}.png', dpi=100)
```

iii. The notes’ runtime table and implementation details support that these dominate wall-clock time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The same answer as 7-b applies: the per-trial slicing/appending loop is the main candidate for vectorization, with a few smaller bookkeeping loops also avoidable.

ii.
```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    ...
    neural_trials.append(trial_neural)
    input_trials.append(env_input)
    output_trials.append(trial_output)
```

iii. The agent did not justify this explicitly, but the code structure makes it the clearest vectorization opportunity.

## 9-c. What processing does the code repeat multiple times?

i. The same answer as 7-c applies: repeated environment-name normalization, repeated trial-boundary calculations, repeated per-trial appends of a constant input vector, and repeated per-session logging/bookkeeping.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
env_name = str(animal_data['envs'][day].squeeze())
...
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
```

iii. This follows from the implementation pattern the agent chose for session-wise conversion and validation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The same answer as 7-d applies: optional figures and logging/timing work are discarded by downstream analyses because only the pickle payload is consumed.

ii.
```python
if do_plot:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
...
fig.savefig(f'processing_{animal}_day{day}.png', dpi=100)
...
print(f"  {animal} done in {animal_time:.1f}s")
```

iii. The trajectory and notes show these were added for developer validation rather than for the dataset itself.
