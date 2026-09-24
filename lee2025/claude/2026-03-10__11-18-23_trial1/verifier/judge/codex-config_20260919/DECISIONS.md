# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes seven animal IDs and loads one extensionless preprocessed joblib file per animal. It indexes the returned outer dictionary by animal ID, then processes every day in the animal's `trace`, `position`, and `envs` arrays. Full mode uses all seven animals; sample mode uses the first two.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

def load_animal_data(animal):
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]

for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)
```

iii. The agent states that the joblib files are preprocessed Python data and that its loading matches the repository's `load_dat` path through `joblib.load`. It validated totals of seven subjects and 207 sessions against the paper.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is treated as one subject. Sessions produced while processing that animal receive `ANIMALS.index(animal)` as their subject index. The saved subject list is always the complete `ANIMALS` list, even in sample mode.

ii.
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)
    subject_id = ANIMALS.index(animal)
    ...
    subject_idx_list.append(subject_id)

'subjects': ANIMALS,
```

iii. The notes identify seven mouse-specific joblib files and report that their per-animal session counts and total unique-cell count match the paper.

## 1-c. How are the data split into sessions?

i. A day (axis 0 of `trace`) is a session. The agent iterates over all days, extracts that day's trace, position, and environment, and skips a day only if it has no registered cells or ultimately has fewer than two trials.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace = d['trace'][day]
    position = d['position'][day]
    ...
    if n_registered == 0:
        continue
    ...
    if n_trials < 2:
        continue
```

iii. The notes describe 207 recording days and treat each continuous approximately 40-minute day as a session, consistent with the paper's session count.

## 1-d. How are the data split into trials?

i. Sessions are cut at starts spaced every 1,800 frames (60 seconds at 30 Hz). Full chunks are retained, and an end chunk is also retained if it is at least 900 frames (30 seconds), so trials can have unequal lengths. Shorter remainders are dropped.

ii.
```python
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The agent cites the requested one-minute trial definition, but separately documents a policy of keeping the final partial trial when it is at least 30 seconds. It notes final lengths as short as 1,666 frames and reports 40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral or signal-quality filtering. Only partial trials shorter than 30 seconds are removed; a session is removed if fewer than two retained trials remain.

ii.
```python
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
if n_trials < 2:
    continue
```

iii. The notes explicitly say “No trial filtering,” apart from segmentation constraints, because the decoder task did not prescribe velocity filtering. The two-trial check enforces the target format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace` array in the preprocessed joblib object. The notes characterize it as an already extracted, binary calcium-event trace with NaNs for cells not registered that day.

ii.
```python
trace = d['trace'][day]
trace_registered = trace[registered_mask]
```

iii. The agent concluded from the paper and repository that fluorescence processing had already been performed and that the stored trace is the rising-phase binary event representation, so delta-F/F need not be recomputed.

## 2-b. How is the `neural` data processed?

i. The code selects registered-cell rows, slices time into trials, and casts each slice to `float32`. It applies no smoothing, rebinning, normalization, rate-map conversion, or deconvolution.

ii.
```python
trace_registered = trace[registered_mask]
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The notes say the binary traces are already preprocessed and should be used as-is. They also cite the paper's inclusion of all cells rather than restricting the conversion to place cells.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is considered registered if its first time point is not NaN. All such cells are kept; days with zero registered cells are skipped. There is no place-cell, activity-count, reliability, or velocity-based filtering.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
if n_registered == 0:
    continue
trace_registered = trace[registered_mask]
```

iii. The agent says NaNs denote unregistered cells and that the reference paper used all cells for its principal analyses. It consciously omitted the reference position-decoder's movement and event-count filters because they were not specified for this downstream task.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. The artificial alignment event is the start of each minute segment, assigned offset 0; neural slices begin at those fixed frame indices.

ii.
```python
neural_trial = trace_registered[:, start:end].astype(np.float32)
...
'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
'off_start': 0.0,
```

iii. The continuous exploration experiment has no discrete stimulus event relevant to the requested segmentation, so the agent uses trial-segment onset as the alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data remain at the native 30 Hz resolution, or 33.333 ms per frame. No temporal rebinning or resampling is performed.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes identify 30 Hz as the acquisition rate and choose native sampling to preserve the aligned neural and position streams.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the per-day environment-name string in `d['envs']`, not directly from the raw `blocked` indices.

ii.
```python
envs = d['envs'].squeeze()
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()
```

iii. The agent found the repository's `get_env_mat` mapping and chose to reuse it, describing it as an identical reference-code function.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `get_env_mat` maps each of ten names to a binary 3x3 occupancy matrix, where 1 means accessible and 0 means blocked. The matrix is flattened row-major to a static nine-element `float32` vector and reused for every trial of the day.

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    ...

env_mat = get_env_mat(env_name).flatten()
input_trial = env_mat.astype(np.float32)
```

iii. The notes justify a nine-part static encoding as the requested environment geometry and say it follows the reference repository. They explicitly document the polarity as 1=open and 0=blocked.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each day's two-row `position` array, containing x and y coordinates for every frame.

ii.
```python
position = d['position'][day]  # (2, n_frames)
pos_bins = bin_position_3x3(position)
```

iii. The notes describe these as DeepLabCut-derived coordinates in a 75 by 75 cm arena and identify them as the relevant time-varying decoder target.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Both coordinate axes are independently assigned to one of three equal spatial bins. The two indices are combined into one of nine categorical labels, sliced by trial, cast to `int64`, and reshaped to `(1, time)`.

ii.
```python
bin_size = (env_size + buffer) / N_POS_BINS
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin
...
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The agent chose the requested 3x3 discretization, using equal 25 cm regions and a row-major mapping consistent with its environment matrix convention.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. With `env_size=75`, the effective thresholds are approximately 25 and 50 cm on each axis. Flooring determines the category; clipping sends out-of-range values to edge bins. The final label is `x_bin * 3 + y_bin`.

ii.
```python
buffer = 1e-5
bin_size = (env_size + buffer) / N_POS_BINS
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin
```

iii. The small buffer is intended to make arena-edge handling safe. The notes state that the chosen ordering is row=x, column=y and call it similar to the spatial binning in the reference code.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural traces are assumed to share frame indices. The same `start:end` boundaries are applied to the precomputed position labels and registered neural trace.

ii.
```python
neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The agent's exact-match spot checks compared converted neural and position trials to their sources, supporting frame-for-frame alignment at 30 Hz.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered neurons are represented by NaNs and removed using the first frame. A day with none is skipped. Short final fragments under 30 seconds and sessions with fewer than two trials are skipped. Coordinate values beyond arena bounds are clipped. The code has no explicit handling for sporadic NaNs inside a registered trace or position array, unknown environment labels, corrupt files, or mismatched stream lengths.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0:
    continue
...
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
x_bin = np.clip(..., 0, N_POS_BINS - 1)
```

iii. The notes explain the dataset's NaN convention and record edge-case checks for partial trials and the animal with fewer sessions. They report no observed errors requiring repair.

## 6-a. What are the most time-consuming steps of the code?

i. Loading/deserializing the large joblib animal objects, copying/casting each neural trial, building a very large in-memory nested dataset, and pickling the roughly 19 GB result dominate normal conversion. Optional plotting adds figure creation and image output. The code times each day, animal, build, and save stages.

ii.
```python
dat = joblib.load(filepath)
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent measured about 21 seconds per animal and 160 seconds for the full conversion. Its notes emphasize efficient vectorized trace/position operations, while the 19,255 MB output makes data movement and serialization inherently substantial.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The animal and day loops are structurally necessary because files, registered-cell sets, and session metadata differ. The inner trial loop could be replaced for complete chunks by reshape/split operations, with the partial tail handled separately. Repeated list-appending of per-session results could likewise be simplified, though views would need care because the current `astype` creates independent arrays.

ii.
```python
for animal in animals:
    ...
    for day in range(n_days):
        ...
        for start in trial_starts:
            neural_trials.append(neural_trial)
            input_trials.append(input_trial)
            output_trials.append(output_trial)
```

iii. The agent claims there are “no inner loops for trace/position,” meaning binning and cell filtering are vectorized; however, it still uses an inner Python loop for trial extraction. It gives no explicit justification for retaining that loop beyond straightforward partial-trial handling.

## 6-c. What processing does the code repeat multiple times?

i. It casts the identical environment vector to `float32` once per trial, repeatedly copies trial-sized neural slices, and later traverses session lists again to flatten them and derive neuron counts. `ANIMALS.index(animal)` performs a repeated linear lookup. In optional plotting, source positions are revisited after converted outputs already exist.

ii.
```python
for start in trial_starts:
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    input_trial = env_mat.astype(np.float32)
...
for s_idx in range(len(neural)):
    n_neurons = neural[s_idx][0].shape[0]
```

iii. The notes do not flag these repetitions as problems. They prioritize clear conversion and validation, and describe the overall operations as vectorized and efficient.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Normal full conversion computes timing and detailed session metadata used mainly for reporting rather than decoder training. It keeps the input geometry even though the supplied decoder is tasked primarily with predicting position from neural activity, but that input is explicitly required by the target schema and therefore is not truly unnecessary. When `--show-processing` is enabled, it generates heatmaps and position/environment figures that are diagnostic only. The imported `sys` module is unused.

ii.
```python
t_day = time.time()
session_info.append({...})
...
if show_processing and len(all_neural) > 0:
    plot_processing(...)
```

iii. The agent treats plots and metadata as sanity-check and provenance aids. Its notes say plots were reviewed during sample validation, so this optional work is intentionally diagnostic even though downstream training discards it.
