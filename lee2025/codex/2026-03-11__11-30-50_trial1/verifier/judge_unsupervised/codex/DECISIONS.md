# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data lazily, one animal file at a time, from `data/`. It finds files whose names match `QLAK-CA1-\d+`, then `joblib.load`s each per-animal file and indexes the top-level animal key. Trials are not loaded from disk directly because the source data have no native trial structure; trials are created later inside `convert_session()`.

ii. 
```python
def get_animals(data_dir: str) -> list[str]:
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and re.fullmatch(r"QLAK-CA1-\d+", name):
            animals.append(name)
    return animals

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

for animal in animals:
    animal_data = load_animal(data_dir, animal)
    ndays = animal_data["trace"].shape[0]
```

iii. In `CONVERSION_NOTES.md`, the agent says the repository’s reference path is to load per-animal joblib files and that it deliberately streams animals one at a time to reduce memory use.

## 1-b. How are the data split into subjects?

i. Subjects are defined by file name: one subject per per-animal file. The subject list is the sorted list of those file names, and each session gets a `subject_idx` pointing back to that list.

ii. 
```python
animals = get_animals(data_dir)

subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}

converted = {
    ...
    "subjects": subjects,
    "subject_idx": [],
    ...
}

converted["subject_idx"].append(subject_to_idx[animal])
```

iii. The notes say session order is “animals in sorted filename order, days in native order within each animal,” so subject identity comes directly from the animal file organization in the reference dataset.

## 1-c. How are the data split into sessions?

i. Sessions are split by day index within each animal file. Each `day_idx` in `animal_data["trace"]` becomes one session.

ii. 
```python
ndays = animal_data["trace"].shape[0]

for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    ...

day = session.day_idx
trace_day = animal_data["trace"][day]
position_day = animal_data["position"][day]
```

iii. The notes say the reference dataset is organized as one 40 minute recording per day, so the agent treated each day as the session boundary.

## 1-d. How are the data split into trials?

i. The source data are continuous sessions, so the agent imposes 1 minute trials. At 30 Hz, that is 1800 frames per trial. It caps each session at nominally 40 minutes, then creates up to 40 consecutive slices. If the recording is a bit short, the last trial can be shorter than 1 minute as long as it has at least 3 frames.

ii. 
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS
NOMINAL_SESSION_SECONDS = 40 * 60
NOMINAL_SESSION_FRAMES = NOMINAL_SESSION_SECONDS * FPS

def get_trial_slices(n_frames_session: int) -> list[tuple[int, int]]:
    usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
    slices = []
    for trial_idx in range(NOMINAL_SESSION_SECONDS // TRIAL_SECONDS):
        start = trial_idx * TRIAL_FRAMES
        end = min(start + TRIAL_FRAMES, usable_frames)
        if end - start >= TEMPORAL_BIN_FRAMES:
            slices.append((start, end))
    return slices
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent justifies this as required trialization because the decoder task requested 1 minute trials even though the source dataset is continuous.

## 1-e. How are trials filtered based on quality controls?

i. The code does not apply any substantive trial-quality filter. The only trial-level exclusion is structural: slices shorter than 3 frames are dropped because they cannot form even one temporal bin.

ii. 
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The notes focus quality control on cell validity and reference decoder settings, but the conversion script does not carry over the reference decoder’s low-velocity frame filtering or activity-threshold cell filtering into the exported trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` field for each day, after removing cells whose session trace is all-NaN/unregistered.

ii. 
```python
trace_day = animal_data["trace"][day]
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The notes explicitly say the released `trace` is already the paper’s binarized rising-event signal and that no `dF/F` recomputation should be done.

## 2-b. How is the `neural` data processed?

i. The code keeps the released binarized event trace, casts it to `float32`, slices it into trials, and averages non-overlapping 3 frame windows to create 100 ms bins. It does not apply the reference decoder’s Gaussian temporal smoothing before pooling.

ii. 
```python
def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

neural_raw = trace_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The notes justify using the released event trace directly and say the 3 frame pooling was chosen to match the reference decoder’s temporal scale. The trajectory shows the agent read the reference `fit_decoder`, which smooths traces with `gaussian_filter1d(..., sigma=temporal_bin_size)` before `AvgPool1d`, but the agent did not implement that part.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells with non-NaN first-frame values are kept. There is no place-cell filtering, no minimum-event filter, and no velocity-dependent frame filter.

ii. 
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The notes argue that the paper retained all registered cells for main analyses and treated NaNs as unregistered cells, but they also note that the reference within-session decoder filters low-velocity frames and low-activity cells internally. The export script omits those latter filters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code aligns each trial to the start of the imposed 1 minute chunk. There is no experimental event marker; the alignment event is simply trial/chunk onset within the continuous session.

ii. 
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
    ...
}
```

iii. The notes say the source recordings are continuous and have no native trial event, so the agent chose chunk start as the only practical alignment point after trialization.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins, produced by averaging every 3 frames from the original 30 Hz stream. This is the only temporal rebinning step in the export script.

ii. 
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0

neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The notes repeatedly justify 100 ms bins by reference to the paper decoder’s `temporal_bin_size=3` frames.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The code derives the decoder input geometry from the raw `blocked` field, not from `envs`.

ii. 
```python
env_label = animal_data["envs"].reshape(-1)[day]
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
...
input_trials.append(open_mask.copy())
```

iii. The trajectory shows the agent initially leaned toward `envs`, then compared `blocked` to `get_env_mat(env)` and found mismatches for shapes like `t`, `l`, `bit donut`, and `glenn`. After that it justified `blocked` as the safer geometry source for the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code converts blocked partition indices into a length 9 binary open-mask vector, with `1=open` and `0=blocked`. If the blocked code is `-1`, all partitions are open. After marking blocked bins, it transposes the 3x3 mask before flattening so the geometry index convention matches the output class convention.

ii. 
```python
def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    return open_mask.reshape(3, 3).T.reshape(-1)
```

iii. The notes and code comment say raw blocked indices are stored in a different 3x3 orientation (“y-major”) than the output class numbering (`x_bin * 3 + y_bin`), so the transpose was added to reconcile the two.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the raw `position` field for each day.

ii. 
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
...
position_raw = position_valid[:, start:end]
```

iii. The notes say this matches the paper/code organization, where `position` stores x-y coordinates aligned to the calcium stream.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. For each trial, the code slices the x-y trajectory with the same frame boundaries used for neural data, averages each non-overlapping 3 frame chunk, computes session-wise x and y maxima over usable frames, and then discretizes the binned x-y coordinates into a 3x3 grid.

ii. 
```python
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)

position_raw = position_valid[:, start:end]
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The notes justify this as adapting the paper’s spatial discretization logic to the requested 3x3 decoder output while keeping trial boundaries aligned to neural data.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each binned x and y coordinate is divided by one third of the session-wise axis maximum, floored to integers, clipped to `0..2`, and combined into a single category `x_bin * 3 + y_bin`, yielding 9 classes.

ii. 
```python
def discretize_position_3x3(position_xy_by_time: np.ndarray, session_max_xy: np.ndarray) -> np.ndarray:
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    denom = np.where(denom <= 0, 1.0, denom)
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```

iii. In the notes, the agent describes this as reusing the reference integer-floor binning rule, but adapted from the paper’s finer grid to the task’s 3x3 categories.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output and neural data use the same session/day, the same trial slices, and the same 3 frame temporal binning, so they are aligned bin-by-bin within each trial.

ii. 
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The notes explicitly list “alignment check” as a planned sanity check and describe neural and behavior as simultaneously acquired at 30 Hz.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/unregistered cells are removed by checking for NaNs in the first frame. Slight session-length irregularities are handled by truncating sessions to 40 minutes and allowing the last trial to be shorter if needed. There is no special imputation or robust handling for missing position samples.

ii. 
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
```

iii. The notes say NaNs encode unregistered cells in the source data and should be treated that way. The trajectory does not show any additional strategy for missing behavioral samples.

## 6-a. What are the most time-consuming steps of the code?

i. The main runtime cost is loading/decompressing the seven large joblib animal files. The next biggest cost is per-session/per-trial temporal binning over long recordings; optional plotting also adds overhead.

ii. 
```python
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    ...
    for day_idx in range(ndays):
        ...
        neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(...)
```

iii. Step 6 of `CONVERSION_NOTES.md` says full-data runtime will be dominated by decompressing the large joblib files and that per-animal streaming was added as a speed optimization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-session trial loop in `convert_session()` could be vectorized by reshaping an entire session into `(n_neurons, n_trials, trial_frames)` and pooling in bulk. Repeating `open_mask.copy()` for every trial is also unnecessary. The plotting loops in `maybe_plot_processing()` are small but also scalar.

ii. 
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    ...
    input_trials.append(open_mask.copy())
```

iii. The notes mention that temporal binning itself was vectorized with `reshape(...).mean(...)`, but the trial-by-trial loop remains.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same geometry vector for every trial in a session, recomputes temporal pooling separately for every trial, and repeats similar bookkeeping for every session. It also stores per-session debug metadata lists in parallel with the main data arrays.

ii. 
```python
for start, end in trial_slices:
    ...
    input_trials.append(open_mask.copy())
    output_trials.append(output_binned)

converted["metadata"]["session_ids"].append(session.session_id)
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
```

iii. The decision follows from the imposed trial structure and from the fact that geometry is static within a session/trial, but the same session-level information is duplicated many times at trial level.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional processing figures, the `debug_info` structure, and the detailed session metadata are not used by downstream decoder training. Repeating a static geometry vector for every trial is also redundant from a learning perspective because it does not vary within a session.

ii. 
```python
if show_processing and first_trial_neural_binned is not None and first_trial_output is not None:
    maybe_plot_processing(...)

debug_info = {
    "env_label": str(env_label),
    "open_mask": open_mask,
    "usable_frames": usable_frames,
    "original_frames": int(trace_valid.shape[1]),
    "trial_slices": trial_slices,
    "session_max_xy": session_max_xy,
}
```

iii. The trajectory and notes describe these as sanity-check and documentation aids rather than required decoder inputs.
