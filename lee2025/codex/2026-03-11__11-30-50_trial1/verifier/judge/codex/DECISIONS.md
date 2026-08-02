# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code loads per-animal joblib files from `data/`, not the `.mat` files. It enumerates filenames matching `QLAK-CA1-\d+`, loads each animal dictionary with `joblib.load`, then iterates over the day axis and later slices each session into 1-minute trials.

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

animals = get_animals(data_dir)
...
animal_data = load_animal(data_dir, animal)
ndays = animal_data["trace"].shape[0]
```

iii. The AI justified this in its notes and trajectory as following the reference repository’s main analysis path, which loads per-animal joblib files. It also noted that joblib loading was expensive but gave exact raw-session and valid-cell counts.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level per-animal files. Each `QLAK-CA1-*` file becomes one subject, and `subjects` is just that sorted filename list.

ii.
```python
animals = get_animals(data_dir)

subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
```

iii. The AI’s notes say each per-animal file contains all sessions for one mouse, so filename-level splitting was treated as the natural subject definition.

## 1-c. How are the data split into sessions?

i. Sessions are split by day within each animal file. The code iterates over `animal_data["trace"].shape[0]`, treating each day as one recording session.

ii.
```python
ndays = animal_data["trace"].shape[0]
...
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    ...
    neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(
        session=session,
        animal_data=animal_data,
        show_processing=do_plot,
    )
```

iii. The AI justified this from the paper/code interpretation that each day is a continuous single-session recording.

## 1-d. How are the data split into trials?

i. The AI imposes artificial trials by cutting each session into consecutive 60-second chunks. It always loops over the nominal 40-minute session, truncates anything beyond 40 minutes, and keeps a final short chunk if it has at least 3 frames.

ii.
```python
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

iii. The AI’s notes explicitly justify a fixed 40 one-minute trialization to match the nominal 40-minute sessions, while minimizing data loss from sessions slightly shorter or longer than 40 minutes.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter such as motion or occupancy checks. The only implicit filtering is that a chunk must contain at least 3 frames to survive temporal binning.

ii.
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. No separate trial-QC justification was given. The notes frame this as a practical constraint from 3-frame temporal pooling rather than a scientific quality-control step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-day `trace` array in each animal dictionary.

ii.
```python
trace_day = animal_data["trace"][day]
```

iii. The AI justified this by noting that the released `trace` field is already the binarized rising-phase calcium-event signal used in the paper’s later analyses.

## 2-b. How is the `neural` data processed?

i. The code keeps only session-valid cells, casts to `float32`, and temporally rebins each trial by non-overlapping 3-frame averaging to 100 ms bins.

ii.
```python
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
...
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    neural_trials.append(neural_binned)
```

iii. The notes and trajectory justify this as matching the paper decoder’s 3-frame temporal pooling and using the released binary event traces directly without recomputing fluorescence-derived signals.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code removes cells whose first timepoint is `NaN`, treating those as session-invalid registrations. It does not apply place-cell, velocity, or activity-threshold filtering during conversion.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The notes justify keeping all cells except unregistered `NaN` entries because the paper’s later analyses reportedly retained all registered cells, and the AI chose not to reproduce decoder-internal activity/velocity filters in the exported dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no behavioral event alignment. Trials are aligned to the start of each consecutive 1-minute chunk, and metadata records that chunk start as the alignment event.

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

iii. The AI justified this because the source recordings are continuous free exploration sessions with no natural per-trial event, so chunk start was used as the only practical alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins produced by averaging each non-overlapping 3-frame block from the native 30 Hz data.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The AI repeatedly justified this in notes and trajectory as matching the paper decoder’s temporal pooling and making the validator tractable.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the raw per-day `blocked` field, not from `envs`.

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. The notes and trajectory say the AI initially considered `envs`, then rejected it because raw `blocked` preserved the actual blocked partitions more faithfully for decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code starts from an all-open 9-vector, sets blocked partitions to `0`, leaves open partitions at `1`, and then transposes the 3x3 layout so the geometry indexing matches the chosen position-class ordering.

ii.
```python
def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    return open_mask.reshape(3, 3).T.reshape(-1)
```

iii. The trajectory records an explicit bug fix here: the AI found that the geometry vector needed a transpose to line up with its output indexing, reducing blocked-bin occupancy from about 17.1% to about 0.004%.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry input is static within a session. The code copies the same 9-element vector into every trial for that session.

ii.
```python
for start, end in trial_slices:
    ...
    input_trials.append(open_mask.copy())
```

iii. The AI justified this because geometry does not vary within a recording session or within the imposed 1-minute chunks.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the per-day `position` array.

ii.
```python
position_day = animal_data["position"][day]
```

iii. The notes describe this as the paper’s aligned x-y behavioral stream.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code temporally bins x-y position by 3-frame averaging, computes session-wise maxima over usable frames, divides each axis into 3 bins using `(session_max + buffer) / 3`, floors and clips the result, and then converts each bin pair into one 0-8 class.

ii.
```python
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
...
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)

def discretize_position_3x3(position_xy_by_time: np.ndarray, session_max_xy: np.ndarray) -> np.ndarray:
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```

iii. The AI justified this as adapting the paper decoder’s session-wise max-based spatial discretization rule to the required 3x3 output grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each averaged x and y coordinate is thresholded by session-specific tercile-width bins derived from the session maxima, then encoded with `class = x_bin * 3 + y_bin`.

ii.
```python
denom = (session_max_xy + BUFFER) / POSITION_BINS
binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
binned = np.clip(binned, 0, POSITION_BINS - 1)
classes = binned[0] * POSITION_BINS + binned[1]
```

iii. The notes justify this as preserving the reference decoder’s coordinate handling while collapsing from finer spatial bins to 3x3 classes.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data use the same per-session trial slices, and both are rebinned with the same 3-frame averaging before output classes are computed.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The AI justified this as maintaining alignment by applying identical chunk boundaries and identical temporal pooling to both streams.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset uses 100 ms bins via 3-frame average pooling; yes, temporal rebinning is applied to both neural and position data.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
...
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The AI explicitly justified 100 ms bins in the notes and trajectory as a decoder-inspired choice.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position are sliced with the same 1-minute trial boundaries and pooled with the same 3-frame windows. The geometry input is static per trial and copied unchanged across all trials in a session.

ii.
```python
trial_slices = get_trial_slices(trace_valid.shape[1])
...
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    input_trials.append(open_mask.copy())
```

iii. The justification given was that common trial slices plus common temporal pooling preserve alignment, while geometry is time-invariant context.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing neuron registrations are removed through the `NaN` mask on `trace`. Sessions are capped at 40 minutes, extra tail frames are dropped, and short end chunks are still retained if they can form at least one 3-frame bin.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
...
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The notes justify these as pragmatic handling choices for registration gaps and small session-length deviations around the nominal 40-minute recording duration.

## 7-a. What are the most time-consuming steps of the code?

i. The dominant cost is loading and decompressing the per-animal joblib files. Secondary costs are the per-session trial loop and optional plotting.

ii.
```python
animal_data = load_animal(data_dir, animal)
...
for day_idx in range(ndays):
    ...
    neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(...)
```

iii. The AI explicitly states in its notes and trajectory that full-data runtime is dominated by decompressing the large joblib animal files.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` could be vectorized by reshaping the full session into trial and bin axes once, instead of repeatedly slicing and calling `temporal_bin_mean` for each trial.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. No explicit justification was recorded for leaving this loop unvectorized. The AI focused its efficiency discussion on file I/O rather than on intra-session vectorization.

## 7-c. What processing does the code repeat multiple times?

i. It repeats the same temporal binning logic twice per trial, repeats `open_mask.copy()` for every trial even though the input is static, and repeats per-session metadata appends in Python loops.

ii.
```python
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
...
input_trials.append(open_mask.copy())
```

iii. The AI did not explicitly justify these repeats. They appear to be a straightforward implementation choice.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional plotting and the debug/metadata bookkeeping are not needed for downstream decoder training. The code also stores `env_label` and several session summary fields that are useful for auditing but not used by the decoder itself.

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

iii. The notes justify these extras as sanity checks and auditability rather than as decoder inputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is the same as in section 6: remove session-invalid `NaN` cells, clip sessions to the nominal 40-minute duration, and keep any trailing chunk long enough for one 3-frame temporal bin.

ii.
```python
valid_cells = get_valid_cell_mask(trace_day)
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
...
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The AI’s justification was pragmatic rather than formal: preserve nominal sessions while handling small raw-data irregularities.

## 9-a. What are the most time-consuming steps of the code?

i. Again, the major cost is joblib deserialization of the animal files; trial-by-trial binning and optional figures are secondary.

ii.
```python
animal_data = load_animal(data_dir, animal)
...
for day_idx in range(ndays):
    ...
    neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(...)
```

iii. The notes explicitly call out joblib decompression as the dominant runtime cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Again, the main vectorization opportunity is the per-trial loop in `convert_session`, especially the repeated slicing and temporal pooling calls.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. No explicit AI justification was recorded for this specific implementation choice.

## 9-c. What processing does the code repeat multiple times?

i. Again, it repeats temporal pooling separately for neural and position on every trial and duplicates a static geometry vector with `.copy()` for every trial.

ii.
```python
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
...
input_trials.append(open_mask.copy())
```

iii. No explicit justification was recorded beyond the general preference for a straightforward, auditable conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Again, optional plotting and audit-focused debug metadata are extra work not needed for decoder training itself.

ii.
```python
if show_processing and first_trial_neural_binned is not None and first_trial_output is not None:
    maybe_plot_processing(...)
...
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
converted["metadata"]["session_usable_frames"].append(debug_info["usable_frames"])
```

iii. The AI justified these as validation and audit artifacts rather than essential converted data.
