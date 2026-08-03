# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the `.mat` files directly. It scanned `data/` for per-animal joblib files whose names match `QLAK-CA1-\d+`, loaded each one with `joblib.load`, and then iterated through each animal's `trace` array to access sessions. Trials were created later from these continuous session arrays.

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
...
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    ndays = animal_data["trace"].shape[0]
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by saying the repository's `main.py` and `load_dat` path operate on the precomputed joblib animal files, so using joblib was the "reference loading path used throughout" the codebase.

## 1-b. How are the data split into subjects?

i. Each per-animal joblib file is treated as one mouse. Subject IDs are the filenames, such as `QLAK-CA1-08`.

ii.
```python
animals = get_animals(data_dir)

subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
...
for animal in animals:
    ...
    converted["subject_idx"].append(subject_to_idx[animal])
```

iii. The notes say each top-level per-animal file contains one animal's full longitudinal dataset, so the filename is the natural subject identifier.

## 1-c. How are the data split into sessions?

i. Each day within an animal is treated as one session. The AI used the first dimension of `animal_data["trace"]` to count sessions and iterated over `day_idx`.

ii.
```python
ndays = animal_data["trace"].shape[0]
...
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
```

iii. The notes explicitly state that the source recordings are continuous single-day sessions and that the native organization is `(n_days, ...)`, so one day equals one session.

## 1-d. How are the data split into trials?

i. The AI imposed 40 nominal 1-minute trials per session. It capped each session to at most 40 minutes (`72000` frames), created 40 consecutive 1800-frame slices, and kept the final slice even if it was shorter than 1800 frames as long as it had at least 3 raw frames.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS
NOMINAL_SESSION_SECONDS = 40 * 60
NOMINAL_SESSION_FRAMES = NOMINAL_SESSION_SECONDS * FPS
...
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

iii. In the notes, the AI said sessions are nominally 40 minutes, so it chose "40 one-minute trials per session," keeping a short 40th trial for shorter recordings and discarding any tiny amount past 40 minutes for longer recordings.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter such as movement, occupancy, or behavioral validity. The only trial-level exclusion is structural: a candidate slice is dropped if it has fewer than 3 raw frames, because the AI later averages in 3-frame bins.

ii.
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The notes do not describe any explicit trial-quality curation in the conversion script. The AI treated sessions as continuous recordings to be chunked mechanically, not screened by trial quality.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is taken from the joblib field `animal_data["trace"][day]`, which the AI understood as session-by-session binarized rising-phase calcium event traces.

ii.
```python
trace_day = animal_data["trace"][day]
```

iii. The notes say the released `trace` signal is already the reference neural signal used by the paper's code: a binarized rising-phase calcium-event trace, not raw fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI kept only valid cells for the session, cast the array to `float32`, and then temporally rebinned each trial by averaging non-overlapping 3-frame windows. Because the joblib arrays are already `(cells, time)`, there is no transpose step.

ii.
```python
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
...
neural_raw = trace_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The notes justify two parts of this choice: keep the released binary event traces directly because that is what the paper uses downstream, and use 3-frame / 100 ms temporal pooling because the reference decoder code bins time in 3-frame windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removed cells that appear absent on that day by checking whether the first frame is `NaN`. It did not apply place-cell filtering or the decoder's low-activity threshold inside conversion.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
...
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The notes say the paper's downstream analyses kept all registered cells and treated `NaN` entries as unregistered cells for that day. The AI therefore removed only day-invalid cells and avoided additional neuron-quality filters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI did not align neural data to an experimental event. Instead, it treated the start of each consecutive 1-minute chunk as the alignment point and recorded that in metadata.

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

iii. The notes frame the dataset as continuous free exploration with no natural trial events, so the AI invented chunk-start alignment as a bookkeeping convention.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 100 ms bins. Yes: the AI rebinned both neural and position streams by averaging each non-overlapping set of 3 raw 30 Hz frames.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
...
def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
...
"time_bin_size": TIME_BIN_MS,
```

iii. The notes explicitly justify 100 ms bins by citing the paper's decoder code, which average-pools in 3-frame windows.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the raw `blocked` field for each day, not from the `envs` categorical labels.

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
...
"geometry_input_source": "raw blocked partition indices from source files",
```

iii. The notes say the raw `blocked` field is the canonical source because the helper `envs -> get_env_mat()` representation can disagree with the actual blocked-partition encoding for some environments.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI turned blocked indices into a 9-element *open-mask* vector, where `1` means open and `0` means blocked. It also transposed the implied 3x3 grid before flattening, so the geometry is remapped into the same index convention used by its output classes. The same static vector is copied into every trial of the session.

ii.
```python
def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    return open_mask.reshape(3, 3).T.reshape(-1)
...
input_trials.append(open_mask.copy())
```

iii. The notes justify using raw blocked partitions and say the transpose is needed to match the AI's chosen output-class ordering.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is taken from `animal_data["position"][day]`, the per-session `(x, y)` trajectory.

ii.
```python
position_day = animal_data["position"][day]
```

iii. The notes identify `position` as the DeepLabCut-derived x-y behavioral stream sampled at the same 30 Hz frame rate as the neural data.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. For each trial, the AI first averaged position in 3-frame windows, then discretized the binned x and y values into a 3x3 grid using each session's maximum observed x and y as the upper range. The output is a single categorical variable with shape `(1, T)`.

ii.
```python
position_valid = position_day.astype(np.float32, copy=False)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
...
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The notes justify this as an adaptation of the paper code's session-wise max-based position discretization rule, reduced from the paper's finer grid to the required 3x3 output.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The thresholds are not fixed at arena thirds. Instead, the AI computes bin widths separately for x and y as `(session_max_xy + BUFFER) / 3`, floors the normalized coordinates, clips them to `[0, 2]`, and combines them as `x_bin * 3 + y_bin`.

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

iii. The notes say this was chosen to match the reference decoder code's handling of position ranges, while adapting the number of spatial bins to 3 per axis.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned by using the same session slices and the same 3-frame temporal binning on both streams, trial by trial.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The notes describe the neural and behavior streams as already synchronized at 30 Hz, so the AI's main alignment step was to slice and rebin them identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled missing/unregistered neurons by removing cells whose first frame is `NaN`. It also capped sessions to 40 nominal minutes, kept short final minute-long chunks if they still had at least 3 frames, and within each chunk silently dropped any 1-2 leftover frames that did not fill a complete 3-frame temporal bin.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
...
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
...
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
...
usable = (arr.shape[-1] // bin_size) * bin_size
trimmed = arr[..., :usable]
```

iii. The notes justify only the NaN-based cell removal and the nominal 40-minute cap. They do not present a separate rationale for keeping undersized final trials or trimming leftover frames inside each trial beyond the 3-frame binning rule.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified loading and decompressing the large per-animal joblib files as the main runtime cost. The rest of the conversion was treated as comparatively cheap vectorized array work.

ii.
```python
for animal in animals:
    animal_start = time.perf_counter()
    animal_data = load_animal(data_dir, animal)
    ...
    animal_elapsed = time.perf_counter() - animal_start
    print(f"Finished {animal} in {animal_elapsed:.2f}s")
```

iii. In the notes, the AI explicitly says full-data runtime "will be dominated by decompressing the 7 large joblib animal files."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunity is the per-trial Python loop in `convert_session`. The code already vectorizes 3-frame averaging within a chunk, but it still iterates over every trial, slices arrays repeatedly, and appends results one at a time.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)

    neural_trials.append(neural_binned)
    input_trials.append(open_mask.copy())
    output_trials.append(output_binned)
```

iii. The notes mention that vectorized 3-frame temporal binning was used, but they do not identify the trial loop itself as a remaining optimization target.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same static geometry vector copy for every trial in a session, and it repeats the same slice-bin-discretize sequence separately for each trial even though the operations could be batched per session.

ii.
```python
for start, end in trial_slices:
    ...
    neural_trials.append(neural_binned)
    input_trials.append(open_mask.copy())
    output_trials.append(output_binned)
```

iii. There is no explicit justification in the notes for this repetition. The notes focus on correctness and memory streaming rather than micro-optimizing repeated per-trial work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores plotting/debug information that the downstream decoder does not need: environment labels, session IDs, original vs usable frame counts, per-session trial/neuron counts, and first-trial previews used only for optional figures. It also keeps optional plotting machinery in the conversion path.

ii.
```python
env_label = animal_data["envs"].reshape(-1)[day]
...
first_trial_neural_binned = None
first_trial_output = None
...
debug_info = {
    "env_label": str(env_label),
    "open_mask": open_mask,
    "usable_frames": usable_frames,
    "original_frames": int(trace_valid.shape[1]),
    "trial_slices": trial_slices,
    "session_max_xy": session_max_xy,
}
...
converted["metadata"]["session_ids"].append(session.session_id)
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
converted["metadata"]["session_usable_frames"].append(debug_info["usable_frames"])
converted["metadata"]["session_trial_counts"].append(len(neural_trials))
converted["metadata"]["session_valid_neuron_counts"].append(int(brain_region_idx.shape[0]))
```

iii. The notes justify these additions as sanity-check and inspection aids, especially for `--show-processing`, rather than as required downstream decoder inputs.
