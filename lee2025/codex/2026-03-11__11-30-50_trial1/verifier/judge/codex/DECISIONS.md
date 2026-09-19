# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not read the MATLAB `.mat` files directly. It scans `/app/data` for per-animal files whose names match `QLAK-CA1-\d+`, loads each one with `joblib`, and then iterates over all day-level sessions inside that loaded animal dictionary.

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

iii. `CONVERSION_NOTES.md` says the repository’s “reference loading path” was the per-animal joblib files and that the script should stream one animal at a time “to limit memory use.” The trajectory also mentions that the compressed joblib loads were slow and dominated runtime.

## 1-b. How are the data split into subjects?

i. Each per-animal file becomes one subject. The subject identifier is the filename itself, and `subjects` is just the sorted list of those animal names.

ii.
```python
animals = get_animals(data_dir)

subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI documented that `data/` contains seven per-animal joblib files and identified the seven subject IDs from those filenames.

## 1-c. How are the data split into sessions?

i. Within each subject, each recording day is treated as one session. The AI uses the first dimension of `animal_data["trace"]` as the number of sessions and iterates `day_idx` from `0` to `ndays - 1`.

ii.
```python
ndays = animal_data["trace"].shape[0]

for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(
        session=session,
        animal_data=animal_data,
        show_processing=do_plot,
    )
```

iii. The notes repeatedly describe the source dataset as “continuous ~40-minute day recordings,” with one session per day.

## 1-d. How are the data split into trials?

i. The AI imposes 60-second artificial trials on each session, but it does so as a fixed nominal 40-trial schedule per session. It caps each session at 40 minutes, creates consecutive 1800-frame slices, and keeps the last slice even if it is shorter than 60 seconds as long as it has at least 3 frames.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS
NOMINAL_SESSION_SECONDS = 40 * 60

def get_trial_slices(n_frames_session: int) -> list[tuple[int, int]]:
    usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
    slices = []
    for trial_idx in range(NOMINAL_SESSION_SECONDS // TRIAL_SECONDS):
        start = trial_idx * TRIAL_FRAMES
        end = min(start + TRIAL_FRAMES, usable_frames)
        if end - start >= TEMPORAL_BIN_FRAMES:
            slices.append((start, end))
    return slices

for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly justified this as: “impose 40 one-minute trials per session,” keep a “shorter final trial if session shorter,” and discard any small tail beyond 40 minutes.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality control based on behavior or neural quality. The only effective trial filter is structural: a chunk is kept if it exists within the nominal 40-minute window and contains at least 3 frames, because 3 frames are needed for one temporal bin.

ii.
```python
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

iii. The notes say the paper’s decoder had velocity and activity thresholds internally, but the conversion script itself intentionally keeps all generated chunks and only requires enough frames to form a 100 ms bin.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data comes from the per-day `trace` array in the loaded animal dictionary.

ii.
```python
trace_day = animal_data["trace"][day]
```

iii. The notes describe `trace` as the released “rise-extracted calcium events” / “binarized rising-phase calcium event trace,” and say this should be used directly rather than recomputing fluorescence features.

## 2-b. How is the `neural` data processed?

i. The AI keeps only session-valid cells, casts to `float32`, and temporally averages the event traces in non-overlapping 3-frame windows, producing 100 ms bins. It does not transpose because the joblib `trace_day` is already `cells x time`.

ii.
```python
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)

def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md` Step 5 says the “released binary CA1 event traces” should be used directly, and that 100 ms bins were chosen to match the paper decoder’s 3-frame temporal pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural-quality filter in the conversion script is dropping cells whose first frame is `NaN`, which the AI treats as session-invalid / unregistered cells. It does not apply place-cell, activity, or movement-based filters in the conversion.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The notes state that the main analyses “keep all valid registered cells” and remove only day-specific `NaN` registrations. They also explicitly say not to apply place-cell filtering in the converter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Neural trials are aligned to the start of each consecutive 1-minute chunk, which the AI records as the temporal alignment event in metadata.

ii.
```python
"metadata": {
    "temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
}
```

iii. The notes say the source data are continuous free-exploration recordings with no native trial event, so chunk start is used as the only alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. Yes: the AI rebins 30 Hz framewise data by averaging every 3 consecutive frames.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0

neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The notes explicitly justify 100 ms bins as matching the reference decoder’s 3-frame temporal pooling and reducing validator/training cost.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives geometry from the raw per-day `blocked` field, not from the helper `envs` label. It still reads `envs` for metadata/debug labeling only.

ii.
```python
env_label = animal_data["envs"].reshape(-1)[day]
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. Step 4 and Step 5 in the notes say the AI found ambiguities in `envs -> get_env_mat()` and therefore chose raw `blocked` as the “safer source” for the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts blocked partition indices into a length-9 static open-mask vector, where `1` means open and `0` means blocked. It also transposes the 3x3 layout before flattening so the geometry indexing matches the output position class indexing.

ii.
```python
def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    return open_mask.reshape(3, 3).T.reshape(-1)

input_trials.append(open_mask.copy())
```

iii. The trajectory shows the AI initially found a geometry mismatch, then concluded the geometry vector “needed a transpose to match the position-bin indexing.” The notes also argue for using raw blocked partitions rather than helper templates.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position classes are derived from the per-day `position` array.

ii.
```python
position_day = animal_data["position"][day]
```

iii. The notes identify `position` as the 2D x-y trajectory stream aligned with `trace`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI casts position to `float32`, limits it to the usable part of the session, computes session-wise maxima for x and y, averages position in 3-frame bins, and then discretizes those averaged x-y coordinates into one 3x3 categorical label per bin.

ii.
```python
position_valid = position_day.astype(np.float32, copy=False)
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)

position_raw = position_valid[:, start:end]
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The notes say this was intended to mirror the paper code’s “session-wise max-based discretization,” adapted from the original 15x15 decoder to the requested 3x3 output grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. For each binned x and y coordinate, the AI divides by `(session_max_xy + 1e-5) / 3`, floors the result, clips to `{0,1,2}`, and combines the bins as `x_bin * 3 + y_bin` to produce class IDs `0..8`.

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

iii. The notes justify this as reusing the reference decoder’s coordinate handling while adapting it to 3 bins per axis.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by construction: both are sliced with the same `trial_slices`, both are rebinned with the same 3-frame averaging, and each trial’s output time axis matches the corresponding neural time axis.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The notes list alignment checks as a required sanity check and later say the raw-vs-converted alignment spot checks passed.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing / imperfect data by dropping session-invalid cells, using `np.nanmax` when computing session position bounds, truncating sessions to a nominal 40-minute maximum, and still keeping a final partial trial if it has at least 3 frames.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)

if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. In Step 5, the AI explicitly framed this as preserving the nominal 40-minute session while minimizing data loss from slightly short or slightly long recordings.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified per-animal `joblib` loading / decompression as the main bottleneck, not the arithmetic on already-loaded arrays.

ii.
```python
def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

for animal in animals:
    animal_start = time.perf_counter()
    animal_data = load_animal(data_dir, animal)
```

iii. `CONVERSION_NOTES.md` Step 6 says full-data runtime would be “dominated by decompressing the 7 large joblib animal files,” and the trajectory repeats that the compressed joblib loads were much slower than expected.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunity left in the code is the per-trial Python loop in `convert_session`. The session could have been reshaped into a trial axis and binned/discretized in bulk instead of slicing one trial at a time.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The notes say the AI deliberately vectorized the inner 3-frame averaging with `reshape(...).mean(...)`, but it left the outer trial loop in Python.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several per-trial operations that are structurally the same for every trial in a session: slicing arrays, averaging into 3-frame bins, discretizing position, and copying the same static geometry vector into every trial list entry.

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

iii. The notes frame the current implementation as a streaming, session-by-session converter, not a bulk vectorized pipeline, so this repeated per-trial work appears to be a deliberate simplicity tradeoff.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes optional plotting / preview generation, timing instrumentation, and debug metadata gathering that are not needed by the downstream decoder input/output tensors. It also carries session labels and several metadata lists for auditability rather than decoding.

ii.
```python
def maybe_plot_processing(...):
    ...
    fig.savefig(f"processing_{session.session_id}.png", dpi=150)

debug_info = {
    "env_label": str(env_label),
    "open_mask": open_mask,
    "usable_frames": usable_frames,
    "original_frames": int(trace_valid.shape[1]),
    "trial_slices": trial_slices,
    "session_max_xy": session_max_xy,
}

elapsed = time.perf_counter() - session_start
print(
    f"  Converted {session.session_id}: {len(neural_trials)} trials, "
    f"{brain_region_idx.shape[0]} neurons, {elapsed:.2f}s"
)
```

iii. The notes emphasize documentation, sanity checks, and processing figures, so these extra computations were intentional for audit/debug purposes rather than for the final decoder dataset itself.
