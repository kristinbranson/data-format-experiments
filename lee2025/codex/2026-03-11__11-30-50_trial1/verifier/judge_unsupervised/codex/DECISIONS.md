# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates per-animal files in `data/`, loads each animal lazily with `joblib.load`, iterates over all recording days in that animal, converts each day into one session, and then splits each session into consecutive trial slices. It never loads the whole cohort into memory at once.

ii. ```python
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
    ...
    for day_idx in range(ndays):
        ...
        neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(...)
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose per-animal streaming to limit memory use and to stay close to the repository’s joblib loading path.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by filename: every `QLAK-CA1-*` file is treated as one mouse, sorted lexicographically, and used directly as the `subjects` list. `subject_idx` is built by mapping each session back to its animal filename.

ii. ```python
subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
...
converted["subject_idx"].append(subject_to_idx[animal])
```

iii. The agent justified this from the raw data layout: one top-level joblib file per mouse, matching the paper’s 7-animal organization.

## 1-c. How are the data split into sessions?

i. Each day index in an animal’s `trace` array is treated as one session. The session id is `"{animal}_day{day_idx:02d}"`.

ii. ```python
@dataclass(frozen=True)
class SessionRecord:
    animal: str
    day_idx: int

    @property
    def session_id(self) -> str:
        return f"{self.animal}_day{self.day_idx:02d}"

...
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
```

iii. The agent’s notes state the source recordings are organized as one continuous recording per day, and that this matches the paper’s “one session per day” structure.

## 1-d. How are the data split into trials?

i. Sessions are artificially trialized into consecutive 1-minute chunks. At 30 Hz, each trial is 1800 frames. The code creates up to 40 such chunks per session, truncated to the nominal first 40 minutes.

ii. ```python
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

iii. The notes say the source has no native trials, so the agent imposed 40 one-minute trials because the task instructions explicitly required 1-minute trials within each 40-minute session.

## 1-e. How are trials filtered based on quality controls?

i. Trials are barely filtered. The code keeps every consecutive 1-minute chunk within the first nominal 40 minutes, except chunks shorter than 3 frames, because those cannot form even one temporal bin. There is no movement-based, occupancy-based, or behavior-based trial rejection.

ii. ```python
usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
...
end = min(start + TRIAL_FRAMES, usable_frames)
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The agent’s rationale was that the source has no native trial QC and that the reference decoder’s stricter filters act on frames/cells, not whole trials. It therefore kept essentially all trialized chunks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-day `trace` matrix. The agent interprets this as the released binarized rising-phase calcium event trace and does not recompute fluorescence features.

ii. ```python
trace_day = animal_data["trace"][day]
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The notes explicitly say the repository and methods treat `trace` as the final neural signal, so recomputing `dF/F` would be inconsistent.

## 2-b. How is the `neural` data processed?

i. The agent removes invalid cells for that day, keeps the binary event trace values, and averages them in non-overlapping 3-frame windows to make 100 ms bins. It does not apply the Gaussian temporal smoothing used inside the reference decoder functions.

ii. ```python
def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

...
neural_raw = trace_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The agent justified the 3-frame averaging from the paper code’s `temporal_bin_size=3`, and argued that storing the released event trace directly best matches the reference signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC consists only of dropping cells whose first timepoint is `NaN` on that day, treating those as unregistered/invalid for the session. The code does not apply the reference decoder’s movement threshold or minimum-event cell threshold.

ii. ```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

...
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. In the notes, the agent argues that the paper’s downstream analyses keep all registered cells and that the decoder-specific activity filters should not be baked into the converted dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of each imposed 1-minute chunk. There is no task event in the raw data, so trial start is the temporal alignment event.

ii. ```python
"temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
...
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
```

iii. The agent notes say this follows directly from the instructions, which define trials as 1-minute splits of otherwise continuous sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins, produced by averaging every 3 frames of the 30 Hz raw streams. Yes, temporal rebinning is applied.

ii. ```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
...
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
...
"time_bin_size": TIME_BIN_MS,
```

iii. The agent cites the reference decoder’s `temporal_bin_size=3` as the reason for using 100 ms bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The geometry input is derived from `animal_data["blocked"][day]`, not from the reference helper that maps `envs` labels to 3x3 templates.

ii. ```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
...
"geometry_input_source": "raw blocked partition indices from source files",
```

iii. The notes say the agent initially considered `envs -> get_env_mat()`, but chose raw `blocked` because it believed the helper used a simplified convention for some shapes.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent converts blocked partition indices into a length-9 open/closed vector, where `1` means open and `0` means blocked. `-1` means no blocked partition. It then transposes the 3x3 layout so the geometry ordering matches the position-class ordering `x_bin * 3 + y_bin`.

ii. ```python
def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    return open_mask.reshape(3, 3).T.reshape(-1)
```

iii. The trajectory shows the transpose was added after a sanity check found that blocked-bin occupancy was too high until the 3x3 grid was transposed.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Geometry is static within a session and trial, so the same 9-element vector is copied into every trial of that session. It is aligned at the trial level, not as a time-varying series.

ii. ```python
for start, end in trial_slices:
    ...
    input_trials.append(open_mask.copy())
```

iii. The agent’s notes say this is intentional because arena geometry does not vary within a recording, and the validator can tile static trial-level inputs across time.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output comes from `animal_data["position"][day]`, i.e. the raw x-y tracking stream for that session.

ii. ```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
```

iii. The notes identify `position` as the DeepLabCut-derived behavioral variable used by the paper’s decoder.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code takes the raw x-y trajectory, slices it into the same trials as neural data, averages it in 3-frame windows, computes session-wise spatial scale factors from the session maxima, and converts each 100 ms x-y sample into a 3x3 categorical location.

ii. ```python
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
...
position_raw = position_valid[:, start:end]
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The agent says this mirrors the reference decoder’s session-wise max-based position binning, adapted from 15x15 to 3x3 because the task requested 9 position classes.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each x and y coordinate is divided by `(session_max_xy + buffer) / 3`, floored, clipped to `[0, 2]`, and then combined into a single class `x_bin * 3 + y_bin`, giving 9 categories.

ii. ```python
denom = (session_max_xy + BUFFER) / POSITION_BINS
denom = np.where(denom <= 0, 1.0, denom)
binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
binned = np.clip(binned, 0, POSITION_BINS - 1)
classes = binned[0] * POSITION_BINS + binned[1]
return classes[np.newaxis, :]
```

iii. The justification in the notes is that this preserves the reference code’s flooring logic while changing the spatial resolution from 15x15 bins to 3x3 bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output position uses the same trial boundaries and the same 3-frame temporal bins as neural data, so each output timepoint corresponds to the same 100 ms interval as the neural timepoint.

ii. ```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The agent explicitly planned a shared temporal slicing/binning path for neural and position to avoid alignment drift.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted dataset is at 100 ms resolution, created by non-overlapping 3-frame mean pooling from the original 30 Hz stream. Rebinning is applied to both neural and position data.

ii. ```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
...
return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The notes tie this directly to the paper code’s temporal binning choice.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are aligned by identical trial slices and identical 3-frame bins. Input geometry is static and copied per trial, so it is implicitly aligned to the full trial duration rather than to individual frames.

ii. ```python
for start, end in trial_slices:
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    ...
    input_trials.append(open_mask.copy())
```

iii. The agent describes geometry as constant contextual input and the other two streams as jointly binned time series.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The code handles a few specific issues: session-invalid cells are removed via `NaN` checks; sessions longer than 40 minutes are truncated; short trailing chunks are dropped if they cannot form a 3-frame bin; and `-1` in `blocked` means “no blocked partitions.” It does not implement broader malformed-data recovery.

ii. ```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
...
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))

if not (blocked_values.size == 1 and blocked_values[0] == -1):
    open_mask[blocked_values.astype(int)] = 0.0
```

iii. The notes frame these as minimal safeguards consistent with the released data representation rather than aggressive cleaning.

## 7-a. What are the most time-consuming steps of the code?

i. The agent identifies per-animal joblib decompression/loading as the main bottleneck. After loading, the largest repeated work is looping over every session and every trial to bin neural and position arrays.

ii. ```python
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    ...
    for day_idx in range(ndays):
        ...
        neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(...)
```

iii. `CONVERSION_NOTES.md` explicitly says full-data runtime is dominated by decompressing the seven large joblib files.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop inside `convert_session` could be vectorized across the whole session instead of slicing and binning each trial separately. The per-sample categorical conversion inside `discretize_position_3x3` is already vectorized.

ii. ```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy)
```

iii. The agent notes mention vectorization as a goal, but it only vectorized within each slice, not across all trial slices at once.

## 7-c. What processing does the code repeat multiple times?

i. It repeatedly calls the same temporal mean-pooling for each trial, repeatedly copies the same static geometry vector into every trial, and reruns the full conversion path separately for sample and full outputs.

ii. ```python
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
...
input_trials.append(open_mask.copy())
```

iii. The agent’s notes acknowledge repeated per-trial processing, but prioritize simplicity and easy validation over deeper refactoring.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores extra debugging/plotting state such as `first_trial_neural_binned`, `first_trial_output`, and `debug_info`, and can render processing figures. Those are useful for inspection but not required by downstream decoding.

ii. ```python
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
```

iii. The trajectory shows these additions were motivated by the instruction to produce visual sanity checks and debugging metadata.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as question 6: the code removes cells marked invalid by `NaN`, truncates or drops undersized temporal segments, and interprets `blocked == -1` as no obstruction. It does not attempt more sophisticated imputation or repair.

ii. ```python
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
...
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
...
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The agent’s justification was that the source files are already preprocessed and mostly well-formed, so only lightweight handling was needed.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading the large joblib files dominates, followed by the session/trial loops that bin neural and position data.

ii. ```python
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    ...
    for day_idx in range(ndays):
        ...
```

iii. The agent says this explicitly in Step 6 of `CONVERSION_NOTES.md`.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the trial loop in `convert_session` is the clearest remaining vectorization target.

ii. ```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    ...
```

iii. The agent partly optimized by vectorizing within-slice pooling, but not across all slices.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: repeated temporal binning by trial, repeated copying of static geometry, and repeated sample/full runs through the same conversion code path.

ii. ```python
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
input_trials.append(open_mask.copy())
```

iii. This repetition reflects a straightforward implementation rather than an attempt to cache shared intermediate results.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: debugging state, timing/metadata bookkeeping, and optional plots are produced for validation but are not needed by the downstream decoder.

ii. ```python
if show_processing and first_trial_neural_binned is not None and first_trial_output is not None:
    maybe_plot_processing(...)
...
converted["metadata"]["session_ids"].append(session.session_id)
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
```

iii. The agent added these because the instructions required process visualization and detailed audit notes.
