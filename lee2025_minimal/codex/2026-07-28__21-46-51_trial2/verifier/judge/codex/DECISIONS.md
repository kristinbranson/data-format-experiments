# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not discover all source files with a glob over `.mat` files. Instead, it hardcodes seven animal IDs, loads `/app/data/<animal>` with `joblib.load`, extracts `envs`, `trace`, `position`, and `blocked` from the loaded dict, and then iterates over each day/session. Trials are created later by slicing each day into 60 s windows.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    envs = flatten_envs(dat["envs"])
    trace = np.asarray(dat["trace"], dtype=np.float32)
    position = np.asarray(dat["position"], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says the agent wanted to "preserv[e] the paper/code preprocessing" and use the provided per-animal files directly. The trajectory shows it inspected the dense `trace` and `position` arrays in those joblib files, then built the converter around that representation rather than around the `.mat` files.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hardcoded animal IDs in `ANIMALS`. Each loaded `/app/data/<animal>` file is treated as one subject, and `subject_id` is the index from `enumerate(ANIMALS)`.

ii.
```python
for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    ...
    subject_idx.append(subject_id)

data = {
    ...
    "subjects": list(ANIMALS),
    "subject_idx": np.array(subject_idx, dtype=np.int64),
}
```

iii. `CONVERSION_NOTES.md` explicitly lists the seven subject IDs and says "Subjects are the seven animal IDs". The trajectory also describes the source as per-animal files.

## 1-c. How are the data split into sessions?

i. Each day along axis 0 of the loaded `trace` and `position` arrays is treated as one session. The code loops over `for day in range(trace.shape[0])`, and each day contributes one session to `neural`, `input`, and `output`.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
position = np.asarray(dat["position"], dtype=np.float32)

for day in range(trace.shape[0]):
    blocked_entry = normalize_blocked_entry(dat["blocked"][day])
    env_name = str(envs[day])
    day_trace = trace[day]
    day_pos = position[day]
    ...
    neural.append(session_trials_neural)
    decoder_input.append(session_trials_input)
    decoder_output.append(session_trials_output)
```

iii. `CONVERSION_NOTES.md` says "I treated each original recording day as one decoder session." Step 47 of the trajectory repeats that the converter would preserve the paper’s day/session structure.

## 1-d. How are the data split into trials?

i. Within each day/session, the code computes `n_trials = T // 1800` and slices contiguous non-overlapping 60 s windows at 30 Hz. Any trailing remainder shorter than 1800 frames is discarded.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
...
n_trials = T // TRIAL_FRAMES
dropped = T - n_trials * TRIAL_FRAMES
...
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. `CONVERSION_NOTES.md` calls this a decoder-specific adaptation because the source recordings are continuous long sessions. It justifies the split as contiguous full 60 s windows, 1800 frames each, with the incomplete tail discarded.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter based on behavior or neural quality. The only trial-level constraints are that incomplete trailing fragments are dropped, and any session with fewer than two full 60 s trials raises an error instead of being exported.

ii.
```python
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
dropped = T - n_trials * TRIAL_FRAMES
stats.total_frames_dropped += int(dropped)
```

iii. The notes justify the minimum-trial logic indirectly through the decoder format requirement and justify dropping the tail as part of the 60 s trialization policy. No separate trial-quality-control rationale is recorded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-animal `trace` array loaded from the joblib file. The code expects `trace` to have shape `(n_days, n_cells, T)`.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
...
if trace.ndim != 3:
    raise ValueError(f"{animal}: trace shape should be (n_days, n_cells, T), got {trace.shape}")
```

iii. `CONVERSION_NOTES.md` says "I used the provided binary rise-event traces directly from `data/<animal>`." That is the agent’s stated source for the exported neural signal.

## 2-b. How is the `neural` data processed?

i. The code keeps the `trace` signal largely as-is. It selects neurons registered on the current day, casts to `float32`, and keeps the day array in `(neurons, time)` order. It does not smooth, deconvolve, re-threshold, or temporally rebin the signal.

ii.
```python
day_trace = trace[day]
registered_today = ~np.isnan(day_trace[:, 0])
...
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
...
session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. `CONVERSION_NOTES.md` explicitly says the agent used "binary rise-event traces directly", and "did not re-deconvolve, smooth, or re-threshold the calcium traces." The notes frame that as matching the paper/methods description of binarized rising-phase events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is kept for a given day only if its first frame is not `NaN`, and the code then checks that kept neurons never contain `NaN`s anywhere and dropped neurons are entirely `NaN`. This acts as a per-session registration filter for unrecorded neurons.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. `CONVERSION_NOTES.md` says "For each session, I kept only neurons registered on that day" and explains that unregistered neurons are all-`NaN` for that day. Step 47 of the trajectory also says it would "drop only neurons that are unregistered for a given day."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The neural data is aligned to the start of each artificial 60 s window produced from a continuous recording day.

ii.
```python
"temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
...
start = trial_idx * TRIAL_FRAMES
stop = start + TRIAL_FRAMES
session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. The notes explicitly describe trialization as a decoder-specific adaptation for continuous recordings, not an alignment to a stimulus or behavioral event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz frame rate, so each bin is `1000 / 30 = 33.33...` ms. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
...
"time_bin_size": TIME_BIN_MS,
```

iii. `CONVERSION_NOTES.md` states that the recordings are split into 60 s windows at 30 Hz and does not mention any temporal resampling. The notes also say the agent used the provided traces directly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from `dat["blocked"][day]`, not from the environment-name string. `envs` is read and tracked for counts and metadata, but the actual input vector comes from the blocked-bin metadata.

ii.
```python
envs = flatten_envs(dat["envs"])
...
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_name = str(envs[day])
env_mask = mask_from_blocked(blocked_entry)
```

iii. `CONVERSION_NOTES.md` says "I derived this from the `blocked` metadata, not only from the environment name," and argues that `blocked` is the authoritative source because the `env` string alone does not fully specify orientation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code normalizes the nested MATLAB-like `blocked` entry into a tuple of blocked bin IDs, converts that into a `3 x 3` mask with `1` for open bins and `0` for blocked bins, then flattens the mask into a static 9D trial input vector.

ii.
```python
def normalize_blocked_entry(entry) -> tuple[int, ...]:
    while isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    arr = np.asarray(entry, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return ()
    arr = arr.astype(int)
    if arr.size == 1 and arr[0] == -1:
        return ()
    return tuple(sorted(arr.tolist()))

def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask
...
input_vec = env_mask.reshape(-1).astype(np.float32)
```

iii. The notes justify this as a decoder-specific adaptation: "Input is a static 9D vector per trial: a flattened `3 x 3` open-bin mask." The justification is that `blocked` preserves the session’s actual geometry and orientation.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry input is not time-varying. The same 9D vector is copied once per trial for every trial in the session, so alignment is only at the trial/session level rather than frame-by-frame.

ii.
```python
input_vec = env_mask.reshape(-1).astype(np.float32)
...
for trial_idx in range(n_trials):
    ...
    session_trials_input.append(input_vec.copy())
```

iii. `CONVERSION_NOTES.md` explicitly says the geometry input is "Static per-trial." The notes also say blocked partitions do not change within a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The decoder output is derived from the `position` array loaded from the per-animal joblib file. The code expects shape `(n_days, 2, T)` and uses each day’s `2 x T` position matrix.

ii.
```python
position = np.asarray(dat["position"], dtype=np.float32)
...
if position.ndim != 3:
    raise ValueError(f"{animal}: position shape should be (n_days, 2, T), got {position.shape}")
...
day_pos = position[day]
```

iii. The trajectory repeatedly refers to using the per-day `position` arrays from the raw files, and `CONVERSION_NOTES.md` frames the decoder output as the mouse’s spatial position.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code first bins the raw `x` and `y` coordinates into a `3 x 3` grid using each axis’s within-session maximum rather than a fixed arena size. It then builds a nearest-valid lookup from the session geometry and projects any binned positions that land in blocked coarse bins onto the nearest open bin before converting to class labels.

ii.
```python
def bin_position_to_grid(position_xy: np.ndarray, n_bins: int = 3) -> np.ndarray:
    max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
    bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)

def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    valid = np.argwhere(env_mask > 0)
    lookup = np.zeros((3, 3, 2), dtype=np.int64)
    for x in range(3):
        for y in range(3):
            if env_mask[x, y] > 0:
                lookup[x, y] = np.array([x, y], dtype=np.int64)
                continue
            dists = np.sum((valid - np.array([x, y])) ** 2, axis=1)
            lookup[x, y] = valid[np.argmin(dists)]
    return lookup

binned_xy = bin_position_to_grid(day_pos, n_bins=3)
lookup = build_nearest_valid_lookup(env_mask)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
```

iii. `CONVERSION_NOTES.md` says the agent "matched the paper code’s binning style" by dividing by per-axis maxima with no min subtraction. It separately justifies the nearest-valid projection as a decoder-task adaptation "analogous" to geometry-aware cleanup in the paper’s within-session decoder.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The thresholding is done by flooring each axis after scaling it into three bins with per-session maxima, clipping to `0..2`, and then converting `(x_bin, y_bin)` to one categorical label with `x_bin * 3 + y_bin`. If the coarse bin is blocked, the label is replaced by the nearest valid open bin first.

ii.
```python
bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
return np.clip(bins, 0, n_bins - 1)
...
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
...
"output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
```

iii. `CONVERSION_NOTES.md` says "Output is a single categorical variable `position_bin` with values `0..8`" and explicitly records the label definition as `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. `position` and `neural` are both taken from the same day/session, remain at the same 30 Hz sampling rate, and are sliced with identical `start:stop` trial windows. That keeps them aligned frame-by-frame within each trial.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The notes justify the whole trialization approach as contiguous windows cut out of the continuous session. Because neural and position are windowed with the same indices, the alignment is implicit.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native frame rate: `33.33...` ms per bin at 30 Hz. No temporal rebinning is performed anywhere in the script.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"time_bin_size": TIME_BIN_MS,
"fps": FPS,
```

iii. The notes describe 30 Hz data split into 60 s windows and do not mention any resampling. The code directly slices native frames.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output position stay aligned because they come from the same day/session arrays and are sliced with the same trial boundaries. Input geometry is static, so the same 9D geometry vector is attached to each trial without a per-frame time axis.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. `CONVERSION_NOTES.md` says the output is time-varying while the geometry input is static per trial. The trajectory also says the converter would use contiguous 60 s windows with a 3x3 geometry input and 3x3 categorical position output.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The script explicitly normalizes malformed/nested `blocked` entries, treats `-1` or empty blocked entries as "no blocked bins", rejects any session with `NaN` positions, removes unregistered neurons via `NaN`-based registration filtering, checks for inconsistent `NaN` patterns, and discards incomplete tail fragments that do not make a full trial.

ii.
```python
def normalize_blocked_entry(entry) -> tuple[int, ...]:
    while isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    arr = np.asarray(entry, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return ()
    ...
    if arr.size == 1 and arr[0] == -1:
        return ()

if np.isnan(day_pos).any():
    raise ValueError(f"{animal} day {day}: position contains NaNs")
...
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(...)
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(...)
...
dropped = T - n_trials * TRIAL_FRAMES
```

iii. The notes justify the neuron handling and tail dropping. Step 32 of the trajectory specifically mentions that `blocked` had a nested MATLAB-style structure that needed careful normalization.

## 7-a. What are the most time-consuming steps of the code?

i. In this implementation, the likely dominant costs are loading each animal file with `joblib.load`, converting full `trace` and `position` arrays into NumPy arrays, and then iterating through every session to bin all position frames and copy every trial slice of neural and output data.

ii.
```python
def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]
...
trace = np.asarray(dat["trace"], dtype=np.float32)
position = np.asarray(dat["position"], dtype=np.float32)
...
binned_xy = bin_position_to_grid(day_pos, n_bins=3)
lookup = build_nearest_valid_lookup(env_mask)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
...
for trial_idx in range(n_trials):
    ...
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The notes and trajectory focus on matching the paper and passing decoder verification, not on optimization. There is no explicit performance justification beyond doing the full conversion and associated sanity checks.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit `for trial_idx in range(n_trials)` loop copies every trial one by one. `build_nearest_valid_lookup` also uses nested Python loops over the 3x3 grid even though there are only a few recurring geometries, and the sample-data builder rescans `subject_idx` session-by-session.

ii.
```python
def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    valid = np.argwhere(env_mask > 0)
    lookup = np.zeros((3, 3, 2), dtype=np.int64)
    for x in range(3):
        for y in range(3):
            ...

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())

for subject_id in range(min(args.sample_subject_count, len(ANIMALS))):
    sessions_for_subject = []
    for sess_idx, subj in enumerate(data["subject_idx"]):
        if int(subj) == subject_id:
            sessions_for_subject.append(sess_idx)
```

iii. No explicit rationale for leaving these as loops is recorded. The overall style suggests the agent prioritized transparent control flow over optimization.

## 7-c. What processing does the code repeat multiple times?

i. The code rebuilds the nearest-valid lookup every session even when the same geometry recurs, copies the same static `input_vec` for every trial in a session, and traverses the exported dataset multiple times again for `summarize_dataset`, sample selection, and printed sanity summaries.

ii.
```python
lookup = build_nearest_valid_lookup(env_mask)
...
for trial_idx in range(n_trials):
    ...
    session_trials_input.append(input_vec.copy())
...
sanity = {
    "summary": summarize_dataset(data),
    ...
}
...
print(json.dumps(sanity, indent=2))
print("Sample summary:")
print(json.dumps(summarize_dataset(sample_data), indent=2))
```

iii. The notes justify the extra summary/statistics work as "sanity checks" and "paper-level count checks." There is no explicit justification for caching or deduplicating the repeated geometry/trial bookkeeping.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations support documentation rather than the final decoder arrays: `ENV_TO_MASK` is defined but never used; `env_counts`, `blocked_patterns_by_env`, `frame_lengths`, `per_animal_unique_neurons`, and `total_frames_reassigned` are collected for metadata/sanity reporting; and the `start` variable in sample selection is computed but never used.

ii.
```python
ENV_TO_MASK = {
    "square": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
    ...
}
...
env_counts = Counter()
blocked_patterns_by_env: dict[str, set[tuple[int, ...]]] = {}
frame_lengths = Counter()
...
per_animal_unique_neurons[animal] = int(np.sum(registered_any_day))
...
stats.total_frames_reassigned += int(np.sum(reassigned))
...
start = 0
for subject_id in range(min(args.sample_subject_count, len(ANIMALS))):
    ...
    start += len(sessions_for_subject)
```

iii. `CONVERSION_NOTES.md` emphasizes extensive sanity checking and environment summaries, which explains why these extra statistics exist even though the downstream decoder does not need them.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is the same as in section 6: nested `blocked` entries are normalized, `-1` is treated as no blockage, sessions with `NaN` positions are rejected, unregistered neurons are removed via `NaN` registration checks, inconsistent `NaN` structure raises an error, and incomplete trailing fragments are discarded.

ii.
```python
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
...
if np.isnan(day_pos).any():
    raise ValueError(f"{animal} day {day}: position contains NaNs")
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(...)
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(...)
...
dropped = T - n_trials * TRIAL_FRAMES
```

iii. The notes justify the neuron-registration filter and discarded tails, and the trajectory explicitly mentions that malformed `blocked` structure had to be normalized before conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The likely hotspots are the same as in 7-a: deserializing the per-animal files, materializing large `trace` and `position` arrays, computing binned/projected position labels for all frames, and copying every trial’s neural/output slices into Python lists.

ii.
```python
dat = load_animal(data_dir, animal)
trace = np.asarray(dat["trace"], dtype=np.float32)
position = np.asarray(dat["position"], dtype=np.float32)
...
binned_xy = bin_position_to_grid(day_pos, n_bins=3)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
...
for trial_idx in range(n_trials):
    ...
```

iii. No separate performance rationale is documented. The recorded justification is correctness and sanity-check coverage, not speed.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The same loops identified in 7-b apply here: the per-trial slicing loop, the nested 3x3 lookup construction, and the rescanning loop used to collect sample-session indices.

ii.
```python
for x in range(3):
    for y in range(3):
        ...

for trial_idx in range(n_trials):
    ...

for sess_idx, subj in enumerate(data["subject_idx"]):
    if int(subj) == subject_id:
        sessions_for_subject.append(sess_idx)
```

iii. The agent did not record any explicit optimization rationale; the implementation favors straightforward loops.

## 9-c. What processing does the code repeat multiple times?

i. The same repeated work from 7-c applies here: geometry lookups are rebuilt per session, static input vectors are recopied per trial, and the exported dataset is re-traversed for summaries and sample reporting.

ii.
```python
lookup = build_nearest_valid_lookup(env_mask)
...
session_trials_input.append(input_vec.copy())
...
"summary": summarize_dataset(data),
...
print(json.dumps(summarize_dataset(sample_data), indent=2))
```

iii. The agent’s stated rationale for the repeated summary work is to support sanity checks and validation reporting.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The same non-essential work from 7-d applies here: unused `ENV_TO_MASK`, extensive metadata/statistics gathering, per-environment bookkeeping, and the unused `start` accumulator in sample-session selection.

ii.
```python
ENV_TO_MASK = {
    ...
}
...
blocked_patterns_by_env.setdefault(env_name, set()).add(blocked_entry)
frame_lengths[T] += 1
...
"total_frames_reassigned_to_valid_bins": stats.total_frames_reassigned,
...
start = 0
...
start += len(sessions_for_subject)
```

iii. The notes explain that these extras exist for documentation and sanity checking, not because the downstream decoder consumes them.
