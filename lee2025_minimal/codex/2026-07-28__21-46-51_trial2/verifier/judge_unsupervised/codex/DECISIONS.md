# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads one preprocessed joblib file per animal from `/app/data`, then pulls the `envs`, `trace`, `position`, and `blocked` fields from each file. It does not read the `.mat` files directly. Trials are not loaded separately; they are created later by slicing each per-day session.

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

iii. In `CONVERSION_NOTES.md`, the AI says it wanted to stay close to the “provided binary rise-event traces directly from `data/<animal>`” and to match the per-day/session structure used by the paper code. The trajectory shows it inspected the joblib layout first and then chose to build the converter around those files.

## 1-b. How are the data split into subjects?

i. The AI treats each animal file as one subject and assigns subjects in the fixed `ANIMALS` list order. `subjects` is exactly that list, and `subject_idx` records which animal each session came from.

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
    ...
}
```

iii. `CONVERSION_NOTES.md` explicitly lists the seven mouse IDs and says these are the subjects in the exported dataset. The trajectory also shows the AI matching the paper code’s `animals = [...]` list.

## 1-c. How are the data split into sessions?

i. The AI treats each recording day as one session. For every subject, it iterates over the first axis of `trace` and `position`; each `day` becomes one exported session.

ii. 
```python
for day in range(trace.shape[0]):
    ...
    day_trace = trace[day]
    day_pos = position[day]
    ...
    neural.append(session_trials_neural)
    decoder_input.append(session_trials_input)
    decoder_output.append(session_trials_output)
```

iii. `CONVERSION_NOTES.md` says “I treated each original recording day as one decoder session,” justified by the paper code iterating over the day/session axis in `trace`, `position`, and `envs`. The methods text also says one session was recorded per day.

## 1-d. How are the data split into trials?

i. The AI converts each continuous session into contiguous non-overlapping 60 s trials. With `FPS = 30`, each trial is 1800 frames. It keeps only complete windows and drops the trailing remainder.

ii. 
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
dropped = T - n_trials * TRIAL_FRAMES

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. `CONVERSION_NOTES.md` calls this a decoder-specific adaptation because the source recordings are continuous sessions, not trials. The justification given is that the task instructions explicitly asked for 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies almost no explicit trial-quality filtering. It only requires at least two full 60 s windows per session, and it discards any incomplete trailing chunk shorter than 60 s. It does not reject trials based on motion, occupancy, or neural activity.

ii. 
```python
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
dropped = T - n_trials * TRIAL_FRAMES
stats.total_frames_dropped += int(dropped)
```

iii. The notes justify this only indirectly: the AI says the recordings are long continuous sessions and the task requires trialization. I did not find a separate justification for skipping stronger per-trial QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` entirely from the raw `trace` variable in each animal file.

ii. 
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
...
day_trace = trace[day]
...
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. `CONVERSION_NOTES.md` states that the AI used the provided binary rise-event traces directly and tied that choice to the paper/methods description that later analyses use the binarized rising phase of calcium transients.

## 2-b. How is the `neural` data processed?

i. The AI does not re-deconvolve, smooth, threshold, or temporally rebin the neural signal. It keeps the binary event trace, casts it to `float32`, removes neurons judged unregistered on that day, and slices the time axis into trials.

ii. 
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
...
registered_today = ~np.isnan(day_trace[:, 0])
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
...
session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. `CONVERSION_NOTES.md` explicitly says: “I used the provided binary rise-event traces directly” and “I did not re-deconvolve, smooth, or re-threshold the calcium traces.” The trajectory shows the AI grounding that in the methods text about the binarized rising-phase vector.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons only by registration / missingness. For each day, it keeps neurons whose first frame is not `NaN`, assumes those are the registered cells for that session, and checks that registered neurons contain no `NaN`s while unregistered neurons are all-`NaN`. It does not apply place-cell, velocity, or activity-threshold filtering.

ii. 
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. `CONVERSION_NOTES.md` says “For each session, I kept only neurons registered on that day” and “No additional place-cell or activity-threshold filtering was applied.” The justification given is to stay faithful to the recorded session content while avoiding `NaN`s.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural trials to the start of each contiguous 60 s window. There is no experimental event marker; trial start itself is treated as the alignment event.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())

"metadata": {
    ...
    "temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
    ...
}
```

iii. The AI’s metadata and notes frame this as a trialization choice forced by the task: the source data are continuous free-exploration recordings, so the start of each artificial 1-minute segment becomes the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI exports the converted data at the native 30 Hz sampling rate, i.e. 33.33 ms per time bin. No temporal rebinning is applied.

ii. 
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "fps": FPS,
    "trial_frames": TRIAL_FRAMES,
    ...
}
```

iii. `CONVERSION_NOTES.md` says the source traces were used directly and that each 60 s trial therefore contains 1800 time bins. No separate justification for not rebinnig appears beyond preserving the raw sampling.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives geometry mainly from the raw `blocked` field, not from `envs`. It still reads `envs`, but only for bookkeeping and sanity summaries.

ii. 
```python
envs = flatten_envs(dat["envs"])
...
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_name = str(envs[day])
env_mask = mask_from_blocked(blocked_entry)
```

iii. `CONVERSION_NOTES.md` says “I derived this from the `blocked` metadata, not only from the environment name.” The trajectory makes the same point after the first conversion attempt failed an internal check based on `env` names alone.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI normalizes the nested MATLAB-style `blocked` entries, converts the blocked-bin list into a `3 x 3` mask with `1` for open bins and `0` for blocked bins, flattens that to length 9, and reuses the same static vector for every trial in the session.

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
...
session_trials_input.append(input_vec.copy())
```

iii. `CONVERSION_NOTES.md` justifies this as a decoder-specific representation: a static 9D geometry vector per trial. It also says `blocked` was chosen because the AI believed the environment string did not fully specify orientation.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives the decoder output from the raw `position` variable.

ii. 
```python
position = np.asarray(dat["position"], dtype=np.float32)
...
day_pos = position[day]
...
binned_xy = bin_position_to_grid(day_pos, n_bins=3)
```

iii. The notes describe the output as mouse position discretized from the source position stream and emphasize that the task asked for a 3 x 3 categorical position output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI bins each session’s raw `x` and `y` positions into a `3 x 3` grid using per-axis maxima and floor division, then projects any bins that land in blocked geometry partitions to the nearest valid open bin for that environment.

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

...
binned_xy = bin_position_to_grid(day_pos, n_bins=3)
lookup = build_nearest_valid_lookup(env_mask)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the max-based binning was chosen to match the paper code’s spatial-scaling logic, and that the nearest-valid-bin projection was added as a decoder-task adaptation so blocked coarse bins would be reassigned to valid positions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI converts each `(x_bin, y_bin)` pair into a single categorical label with the formula `x_bin * 3 + y_bin`, producing categories `0..8`.

ii. 
```python
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)

"output_names": ["position_bin"],
"output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
```

iii. `CONVERSION_NOTES.md` explicitly documents the `x_bin * 3 + y_bin` label definition and lists the corresponding `x0_y0` ... `x2_y2` output labels.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output positions to neural data by using exactly the same per-session frame indices and the same 60 s trial boundaries. It does not introduce any extra lag, interpolation, or temporal pooling.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The justification is implicit in both the notes and metadata: the data are continuous, recorded at one shared 30 Hz frame clock, so the AI simply slices neural and position streams with the same `[start:stop]` windows.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several quirks explicitly: it normalizes nested `blocked` entries, interprets `-1` as “no blocked bins,” drops neurons that are all `NaN` on a day, raises errors if position contains `NaN`s or if trace missingness is inconsistent, drops incomplete trailing trial fragments, and snaps blocked coarse-bin positions to the nearest open bin instead of leaving them missing.

ii. 
```python
def normalize_blocked_entry(entry) -> tuple[int, ...]:
    ...
    if arr.size == 1 and arr[0] == -1:
        return ()
    return tuple(sorted(arr.tolist()))

if np.isnan(day_pos).any():
    raise ValueError(f"{animal} day {day}: position contains NaNs")

if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

projected_xy = lookup[binned_xy[0], binned_xy[1]].T
```

iii. `CONVERSION_NOTES.md` describes the `blocked` normalization and nearest-valid-bin cleanup as practical fixes for dataset quirks and coarse 3 x 3 discretization artifacts. The trajectory shows the AI discovering the MATLAB-style nested `blocked` field and then adding normalization logic around it.

## 6-a. What are the most time-consuming steps of the code?

i. The slowest parts are loading the large animal files, iterating over every session/day, slicing every session into trials, computing position bin assignments for all frames, and then serializing / summarizing the full converted dataset. The code also scans all outputs again in `summarize_dataset`.

ii. 
```python
for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    ...
    for day in range(trace.shape[0]):
        ...
        binned_xy = bin_position_to_grid(day_pos, n_bins=3)
        ...
        for trial_idx in range(n_trials):
            ...
            session_trials_neural.append(session_neural[:, start:stop].copy())
            session_trials_output.append(output_position[np.newaxis, start:stop].copy())

with open(args.full_out, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. I did not find an explicit performance discussion in `CONVERSION_NOTES.md`. This is inferred from the code structure and from the trajectory, where full-dataset scans and conversions were the long-running steps.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial append loop could have been replaced with reshaping / splitting whole session arrays. The nested loops in `build_nearest_valid_lookup` are tiny but still manual. The nested loops in `summarize_dataset` and the loop collecting sample sessions also re-scan Python lists element by element.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())

for x in range(3):
    for y in range(3):
        ...

for session in data["output"]:
    for trial in session:
        vals, counts = np.unique(trial[0], return_counts=True)
        ...
```

iii. No explicit justification was given by the AI. This is a code-level efficiency observation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several scans of already-built structures: it summarizes the converted data more than once, scans `subject_idx` again to make the sample dataset, rebuilds geometry lookup state for every session even though there are only 10 geometries, and re-traverses all outputs to compute class counts.

ii. 
```python
sanity = {
    "summary": summarize_dataset(data),
    ...
}

sample_session_indices = []
for subject_id in range(min(args.sample_subject_count, len(ANIMALS))):
    sessions_for_subject = []
    for sess_idx, subj in enumerate(data["subject_idx"]):
        if int(subj) == subject_id:
            sessions_for_subject.append(sess_idx)

print(json.dumps(summarize_dataset(sample_data), indent=2))
```

iii. There is no explicit justification for these repeated passes. They appear to be convenience / reporting logic rather than a deliberate algorithmic choice.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter does a fair amount of bookkeeping that is not needed by decoder training: environment counts, blocked-pattern inventories, frame-length histograms, reassignment counters, dropped-tail statistics, full dataset summaries, and sample-dataset generation. The `reassigned` mask is only used for a count statistic. `ENV_TO_MASK` is defined but never used in the final conversion.

ii. 
```python
ENV_TO_MASK = {
    "square": ...,
    ...
}

env_counts[env_name] += 1
blocked_patterns_by_env.setdefault(env_name, set()).add(blocked_entry)
frame_lengths[T] += 1
...
reassigned = np.any(projected_xy != binned_xy, axis=0)
stats.total_frames_reassigned += int(np.sum(reassigned))
...
sample_data = subset_sessions(data, sample_session_indices)
```

iii. The notes justify some of this as sanity checking and validation against paper-level counts, but it is not needed for the downstream decoder itself.
