# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, then loads one Python `joblib` file per animal from `/app/data/<animal>`. It does not enumerate the `.mat` files or use `h5py`; instead it reads the preconverted joblib dictionary for each animal and then iterates over the day/session axis inside that object.

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

iii. The justification in `CONVERSION_NOTES.md` is that the agent wanted to preserve the paper's day/session structure while using the provided animal files directly. In the trajectory it explicitly concluded that the repository "exposes exactly the fields we need" in the joblib dataset and proceeded from there rather than reconstructing the `.mat` loading path.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is treated as one subject. The output `subjects` list is just `ANIMALS`, and `subject_idx` is the index of that animal in the list.

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
```

iii. `CONVERSION_NOTES.md` states that the subjects are the seven animal IDs and lists them explicitly. The trajectory summary also says "Each original recording day was kept as a session," implying each animal file is the subject container.

## 1-c. How are the data split into sessions?

i. The AI treats each day in the `trace`/`position` arrays as one session. It iterates over `range(trace.shape[0])`, with one output session per day.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
position = np.asarray(dat["position"], dtype=np.float32)

for day in range(trace.shape[0]):
    ...
    day_trace = trace[day]
    day_pos = position[day]
    ...
    neural.append(session_trials_neural)
    decoder_input.append(session_trials_input)
    decoder_output.append(session_trials_output)
```

iii. `CONVERSION_NOTES.md` says "I treated each original recording day as one decoder session" and says this matches the paper code's iteration over day/session axes in `trace`, `position`, and `envs`.

## 1-d. How are the data split into trials?

i. Sessions are split into contiguous, non-overlapping 60 s trials at 30 Hz, so each trial is 1800 frames. The code drops any trailing remainder that does not fill a full 60 s window.

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

iii. `CONVERSION_NOTES.md` says the source recordings are continuous, so the agent "split each session into contiguous full 60 s windows" and discarded incomplete tails. The trajectory also says this was the one explicit decoder-task adaptation.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a trial-quality screen beyond requiring at least two complete 60 s trials in a session. Incomplete tail frames are discarded rather than kept as shorter trials.

ii.
```python
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
dropped = T - n_trials * TRIAL_FRAMES
stats.total_frames_dropped += int(dropped)
```

iii. The explicit justification comes from the decoder-format requirement in the instructions and from `CONVERSION_NOTES.md`, which emphasizes keeping only complete 60 s windows. There is no additional trial-QC rationale documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are derived from `dat["trace"]` in the per-animal joblib files. The AI interprets this field as binary rising-phase calcium-event activity.

ii.
```python
dat = load_animal(data_dir, animal)
trace = np.asarray(dat["trace"], dtype=np.float32)
...
day_trace = trace[day]
```

iii. `CONVERSION_NOTES.md` says "I used the provided binary rise-event traces directly from `data/<animal>`" and that this matched the paper/methods description that analyses use binarized rising phases of calcium transients.

## 2-b. How is the `neural` data processed?

i. The AI does very little signal processing. It keeps the session slice of `trace`, removes unregistered neurons, casts to `float32`, and leaves the native frame rate unchanged. Because the joblib array is already `(day, cell, time)`, it does not transpose.

ii.
```python
day_trace = trace[day]
registered_today = ~np.isnan(day_trace[:, 0])
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. `CONVERSION_NOTES.md` says it "did not re-deconvolve, smooth, or re-threshold the calcium traces" and used the provided binary rise-event traces directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered per session/day using NaN-based registration status. The AI keeps neurons whose first frame is not NaN, checks that retained neurons contain no NaNs anywhere, and checks that excluded neurons are NaN throughout the session.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. `CONVERSION_NOTES.md` says the source files mark unregistered neurons as all-NaN for that day and that these were dropped from each session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external behavioral or stimulus event. The AI treats the start of each contiguous 60 s trial window as the alignment event and records that in metadata.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
```

iii. The justification is implicit: `CONVERSION_NOTES.md` says the source recordings are continuous long sessions rather than pre-segmented trials, so the trial window itself becomes the anchor for decoder formatting.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at 30 Hz, with `1000 / 30 = 33.33 ms` time bins. No temporal rebinning or downsampling is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"time_bin_size": TIME_BIN_MS,
```

iii. `CONVERSION_NOTES.md` says the agent used the provided session data directly and only trialized it; it did not document any temporal resampling step.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from `dat["blocked"]`, not from `envs` alone. `envs` is read for summaries, but the actual input geometry comes from the blocked-bin metadata.

ii.
```python
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_name = str(envs[day])
env_mask = mask_from_blocked(blocked_entry)
...
input_vec = env_mask.reshape(-1).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` states that the `blocked` field was treated as the authoritative geometry/orientation source because the environment name alone did not fully specify orientation in this dataset.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI normalizes the blocked metadata into a sorted tuple of blocked bin IDs, converts that into a `3 x 3` mask with `1` for open bins and `0` for blocked bins, flattens it to length 9, and reuses the same static vector for every trial in that session.

ii.
```python
def normalize_blocked_entry(entry) -> tuple[int, ...]:
    ...
    if arr.size == 1 and arr[0] == -1:
        return ()
    return tuple(sorted(arr.tolist()))

def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask

input_vec = env_mask.reshape(-1).astype(np.float32)
...
session_trials_input.append(input_vec.copy())
```

iii. `CONVERSION_NOTES.md` says the input is a static 9D vector per trial, with `1` meaning open and `0` meaning blocked, and that this was chosen to preserve session geometry while fitting the decoder task.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from `dat["position"]`, specifically the per-day `(x, y)` trajectories.

ii.
```python
position = np.asarray(dat["position"], dtype=np.float32)
...
day_pos = position[day]
```

iii. The code itself shows this directly. `CONVERSION_NOTES.md` discusses position discretization and says the raw x/y positions were binned into a `3 x 3` spatial output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first bins `x` and `y` coordinates into `3 x 3` coarse bins using per-axis maxima and no minimum subtraction. It then projects any timepoints that land in blocked bins onto the nearest valid open bin for that session geometry.

ii.
```python
def bin_position_to_grid(position_xy: np.ndarray, n_bins: int = 3) -> np.ndarray:
    max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
    bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)

lookup = build_nearest_valid_lookup(env_mask)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
reassigned = np.any(projected_xy != binned_xy, axis=0)
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the agent believed this matched the paper code's max-based binning style, and that nearest-valid-bin reassignment was a decoder-task adaptation "in the spirit of" the paper's geometry-aware decoder evaluation.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI converts each frame to one of 9 categories after coarse binning and blocked-bin cleanup. The category index is `x_bin * 3 + y_bin`, and the labels are named `x0_y0` through `x2_y2`.

ii.
```python
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
...
"output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
"metadata": {
    ...
    "output_representation": "single categorical position label 0..8 with label index = x_bin * 3 + y_bin",
```

iii. `CONVERSION_NOTES.md` explicitly documents this label definition and the 9 category names.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned frame-for-frame within each day/session. The AI assumes they already share the same number of frames, checks that, and slices both with the same trial boundaries.

ii.
```python
if trace.shape[2] != position.shape[2]:
    raise ValueError(f"{animal}: trace and position disagree on number of frames")

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The justification is implicit in both the code and notes: the source recordings are continuous and already synchronized, so identical frame slicing is used for both signals.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly fails fast instead of repairing data. It normalizes nested/`-1` blocked entries, drops incomplete trailing trial fragments, removes unregistered neurons, and raises errors if positions contain NaNs or if the NaN pattern for traces is inconsistent with the registration assumption.

ii.
```python
if np.isnan(day_pos).any():
    raise ValueError(f"{animal} day {day}: position contains NaNs")

registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

dropped = T - n_trials * TRIAL_FRAMES
```

iii. `CONVERSION_NOTES.md` only explicitly justifies dropping unregistered neurons and dropping incomplete tails. The stricter fail-fast checks are not separately justified there; they appear to have been added as sanity checks.

## 6-a. What are the most time-consuming steps of the code?

i. The heavy steps are loading each full animal file, scanning every session's full trace/position arrays, doing per-frame position binning and blocked-bin projection, and then copying every trial slice into new arrays.

ii.
```python
for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    ...
    for day in range(trace.shape[0]):
        ...
        binned_xy = bin_position_to_grid(day_pos, n_bins=3)
        lookup = build_nearest_valid_lookup(env_mask)
        projected_xy = lookup[binned_xy[0], binned_xy[1]].T
        ...
        for trial_idx in range(n_trials):
            ...
            session_trials_neural.append(session_neural[:, start:stop].copy())
            session_trials_input.append(input_vec.copy())
            session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. This is not explicitly discussed in `CONVERSION_NOTES.md`. It is inferred from the code structure and from the trajectory, which focused on full-dataset counts and large frame-level reassignment totals.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing/copy loop is the clearest vectorization target. The per-bin loop in `mask_from_blocked` and the `3 x 3` nearest-valid lookup construction are also loops, though small. The session and subject loops are structurally necessary.

ii.
```python
for idx in blocked_bins:
    mask[idx // 3, idx % 3] = 0.0

for x in range(3):
    for y in range(3):
        ...

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The AI did not document a justification for keeping these loops. The code reads as straightforward implementation rather than an optimized one.

## 6-c. What processing does the code repeat multiple times?

i. It repeatedly rebuilds session-level geometry lookup data, repeatedly copies the same static `input_vec` once per trial, and separately recomputes dataset summaries for the full and sample exports.

ii.
```python
lookup = build_nearest_valid_lookup(env_mask)
...
for trial_idx in range(n_trials):
    ...
    session_trials_input.append(input_vec.copy())

sanity = {
    "summary": summarize_dataset(data),
    ...
}
...
print(json.dumps(summarize_dataset(sample_data), indent=2))
```

iii. There is no explicit justification in the notes. The repeated work appears to support clarity and sanity reporting rather than efficiency.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script tracks extensive sanity statistics and metadata that `train_decoder.py` does not use, builds a sample dataset in the same run as the full dataset, calls `gc.collect()`, and defines `ENV_TO_MASK` even though the conversion logic actually uses `blocked` plus `mask_from_blocked` instead.

ii.
```python
ENV_TO_MASK = {
    "square": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
    ...
}

stats = ConversionStats()
per_animal_unique_neurons = {}
per_subject_sessions = Counter()
env_counts = Counter()
blocked_patterns_by_env: dict[str, set[tuple[int, ...]]] = {}
frame_lengths = Counter()
dropped_seconds_by_session = []
...
del dat, trace, position
gc.collect()
...
sample_data = subset_sessions(data, sample_session_indices)
```

iii. `CONVERSION_NOTES.md` does justify the sanity-checking effort in general, but not these particular extras. They appear to have been added for validation convenience rather than because downstream decoding required them.
