# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers the seven extensionless `QLAK-CA1-<number>` files in `./data`, loads each per-animal joblib dictionary, and iterates every day. Full mode processes all 207 sessions; sample mode stops after two sessions.

ii.
```python
for name in sorted(os.listdir(data_dir)):
    if os.path.isfile(path) and re.fullmatch(r"QLAK-CA1-\d+", name):
        animals.append(name)
...
return joblib.load(os.path.join(path, animal))[animal]
...
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    for day_idx in range(animal_data["trace"].shape[0]):
```

iii. The notes say these are the same per-animal joblib files used by the repository's `load_dat` path. Loading one animal at a time limits memory and reproduced the paper's counts of 7 mice and 207 sessions.

## 1-b. How are the data split into subjects?

i. Each matching file is one mouse. Sorted filenames are used verbatim as subject IDs, and a filename-to-index mapping supplies each session's `subject_idx`.

ii.
```python
subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
...
converted["subject_idx"].append(subject_to_idx[animal])
```

iii. The notes identify the data as seven per-animal files and report that the resulting subject/session counts match the source and paper.

## 1-c. How are the data split into sessions?

i. Every day along the first dimension of an animal's `trace` array becomes one output session, ordered by sorted animal and then native day index.

ii.
```python
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    ...
    converted["neural"].append(neural_trials)
```

iii. The agent understood each day as one continuous recording session and preserved all 207 released days, including the mouse with only 21 sessions.

## 1-d. How are the data split into trials?

i. A nominal 40-minute session is divided into up to 40 consecutive 60-second chunks (1,800 native frames). Unlike a complete-chunk rule, a shortened last chunk is retained whenever it contains at least three frames; frames after 40 minutes are discarded.

ii.
```python
usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
for trial_idx in range(NOMINAL_SESSION_SECONDS // TRIAL_SECONDS):
    start = trial_idx * TRIAL_FRAMES
    end = min(start + TRIAL_FRAMES, usable_frames)
    if end - start >= TEMPORAL_BIN_FRAMES:
        slices.append((start, end))
```

iii. The notes justify this as preserving the nominal 40-minute experimental duration while minimizing loss; thus all sessions have 40 trials, with shortened final trials in three animals.

## 1-e. How are trials filtered based on quality controls?

i. No behavioral or recording-quality trial filtering is applied. Chunks are retained if they contain at least one three-frame temporal bin; only data beyond 40 minutes are excluded.

ii.
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. The notes do not identify a trial-level QC criterion. They prioritize retaining the nominal 40 chunks and report validator success.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace`, described as the released binarized rising-phase calcium-event trace.

ii.
```python
trace_day = animal_data["trace"][day]
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. The paper/code exploration concluded that `trace` is already an event representation, so fluorescence or dF/F should not be recomputed.

## 2-b. How is the `neural` data processed?

i. Session-valid cells are selected, data are cast to float32, trial slices are taken, and every non-overlapping group of three 30 Hz frames is averaged to produce 100 ms samples.

ii.
```python
neural_raw = trace_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
...
return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The agent says three-frame average pooling matches temporal binning in the paper's decoder, reduces storage/training cost, and retains neural/behavior alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained for a day when its first trace sample is not NaN. No place-cell, activity-count, velocity, or other neural QC filter is applied.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. The agent interpreted NaNs as absent/unregistered cells and argued that all recorded cells should otherwise be included; it explicitly chose not to apply place-cell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental-event alignment. Time zero is the start of each consecutive one-minute chunk, with metadata describing offsets of 0 to 60 seconds.

ii.
```python
"temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The recordings are continuous free exploration without a trial event, so the agent used artificial chunk starts as the alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 100 ms. Both neural activity and position coordinates are averaged over non-overlapping groups of three native 30 Hz frames.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The notes cite the reference decoder's three-frame pooling and computational savings as justification.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from each day's raw `blocked` entry. `envs` is retained only as a metadata label.

ii.
```python
env_label = animal_data["envs"].reshape(-1)[day]
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. The agent found disagreements between abstract `envs` templates and the raw blocked partitions and chose `blocked` as the more direct description of actual session geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A length-nine vector starts at one (open); blocked indices are set to zero. The 3x3 vector is transposed/flattened to match the agent's x-major output class order. A copy is stored as a static 1D input for each trial.

ii.
```python
open_mask = np.ones(9, dtype=np.float32)
if not (blocked_values.size == 1 and blocked_values[0] == -1):
    open_mask[blocked_values.astype(int)] = 0.0
return open_mask.reshape(3, 3).T.reshape(-1)
...
input_trials.append(open_mask.copy())
```

iii. The notes call this a compact binary open/closed geometry representation and explain the transpose as reconciling raw y-major partitions with `x_bin * 3 + y_bin` outputs.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each day's two-coordinate `position` array.

ii.
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
```

iii. The notes identify the raw x-y trajectory as frame-aligned with `trace` and the direct source for the requested position output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are cast to float32, sliced identically to neural data, averaged within three-frame windows, then converted to one integer categorical row with values 0 through 8.

ii.
```python
position_raw = position_valid[:, start:end]
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. The agent chose one nine-class time-varying variable because it directly expresses the requested 3x3 decoding target.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided by one third of that session's maximum observed coordinate (plus a small buffer), floored and clipped to 0–2. The class is `x_bin * 3 + y_bin`.

ii.
```python
denom = (session_max_xy + BUFFER) / POSITION_BINS
binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
binned = np.clip(binned, 0, POSITION_BINS - 1)
classes = binned[0] * POSITION_BINS + binned[1]
```

iii. The notes say this adapts the reference decoder's session-max flooring rule from 15x15 to the requested 3x3 grid.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural streams use the same trial start/end indices and the same three-frame mean bins, yielding one output class per neural time bin.

ii.
```python
neural_raw = trace_valid[:, start:end]
position_raw = position_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
```

iii. The agent states the source streams are frame-synchronous and reports raw-to-converted spot checks confirming common bin alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neuron-day entries marked missing are removed using first-frame NaNs. Session coordinate maxima use `nanmax`; bin indices are clipped; sessions shorter than 40 minutes retain a shortened final trial; excess frames and incomplete three-frame tails are discarded. No general imputation is performed.

ii.
```python
valid_cells = ~np.isnan(trace_day[:, 0])
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
usable = (arr.shape[-1] // bin_size) * bin_size
trimmed = arr[..., :usable]
binned = np.clip(binned, 0, POSITION_BINS - 1)
```

iii. The notes treat NaN cell traces as registration absence, preserve shortened recordings to minimize loss, and report explicit spot checks of shortened sessions.

## 6-a. What are the most time-consuming steps of the code?

i. Loading/decompressing the large joblib animal files dominates conversion; optional plotting and serializing the multi-gigabyte pickle add secondary costs.

ii.
```python
animal_data = load_animal(data_dir, animal)
...
with open(out_path, "wb") as f:
    pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly identify decompression of seven large joblib files as the runtime bottleneck and time each animal/session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The animal/day loops are appropriate for file and session organization. The 40-trial loop and repeated per-trial slicing/mean operations could be vectorized for full-length sessions by reshaping the entire session, though shortened final trials complicate this. Trial geometry copies could also be constructed without a loop.

ii.
```python
for day_idx in range(ndays):
    ...
for start, end in trial_slices:
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES)
    input_trials.append(open_mask.copy())
```

iii. The agent only flags I/O as important and already vectorizes within-bin averaging via reshape/mean; it does not discuss vectorizing the trial loop.

## 6-c. What processing does the code repeat multiple times?

i. For every trial it separately slices neural and position arrays, computes usable multiples of three, reshapes/averages both streams, copies the same geometry vector, and casts outputs. Session metadata and status printing are also repeatedly appended.

ii.
```python
for start, end in trial_slices:
    neural_binned = temporal_bin_mean(trace_valid[:, start:end], TEMPORAL_BIN_FRAMES)
    position_binned = temporal_bin_mean(position_valid[:, start:end], TEMPORAL_BIN_FRAMES)
    input_trials.append(open_mask.copy())
```

iii. The notes emphasize streaming and vectorized pooling but give no specific justification for repeating these per-trial operations beyond straightforward organization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Conversion computes debug metadata such as environment labels, frame counts, session maxima, and trial slices; `trial_slices` is returned in `debug_info` but not saved. Optional plots assemble multiple previews used only for diagnostics. Timing/status calculations do not affect the dataset.

ii.
```python
debug_info = {
    "env_label": str(env_label),
    "open_mask": open_mask,
    "trial_slices": trial_slices,
    "session_max_xy": session_max_xy,
}
...
elapsed = time.perf_counter() - session_start
```

iii. The agent intentionally added these as sanity checks, audit metadata, processing plots, and runtime diagnostics; they are not claimed to be decoder features.
