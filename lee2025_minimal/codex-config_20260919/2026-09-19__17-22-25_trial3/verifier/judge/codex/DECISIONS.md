# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of seven animals, then loads one joblib file per animal from `/app/data/<animal>`. From each loaded object it reads `trace`, `position`, `envs`, and `blocked`, and then iterates through recording days to build sessions and trials.

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
...
for animal_index, animal in enumerate(ANIMALS):
    source_path = data_dir / animal
    print(f"Loading {source_path}", flush=True)
    source = joblib.load(source_path)[animal]
    traces = source["trace"]
    positions = source["position"]
    environments = np.asarray(source["envs"]).squeeze()
    blocked = source["blocked"]
```

iii. The trajectory says the "source joblib files already contain the final, rise-extracted calcium-event trains and frame-aligned head position used by the paper" (step 11). The AI also states it confirmed each source day is a frame-aligned ~40-minute recording and therefore used those joblib files directly rather than reconstructing sessions from the `.mat` files (step 18).

## 1-b. How are the data split into subjects?

i. Subjects are defined by the fixed `ANIMALS` list. The output `subjects` field is just `ANIMALS.copy()`, and every session gets the integer index of the corresponding animal from that list.

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
...
for animal_index, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(animal_index)
...
"subjects": ANIMALS.copy(),
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. In the trajectory the AI repeatedly describes the dataset as seven animals and reports "207 recording-day sessions" across those animals (steps 29 and 39). It therefore treated each named animal as a subject and each day within that animal as a session.

## 1-c. How are the data split into sessions?

i. Each recording day inside an animal file becomes one session. The code loops over `range(traces.shape[0])`, where each index corresponds to one day/session, and appends one session entry to `neural`, `input`, and `output`.

ii.
```python
if traces.shape[0] != positions.shape[0] or traces.shape[0] != len(blocked):
    raise ValueError(f"Inconsistent session count for {animal}")

for day in range(traces.shape[0]):
    trace = traces[day]
    position = positions[day]
    ...
    neural.append(session_neural)
    decoder_input.append(session_input)
    decoder_output.append(session_output)
    subject_idx.append(animal_index)
```

iii. The AI states: "I’m treating each recording day as one decoder session" (step 18). Later it summarizes the result as "207 recording-day sessions" (step 29).

## 1-d. How are the data split into trials?

i. After temporal pooling to 10 Hz, each session is split into consecutive 600-bin chunks, intended to represent 60-second trials. The code keeps a final shorter chunk if it is at least 30 seconds long, and drops only shorter trailing overhangs.

ii.
```python
SOURCE_FPS = 30
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * SOURCE_FPS / POOL_FRAMES)
MIN_FINAL_TRIAL_SECONDS = 30.0
...
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
    stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
    duration_s = (stop - start) * TIME_BIN_MS / 1000.0
    if duration_s < MIN_FINAL_TRIAL_SECONDS:
        break
    session_neural.append(
        np.ascontiguousarray(pooled_neural[:, start:stop], dtype=np.float32)
    )
    session_input.append(geometry.copy())
    session_output.append(labels[np.newaxis, start:stop].copy())
```

iii. The AI says it wanted contiguous 1-minute trials and therefore kept continuous behavior within each minute rather than removing low-speed samples (step 18). It later states its policy explicitly: "Each nominal 40-minute recording yields 40 trials; only tiny post-40-minute overhangs are dropped, while the three slightly short recordings retain their ~55.5-second final trial" (step 22).

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a behavioral or artifact quality-control filter to ordinary trials. The only trial-level filtering is dropping the final chunk if it is shorter than 30 seconds; it also rejects entire sessions if fewer than two trials remain.

ii.
```python
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
    stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
    duration_s = (stop - start) * TIME_BIN_MS / 1000.0
    if duration_s < MIN_FINAL_TRIAL_SECONDS:
        break
    ...

if len(session_neural) < 2:
    raise ValueError(f"Too few trials for {animal}, day {day}")
```

iii. The trajectory says the AI deliberately kept low-speed samples because removing them would "destroy the requested minute trials" (step 18), so it avoided per-trial behavioral masking. Step 22 explains the only filtering it kept at the trial level was removing tiny trailing overhangs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the `trace` field in each animal joblib file. It interprets `trace` as rise-extracted binary calcium-event trains rather than raw fluorescence.

ii.
```python
source = joblib.load(source_path)[animal]
traces = source["trace"]
...
"neural_signal": "rise-extracted binary calcium events",
```

iii. The trajectory explicitly says the "source files already contain the paper’s final rise-extracted binary calcium events" and therefore no signal extraction would be redone (step 11).

## 2-b. How is the `neural` data processed?

i. The AI first computes a movement mask from position, then uses that mask to count events for neuron filtering. For retained neurons it casts to `float32`, applies a 1-D Gaussian smoothing filter with `sigma=3` frames, and then averages non-overlapping groups of three source frames to produce 10 Hz neural time bins.

ii.
```python
def moving_samples(position: np.ndarray) -> np.ndarray:
    bin_cm = ARENA_SIZE_CM / PAPER_DECODER_SPATIAL_BINS
    position_15 = position / bin_cm
    speed_bins_s = np.linalg.norm(np.diff(position_15, axis=1), axis=0) * SOURCE_FPS
    speed_bins_s = gaussian_filter1d(
        speed_bins_s, sigma=VELOCITY_FILTER_SIGMA_FRAMES
    )
    moving = np.zeros(position.shape[1], dtype=bool)
    moving[1:] = speed_bins_s > (VELOCITY_THRESHOLD_CM_S / bin_cm)
    return moving
...
selected = np.asarray(trace[keep_cells, :usable_frames], dtype=np.float32)
smoothed = gaussian_filter1d(
    selected, sigma=NEURAL_FILTER_SIGMA_FRAMES, axis=1
)
pooled_neural = smoothed.reshape(
    selected.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
```

iii. Step 22 states the chosen preprocessing explicitly: "paper-matched Gaussian smoothing (σ=3 frames) followed by non-overlapping 3-frame averaging (10 Hz)." Step 11 says this was meant to preserve the paper decoder’s smoothing/downsampling behavior while adapting to the requested output format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is kept only if it is registered on that source day and has more than five calcium events during periods when the mouse is moving faster than 5 cm/s. If no such cells remain, the session is rejected with an error.

ii.
```python
moving = moving_samples(position)
registered = np.isfinite(trace[:, 0])
moving_event_count = np.nansum(trace[:, moving], axis=1)
keep_cells = registered & (moving_event_count > MIN_MOVING_EVENTS)
if not np.any(keep_cells):
    raise ValueError(f"No eligible cells for {animal}, day {day}")
```

iii. The AI explains this in two places: step 18 says it is "dropping only unregistered/insufficiently active cells as the paper’s position decoder does," and step 22 restates the exact threshold as the "paper decoder’s >5-event moving-period neuron threshold."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no task-event alignment. The AI treats the start of each contiguous 1-minute recording segment as the alignment point, after pooling the continuous recording into 100 ms bins.

ii.
```python
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
    stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
    ...
    session_neural.append(
        np.ascontiguousarray(pooled_neural[:, start:stop], dtype=np.float32)
    )
...
"temporal_alignment_event": "start of each contiguous 1-minute recording segment",
"off_start": 0.0,
"off_end": 60.0,
```

iii. Step 11 says the source files already contain "frame-aligned head position" with the neural activity. Step 18 then says the recordings are continuous and that the AI kept contiguous 1-minute chunks rather than event-based windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. The AI rebins from 30 Hz to 10 Hz by averaging non-overlapping groups of three source frames; no further temporal rebinning is applied afterward.

ii.
```python
SOURCE_FPS = 30
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
...
def pool_position(position: np.ndarray, usable_frames: int) -> np.ndarray:
    return position[:, :usable_frames].reshape(2, -1, POOL_FRAMES).mean(axis=2)
...
pooled_neural = smoothed.reshape(
    selected.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
...
"time_bin_size": TIME_BIN_MS,
```

iii. Step 22 explicitly says the data use "non-overlapping 3-frame averaging (10 Hz)." Step 11 says the AI chose this to retain the paper decoder’s temporal binning while keeping the converted dataset manageable.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input for environment geometry is derived from the per-day `blocked` entries in the joblib file. The `envs` field is read too, but only used for metadata, not as a decoder input.

ii.
```python
positions = source["position"]
environments = np.asarray(source["envs"]).squeeze()
blocked = source["blocked"]
...
geometry = blocked_mask(blocked[day])
```

iii. In step 13 the AI explicitly inspects the `blocked` contents across several named environments and confirms their structure. Step 22 summarizes the final choice as "nine static 'blocked-bin' indicators."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI flattens the `blocked` values, ignores negative sentinel values, and writes a length-9 binary mask where `1` means that spatial bin is blocked. That vector is constant for every trial from the same recording day.

ii.
```python
def blocked_mask(blocked: object) -> np.ndarray:
    mask = np.zeros(N_SPATIAL_BINS**2, dtype=np.float32)
    values = np.asarray(blocked, dtype=float).reshape(-1)
    for value in values[np.isfinite(values)]:
        index = int(value)
        if index >= 0:
            if index >= mask.size:
                raise ValueError(f"Invalid blocked-bin index {index}")
            mask[index] = 1.0
    return mask
...
geometry = blocked_mask(blocked[day])
...
session_input.append(geometry.copy())
```

iii. Step 22 states the final representation as "nine static 'blocked-bin' indicators." The code comment also says geometry is static during a recording day, so the AI chose a 1-D vector rather than repeating it across time within each trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in the joblib file. The code uses the 2-D coordinates for each recording day.

ii.
```python
positions = source["position"]
...
position = positions[day]
```

iii. Step 11 says the joblib files contain "frame-aligned head position used by the paper," which is the signal the AI uses for the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first averages every non-overlapping group of three source frames to get pooled 2-D positions, then discretizes those pooled positions into a 3x3 grid over the 75 cm arena.

ii.
```python
def pool_position(position: np.ndarray, usable_frames: int) -> np.ndarray:
    return position[:, :usable_frames].reshape(2, -1, POOL_FRAMES).mean(axis=2)

def discretize_position(position: np.ndarray) -> np.ndarray:
    xy = np.floor(
        position / ((ARENA_SIZE_CM + np.finfo(np.float64).eps * 64) / N_SPATIAL_BINS)
    ).astype(np.int64)
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
    return xy[0] * N_SPATIAL_BINS + xy[1]
...
pooled_position = pool_position(position, usable_frames)
labels = discretize_position(pooled_position)
```

iii. Step 22 says the AI chose "3×3 binning over the documented 75 cm arena" together with "non-overlapping 3-frame averaging (10 Hz)." This is presented as an adaptation of the paper decoder to the requested 3x3 output task.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each pooled `x` and `y` coordinate is converted to one of three bins by floor division over the arena extent and clipping into `[0, 2]`. The final category label is `3 * x_bin + y_bin`, yielding labels `0..8`.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    xy = np.floor(
        position / ((ARENA_SIZE_CM + np.finfo(np.float64).eps * 64) / N_SPATIAL_BINS)
    ).astype(np.int64)
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
    return xy[0] * N_SPATIAL_BINS + xy[1]
...
output_values = [
    f"bin_{x * N_SPATIAL_BINS + y}_(x{x}_y{y})"
    for x in range(N_SPATIAL_BINS)
    for y in range(N_SPATIAL_BINS)
]
```

iii. The trajectory does not contain a fuller separate defense of the `3 * x_bin + y_bin` label order. The visible rationale is step 22's statement that the decoder output should be 3x3 spatial bins, plus the code comment that this is the dataset's "documented 0..8 grid order."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned with neural data by using the same `usable_frames`, the same 3-frame pooling boundaries, and the same per-trial `start`/`stop` indices. The categorical labels are sliced in lockstep with `pooled_neural`.

ii.
```python
usable_frames = (trace.shape[1] // POOL_FRAMES) * POOL_FRAMES
selected = np.asarray(trace[keep_cells, :usable_frames], dtype=np.float32)
...
pooled_neural = smoothed.reshape(
    selected.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
pooled_position = pool_position(position, usable_frames)
labels = discretize_position(pooled_position)
...
session_output.append(labels[np.newaxis, start:stop].copy())
```

iii. Step 11 says neural activity and position are already frame-aligned in the source data. The AI then preserves that alignment by applying the same pooling and trial slicing to both signals.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI treats missing cross-day registrations as NaN-marked neurons and removes unregistered cells using `np.isfinite(trace[:, 0])`. It uses `np.nansum` when counting movement-period events, drops incomplete trailing 3-frame groups, discards a final trial if it is shorter than 30 seconds, and raises hard errors for inconsistent session counts, trace/position length mismatches, no eligible neurons, or too few trials.

ii.
```python
if traces.shape[0] != positions.shape[0] or traces.shape[0] != len(blocked):
    raise ValueError(f"Inconsistent session count for {animal}")
...
if trace.shape[1] != position.shape[1]:
    raise ValueError(f"Unaligned trace/position for {animal}, day {day}")
...
registered = np.isfinite(trace[:, 0])
moving_event_count = np.nansum(trace[:, moving], axis=1)
keep_cells = registered & (moving_event_count > MIN_MOVING_EVENTS)
if not np.any(keep_cells):
    raise ValueError(f"No eligible cells for {animal}, day {day}")
...
usable_frames = (trace.shape[1] // POOL_FRAMES) * POOL_FRAMES
...
if duration_s < MIN_FINAL_TRIAL_SECONDS:
    break
...
if len(session_neural) < 2:
    raise ValueError(f"Too few trials for {animal}, day {day}")
```

iii. Step 18 says the AI confirmed "absent cross-day registrations [are] encoded as NaNs." Step 22 explains the handling of minor length mismatches at the end of recordings: tiny overhangs are dropped, while final chunks of about 55.5 seconds are kept.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive parts of this code are loading the large per-animal joblib files, smoothing and pooling the neural traces for every day, and writing the final 6.2 GB pickle.

ii.
```python
for animal_index, animal in enumerate(ANIMALS):
    source_path = data_dir / animal
    print(f"Loading {source_path}", flush=True)
    source = joblib.load(source_path)[animal]
...
smoothed = gaussian_filter1d(
    selected, sigma=NEURAL_FILTER_SIGMA_FRAMES, axis=1
)
pooled_neural = smoothed.reshape(
    selected.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
...
with output_path.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory does not contain a formal profile, but it does show the conversion producing a 6.2 GB pickle (step 34) and later describes the end-to-end run as "computationally heavier" because of per-session processing across all 207 sessions (step 46).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit Python loop over blocked indices in `blocked_mask` could be replaced with indexed assignment on the whole array, and the trial-building loop could be vectorized into reshaping/splitting operations instead of repeated Python `append` calls. The current code keeps both as Python loops.

ii.
```python
for value in values[np.isfinite(values)]:
    index = int(value)
    if index >= 0:
        if index >= mask.size:
            raise ValueError(f"Invalid blocked-bin index {index}")
        mask[index] = 1.0
...
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
    stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
    ...
    session_neural.append(
        np.ascontiguousarray(pooled_neural[:, start:stop], dtype=np.float32)
    )
    session_input.append(geometry.copy())
    session_output.append(labels[np.newaxis, start:stop].copy())
```

iii. The trajectory does not show the AI discussing vectorization or runtime optimization at this granularity; this observation is from the written code itself.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same trial-slicing logic for every session and repeatedly copies the same static geometry vector once per trial. It also repeats movement-mask computation and neuron eligibility checks for every recording day.

ii.
```python
moving = moving_samples(position)
registered = np.isfinite(trace[:, 0])
moving_event_count = np.nansum(trace[:, moving], axis=1)
keep_cells = registered & (moving_event_count > MIN_MOVING_EVENTS)
...
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
    stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
    ...
    session_input.append(geometry.copy())
    session_output.append(labels[np.newaxis, start:stop].copy())
```

iii. The only explicit rationale in the code is the comment that geometry is static within a recording day, so the AI chose to attach a copy of that same 1-D vector to every trial. The trajectory does not contain a separate explanation of the repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `envs` and stores a large `session_info` metadata structure even though the decoder does not use them. It also computes and records detailed provenance fields such as blocked-bin lists, source frame counts, and per-trial lengths that are not consumed by downstream decoding.

ii.
```python
environments = np.asarray(source["envs"]).squeeze()
...
session_info.append(
    {
        "subject": animal,
        "source_day_index": day,
        "environment": str(environments[day]),
        "blocked_bins": np.flatnonzero(geometry).astype(int).tolist(),
        "source_frames": int(trace.shape[1]),
        "registered_neurons": int(registered.sum()),
        "retained_neurons": int(keep_cells.sum()),
        "n_trials": len(session_neural),
        "trial_timepoints": [int(x.shape[1]) for x in session_neural],
    }
)
...
"metadata": {
    ...
    "session_info": session_info,
},
```

iii. The trajectory frames this as provenance retention rather than a required decoder input: step 29 says the dataset keeps "source-day/environment provenance retained in metadata." There is no indication that the AI believed this metadata would be used by the downstream decoder itself.
