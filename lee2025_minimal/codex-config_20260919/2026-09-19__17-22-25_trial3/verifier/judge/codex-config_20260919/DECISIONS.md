# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses a fixed list of seven animal IDs and loads each extensionless joblib file from the data directory. It indexes the loaded dictionary by animal ID and obtains the per-day `trace`, `position`, `envs`, and `blocked` collections. It iterates every day for every listed animal.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_index, animal in enumerate(ANIMALS):
    source_path = data_dir / animal
    source = joblib.load(source_path)[animal]
    traces = source["trace"]
    positions = source["position"]
    environments = np.asarray(source["envs"]).squeeze()
    blocked = source["blocked"]
```

iii. The trajectory says the AI inspected the available sources and concluded that these files already contain the paper's final rise-extracted calcium-event trains and frame-aligned positions. It therefore chose not to repeat signal extraction and reported that conversion covered all seven animals and 207 recording days.

## 1-b. How are the data split into subjects?

i. Each ID in the fixed `ANIMALS` list is one subject. The loop index is recorded as `subject_idx` for every session belonging to that animal.

ii.
```python
for animal_index, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(animal_index)
...
"subjects": ANIMALS.copy(),
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The AI treated each per-animal source file as containing that animal's recording days. Its trajectory explicitly describes processing seven animals and retaining subject/day provenance.

## 1-c. How are the data split into sessions?

i. One recording day for one animal is one output session. The day index runs over the first dimension of the trace array, and corresponding position, environment, and blocked-geometry entries use the same index.

ii.
```python
for day in range(traces.shape[0]):
    trace = traces[day]
    position = positions[day]
    ...
    geometry = blocked_mask(blocked[day])
    ...
    neural.append(session_neural)
```

iii. The trajectory states that each source day is a frame-aligned roughly 40-minute recording and that the AI intentionally treated each recording day as a decoder session with a day-specific CA1 population.

## 1-d. How are the data split into trials?

i. After three-frame pooling, sessions are sliced consecutively into nominal 600-bin (60-second) trials. Unlike the human reference, the AI retains a final partial trial if it is at least 30 seconds and drops it if shorter. It raises an error if a session has fewer than two retained trials.

ii.
```python
TRIAL_BINS = int(TRIAL_SECONDS * SOURCE_FPS / POOL_FRAMES)
MIN_FINAL_TRIAL_SECONDS = 30.0
...
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
    stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
    duration_s = (stop - start) * TIME_BIN_MS / 1000.0
    if duration_s < MIN_FINAL_TRIAL_SECONDS:
        break
    session_neural.append(pooled_neural[:, start:stop])
...
if len(session_neural) < 2:
    raise ValueError(...)
```

iii. The AI reasoned that tiny post-40-minute overhangs should be discarded, while three slightly short recordings should keep their approximately 55.5-second final segments. It reported 40 trials per session and 8,280 total trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial rejection. Only terminal segments shorter than 30 seconds are discarded, and sessions with fewer than two resulting trials cause conversion to fail.

ii.
```python
if duration_s < MIN_FINAL_TRIAL_SECONDS:
    break
...
if len(session_neural) < 2:
    raise ValueError(f"Too few trials for {animal}, day {day}")
```

iii. The trajectory justifies this as retaining meaningful near-complete recordings while removing tiny acquisition overhangs and satisfying the downstream requirement for at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace` entry in the per-animal joblib source. The module docstring identifies this as final rise-extracted calcium-event trains.

ii.
```python
source = joblib.load(source_path)[animal]
traces = source["trace"]
...
trace = traces[day]
```

iii. The AI concluded from the repository and data inspection that `trace` already contains the paper's final binary calcium events, so raw imaging need not be reprocessed.

## 2-b. How is the `neural` data processed?

i. Retained neuron traces are truncated to a multiple of three frames, converted to float32, Gaussian-smoothed along time with sigma 3 source frames, and averaged over non-overlapping groups of three frames. Smoothing occurs before trial splitting.

ii.
```python
usable_frames = (trace.shape[1] // POOL_FRAMES) * POOL_FRAMES
selected = np.asarray(trace[keep_cells, :usable_frames], dtype=np.float32)
smoothed = gaussian_filter1d(
    selected, sigma=NEURAL_FILTER_SIGMA_FRAMES, axis=1
)
pooled_neural = smoothed.reshape(
    selected.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
```

iii. The trajectory says this was chosen to match the paper's position decoder: sigma-3-frame smoothing followed by non-overlapping three-frame averaging. Processing the continuous session before splitting avoids artificial filter boundaries at trial edges.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron must be registered on the day, represented by a finite first trace sample, and must have more than five events during moving samples. Movement is computed from position in the paper's 15-bin coordinates, with Gaussian-smoothed speed and a threshold equivalent to 5 cm/s.

ii.
```python
moving = moving_samples(position)
registered = np.isfinite(trace[:, 0])
moving_event_count = np.nansum(trace[:, moving], axis=1)
keep_cells = registered & (moving_event_count > MIN_MOVING_EVENTS)
...
speed_bins_s = np.linalg.norm(np.diff(position_15, axis=1), axis=0) * SOURCE_FPS
speed_bins_s = gaussian_filter1d(speed_bins_s, sigma=5)
moving[1:] = speed_bins_s > (VELOCITY_THRESHOLD_CM_S / bin_cm)
```

iii. The AI states that absent cross-day registrations are NaN and that the paper's within-session position decoder retains cells with more than five events during movement. It deliberately applies that neuron criterion while retaining slow time samples themselves.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. The artificial alignment event is the start of each contiguous recording segment; neural trials begin at consecutive 60-second boundaries after session-wide processing.

ii.
```python
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
    stop = min(start + TRIAL_BINS, pooled_neural.shape[1])
    session_neural.append(pooled_neural[:, start:stop])
...
"temporal_alignment_event": "start of each contiguous 1-minute recording segment",
```

iii. The trajectory identifies the data as continuous free-exploration recordings, so it uses segment starts rather than a stimulus or behavioral event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms (10 Hz). The native data are 30 Hz, and every three consecutive frames are averaged after Gaussian smoothing.

ii.
```python
SOURCE_FPS = 30
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
...
pooled_neural = smoothed.reshape(
    selected.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
```

iii. The AI explicitly chose 10 Hz to preserve what it interpreted as the paper decoder's smoothing/downsampling behavior and to keep the full converted dataset practical in size.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the per-session `blocked` entry, containing blocked spatial-bin indices. The environment identity in `envs` is retained only in metadata.

ii.
```python
blocked = source["blocked"]
...
geometry = blocked_mask(blocked[day])
...
"environment": str(environments[day]),
```

iii. The AI interpreted the requested environment geometry as which of the arena's nine grid positions are blocked and chose nine static blocked-bin indicators.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Finite blocked values are flattened and converted into a length-nine float32 multi-hot mask, with 1 meaning blocked. Negative values mean no blocked bin. A copy of the static vector is stored for each trial.

ii.
```python
mask = np.zeros(N_SPATIAL_BINS**2, dtype=np.float32)
values = np.asarray(blocked, dtype=float).reshape(-1)
for value in values[np.isfinite(values)]:
    index = int(value)
    if index >= 0:
        mask[index] = 1.0
...
session_input.append(geometry.copy())
```

iii. The AI chose the target format's supported one-dimensional static-per-trial representation, avoiding redundant repetition across time, and validates out-of-range nonnegative indices.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from each recording day's frame-aligned two-dimensional `position` array.

ii.
```python
positions = source["position"]
...
position = positions[day]
```

iii. The trajectory says position is already frame-aligned with the rise-extracted neural events and records behavior throughout each recording day.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is truncated to the same usable native-frame count as neural data, reshaped into consecutive groups of three frames, and averaged within each group. The pooled x-y coordinates are then discretized into one categorical label per time bin.

ii.
```python
def pool_position(position, usable_frames):
    return position[:, :usable_frames].reshape(2, -1, POOL_FRAMES).mean(axis=2)
...
pooled_position = pool_position(position, usable_frames)
labels = discretize_position(pooled_position)
```

iii. The AI says three-frame coordinate averaging matches the temporal pooling used by the paper's position decoder and keeps behavioral labels at the same 10 Hz resolution as neural activity.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is floor-divided into thirds of the documented 75 cm arena, clipped to 0–2, and encoded as `3*x_bin + y_bin`. A small epsilon in the divisor keeps an exact 75 cm point in the last valid bin.

ii.
```python
xy = np.floor(
    position / ((ARENA_SIZE_CM + np.finfo(np.float64).eps * 64) / N_SPATIAL_BINS)
).astype(np.int64)
xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
return xy[0] * N_SPATIAL_BINS + xy[1]
```

iii. The AI describes this as the paper dataset's documented 0–8 grid order and says its upper-bound treatment matches the reference rate-map code.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Raw trace and position lengths must match. Both streams use the same truncated native-frame extent and three-frame groups; their pooled arrays are sliced with identical trial start and stop indices.

ii.
```python
if trace.shape[1] != position.shape[1]:
    raise ValueError(f"Unaligned trace/position for {animal}, day {day}")
...
pooled_position = pool_position(position, usable_frames)
...
session_neural.append(pooled_neural[:, start:stop])
session_output.append(labels[np.newaxis, start:stop].copy())
```

iii. The AI relied on the source's frame alignment and preserved it by applying exactly corresponding temporal pooling and slicing to both streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered neurons are excluded using finite-value status. Non-finite blocked entries and negative no-block sentinels are ignored, while invalid high blocked indices raise an error. The code validates session counts, trace-position length, existence of eligible neurons, and minimum trial count. It truncates incomplete three-frame groups and drops terminal trials shorter than 30 seconds.

ii.
```python
if traces.shape[0] != positions.shape[0] or traces.shape[0] != len(blocked):
    raise ValueError(...)
registered = np.isfinite(trace[:, 0])
...
for value in values[np.isfinite(values)]:
    ...
    if index >= mask.size:
        raise ValueError(...)
usable_frames = (trace.shape[1] // POOL_FRAMES) * POOL_FRAMES
```

iii. The AI sought to distinguish expected missing cross-day registrations from structural corruption. It documents truncation/partial-trial policies and fails loudly on inconsistent alignment or unusable sessions.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant steps are loading large joblib sources, Gaussian filtering all retained neural traces, and serializing the roughly 6.2 GB output. Movement-speed filtering is smaller but repeated per session.

ii.
```python
source = joblib.load(source_path)[animal]
...
smoothed = gaussian_filter1d(selected, sigma=NEURAL_FILTER_SIGMA_FRAMES, axis=1)
...
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory initially identified data loading and preprocessing scale as important, monitored conversion through all animals, and later reported the full dataset size. It did not provide a formal timing profile.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The small loop over blocked indices could be replaced with validated advanced indexing. The animal/day and trial loops organize heterogeneous files, neuron counts, and trial lengths and are not useful candidates for simple whole-dataset vectorization.

ii.
```python
for value in values[np.isfinite(values)]:
    index = int(value)
    ...
    mask[index] = 1.0
...
for day in range(traces.shape[0]):
...
for start in range(0, pooled_neural.shape[1], TRIAL_BINS):
```

iii. The trajectory does not explicitly discuss loop vectorization. The computationally heavy neural smoothing and pooling are already vectorized across neurons and time.

## 6-c. What processing does the code repeat multiple times?

i. It recomputes movement speed/masks, cell eligibility, neural smoothing/pooling, position pooling, geometry encoding, and trial slicing independently for every recording day. Static geometry is copied once per trial, and output metadata recomputes simple counts from masks and trial arrays.

ii.
```python
for day in range(traces.shape[0]):
    moving = moving_samples(position)
    ...
    smoothed = gaussian_filter1d(...)
    ...
    geometry = blocked_mask(blocked[day])
    for start in range(...):
        session_input.append(geometry.copy())
```

iii. The trajectory does not identify avoidable repeated processing. Most repetition is required because each recording day has distinct position, activity, geometry, and retained neurons.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `envs` only to store environment names in metadata and computes a movement mask solely for neuron filtering; the mask and speed are not saved. It also repeatedly copies static geometry vectors, though the target representation requires a per-trial entry. No major transformed neural or behavioral array is computed and then discarded.

ii.
```python
environments = np.asarray(source["envs"]).squeeze()
...
moving = moving_samples(position)
moving_event_count = np.nansum(trace[:, moving], axis=1)
...
"environment": str(environments[day]),
```

iii. The trajectory does not explicitly discuss discarded processing. The movement computation is nevertheless necessary to implement the AI's chosen paper-derived neuron quality filter, and the environment identity provides provenance even though the decoder does not consume it.
