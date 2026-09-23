# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads one paper-provided joblib file for each of seven hard-coded subjects. It unwraps the subject-keyed dictionary and converts `trace`, `position`, and `envs` to NumPy arrays; `blocked` is accessed per day. It validates dimensional and session/time consistency.

ii.
```python
for subject_number, subject in enumerate(SUBJECTS):
    source_path = data_dir / subject
    wrapped = joblib.load(source_path)
    source = wrapped[subject]
    traces = np.asarray(source["trace"])
    positions = np.asarray(source["position"])
    environments = np.asarray(source["envs"]).reshape(-1)
```

iii. The trajectory says the source joblib data already contain the paper's final binary rising-phase events and aligned x-y tracking, so the agent deliberately starts from these products rather than repeating unavailable upstream image processing.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the seven entries in the hard-coded `SUBJECTS` list; one source file and one top-level dictionary key correspond to each subject. Session-level `subject_idx` uses the enumeration index.

ii.
```python
SUBJECTS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
            "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
for subject_number, subject in enumerate(SUBJECTS):
    ...
    subject_idx.append(subject_number)
```

iii. The trajectory identifies session-level cell registration as a key choice and reports seven subjects. No more specific justification for hard-coding the list is recorded.

## 1-c. How are the data split into sessions?

i. Each recording day (`traces.shape[0]`) becomes one output session. Neural, input, output, subject index, region index, and metadata are appended once per day.

ii.
```python
for day in range(traces.shape[0]):
    position = positions[day]
    ...
    neural.append([...])
    decoder_input.append([...])
    decoder_output.append([...])
```

iii. The code's conversion notes state that a source recording day is one session; this preserves the source day organization and its day-specific registered-cell set and environment.

## 1-d. How are the data split into trials?

i. Continuous days are divided into consecutive, non-overlapping 60-second trials (1,800 raw frames). Only complete trials are used; the short tail is discarded. After 1-second pooling, every trial has 60 samples.

ii.
```python
n_trials = total_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The agent states that the task requires one-minute segmentation and that short final segments are discarded rather than padded or assigned a different bin duration.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level behavioral quality filter. Incomplete tails are excluded, and the converter rejects a day with fewer than two complete trials.

ii.
```python
if n_trials < 2:
    raise ValueError(f"Fewer than two complete trials for {subject}, day {day}")
```

iii. Complete trials are retained to satisfy fixed duration, and the two-trial check enforces the decoder format requirement. The trajectory only explicitly mentions retaining full one-minute trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the joblib source's `trace` array, described by the agent as paper-provided binary rising-phase calcium events.

ii.
```python
traces = np.asarray(source["trace"])
selected = traces[day, cell_mask, :].astype(np.float32, copy=True)
```

iii. The agent concluded from the paper repository and array inspection that these traces are already the final event representation and therefore did not redo raw imaging extraction.

## 2-b. How is the `neural` data processed?

i. Selected traces are cast to float32, temporally Gaussian-smoothed with sigma 3 source frames, truncated to complete trials, and mean-pooled over non-overlapping 30-frame (1-second) bins. Trial arrays are made contiguous in neuron-by-time order.

ii.
```python
smoothed = np.empty_like(selected)
gaussian_filter1d(selected, sigma=3, axis=1, output=smoothed)
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The agent says sigma-3 smoothing matches the repository's within-day position decoder, while 1-second pooling makes the common decoder tractable for long recordings while retaining an event-rate representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. On each day, cells are retained only if they have more than five summed events during frames where estimated speed exceeds 5 cm/s. Speed derives from frame differences at 30 Hz and is Gaussian-filtered with sigma 5. NaN/unregistered cells fail the summed-event comparison. Zero qualifying cells cause an error.

ii.
```python
moving = moving_mask(position)
events_while_moving = np.sum(traces[day][:, moving], axis=1)
cell_mask = events_while_moving > 5
```

iii. The agent states this reproduces the paper repository's `decode_position_within` locomotion criterion and activity filter. The trajectory notes that it corrected a NumPy advanced-indexing axis-order issue while implementing it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Each artificial trial is aligned to the start of its consecutive complete 60-second segment (`off_start=0`, `off_end=60`).

ii.
```python
"temporal_alignment_event": "start of each consecutive complete 60-second segment",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The recording is continuous and the requested trials are fixed-duration segments, so segment onset is the only alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 1,000 ms. Native 30 Hz data are rebinned by averaging each non-overlapping group of 30 frames.

ii.
```python
TIME_BIN_FRAMES = 30
TIME_BIN_MS = 1000.0
...reshape(..., TIME_BIN_FRAMES).mean(axis=3, dtype=np.float32)
```

iii. The agent chose coarser bins to make decoder training tractable on 40-minute recordings, while viewing the mean as a preserved event-rate representation.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from `source["blocked"][day]`, whose entries identify blocked sectors; `-1` means none blocked.

ii.
```python
geometry, blocked_bins = blocked_geometry(source["blocked"][day])
raw = np.asarray(blocked_for_day[0]).reshape(-1)
```

iii. The agent identified `blocked` as the source environment variable and inspected actual occupancy to resolve its indexing convention.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Nonnegative blocked indices set ones in a 3x3 y-major matrix. That matrix is transposed and flattened to an x-major nine-element float32 indicator vector. A copy of the static vector is assigned to every trial in the day.

ii.
```python
blocked_yx = np.zeros((3, 3), dtype=np.float32)
for value in raw:
    if int(value) >= 0:
        blocked_yx.flat[int(value)] = 1.0
vector = blocked_yx.T.reshape(-1)
decoder_input.append([geometry.copy() for _ in range(n_trials)])
```

iii. The trajectory says zero-occupancy bins showed that geometry indexing was transposed relative to stored `(x,y)` positions; the transpose was intended to keep geometry and labels in one x-major convention.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from the per-day two-coordinate `position` array in the joblib source.

ii.
```python
positions = np.asarray(source["position"])
position = positions[day]
classes = position_classes(position, used_frames)
```

iii. The agent describes the source positions as already timestamp-aligned x-y tracking.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are truncated with neural data, averaged within each 30-frame/1-second bin, then converted to one categorical class. The classes are reshaped into 60-sample trials with a singleton output dimension.

ii.
```python
mean_position = position[:, :used_frames].reshape(2, -1, 30).mean(axis=2)
classes = position_classes(position, used_frames).reshape(n_trials, 60)
[np.ascontiguousarray(classes[trial][None, :]) for trial in range(n_trials)]
```

iii. Position averaging mirrors the neural temporal pooling so each output represents the same one-second interval.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Mean x and y are floor-divided into 25-cm sectors of the 75-cm arena, clipped to indices 0-2, and encoded x-major as `3*x_bin + y_bin`, yielding classes 0-8.

ii.
```python
xy_bin = np.floor(mean_position / SPATIAL_BIN_CM).astype(np.int64)
xy_bin = np.clip(xy_bin, 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy_bin[0] + xy_bin[1]).astype(np.int64)
```

iii. Clipping handles exact 75-cm boundary values. The x-major convention was selected to match the agent's transposed geometry representation.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural arrays are validated to have equal raw time lengths, truncated to the same `used_frames`, averaged in the same 30-frame bins, and reshaped using the same trial boundaries.

ii.
```python
if traces.shape[2] != positions.shape[2]:
    raise ValueError(...)
binned_neural = smoothed[:, :used_frames].reshape(..., 30).mean(axis=3)
mean_position = position[:, :used_frames].reshape(2, -1, 30).mean(axis=2)
```

iii. The agent regarded the source streams as already timestamp-aligned and preserved alignment through identical truncation, bin widths, and trial boundaries.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/unregistered cells contain NaNs and fail the `>5` activity test. Exact arena boundaries are clipped into valid spatial bins. Incomplete recording tails are discarded. Missing files, inconsistent shapes/counts/times, no qualifying cells, or fewer than two complete trials raise explicit errors rather than being silently repaired.

ii.
```python
events_while_moving = np.sum(traces[day][:, moving], axis=1)
cell_mask = events_while_moving > 5
xy_bin = np.clip(xy_bin, 0, N_SPATIAL_BINS - 1)
if traces.shape[2] != positions.shape[2]:
    raise ValueError(...)
```

iii. Comments say NaN propagation deliberately matches the repository implementation. The complete-tail and boundary policies avoid padding and invalid category indices, while validation prevents malformed data from propagating.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large joblib objects and applying Gaussian filters to every selected neural trace are the main expensive operations; serialization and construction of thousands of trial arrays also contribute. The trajectory confirms conversion proceeded animal by animal and then ran full decoder validation.

ii.
```python
wrapped = joblib.load(source_path)
gaussian_filter1d(selected, sigma=3, axis=1, output=smoothed)
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent did not explicitly rank runtime stages. Its use of per-subject loading, explicit output buffers, deletion, and garbage collection shows awareness of the cost and memory footprint.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over blocked indices could use indexed assignment directly. Trial-list construction loops could potentially be replaced by array views, but lists are required by the target format. Subject/day loops are appropriate because files, cell counts, and session metadata vary.

ii.
```python
for value in raw:
    idx = int(value)
    if idx >= 0:
        blocked_yx.flat[idx] = 1.0
[np.ascontiguousarray(binned_neural[:, trial, :]) for trial in range(n_trials)]
```

iii. The trajectory provides no explicit vectorization discussion. These are code-level observations; the numerically heavy smoothing, binning, speed, and class calculations are already vectorized.

## 6-c. What processing does the code repeat multiple times?

i. Moving-speed calculation, activity filtering, smoothing, binning, position classification, and geometry encoding repeat once per day. Geometry is copied once per trial, and neural/output trials are copied into contiguous arrays.

ii.
```python
for day in range(traces.shape[0]):
    moving = moving_mask(position)
    ...
decoder_input.append([geometry.copy() for _ in range(n_trials)])
```

iii. Day-level repetition is needed because positions, valid cells, duration, and geometry differ by day. The agent did not record a separate justification for copying static geometry per trial; it follows the required nested format.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and stores `envs` only to write session metadata, computes `blocked_bins` only for metadata, computes several session statistics, and runs `gc.collect()` per subject. These do not affect decoder tensors. Copies for contiguity and static geometry increase work but support safe standalone trial arrays.

ii.
```python
environments = np.asarray(source["envs"]).reshape(-1)
geometry, blocked_bins = blocked_geometry(source["blocked"][day])
"environment": str(environments[day]),
"blocked_bins_x_major": blocked_bins,
gc.collect()
```

iii. The agent says conversion decisions should be explicit and reproducible; the extra metadata serves that auditability even though the downstream decoder does not use it. No explicit trajectory justification is given for forced garbage collection.
