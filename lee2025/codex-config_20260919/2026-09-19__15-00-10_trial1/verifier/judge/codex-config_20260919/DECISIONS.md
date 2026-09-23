# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter uses the seven hard-coded animal IDs and loads each extensionless joblib file sequentially. It indexes the top-level dictionary by animal ID, processes every day, then frees that animal before loading the next. Full mode processes all 207 sessions; sample mode stops after two sessions.

ii.
```python
for animal_index, animal in enumerate(ANIMALS):
    source = joblib.load(DATA_DIR / animal)[animal]
    ndays = source["trace"].shape[0]
    for day in range(ndays):
        ...
    del source
    gc.collect()
```

iii. The notes say joblib is the reference repository's default loader and preserves the same values as the paired MATLAB files; sequential loading bounds memory. The agent later spot-checked joblib against dereferenced MATLAB arrays.

## 1-b. How are the data split into subjects?

i. Each named joblib file/top-level dictionary entry is one mouse. The fixed `ANIMALS` list is also written to `subjects`, and its loop index is assigned to every session from that mouse.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subject_idx.append(animal_index)
...
"subjects": ANIMALS.copy(),
```

iii. The agent found these seven animals explicitly listed in the paper repository and confirmed the source files have one top-level entry per animal.

## 1-c. How are the data split into sessions?

i. Each day along axis 0 of an animal's object arrays is one output session. Trace, position, blocked geometry, and environment are indexed with the same `day` value.

ii.
```python
ndays = source["trace"].shape[0]
for day in range(ndays):
    trace = source["trace"][day]
    position = source["position"][day]
    blocked = blocked_mask_x_y(source["blocked"][day])
```

iii. The notes identify a recording day as the natural session and report 207 days/sessions, matching the paper data.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided from frame zero into non-overlapping 60-second chunks. At 30 Hz this is 1,800 raw frames, or 600 bins after 3-frame pooling. Only the incomplete terminal chunk is discarded.

ii.
```python
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
for trial in range(ntrials):
    start = trial * TRIAL_TIMEPOINTS
    end = start + TRIAL_TIMEPOINTS
    neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
```

iii. One-minute trials were required by the task. The notes justify starting at frame zero, retaining every complete minute, and recording the omitted tail rather than padding or fabricating samples.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial rejection. Every complete 60-second chunk is kept, and every session has at least two trials; only terminal partial minutes are omitted.

ii.
```python
ntrials = trace.shape[1] // RAW_TRIAL_FRAMES
n_raw_keep = ntrials * RAW_TRIAL_FRAMES
...
assert len(neural_trials) >= 2
```

iii. The paper describes continuous free exploration and no trial rejection. The agent retained all complete chunks to avoid changing occupancy or breaking contiguous time.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace`, the released binary significant calcium-transient rising-phase array. Rows are longitudinal cells and columns are 30 Hz frames.

ii.
```python
trace = source["trace"][day]
registered = np.isfinite(trace[:, 0])
pooled_neural = pool_neural(trace, registered, n_raw_keep)
```

iii. The agent concluded these traces were already extracted from fluorescence, so delta-F/F or transient detection should not be recomputed.

## 2-b. How is the `neural` data processed?

i. Registered trace rows are selected, Gaussian-smoothed along time with sigma 3 source frames, averaged in non-overlapping groups of three, cast/kept as contiguous float32, and then sliced into trials.

ii.
```python
source = np.asarray(trace[registered], dtype=np.float32)
smoothed = np.empty_like(source, dtype=np.float32)
gaussian_filter1d(source, sigma=SMOOTH_SIGMA_FRAMES, axis=1, output=smoothed)
pooled = smoothed[:, :n_raw_keep].reshape(
    smoothed.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
```

iii. The agent chose the smoothing/pooling used by the repository's within-session decoder, arguing it matched the paper decoder and reduced output from roughly 20 GB to 6.7 GB. Smoothing is performed over the whole session to avoid trial-edge artifacts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained for a session when its first trace value is finite; the code asserts the entire selected trace is finite and binary. All-NaN unregistered rows are excluded, but there is no place-cell, minimum-activity, or running-speed filter.

ii.
```python
registered = np.isfinite(trace[:, 0])
assert registered.any()
assert np.isfinite(trace[registered]).all()
assert np.isin(trace[registered], (0.0, 1.0)).all()
```

iii. The notes state unregistered cell/day rows are entirely NaN. They treat place-cell reliability as downstream analysis and reject running/activity filtering because it would destroy contiguous one-minute trials and stationary position labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. The declared event is the start of each consecutive 60-second segment; neural trials begin at pooled indices 0, 600, 1200, and so on.

ii.
```python
start = trial * TRIAL_TIMEPOINTS
end = start + TRIAL_TIMEPOINTS
neural_trials.append(np.ascontiguousarray(pooled_neural[:, start:end]))
...
"temporal_alignment_event": "start of each consecutive 60-second recording segment",
```

iii. These are artificial trials cut from continuous exploration, so the agent found no stimulus or behavioral event to align to.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms (10 Hz). Three consecutive 30 Hz source frames are mean-pooled after neural smoothing; position is mean-pooled over the identical frame groups.

ii.
```python
SOURCE_HZ = 30
POOL_FRAMES = 3
TARGET_HZ = SOURCE_HZ / POOL_FRAMES
...
"time_bin_size": 100.0,
```

iii. The agent selected the repository decoder's 3-frame pooling as reference-equivalent preprocessing and cited substantially lower memory and training cost.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from `source["blocked"][day]`, a nested list of blocked indices in a 3×3 arena; `-1` denotes the all-open square.

ii.
```python
blocked = blocked_mask_x_y(source["blocked"][day])
...
source_indices = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
```

iii. The notes identify `blocked` as the direct representation requested by the task; environment names are retained only as metadata.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Valid indices set ones in a 3×3 zero mask. The source row-major `(y,x)` mask is transposed and flattened in `(x,y)` order, producing nine float32 blockedness features. A copy of the static vector is stored for every trial.

ii.
```python
source_y_x = np.zeros((3, 3), dtype=np.float32)
valid = source_indices[source_indices >= 0].astype(np.int64)
source_y_x.reshape(-1)[valid] = 1.0
return source_y_x.T.reshape(-1).copy()
...
input_trials.append(blocked.copy())
```

iii. The agent chose the transpose so geometry dimensions use the same `x*3+y` order as its position classes, and reported that blocked classes had zero occupancy across geometries.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from the day's two-row `position` array, containing aligned x and y coordinates in centimeters.

ii.
```python
position = source["position"][day]
pooled_xy, position_classes = pool_position(position, n_raw_keep)
```

iii. The agent identified position as a finite, frame-aligned 30 Hz behavioral stream covering the 75×75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. X and y are averaged separately over the same non-overlapping three-frame groups as neural data, converted to 25 cm bins, clipped to 0–2, and encoded as one categorical row using `3*x_bin + y_bin`.

ii.
```python
pooled_xy = trimmed.reshape(2, -1, POOL_FRAMES).mean(axis=2)
xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
classes = (N_POSITION_BINS * xy_bins[0] + xy_bins[1]).astype(np.int8)
```

iii. The agent says identical pooling preserves alignment and chose x-major class ordering to agree with its transposed geometry feature ordering.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is assigned to `[0,25)`, `[25,50)`, or `[50,75]` cm via floor division; out-of-range/boundary results are clipped. The two axis bins form classes 0–8 in x-major order.

ii.
```python
POSITION_BIN_CM = ARENA_SIZE_CM / N_POSITION_BINS
xy_bins = np.floor(pooled_xy / POSITION_BIN_CM).astype(np.int8)
np.clip(xy_bins, 0, N_POSITION_BINS - 1, out=xy_bins)
classes = (3 * xy_bins[0] + xy_bins[1]).astype(np.int8)
```

iii. Equal 25 cm divisions directly implement the requested 3×3 discretization; clipping safely handles a coordinate exactly at 75 cm.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Raw trace and position lengths must match. Both streams retain the same complete raw-frame prefix, pool identical three-frame groups, and are sliced with the same 600-bin trial boundaries.

ii.
```python
assert trace.shape[1] == position.shape[1]
...
assert pooled_neural.shape[1] == position_classes.size == ntrials * TRIAL_TIMEPOINTS
...
output_trials.append(position_classes[None, start:end].copy())
```

iii. The notes report direct raw comparisons, including bins on both sides of trial boundaries, to rule out skipped or duplicated samples.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Entirely NaN unregistered cell rows are omitted. Selected traces and positions must otherwise be finite, binary traces are enforced, and equal stream lengths are required. Incomplete terminal minutes are dropped and documented. Unexpected partial missingness is not imputed; assertions fail instead.

ii.
```python
registered = np.isfinite(trace[:, 0])
assert np.isfinite(trace[registered]).all()
assert np.isfinite(position).all()
assert trace.shape[1] == position.shape[1]
...
"dropped_terminal_frames": int(trace.shape[1] - n_raw_keep),
```

iii. Exploration showed NaNs encode non-registration rather than corrupt samples. The agent preferred exclusion of those rows and explicit validation over filling missing neural or behavioral values.

## 6-a. What are the most time-consuming steps of the code?

i. Loading compressed joblib files, Gaussian filtering all registered cell traces, and serializing the multi-gigabyte pickle dominate. The full run reported about 194 seconds before serialization and 5.7 seconds for writing.

ii.
```python
source = joblib.load(DATA_DIR / animal)[animal]
gaussian_filter1d(source, sigma=SMOOTH_SIGMA_FRAMES, axis=1, output=smoothed)
...
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes measured sample and full timings and explain that large compressed neural arrays plus whole-session smoothing drive runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Cell/time smoothing and three-frame pooling were already vectorized. The remaining Python trial loop merely creates required per-trial arrays and could be replaced by reshaping plus list conversion, but this would offer limited benefit. Animal/day loops reflect heterogeneous objects and sequential memory management.

ii.
```python
pooled = smoothed[:, :n_raw_keep].reshape(
    smoothed.shape[0], -1, POOL_FRAMES
).mean(axis=2, dtype=np.float32)
for trial in range(ntrials):
    ...
```

iii. The agent explicitly replaced time-bin loops with reshape/mean and vectorized filtering across cells and time; it retained outer loops to bound memory and construct the nested target format.

## 6-c. What processing does the code repeat multiple times?

i. Geometry is copied once per trial, validation iterates over every trial after construction, and trial slicing repeats three append operations. Expensive convolution and pooling are not repeated per trial: each is done once per full session.

ii.
```python
for trial in range(ntrials):
    neural_trials.append(...)
    input_trials.append(blocked.copy())
    output_trials.append(...)
...
for neural, inputs, outputs in zip(neural_trials, input_trials, output_trials):
    assert ...
```

iii. The notes specifically justify whole-session processing to avoid repeated convolution and boundary artifacts. Repeated geometry copies and validation are accepted for isolation and correctness.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `pooled_xy` is returned and retained only for optional processing plots, yet is computed even when plotting is disabled and then deleted. Environment strings and extensive provenance are metadata rather than decoder features. No rate maps or place-cell computations are performed.

ii.
```python
pooled_xy, position_classes = pool_position(position, n_raw_keep)
...
if show_processing and processing_plots < 2:
    plot_processing(..., pooled_xy, ...)
...
del pooled_neural, pooled_xy, position_classes
```

iii. The agent deliberately avoided source rate maps because they are derived from trace/position and could leak the target. It considered diagnostic plots and provenance worthwhile, although `pooled_xy` is unnecessary in ordinary full conversion after classes are calculated.
