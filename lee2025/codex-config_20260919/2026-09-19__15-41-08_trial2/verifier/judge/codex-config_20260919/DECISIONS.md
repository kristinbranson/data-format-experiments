# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter uses a fixed list of seven animal IDs and loads one extensionless joblib file per animal. Each file contains a dictionary keyed by that animal ID; the payload's `trace`, `position`, `envs`, and `blocked` fields supply every recording day. Full mode selects every day, while sample mode deliberately selects only the first two days of the first animal.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_idx, animal in enumerate(ANIMALS):
    payload = joblib.load(DATA_DIR / animal)[animal]
    trace = np.asarray(payload["trace"])
    position = np.asarray(payload["position"])
    envs = np.asarray(payload["envs"]).squeeze()
    selected_days = range(trace.shape[0]) if targets[animal] is None else sorted(targets[animal])
```

iii. The agent found that the supplied joblib objects are the format used by the authors' `load_dat` path and contain the same primary fields as the duplicate MATLAB files. Loading each animal once also amortizes decompression and limits repeated I/O. Exact full-data totals (7 animals and 207 sessions) were asserted during validation.

## 1-b. How are the data split into subjects?

i. One source file and one payload dictionary correspond to one mouse. Subject order and identifiers come from the fixed `ANIMALS` list, and every emitted day receives that animal's list index in `subject_idx`.

ii.
```python
for animal_idx, animal in enumerate(ANIMALS):
    payload = joblib.load(DATA_DIR / animal)[animal]
    ...
    subject_idx.append(animal_idx)
...
"subjects": ANIMALS.copy(),
```

iii. The notes report seven animal files whose names are the mouse identifiers. This matches the native per-animal organization and keeps session-to-subject provenance explicit.

## 1-c. How are the data split into sessions?

i. Each recording day (axis 0 of `trace` and `position`) becomes one output session. The code iterates every day for each mouse in full mode and records the animal/day identity in `session_info`.

ii.
```python
selected_days = range(trace.shape[0]) if targets[animal] is None else sorted(targets[animal])
for day in selected_days:
    ...
    session_id = f"{animal}_day{day:02d}"
```

iii. The agent identified days as the independent recording sessions in the reference code and data. The resulting 207 sessions match the source/reference total.

## 1-d. How are the data split into trials?

i. A continuous day is divided into non-overlapping complete 60-second windows. At 30 Hz these are 1,800 raw frames; after 3-frame pooling each trial has 600 time bins. Any final incomplete minute is discarded.

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
TIMEPOINTS_PER_TRIAL = RAW_TRIAL_FRAMES // POOL_FRAMES
...
n_trials = n_raw_frames // RAW_TRIAL_FRAMES
n_used_frames = n_trials * RAW_TRIAL_FRAMES
...
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
```

iii. The task explicitly defines trials as one-minute pieces. The notes state that frame 0 is the common origin and boundary checks confirmed no overlap or gap. Complete-window slicing produced 8,187 trials.

## 1-e. How are trials filtered based on quality controls?

i. No complete trial is filtered for behavior, speed, neural activity, or geometry. Sessions with fewer than two complete trials cause an error, and only the incomplete tail is omitted.

ii.
```python
n_trials = n_raw_frames // RAW_TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day} has fewer than two complete trials")
```

iii. The agent reasoned that the paper decoder's moving-frame mask is classifier-specific and would destroy the requested contiguous one-minute time series. It therefore retained all complete windows and treated blocked-bin occupancy as a diagnostic, not a rejection rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the payload's `trace` array for each recording day. These values are supplied binary rising-phase calcium events, with NaNs marking cells absent on a day.

ii.
```python
trace = np.asarray(payload["trace"])
...
binned_neural = process_neural(trace[day], present, n_used_frames)
```

iii. The agent concluded that upstream fluorescence processing and event extraction had already been performed, so dF/F should not be recomputed.

## 2-b. How is the `neural` data processed?

i. Day-present cells are selected and cast to float32. Their full-session event streams are Gaussian-smoothed along time with sigma 3 native frames, cropped to the complete-minute extent, and mean-pooled over non-overlapping groups of three frames. Trial slices are stored neuron-by-time as contiguous float32 arrays.

ii.
```python
selected = np.asarray(raw_trace[present, :], dtype=np.float32)
gaussian_filter1d(selected, sigma=POOL_FRAMES, axis=1, output=selected)
selected = selected[:, :n_used_frames]
return selected.reshape(selected.shape[0], -1, POOL_FRAMES).mean(axis=2)
```

iii. The agent chose this because it found Gaussian sigma-3 smoothing followed by 3-frame average pooling in the authors' position-decoder path. It smoothed the continuous physical session before cropping to avoid an artificial boundary at the last retained minute.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained for a day when its first sample is finite, which selects registered cells and removes absent all-NaN registrations. The code rejects a selected trace if any intermittent NaN/Inf remains. It applies no place-cell, event-count, or activity threshold, so even zero-event registered neurons remain.

ii.
```python
present = np.isfinite(trace[day, :, 0])
if not present.any():
    raise ValueError(f"{animal} day {day} has no registered neurons")
...
if not np.isfinite(selected).all():
    raise ValueError("Registered neural traces contain intermittent NaN/Inf")
```

iii. The paper states that all manually curated cells were included in subsequent population analyses. The notes distinguish absent all-NaN registrations from low-activity but valid cells and report that the latter were intentionally retained.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Neural and position acquisition are already frame-aligned; artificial trials start at session frame 0 and each subsequent minute boundary. Metadata names the start of each one-minute window as the alignment event with offsets 0 to 60 seconds.

ii.
```python
"temporal_alignment_event": "start of each contiguous one-minute window within the recording session",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The notes say calcium and video were simultaneously acquired at 30 Hz and timestamp-aligned, so no lag correction or interpolation was required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 100 ms (10 Hz). Both neural activity and position are rebinned by averaging disjoint groups of three native 30 Hz frames; neural activity is also Gaussian-smoothed first.

ii.
```python
FPS = 30
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
...
"time_bin_size": TIME_BIN_MS,
```

iii. The agent justified 100 ms as the temporal bin used by the authors' position-decoder implementation and applied identical raw-frame groupings to neural and behavioral streams.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from `payload["blocked"][day][0]`, the per-day list of blocked partition indices. A lone `-1` denotes an open arena with no blocked partitions.

ii.
```python
geometry = blocked_vector(payload["blocked"][day][0])
```

iii. The agent treated the native `blocked` field as authoritative because environment names and the reference helper do not preserve all rotated/flipped daily geometries.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code creates nine float32 bits with 1 for blocked and 0 for accessible. It fills the native row-major `(y,x)` grid, transposes it, then flattens in the output's x-first order. The static vector is copied into every trial of that day.

ii.
```python
native_yx = np.zeros(9, dtype=np.float32)
if values.size == 1 and float(values[0]) == -1.0:
    return native_yx
native_yx[indices] = 1.0
return np.ascontiguousarray(native_yx.reshape(3, 3).T.ravel())
...
session_input.append(geometry.copy())
```

iii. Testing the eight square-grid transforms showed that transposition reduced mouse samples assigned to blocked cells from 17.14% to 0.0031%. The agent therefore used it to put native matrix indices in the same coordinate convention as position labels.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output comes from the per-day two-coordinate `position` stream in the animal payload.

ii.
```python
position = np.asarray(payload["position"])
...
binned_position = pool_position(position[day], n_used_frames)
```

iii. The source contains aligned x/y mouse coordinates in centimeters for every 30 Hz frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is cast to float32, checked for finite values, and mean-pooled over the same non-overlapping three-frame groups as neural activity. Pooled coordinates are clipped into the 75 cm arena, divided into 25 cm bins, and converted to one integer class per time bin.

ii.
```python
selected = np.asarray(raw_position[:, :n_used_frames], dtype=np.float32)
return selected.reshape(2, -1, POOL_FRAMES).mean(axis=2)
...
xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
labels = xy[0] * SPATIAL_BINS + xy[1]
```

iii. The agent viewed position pooling as part of matching the paper decoder's temporal processing. It selected the x-first class order used in its reading of reference map accumulation and paired it with matching output value names.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is assigned to `[0,25)`, `[25,50)`, or `[50,75]` cm. Values are clipped before floor division, including exact 75 cm values just below the upper endpoint. The final class is `x_bin * 3 + y_bin`, yielding 0 through 8.

ii.
```python
upper = np.nextafter(np.float32(ARENA_CM), np.float32(0.0))
xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
labels = xy[0] * SPATIAL_BINS + xy[1]
```

iii. Equal 25 cm divisions directly implement the requested 3x3 grid. Clipping handles boundary noise and guarantees valid labels; x-first names document the chosen permutation.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position begin at the same source frame, use identical three-frame pooling groups, and are sliced by the same 600-bin trial ranges. Shape assertions check equality before trial creation.

ii.
```python
binned_position = pool_position(position[day], n_used_frames)
labels = position_classes(binned_position)
binned_neural = process_neural(trace[day], present, n_used_frames)
if binned_neural.shape[1] != labels.size:
    raise AssertionError("Neural and position bins are temporally misaligned")
```

iii. The streams were natively synchronized. Independent raw checks and plotted overlays reportedly found no lag, overlap, or trial-boundary gap.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Expected all-NaN neuron registrations are removed per day. Unexpected intermittent neural NaN/Inf, nonfinite retained position, invalid blocked indices, missing neurons, too few trials, or shape/range mismatches raise errors rather than being imputed. Exact arena-boundary positions are clipped, and incomplete final trial tails are discarded and recorded in metadata.

ii.
```python
present = np.isfinite(trace[day, :, 0])
...
if not np.isfinite(selected).all():
    raise ValueError("Registered neural traces contain intermittent NaN/Inf")
...
xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
```

iii. The agent found no intermittent missing samples, so it avoided inventing imputation. It considered absent registrations and incomplete tails expected structural features, while treating other corruption as a hard validation failure.

## 6-a. What are the most time-consuming steps of the code?

i. Per-animal joblib loading/decompression is the dominant I/O cost, while full-session Gaussian filtering and conversion/copying of the large trace arrays are the main compute and memory costs. Serializing the 6.2 GiB pickle is another material step.

ii.
```python
load_start = time.perf_counter()
payload = joblib.load(DATA_DIR / animal)[animal]
...
gaussian_filter1d(selected, sigma=POOL_FRAMES, axis=1, output=selected)
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes measured 8.88 seconds loading versus 1.40 seconds processing for the two-session sample, and 161 seconds conversion plus about 6 seconds saving for the full run.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Smoothing, pooling, class construction, and per-session diagnostics are already vectorized. The remaining trial loop only slices/copies arrays into the nested-list format and could be expressed as reshapes plus list construction, but individual arrays/lists are ultimately required. Animal/day loops reflect heterogeneous session and neuron dimensions and are not natural dense-array vectorization targets. Validation's per-trial loop could be consolidated but is not conversion-critical.

ii.
```python
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
    session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
    session_input.append(geometry.copy())
    session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The agent explicitly used reshape/mean vectorization and session-wide filtering to avoid per-trial numerical processing. It regarded repeated Gaussian work as the important optimization target.

## 6-c. What processing does the code repeat multiple times?

i. Static geometry is copied once per trial; per-trial contiguous copies are made for neural and output slices; validation later loops over every stored trial to repeat shape and finiteness checks. The code also computes global class and blocked-occupancy summaries while converting. It does not repeat loading or smoothing within an animal/day.

ii.
```python
session_input.append(geometry.copy())
...
for n, i, o in zip(data["neural"][s], data["input"][s], data["output"][s]):
    ...
    if not np.isfinite(n).all() or not np.isfinite(i).all() or not np.isfinite(o).all():
```

iii. The notes emphasize that each animal is loaded once, each selected session is cast once, and its Gaussian filter is run once. The remaining repetition is required by the target nested structure or retained as validation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `envs` is loaded only to populate provenance metadata. Class counts and blocked-position overlap are computed solely for printed diagnostics. Optional processing plots re-read/display source portions but are capped at two sessions. These products do not affect decoder arrays. The full-session Gaussian filter also computes values in the incomplete tail, though this is intentional so retained edge bins have the correct continuous-session filter context.

ii.
```python
envs = np.asarray(payload["envs"]).squeeze()
...
occupied_geometry = geometry[labels]
blocked_position_count += int(occupied_geometry.sum())
class_counts += np.bincount(labels, minlength=9)
...
if show_processing and plot_count < 2:
    save_processing_plot(...)
```

iii. The agent retained these operations as sanity checks and provenance. It specifically justified smoothing before tail removal to avoid a synthetic filter boundary, despite discarding the tail afterward.
