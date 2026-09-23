# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses a fixed list of seven animals and loads one extensionless joblib bundle per animal from `/app/data`. From each animal record it extracts the complete `trace`, `position`, `blocked`, and `envs` arrays, processes every day, and stops after two sessions only in `--sample` mode.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for subject_index, animal in enumerate(ANIMALS):
    bundle = joblib.load(DATA_DIR / animal)
    record = bundle[animal]
    trace = record["trace"]
    position = record["position"]
    blocked = record["blocked"]
    envs = np.asarray(record["envs"]).reshape(-1)
```

iii. The notes say that both joblib and matching MATLAB files exist and contain the same converted arrays. The AI chose joblib because the paper's loader supports it and it loads substantially faster. It checked for 207 sessions, 8,187 complete trials, and 69,744 session-neuron instances.

## 1-b. How are the data split into subjects?

i. Each named joblib file is one mouse. The hard-coded order defines `subjects`, and the enumeration index is appended once per session to `subject_idx`.

ii.
```python
"subjects": ANIMALS.copy(),
...
for subject_index, animal in enumerate(ANIMALS):
    ...
    data["subject_idx"].append(subject_index)
```

iii. The AI found that the repository and main analysis consistently identify these seven animals and that each source bundle is keyed by its animal ID.

## 1-c. How are the data split into sessions?

i. Each recording day within a mouse is made a separate target session. All neural, input, output, subject-index, brain-region, and provenance entries are appended in animal/day order.

ii.
```python
for day in range(trace.shape[0]):
    result = convert_session(trace[day], position[day], blocked[day],
                             animal, subject_index, day, str(envs[day]))
    neural_trials, input_trials, output_trials, valid_cells, session_info = result
    data["neural"].append(neural_trials)
    data["input"].append(input_trials)
    data["output"].append(output_trials)
```

iii. The AI reasoned that an animal-day is the natural session because geometry and the valid CellReg population are stable within that day, while registered-cell availability can change between days.

## 1-d. How are the data split into trials?

i. Continuous sessions are split from frame zero into consecutive, non-overlapping 1,800-frame (60-second) blocks. Only complete blocks are retained; the final incomplete tail is dropped. After the AI's temporal pooling, each trial has 600 output time bins.

ii.
```python
n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
n_used_frames = n_trials * SOURCE_FRAMES_PER_TRIAL
selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
...
for trial in range(n_trials):
    start = trial * SOURCE_FRAMES_PER_TRIAL
    stop = start + SOURCE_FRAMES_PER_TRIAL
```

iii. There is no native trial hierarchy. The instructions define trials as one-minute segments, so the AI used exact complete minutes without padding or overlap and recorded discarded-tail counts in metadata.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral or trial-quality filtering. A session must contain at least two complete trials, and malformed/nonfinite sessions raise errors. Incomplete tail frames are excluded but are not treated as a trial.

ii.
```python
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: fewer than two complete trials")
...
if not np.isfinite(position_day).all():
    raise ValueError(f"{animal} day {day}: position contains NaN/Inf")
```

iii. The notes state that no trial curation was reported for continuous free exploration. The AI deliberately did not apply the paper decoder's speed filter because deleting noncontiguous frames would violate complete elapsed one-minute trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from each day's `trace` array in the animal joblib record. These are already preprocessed binary rising-phase calcium-event traces, not raw fluorescence or supplied rate maps.

ii.
```python
trace = record["trace"]
...
result = convert_session(trace[day], ...)
```

iii. The AI found in the methods and source code that trace extraction, smoothing for event detection, noise normalization, and thresholding had already been performed; therefore it did not recompute delta-F/F.

## 2-b. How is the `neural` data processed?

i. After selecting valid day-cells and complete source frames, the AI casts traces to float32, reshapes them as `(cell, trial, 1800)`, Gaussian-smooths each trial independently with sigma 3 source frames and reflect boundaries, then averages non-overlapping groups of three frames to `(cell, trial, 600)`. Trial arrays are made contiguous.

ii.
```python
selected = np.asarray(trace_day[valid_cells, :n_used_frames], dtype=np.float32)
selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
smoothed = gaussian_filter1d(selected, sigma=3.0, axis=2, mode="reflect")
pooled_neural = smoothed.reshape(
    valid_cells.size, n_trials, POOLED_BINS_PER_TRIAL, POOL_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The AI justified this as matching `fit_decoder`/`decode_position_within` in the paper repository. It intentionally smooths within each one-minute trial to prevent information crossing train/validation trial boundaries.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells absent on a recording day are removed. The code identifies retained cells by finiteness at the first frame, relying on the verified invariant that a CellReg cell is either finite for the whole day or NaN for the whole day. It retains all such cells, with no place-cell, activity-count, or speed filter, and verifies the converted arrays are finite.

ii.
```python
valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
...
if not np.isfinite(neural_trial).all():
    raise ValueError(f"{animal} day {day}: retained neural data contain NaN/Inf")
```

iii. The notes distinguish upstream manual cell QC from decoder-specific feature selection. The paper says all manually curated cells were included broadly; the AI therefore treats all-day NaNs as absent registration but does not impose the reference decoder's `>5` event threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Frame zero of each artificial one-minute segment is the alignment event, with offsets 0 to 60 seconds. Neural smoothing is performed independently within those boundaries.

ii.
```python
"temporal_alignment_event": "start of each non-overlapping 1-minute session segment",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The experiment is continuous free exploration and supplies no trial event. The AI used the segmentation boundary required by the task and preserved synchronized source-frame boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100-ms bins (10 Hz). The native 30-Hz data are rebinned by non-overlapping three-frame mean pooling after neural smoothing.

ii.
```python
POOL_FRAMES = 3
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES
...
"time_bin_size": 100.0,
"temporal_pool_frames": POOL_FRAMES,
```

iii. The AI chose the paper decoder's `temporal_bin_size=3`, believing decoder-specific smoothing/pooling should be retained. Its earlier notes had recognized that this processing need not be baked into converted data, but Step 5 ultimately chose to apply it.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived directly from the per-day nested `blocked` field, whose values are indices into the nine arena partitions; `-1` denotes no blocked partitions.

ii.
```python
blocked = record["blocked"]
...
geometry, blocked_indices = blocked_vector(blocked_entry)
```

iii. The AI preferred the native blocked indices over deriving geometry from environment strings or transformed display matrices, because the indices directly encode the task variable in canonical partition order.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The indices are validated as integral and within 0–8, then converted to a nine-element float32 multi-hot vector where 1 means blocked. `[-1]` becomes all zeros. A copy of this static vector is stored for every trial.

ii.
```python
if values.size == 1 and values[0] == -1:
    indices = []
...
vector = np.zeros(9, dtype=np.float32)
vector[indices] = 1.0
...
input_trials.append(geometry.copy())
```

iii. This directly satisfies the requested static per-trial geometry input and gives each partition an independent decoder feature.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position output is derived from each day's synchronized `position` field, a `(2, frames)` x/y coordinate stream.

ii.
```python
position = record["position"]
...
result = convert_session(..., position[day], ...)
```

iii. The AI identified `position` as the framewise DeepLabCut-derived behavioral stream already aligned with `trace` at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Within every trial, x and y are mean-pooled over the same non-overlapping three-frame windows used for neural pooling. The pooled coordinates are divided into 25-cm bins, clipped to the 3×3 arena, then flattened in row-major geometry order as `3*y_bin + x_bin` and stored as a `(1, 600)` uint8 array.

ii.
```python
pooled = position_trial.reshape(2, POOLED_BINS_PER_TRIAL, POOL_FRAMES).mean(axis=2)
xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
```

iii. The task overrides the paper's 15×15 target with 3×3 classes. The AI pooled coordinates before categorization to mirror the reference decoder and chose `y*3+x` after an empirical geometry-orientation check exposed an earlier x/y flattening error.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis uses thresholds at 25 and 50 cm: effectively `[0,25)`, `[25,50)`, and `[50,75]`. Floor division determines bins and clipping maps exact 75-cm/border excursions into the outer bin. The class is `3*y+x`, from 0 through 8.

ii.
```python
SPATIAL_BIN_CM = 25.0
xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
np.clip(xy_bin, 0, 2, out=xy_bin)
labels = (xy_bin[1] * 3 + xy_bin[0]).astype(np.uint8)[None, :]
```

iii. Three equal 25-cm partitions implement the specified 3×3 discretization. Clipping preserves synchronized boundary samples rather than dropping them.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Trace and position use identical 30-Hz day/frame axes, identical 1,800-frame trial slices, and identical non-overlapping three-frame pooling windows. Thus each 100-ms position class corresponds to the same source frames as its neural column.

ii.
```python
start = trial * SOURCE_FRAMES_PER_TRIAL
stop = start + SOURCE_FRAMES_PER_TRIAL
_, labels = pool_position(position_day[:, start:stop])
neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
```

iii. The paper and code describe simultaneous timestamped acquisition, and the released arrays have identical frame counts. The AI verified exact reconstructed values on multiple sessions and trials.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Whole-day NaN cell channels are treated as absent and removed; nonfinite positions, retained neural values, incompatible shapes/counts, invalid blocked indices, missing neurons, or too-short sessions cause explicit errors rather than imputation. Incomplete trial tails are dropped, exact coordinate 75 is clipped into range, and provenance records discarded frames and retained source-cell indices.

ii.
```python
valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
if trace_day.shape[1] != position_day.shape[1]:
    raise ValueError("... trace and position frame counts differ")
if not np.isfinite(position_day).all():
    raise ValueError("... position contains NaN/Inf")
...
"discarded_tail_frames": int(n_source_frames - n_used_frames),
```

iii. Exploration established that missing cells are consistently all-NaN for a day and positions are finite. The AI avoids inventing data, excludes only structurally incomplete tails, and uses strict checks to make unexpected corruption visible.

## 6-a. What are the most time-consuming steps of the code?

i. Loading and decompressing the seven large joblib animal bundles and processing their large trace arrays (especially Gaussian filtering) dominate. The complete conversion took 167.71 seconds; the final 6.173-GiB pickle write took 5.87 seconds. A sample animal load was reported as about 9 seconds, while individual session conversion typically took roughly 0.2–0.6 seconds.

ii.
```python
bundle = joblib.load(DATA_DIR / animal)
...
smoothed = gaussian_filter1d(selected, sigma=NEURAL_SMOOTH_SIGMA_FRAMES,
                             axis=2, mode="reflect")
...
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI instrumented animal loading, per-session conversion, output writing, and total runtime. Its notes specifically characterize native loading as expensive and use batched filtering to keep per-session processing modest.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The expensive neural operation is already vectorized across cells and trials in one SciPy call. The remaining per-trial loop could be reduced by pooling all used positions for a session in one reshape/mean operation and then constructing trial views/lists, although target-format lists and per-trial validation still require some iteration. The final validation loop also iterates over every saved trial.

ii.
```python
smoothed = gaussian_filter1d(selected, ..., axis=2, ...)
...
for trial in range(n_trials):
    _, labels = pool_position(position_day[:, start:stop])
    ...
for neural, inp, out in zip(data["neural"][s], data["input"][s], data["output"][s]):
    ...
```

iii. The AI explicitly optimized neural smoothing and pooling with reshape-based batching. It kept the trial loop to build the required nested list structure and perform trial-specific checks.

## 6-c. What processing does the code repeat multiple times?

i. `pool_position` and finite/range checks run once per trial; the same session-static geometry is copied once per trial; shape checks traverse all trials again after conversion. With `--show-processing`, smoothing and position pooling are recomputed for the diagnostic plot for up to two sessions.

ii.
```python
for trial in range(n_trials):
    _, labels = pool_position(position_day[:, start:stop])
    ...
    input_trials.append(geometry.copy())
...
smooth = gaussian_filter1d(raw, sigma=NEURAL_SMOOTH_SIGMA_FRAMES, axis=1)
pooled_xy, labels = pool_position(position_day[:, :SOURCE_FRAMES_PER_TRIAL])
```

iii. Most repetition supports the nested target format, validation, or optional visual diagnostics. The AI judged the measured runtime acceptable and did not pursue further optimization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal conversion, `pool_position` computes and returns pooled coordinates even though the caller discards them and saves only labels. Extensive `session_info` provenance is saved but is not used by the supplied decoder. Optional processing plots recompute transformations solely for diagnostics. Source `envs` are preserved only as metadata. The major neural smoothing/pooling is not discarded—it directly changes saved data—though it is unnecessary relative to the human conversion.

ii.
```python
def pool_position(...):
    ...
    return pooled, labels
...
_, labels = pool_position(position_day[:, start:stop])
...
"session_info": [],
```

iii. The AI retained provenance and optional plots to support independent sanity checks. It deliberately omitted large unused native fields such as maps, spatial footprints, and centroids, and released animal arrays after use.
