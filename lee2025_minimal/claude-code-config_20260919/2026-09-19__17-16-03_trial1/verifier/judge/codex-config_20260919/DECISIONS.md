# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes seven animal identifiers, loads one extensionless joblib file per animal, indexes the top-level dictionary by the animal ID, and reads the `trace`, `position`, `envs`, and `blocked` arrays. It processes every day in every animal file.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position = dat['trace'], dat['position']
envs, blocked = dat['envs'], dat['blocked']
```

iii. The trajectory shows that the agent listed `/app/data`, inspected a loaded file's keys and array shapes, and counted 207 total sessions before implementing this loader. It justified the choice from the observed joblib organization: one file per mouse containing all days.

## 1-b. How are the data split into subjects?

i. Each identifier in `ANIMALS` is one subject. `subjects` preserves that list, and each output session gets the current enumeration index in `subject_idx`.

ii.
```python
'subjects': list(ANIMALS), 'subject_idx': [],
for animal_idx, animal in enumerate(ANIMALS):
    ...
    data['subject_idx'].append(animal_idx)
```

iii. The agent inspected all seven files and reported their day counts. It treated the file/key names as stable subject identifiers.

## 1-c. How are the data split into sessions?

i. A recording day is a session. The first dimension of `trace` determines the number of days, and one session entry is appended for every day.

ii.
```python
n_days = trace.shape[0]
for day in range(n_days):
    neural, keep = session_neural(trace[day])
    ...
    data['neural'].append(neural_trials)
```

iii. Inspection showed arrays organized as day × cell × frame (and day × coordinate × frame for position). The final response explicitly described one session per mouse-day and confirmed 207 sessions.

## 1-d. How are the data split into trials?

i. After 3-frame temporal pooling, each session is cut into consecutive, non-overlapping 600-bin (60-second) trials. The shorter remainder is dropped, and the minimum available length of neural and position streams controls the trial count.

ii.
```python
TRIAL_BINS = TRIAL_SECONDS * FPS // TEMPORAL_BIN_FRAMES
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
```

iii. The task explicitly requires one-minute trials. The agent stated that incomplete remainders are dropped and verified 39–40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No completed one-minute trial is quality-filtered. Low-speed frames are deliberately retained; only the incomplete terminal remainder is omitted.

ii.
```python
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
```

iii. The agent noted that the paper's decoder uses a speed threshold but reasoned that removing frames would conflict with contiguous one-minute trials and that position remains defined during immobility.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace`, described as the binarized rising phase of calcium transients.

ii.
```python
trace, position = dat['trace'], dat['position']
neural, keep = session_neural(trace[day])
```

iii. The agent inspected trace values and the paper/code, concluding that `trace` is the paper's neural “firing rate” representation rather than raw fluorescence.

## 2-b. How is the `neural` data processed?

i. Registered, sufficiently active traces are cast to float32, Gaussian-smoothed over time with sigma 3 frames, then averaged in non-overlapping 3-frame bins.

ii.
```python
trace = trace_day[keep].astype(np.float32)
smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
return pool_time(smoothed).astype(np.float32), keep
```

iii. The agent traced the paper repository's `decode_position_within`/`fit_decoder` path and said this mirrors its smoothing and `AvgPool1d(kernel_size=3, stride=3)` preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is kept on a day only if its first sample is non-NaN (registered) and it has more than five transient events over the session.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
events = np.nansum(trace_day, axis=1)
keep = registered & (events > MIN_EVENTS)
```

iii. The trajectory checked that NaNs are all-or-none per cell/day and compared registered counts with counts after the paper's `cell_threshold=5`. The agent identified the latter as the within-session decoding QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Time zero is the start of each artificial one-minute trial; identical slices are applied to the already pooled neural and position series.

ii.
```python
sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
output_trials.append(bins[sl][np.newaxis, :])
```

iii. The agent described the sessions as continuous free exploration with no discrete trial event and recorded trial start as the alignment event in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms. Three native 30 Hz frames are average-pooled per output bin after neural smoothing.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
return x[..., :n * TEMPORAL_BIN_FRAMES].reshape(
    x.shape[:-1] + (n, TEMPORAL_BIN_FRAMES)).mean(-1)
```

iii. The agent attributed the 3-frame pooling to the paper decoder and verified that converted trials contain 600 timepoints.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the per-day `blocked` entry; `envs` is used only for session metadata.

ii.
```python
envs, blocked = dat['envs'], dat['blocked']
geometry = blocked_vector(blocked[day])
```

iii. The trajectory cross-checked all blocked indices against the repository's environment matrices and found them consistent.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices become a length-nine float32 multi-hot vector, with one for each walled-off partition. The `-1` sentinel produces all zeros. A copy of the static vector is stored for every trial.

ii.
```python
idx = np.array(blocked_day[0]).ravel().astype(int)
vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
if idx.size and idx[0] != -1:
    vec[idx] = 1.0
input_trials.append(geometry.copy())
```

iii. The agent reasoned that all nine components should remain for a complete geometry description, even though one partition is never blocked in the supplied geometries.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each day's two-coordinate `position` array.

ii.
```python
trace, position = dat['trace'], dat['position']
bins = session_position_bins(position[day], bin_width)
```

iii. The agent inspected coordinate ranges and occupancy by blocked partition to establish that the coordinates represent x and y position in centimeters.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. X and y coordinates are averaged over each 3-frame temporal bin, divided by an animal-specific spatial width, cast to integer bins, clipped to 0–2, and combined as `3*y + x`.

ii.
```python
pooled = pool_time(position_day.astype(np.float64))
xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. The agent followed the paper's within-animal position-decoding code and checked that the resulting orientation agrees with the dataset's blocked indices.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. One grid is computed per animal with spatial bin width `(maximum position over all its sessions + 1e-15) / 3`. Flooring by integer conversion and clipping creates three bins per axis and labels 0–8 in row-major order.

ii.
```python
bin_width = (np.nanmax(position) + BUFFER) / N_SPATIAL_BINS
xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. The agent said the animal-wide maximum reproduces `decode_position_within`, maintains one grid across an animal's geometries, and aligns the categories with physical partitions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both streams begin frame-aligned at 30 Hz, are pooled independently over the same non-overlapping groups of three frames, and are cut with identical trial slices. The shorter pooled stream bounds the number of trials.

ii.
```python
neural, keep = session_neural(trace[day])
bins = session_position_bins(position[day], bin_width)
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
```

iii. The agent verified the source shapes and preserved one position category per neural time bin.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered all-NaN cell/day rows are excluded; low-event cells are also excluded. Pooling discards fewer than three residual frames, trial construction discards the incomplete final minute, spatial categories are clipped to valid bounds, and neural/position length mismatch is guarded by `min`. No position imputation is performed.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
keep = registered & (events > MIN_EVENTS)
n = x.shape[-1] // TEMPORAL_BIN_FRAMES
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
```

iii. The agent empirically established that trace NaNs occur by whole cell/day and regarded them as non-registration, not values to impute. Its checks found no position NaNs in the inspected data.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive operations are loading the large joblib arrays, Gaussian filtering every retained cell's full session trace, materializing pooled/trial arrays, and serializing the large pickle. The conversion trajectory took several minutes and produced a multi-gigabyte-style full dataset workload.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
pickle.dump(data, f, protocol=4)
```

iii. The agent did not explicitly rank timing within the script, but its long background conversion and memory checks support these as the dominant I/O and full-array operations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The animal/day loops are needed for variable session and neuron counts, but the inner trial loop is regular slicing and could be replaced by reshape/split operations (with copies only where required). Metadata assembly would still require session-level iteration.

ii.
```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geometry.copy())
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. The trajectory contains no explicit efficiency justification for this loop; it is a straightforward, readable implementation of variable-length nested output lists.

## 6-c. What processing does the code repeat multiple times?

i. `pool_time` separately computes the same three-frame grouping for neural and position streams in every session. The static geometry is copied once per trial, and diagnostic/session metadata repeatedly computes simple per-session summaries.

ii.
```python
return pool_time(smoothed).astype(np.float32), keep
pooled = pool_time(position_day.astype(np.float64))
input_trials.append(geometry.copy())
```

iii. The agent did not discuss repeated work. Separate pooling calls are necessary because the values differ, though their shared bin boundaries could be computed once; geometry copies favor safe independent trial objects.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `envs` only to create descriptive `session_info`, builds extensive metadata not consumed by the decoder features, makes a separate geometry copy for every trial, and forces every neural trial contiguous. These are useful for provenance or robustness but not needed for the downstream feature/target computation itself.

ii.
```python
envs, blocked = dat['envs'], dat['blocked']
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
input_trials.append(geometry.copy())
session_info.append({'environment': str(envs[day][0]), ...})
```

iii. The trajectory shows deliberate attention to descriptive metadata and verification. It provides no claim that these operations are needed by decoder training.
