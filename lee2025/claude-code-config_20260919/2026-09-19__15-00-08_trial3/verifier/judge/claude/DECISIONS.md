# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib files in `/app/data/` using `joblib.load()`. Each file contains a dictionary keyed by the animal name, with fields `trace`, `position`, `envs`, and `blocked`. The 7 animals are listed explicitly in a constant `ANIMALS`. Animals can be processed in parallel using joblib's `Parallel`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
```

iii. From CONVERSION_NOTES.md Step 1: "load_dat(animal, p, format) ... Loads one animal's dict (joblib or MATLAB v7.3 via mat73)." The AI chose joblib files because they are the same data as the `.mat` files and load directly into Python dictionaries.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (animal). The 7 animal IDs are hardcoded in the `ANIMALS` list. Each animal is processed independently via `process_animal()`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
data['subjects'] = list(ANIMALS)
...
data['subject_idx'].append(ANIMALS.index(s['animal']))
```

iii. The AI identified that each file corresponds to one animal and used the filenames as subject identifiers. The subject list always contains all 7 animals regardless of `--sample` mode.

## 1-c. How are the data split into sessions?

i. Within each animal's data, `trace` has shape `(n_days, n_cells, n_frames)`. Each day (first axis index) becomes a separate session. Sessions are iterated via `for day in days`.

ii.
```python
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
if days is None:
    days = list(range(trace.shape[0]))
...
for day in days:
    res = process_session(trace[day], position[day], blocked[day], str(envs[day]),
                          session_id=f'{animal}_day{day:02d}', ...)
```

iii. From CONVERSION_NOTES.md Step 2: "6 animals x 31 + 1 animal (QLAK-CA1-51) x 21 -> 207 sessions". Each recording day becomes a session.

## 1-d. How are the data split into trials?

i. Each ~40-minute session is split into consecutive 1-minute blocks of 600 time bins (100 ms each). Within each block, only bins where the mouse was moving (speed > 5 cm/s) are retained, making trials variable-length. The final partial block (~18 s) is kept as a shorter trial. Trials with fewer than 30 retained bins (3 seconds) are dropped.

ii.
```python
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TBIN))    # 600 bins of 100 ms
MIN_TRIAL_BINS = 30                                    # drop trials with < 3 s of movement
...
for start in range(0, nb, TRIAL_BINS):
    idx = np.arange(start, min(start + TRIAL_BINS, nb))
    sel = idx[moving[idx]]
    if len(sel) < MIN_TRIAL_BINS:
        n_dropped += 1
        continue
    trials_neural.append(np.ascontiguousarray(neural[:, sel]))
```

iii. From CONVERSION_NOTES.md Step 5: "Trials = consecutive 1 min blocks (600 bins of 100 ms) of the session, as required by the task. The final partial block of each session (~18 s) is kept as a shorter trial so that no data is discarded. Trials retaining fewer than 30 bins (3 s of movement) after the speed filter are dropped as unusable."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on two criteria: (1) the speed filter removes individual time bins where the mouse moves <= 5 cm/s, and (2) 1-minute blocks with fewer than 30 retained bins after speed filtering are dropped entirely. Additionally, sessions with fewer than 2 usable trials are skipped.

ii.
```python
moving = speed_b > V_THRESH    # velocity filter (reference: v_thresh=5)
...
if len(sel) < MIN_TRIAL_BINS:
    n_dropped += 1
    continue
...
if len(s['neural']) < 2:
    print(f"  !! skipping session {s['session_id']}: only {len(s['neural'])} usable trials")
    continue
```

iii. From CONVERSION_NOTES.md Step 5: "Speed > 5 cm/s filter ... matches the reference decoding analysis." The minimum 30-bin threshold and minimum 2-trials-per-session rule ensure sufficient data for decoder training.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in the joblib files, which contains binarized calcium-transient rising-phase event trains. Shape is `(n_days, n_cells, n_frames)` with values {0, 1} or NaN (for unregistered cells).

ii.
```python
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
...
n_cells, n_frames = trace.shape
...
registered = ~np.isnan(trace[:, 0])    # NaN is all-or-none per (cell, day)
raw = trace[registered][:, :nb * TBIN].astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 1: "The dataset already contains the binarized rising-phase event trains ('trace'), which the paper treats as the firing rate. No further neuron quality filtering exists beyond [cell curation]."

## 2-b. How is the `neural` data processed?

i. The AI applies: (1) Gaussian smoothing with sigma=3 frames along the time axis, (2) average pooling over 3-frame (100 ms) bins, (3) multiplication by 30 (FPS) to convert to events/s. This exactly follows the reference `fit_decoder` pipeline.

ii.
```python
TRACE_SIGMA = 3.0      # frames, Gaussian smoothing of the traces (fit_decoder)
...
smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)     # (n_reg, nb) in events/s
```

iii. From CONVERSION_NOTES.md Step 5: "Time bin = 100 ms (3 frames at 30 Hz), with a 3-frame Gaussian smoothing before pooling. This is exactly the reference decoder's temporal_bin_size=3 + gaussian_filter1d(sigma=3)." And "Units: events/s (Hz), i.e. the pooled mean binary value x 30 fps, as get_rate_maps does."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) cells not registered on a given day (all-NaN trace) are removed, (2) of the registered cells, those with <= 5 events during the retained moving bins are dropped. This matches the reference `decode_position_within` function's `cell_threshold=5`.

ii.
```python
registered = ~np.isnan(trace[:, 0])                            # NaN is all-or-none per (cell, day)
raw = trace[registered][:, :nb * TBIN].astype(np.float32)
events_b = bin_time(raw, nb, how='sum')                        # events per 100 ms bin (for curation)
...
active = events_b[:, moving].sum(axis=1) > CELL_THRESH
cell_idx = np.where(registered)[0][active]
neural = neural[active]
```

iii. From CONVERSION_NOTES.md Step 5: "Cell curation: registered that day AND > 5 events during the retained moving bins, matching decode_position_within's cell_idx. No place-cell selection (the paper includes all cells)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus event to align to. The sessions are continuous 40-minute recordings with no trial structure. Trials are created by cutting the session into consecutive 1-minute blocks. Within each block, only the time bins where the mouse was moving are retained (in order). The alignment event is effectively the start of each 1-minute block.

ii.
```python
for start in range(0, nb, TRIAL_BINS):
    idx = np.arange(start, min(start + TRIAL_BINS, nb))
    sel = idx[moving[idx]]
    ...
    trials_neural.append(np.ascontiguousarray(neural[:, sel]))
```

iii. From metadata: "Start of each 1-minute trial. Sessions are continuous 40-min recordings cut into consecutive 1-min blocks; within a block only the time bins in which the mouse ran faster than 5 cm/s are kept, so trials contain <= 600 time bins."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 100 ms (3 frames at 30 Hz). Temporal rebinning is applied: the native 30 Hz data is average-pooled over 3-frame bins after Gaussian smoothing.

ii.
```python
FPS = 30.0             # acquisition rate
TBIN = 3               # frames per time bin -> 100 ms
...
smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
```
```python
def bin_time(x, nbins, how='mean'):
    x = x[..., :nbins * TBIN]
    x = x.reshape(x.shape[:-1] + (nbins, TBIN))
    return x.mean(axis=-1) if how == 'mean' else x.sum(axis=-1)
```

iii. From CONVERSION_NOTES.md Step 1: "Temporal binning (inside fit_decoder/test_decoder): traces are smoothed with gaussian_filter1d(sigma=3 frames) and average-pooled over temporal_bin_size=3 frames -> 100 ms bins at the 30 Hz acquisition rate."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Input is derived from the `blocked` field of each animal's data, which contains the indices of blocked partitions in the 3x3 grid for each session. A value of -1 means nothing is blocked (square geometry).

ii.
```python
blocked_vec, blocked_ids = blocked_vector(blocked_entry)
...
def blocked_vector(blocked_entry):
    e = blocked_entry
    if isinstance(e, (list, tuple)):
        e = e[0]
    idx = np.atleast_1d(np.asarray(e, dtype=float)).ravel()
    idx = idx[idx >= 0].astype(int)
    vec = np.zeros(N_GRID * N_GRID, dtype=np.float32)
    vec[idx] = 1.0
    return vec, tuple(sorted(idx.tolist()))
```

iii. From CONVERSION_NOTES.md Step 4: "blocked equals the zeros of flipud(get_env_mat(env)) for all 10 geometries... I use the per-session blocked field, which I verified directly against occupancy."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted to a 9-dimensional binary vector where 1 indicates a blocked partition and 0 indicates an open partition. The vector is static per trial (same for all trials within a session). Negative indices (-1, indicating no blocked partitions) are filtered out.

ii.
```python
vec = np.zeros(N_GRID * N_GRID, dtype=np.float32)
vec[idx] = 1.0
return vec, tuple(sorted(idx.tolist()))
...
trials_input.append(blocked_vec.copy())
```

iii. From CONVERSION_NOTES.md Step 5: "Input = 9-dim binary geometry vector (1 = blocked), static per trial, exactly the Decoder Task specification."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` field, which contains 2D head position coordinates (x, y) in cm, shape `(2, n_frames)`, range [0, 75].

ii.
```python
position = dat['position']
...
pos_b = bin_time(position, nb)    # (2, nb), cm
part, n_fixed = discretize_position(pos_b, blocked_ids)
```

iii. From CONVERSION_NOTES.md Step 2: "position: (n_days, 2, n_frames) float64, x-y head position in cm, range exactly [0, 75] in both dims, no NaNs."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is first average-pooled over 3-frame bins (same binning as neural data), then discretized into a 3x3 grid. The discretization uses `floor(pos / 25)` with clipping to [0, 2], computing partition = 3*row + col where row = floor(y/25) and col = floor(x/25).

ii.
```python
pos_b = bin_time(position, nb)                                 # (2, nb), cm
part, n_fixed = discretize_position(pos_b, blocked_ids)        # (nb,) partition index 0..8
...
def discretize_position(pos_binned, blocked_ids):
    xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    part = N_GRID * yb + xb
```

iii. From CONVERSION_NOTES.md Step 5: "Output = single categorical variable with 9 values (the 3x3 partition), time-varying... Value names r0c0...r2c2 (row = North-South = position dim 1, column = West-East = position dim 0)."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using fixed 25 cm bin edges (75 cm / 3 = 25 cm per partition). The formula `partition = 3 * floor(y/25) + floor(x/25)` creates 9 categories (0-8). Positions exactly at the 75 cm boundary are clipped into bin 2. Samples that land in a blocked partition (due to tracking noise at partition walls) are reassigned to the nearest open partition based on Euclidean distance.

ii.
```python
PART_SIZE = ARENA_SIZE / N_GRID   # 25 cm
...
xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
part = N_GRID * yb + xb
...
if len(blocked_ids):
    bad = np.isin(part, blocked_ids)
    ...
    open_ids = np.array([p for p in range(N_GRID * N_GRID) if p not in blocked_ids])
    d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None, :, :], axis=2)
    part[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. From CONVERSION_NOTES.md Step 5: "average-pool 3 frames -> partition = 3*floor(y/25) + floor(x/25), clipped to [0,2] per axis; rare labels in a blocked partition reassigned to the nearest open partition."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are acquired at the same 30 Hz rate (frame i of position corresponds to frame i of trace). Both are binned identically using the same 3-frame average-pooling via `bin_time()`. The speed filter is applied to both: only bins where the mouse is moving are retained, and the same bin indices are used to slice both neural and output arrays within each trial.

ii.
```python
pos_b = bin_time(position, nb)         # (2, nb), cm
...
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)     # (n_reg, nb) in events/s
...
trials_neural.append(np.ascontiguousarray(neural[:, sel]))
...
trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. From CONVERSION_NOTES.md Step 4: "position frame i is aligned to trace frame i (both streams in the files have identical length)." Both are binned and filtered identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) NaN traces for unregistered cells are filtered out. (2) Positions exactly at 75 cm are clipped to bin 2 (198 frames of 14.9M). (3) Samples landing in blocked partitions after binning (14 frames total) are reassigned to the nearest open partition. (4) Frame counts not divisible by 3 have trailing frames dropped (<0.003%). (5) The final partial 1-minute block is kept as a shorter trial rather than discarded. (6) Two sessions have neurons that pass the >5 events threshold but are silent in all retained trials; these are left as-is.

ii.
```python
xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
...
x = x[..., :nbins * TBIN]    # truncate to complete bins
...
for start in range(0, nb, TRIAL_BINS):
    idx = np.arange(start, min(start + TRIAL_BINS, nb))  # handles partial last block
```

iii. From CONVERSION_NOTES.md Step 10 Check 5: "Position values exactly at the 75 cm arena edge (198 frames of 14.9M) are clipped into bin 2." and "Samples whose averaged position falls in a blocked partition (14 frames in the whole dataset) are reassigned to the nearest open partition."

## 6-a. What are the most time-consuming steps of the code?

i. File loading is the dominant cost. Loading one animal's joblib file takes 6-16 seconds. Per-session processing (Gaussian smoothing, binning, speed computation) takes 0.32-0.59 seconds per session. Total conversion time is ~41 seconds for all 207 sessions.

ii. N/A (timing is reported in stdout, not in code structure)

iii. From CONVERSION_NOTES.md Step 6: "Animals are processed in parallel (--jobs, default 7). Result: 41 s for the full dataset (207 sessions), of which ~17 s is per-animal processing (0.54-0.59 s/session) and ~16 s is file loading."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-cutting loop iterates over 1-minute blocks within each session, applying the speed filter and slicing arrays. This could potentially be vectorized but is already fast relative to file I/O. The per-session loop within each animal is sequential.

ii.
```python
for start in range(0, nb, TRIAL_BINS):
    idx = np.arange(start, min(start + TRIAL_BINS, nb))
    sel = idx[moving[idx]]
    ...
```

iii. From CONVERSION_NOTES.md Step 6: "The naive implementation would loop over frames; all binning is instead done with a single reshape(...).mean(-1)." The main vectorization was already applied to the binning step.

## 6-c. What processing does the code repeat multiple times?

i. The speed computation and binning are done once per session. However, `bin_time` is called separately for neural traces, position, and speed - each performing a reshape-and-reduce on different arrays. The `events_b` (sum-binned raw events for cell curation) and the smoothed+averaged neural array are computed separately from the same raw data.

ii.
```python
speed_b = bin_time(speed, nb)
pos_b = bin_time(position, nb)
events_b = bin_time(raw, nb, how='sum')
smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
```

iii. No explicit justification given for this structure. Each `bin_time` call processes a different signal or uses a different aggregation method (mean vs sum), so the repetition is necessary.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `cell_idx` (indices of kept cells into the animal's full cell list) and stores it in session metadata, but this is only informational and not used by the decoder. The `n_blocked_fixed` count and detailed `session_info` metadata are also computed but not used downstream. The blocked-partition reassignment for output labels (affecting 14 frames total) is arguably unnecessary given the speed filter removes most of those frames.

ii.
```python
cell_idx = np.where(registered)[0][active]    # stored in session_info but not used by decoder
...
'n_blocked_fixed': n_fixed,    # diagnostic counter
```

iii. From CONVERSION_NOTES.md Step 10: these are documented as diagnostic/bookkeeping information to support validation and reproducibility.
