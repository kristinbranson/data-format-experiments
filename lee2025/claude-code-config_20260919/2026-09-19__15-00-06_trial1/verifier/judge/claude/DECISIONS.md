# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a `.mat` file (MATLAB v7.3 / HDF5) in the data directory. The agent hardcodes the 7 animal IDs in an `ANIMALS` list and opens each file via `h5py`. Within each file, sessions (days) are read lazily one at a time via object references to `trace`, `position`, `envs`, and `blocked`. The 7 animals are processed in parallel using `multiprocessing.Pool`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
# ...
def read_session(f, day):
    position = f[f['position'][day, 0]][()].astype(np.float64)
    trace = f[f['trace'][day, 0]][()].astype(np.float32)
    env = _h5_str(f, f['envs'][0, day])
    blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
    blocked = blocked[blocked >= 0].astype(int)
    return position, trace, env, blocked
# ...
def process_animal(task):
    animal, days, plot_days = task
    with h5py.File(os.path.join(DATA_DIR, f"{animal}.mat"), 'r') as f:
        n_days = f['trace'].shape[0]
        day_list = range(n_days) if days is None else days
        for day in day_list:
            position, trace, env, blocked = read_session(f, day)
```

iii. The agent chose HDF5 lazy per-day reads over loading entire joblib files to reduce memory usage (~0.3 GB per worker vs 9-25 GB). The CONVERSION_NOTES.md documents this choice in Step 5 Decision 9, noting the `.mat` and joblib contents are byte-identical.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject. The agent hardcodes the 7 animal IDs in the `ANIMALS` list. Subjects are identified by their animal ID strings.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
# ...
subjects = sorted({s['animal'] for s in all_sessions})
```

iii. The 7 animal IDs are taken from `main.py` in the reference code. Each file contains all recording sessions for one animal.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording days. Each day is a separate session. The number of days is read from the file's `trace` array shape. Each day's data (trace, position, envs, blocked) is read separately.

ii.
```python
n_days = f['trace'].shape[0]
day_list = range(n_days) if days is None else days
for day in day_list:
    position, trace, env, blocked = read_session(f, day)
    sess = process_session(position, trace, env, blocked,
                           session_id=f"{animal}_day{day:02d}", ...)
```

iii. A recording day is a natural session boundary since the geometry (decoder input) changes daily and cell identity is only stable within a day.

## 1-d. How are the data split into trials?

i. After temporal smoothing and binning to 100 ms bins (3-frame pools at 30 Hz → 600 bins per 60 s), each session is split into contiguous 1-minute blocks (BINS_PER_TRIAL = 600). Within each block, only bins where the animal was running (speed > 5 cm/s) are retained. The trailing partial block at the end of a session is kept if it spans >= 10 s (MIN_TAIL_BINS = 100 bins). This means trials have **variable length** after immobility removal.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / BIN_FRAMES))  # 600
MIN_TAIL_BINS = 100
MIN_TRIAL_BINS = 30
# ...
starts = list(range(0, n_bins - BINS_PER_TRIAL + 1, BINS_PER_TRIAL))
bounds = [(s, s + BINS_PER_TRIAL) for s in starts]
tail_start = len(starts) * BINS_PER_TRIAL
if n_bins - tail_start >= MIN_TAIL_BINS:
    bounds.append((tail_start, n_bins))

for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    if idx.size < MIN_TRIAL_BINS:
        continue
    neural_trials.append(np.ascontiguousarray(neural[:, idx]))
    output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

iii. The agent justifies this by following the reference paper's `decode_position_within` which excludes immobility frames (speed <= 5 cm/s). The 1-minute block definition comes from the Decoder Task instructions. The trailing partial block retention preserves data. The MIN_TRIAL_BINS threshold ensures each trial has enough data for meaningful decoding.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 30 running bins (3 seconds of running data after speed filtering) are dropped. Sessions with fewer than 2 usable trials are dropped (though none actually are in this dataset).

ii.
```python
MIN_TRIAL_BINS = 30
# ...
if idx.size < MIN_TRIAL_BINS:
    continue
# ...
if len(neural_trials) < 2:
    return None
```

iii. The minimum 2 trials per session is required by the decoder validation (train/test split). The minimum trial length ensures meaningful decoding data per trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the `.mat` file, which contains binarised calcium-transient rising phases with shape `(T, n_cells)` in HDF5 orientation.

ii.
```python
trace = f[f['trace'][day, 0]][()].astype(np.float32)
```

iii. The `trace` variable is already the final binarised event vector (values in {0, 1}, NaN for unregistered cells). No dF/F computation is needed; the paper states "This binary vector was treated as the firing rate in all further analyses."

## 2-b. How is the `neural` data processed?

i. The agent applies three processing steps from the reference paper's decoder pipeline:
1. Remove unregistered cells (all-NaN columns)
2. Remove cells with <= 5 binarised events while running (the reference's `cell_threshold=5`)
3. Gaussian smoothing with sigma=3 frames, then average-pooling over 3 frames to produce 100 ms bins

After these steps, only timepoints where the animal was running (speed > 5 cm/s) are retained.

ii.
```python
# Cell curation
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
events_running = trace[moving].sum(axis=0)
keep_cell = events_running > CELL_THRESH
trace = trace[:, keep_cell]

# Temporal smoothing + binning
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)  # sigma=3
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
```

iii. The agent explains this follows `fit_decoder` from the reference code (`gaussian_filter1d(traces, sigma=temporal_bin_size)` then `AvgPool1d(kernel=3, stride=3)`). One deliberate difference: smoothing is applied to the intact 30 Hz session before discarding immobility, while the reference smooths after subsetting, which convolves across temporal discontinuities.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied per session:
1. Cells not registered (all-NaN trace) are removed
2. Cells with <= 5 binarised events during running frames are removed (reference's `cell_threshold=5`)

This removes ~1.26% of registered cell-sessions (882 of 69,744).

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]

speed = running_speed(position)
moving = speed > V_THRESH
events_running = trace[moving].sum(axis=0)
keep_cell = events_running > CELL_THRESH  # CELL_THRESH = 5
trace = trace[:, keep_cell]
```

iii. The agent cites `decode_position_within`: `cell_idx = traces[vel_idx].sum(axis=0) > 5`. No place-cell selection is applied, per the paper's statement that "all cells" are included in analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The experiment is continuous free foraging with no stimulus events. Neural and behavioral data are frame-aligned at 30 Hz by the authors' DAQ. Trials are artificial 1-minute blocks of the continuous recording.

ii. N/A (no alignment code needed)

iii. The agent documents that "imaging and behaviour were acquired on the same DAQ at 30 Hz and are frame-aligned in the source data." This was verified by reproducing the dataset's stored rate maps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 100 ms time bins (10 Hz). The original 30 Hz data is Gaussian-smoothed (sigma=3 frames) then average-pooled over 3 frames, following `fit_decoder(temporal_bin_size=3)`.

ii.
```python
BIN_FRAMES = 3
SMOOTH_SIGMA = 3
TIME_BIN_MS = 1000.0 * BIN_FRAMES / FPS  # 100 ms

smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
```

iii. This exactly matches the reference decoder's `fit_decoder` function which uses `temporal_bin_size=3` with `fps=30`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the `.mat` file, which contains indices of blocked reward locations for each session. The `envs` variable (geometry name) is used to cross-check the blocked list.

ii.
```python
blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
blocked = blocked[blocked >= 0].astype(int)

# Cross-check
env = _h5_str(f, f['envs'][0, day])
expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected)
```

iii. The `blocked` variable stores which of the 9 partitions are walled off. The `-1` sentinel for "nothing blocked" (the square geometry) is filtered to an empty array.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked partition indices are converted to a 9-dimensional binary vector (1 = blocked, 0 = open). This is static per session (all trials in a session share the same geometry).

ii.
```python
geometry = np.zeros(NBINS * NBINS, dtype=np.float32)
geometry[blocked] = 1.0
# ...
input_trials.append(geometry.copy())
```

iii. One-hot encoding allows the decoder to independently represent each blocked partition. The cross-check against `get_env_mat` ensures the blocked list matches the geometry name for all 207 sessions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the `.mat` file, which contains 2D head coordinates (x, y) in cm at 30 Hz, tracked by DeepLabCut.

ii.
```python
position = f[f['position'][day, 0]][()].astype(np.float64)  # (T, 2)
```

iii. The position variable records the animal's head location in the 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is average-pooled over 3 frames (same as neural data), then discretized into a 3x3 grid. Bin assignment uses `floor(position / 25 cm)` clipped to [0, 2], with `bin = 3 * ybin + xbin`. Samples that land in a blocked partition (~0.006%) are snapped to the nearest open partition.

ii.
```python
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
pos_bin = position_to_bin(position_binned_cm).astype(np.int64)
pos_bin = snap_to_open_bins(pos_bin, blocked)

def position_to_bin(position_cm):
    b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
    return NBINS * b[:, 1] + b[:, 0]
```

iii. The 25 cm bin size comes from 75 cm / 3 bins. The agent verified that using the fixed arena size (rather than per-session max) is necessary to reproduce the dataset's stored occupancy maps. The snap-to-open-bins handles DeepLabCut tracking artifacts near partition walls.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D position is discretized into 9 categories using a 3x3 grid over the 75 cm arena. Each axis is divided into 3 equal 25 cm bins. The bin index is `3 * floor(y/25) + floor(x/25)`, clipped to [0, 2] per axis. Positions in blocked partitions are snapped to the nearest open partition.

ii.
```python
BIN_SIZE_CM = ARENA_CM / NBINS  # 25 cm

def position_to_bin(position_cm):
    b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
    return NBINS * b[:, 1] + b[:, 0]

def snap_to_open_bins(pos_bin, blocked):
    if blocked.size == 0:
        return pos_bin
    open_bins = np.setdiff1d(np.arange(NBINS * NBINS), blocked)
    # ...snap blocked samples to nearest open bin...
    return lut[pos_bin]
```

iii. The `snap_to_open_bins` function addresses ~152 samples (0.006%) where DeepLabCut tracked the head slightly past a partition wall. The agent traces this to the reference decoder's `temp_maps` cleaning.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are both average-pooled over the same 3-frame bins from the original 30 Hz frame-aligned data, then the same running-speed mask is applied to both. Both are split into trials using the same index arrays.

ii.
```python
# Same pooling for both neural and position
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
# Same running mask
idx = np.flatnonzero(moving_bins[s:e]) + s
neural_trials.append(neural[:, idx])
output_trials.append(pos_bin[idx][np.newaxis, :])
```

iii. Frame-for-frame alignment is guaranteed by the DAQ and preserved through identical pooling and masking operations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Unregistered cells (all-NaN traces): removed per session, with an assertion that registered cells never have partial NaNs
- `blocked = -1` (open square): filtered to an empty array, resulting in an all-zero input vector
- Position at arena edge (75.0 cm): `np.clip` ensures valid bin assignment
- Frame 0 has no velocity: speed set to 0, always excluded (matches reference)
- Session length not divisible by 3 frames: trailing 1-2 frames dropped
- Session length not divisible by 600 bins: trailing partial block kept if >= 10s
- Head tracked past partition wall: ~152 samples snapped to nearest open partition
- QLAK-CA1-51 has 21 days (not 31): handled generically by reading n_days from file

ii.
```python
# Edge at arena boundary
b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
# Frame 0
speed = np.zeros(position.shape[0])
speed[1:] = gaussian_filter1d(...)
# Partial NaN assertion
assert not np.any(np.isnan(trace)), "a registered cell has partial NaNs"
# Snap to open bins
pos_bin = snap_to_open_bins(pos_bin, blocked)
```

iii. The agent documents in CONVERSION_NOTES.md Step 10 that the snap-to-open-bins fix was found during the first full conversion when sanity checks detected 127 samples in blocked partitions. All other edge cases are documented in Step 10 Check 5.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and processing the `.mat` files (HDF5 I/O + Gaussian smoothing + array operations). The full conversion takes ~44 seconds with 7 parallel workers. Pickle writing (~3 GB) takes ~4 seconds.

ii.
```python
# Per-session timing
t0 = time.time()
position, trace, env, blocked = read_session(f, day)
sess = process_session(position, trace, env, blocked, ...)
print(f"... | {time.time()-t0:.2f}s", flush=True)
```

iii. The agent notes ~2 seconds per session. With 7 parallel workers (one per animal), the longest animal (31 sessions) takes ~1 minute, well under the 15-minute budget.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `snap_to_open_bins` function uses a Python loop over blocked partitions to build a lookup table, but this is negligible (at most 4 blocked partitions per geometry). The trial-splitting loop is sequential but each iteration is small. No significant vectorization opportunities remain.

ii.
```python
def snap_to_open_bins(pos_bin, blocked):
    # ...
    for b in blocked:  # at most 4 iterations
        d = np.linalg.norm(coords[open_bins] - coords[b], axis=1)
        lut[b] = open_bins[np.argmin(d)]
    return lut[pos_bin]
```

iii. The agent already vectorized the main processing (position binning, speed computation, temporal pooling via reshape+mean). The remaining loops operate on tiny arrays.

## 6-c. What processing does the code repeat multiple times?

i. The `blocked_from_env` cross-check is computed for every session (207 times), but it is a trivial dictionary lookup + array operation. No significant repeated processing is evident.

ii.
```python
expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected)
```

iii. This is a correctness check, not a computational bottleneck.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `snap_to_open_bins` processing affects only 152 of 2,569,722 samples (0.006%). The diagnostic plotting code (`--show-processing`) stores extra arrays in memory that are only used for visualization. The blocked-vs-env cross-check is a safety check not needed for the output.

ii.
```python
if show_processing:
    result['_plot'] = dict(position=position, speed=speed, moving=moving,
                           neural=neural, pos_bin=pos_bin, moving_bins=moving_bins,
                           position_binned_cm=position_binned_cm)
```

iii. The extra data stored for plotting is only generated when `--show-processing` is enabled. The cross-check ensures data integrity. Neither significantly impacts runtime.
