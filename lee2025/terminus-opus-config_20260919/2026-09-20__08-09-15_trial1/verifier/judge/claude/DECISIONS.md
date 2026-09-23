# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib` from extensionless joblib files in `/app/data/`, rather than the `.mat` files. Each file is a dictionary keyed by the animal ID, containing fields `trace`, `position`, `envs`, and `blocked`. The animal names are hardcoded in a list. The reference code's `load_dat` function supports both formats; the AI chose joblib for speed.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
traces = dat['trace']
positions = dat['position']
envs = np.array(dat['envs']).ravel()
blocked = dat['blocked']
```

iii. The AI notes that the data directory contains both `.mat` and joblib files with identical content, and the reference code's `load_dat` function defaults to joblib format. Joblib is faster to load. Documented in CONVERSION_NOTES.md Step 2.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The subject names are the 7 hardcoded animal IDs from the reference code's `main.py`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for ai, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
```

iii. The animal list matches the 7 mice described in the paper and reference code. Each file contains all recording sessions for one animal.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal's data becomes a separate session. The number of days is determined by the shape of the `position` array (`ndays = positions.shape[0]`).

ii.
```python
ndays = positions.shape[0]
...
for d in range(ndays):
    trace_day = np.asarray(traces[d])
    position_day = np.asarray(positions[d])
```

iii. This matches the reference code's per-day analysis structure and yields 207 total sessions (31 per mouse except QLAK-CA1-51 with 21), matching the paper.

## 1-d. How are the data split into trials?

i. Each session's temporally-binned data (100ms bins after 3-frame average pooling) is split into consecutive 1-minute windows of 600 bins each. The trailing partial window is dropped. Within each trial, only bins where the animal was moving (speed > 5 cm/s) are retained, so trials have variable numbers of timepoints.

ii.
```python
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES))   # 600 bins
...
n_full_trials = nbins // TRIAL_BINS
for t in range(n_full_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    if m.sum() < MIN_TRIAL_BINS:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
    output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
```

iii. The 1-minute trial duration is specified in the task instructions. The speed-based filtering within trials follows the reference `decode_position_within` function, which excludes immobile frames from decoding.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 30 moving bins (3 seconds of moving data) are dropped. Sessions with fewer than 2 trials after filtering are also dropped.

ii.
```python
MIN_TRIAL_BINS = 30
...
if m.sum() < MIN_TRIAL_BINS:
    continue
...
if len(neural_trials) < 2:
    print(f'  WARNING session {animal} day {d} has < 2 trials, skipped')
    continue
```

iii. The minimum of 30 moving bins ensures trials have enough data for meaningful decoding. The minimum 2 trials per session requirement comes from the task instructions ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). In practice, no sessions are dropped (minimum is 22 trials).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field, which contains binary rising-phase calcium event traces with shape `(n_cells, n_frames)`.

ii.
```python
trace_day = np.asarray(traces[d])
```

iii. The `trace` field contains pre-extracted binary (0/1) rising-phase transient vectors. The paper states "This binary vector was treated as the firing rate in all further analyses."

## 2-b. How is the `neural` data processed?

i. The AI applies the reference code's decoding preprocessing pipeline:
1. Filter to registered cells (non-NaN) and cells with >5 events while moving
2. Cast to float32
3. Gaussian smoothing with sigma=3 frames along time axis
4. Average pooling over 3 frames (100ms bins), matching `AvgPool1d(kernel_size=3, stride=3)` in `fit_decoder`
5. Multiply by 30 (FPS) to convert to event rate in Hz
6. Remove non-moving bins (speed ≤ 5 cm/s)

ii.
```python
tr = trace_day[keep_cells].astype(np.float32)
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS  # event rate in Hz
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
```

Where `bin_frames` does:
```python
def bin_frames(x, nframes_per_bin=TEMPORAL_BIN_FRAMES):
    T = x.shape[-1]
    nbins = T // nframes_per_bin
    x = x[..., :nbins * nframes_per_bin]
    return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)
```

iii. The AI documents that this processing matches the reference `fit_decoder` and `decode_position_within` functions. The Gaussian smoothing and temporal binning are from `fit_decoder(temporal_bin_size=3)`, and the speed filtering is from `decode_position_within(v_thresh=5)`. The one deliberate deviation is smoothing the continuous trace before removing immobile frames (rather than after), which the AI argues is more correct to avoid mixing activity across temporal discontinuities.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering criteria are applied:
1. Cells must be registered on that day (trace not all-NaN)
2. Cells must have more than 5 events during moving frames (`cell_threshold=5` from `decode_position_within`)

ii.
```python
nan_all = np.isnan(trace_day).all(axis=1)
registered = ~nan_all
speed = compute_speed(position_day)
moving = speed > V_THRESH
events_moving = np.nansum(trace_day[:, moving], axis=1)
keep_cells = registered & (events_moving > CELL_THRESHOLD)
```

iii. The AI notes this matches `decode_position_within`'s curation: "cells with > 5 events during moving frames within that day". This removes 882/69,744 = 1.3% of cell-sessions. The AI also verifies that no cells are partially NaN (`assert np.array_equal(nan_any, nan_all)`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous free foraging with no experimenter-defined trials or stimulus events. Position and neural data are already frame-aligned in the released data (both acquired at 30 Hz on the same DAQ). Trials are artificial 1-minute segments of the continuous recording.

ii. N/A (no alignment code needed)

iii. The AI verifies: "streams are already aligned 1:1; no additional alignment needed" and confirms by reproducing the published Bayesian decoding error to within 0.15 cm (Step 12 of CONVERSION_NOTES.md).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100ms time bins (3 frames at 30 Hz), matching the reference code's `temporal_bin_size=3` parameter in `fit_decoder`/`test_decoder`. This is achieved via Gaussian smoothing (sigma=3 frames) followed by average pooling over 3 frames.

ii.
```python
FPS = 30.0
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100 ms
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES))  # 600
...
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS
```

iii. The AI identifies this binning from the reference `fit_decoder` function and notes it is "identical to `fit_decoder`/`test_decoder`: gaussian_filter1d(trace, sigma=3 frames) then AvgPool1d(kernel=3, stride=3)".

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the `blocked` field in each animal's data, which contains indices of blocked partition positions for each recording session.

ii.
```python
blocked = dat['blocked']
...
blocked_day = blocked[d]
...
geometry = blocked_to_vector(blocked_day)
```

iii. The AI notes that the `blocked` field is the per-session ground truth (not `get_env_mat(env)`, which does not capture vertically flipped presentations). This was verified by checking that occupancy is zero in blocked partitions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-dimensional binary vector (1=blocked, 0=open). If no positions are blocked (value -1), the vector is all zeros. The vector is static per trial (same for all trials in a session).

ii.
```python
def blocked_to_vector(blocked_day):
    b = np.atleast_1d(np.asarray(blocked_day[0]).ravel()).astype(float)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if b.size and b[0] >= 0:
        vec[b.astype(int)] = 1.0
    return vec
...
input_trials.append(geometry.copy())
```

iii. The partition indexing convention is `p = 3*ybin + xbin`, verified empirically: "occupancy in blocked partitions is 0.0000 for `3*y+x` versus ~0.19 for the alternative".

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field, which contains 2D (x, y) coordinates in cm at 30 Hz.

ii.
```python
position_day = np.asarray(positions[d])  # (2, n_frames)
```

iii. Position is in the 75x75 cm arena, range [0, 75]. The AI verified no NaNs in position data.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first temporally binned (mean over 3-frame windows, same as neural data), then discretized into a 3x3 grid using the reference code's formula: `floor(position / ((arena_max + buffer) / n_bins))`.

ii.
```python
pos_binned = bin_frames(position_day)  # (2, nbins), cm
...
def position_to_class(position_binned_xy):
    bin_down = (ARENA_SIZE + BUFFER) / N_SPATIAL_BINS  # 25 cm
    xy = np.floor(position_binned_xy / bin_down).astype(int)
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. The AI uses the reference `decode_position_within` formula with `n_bins=3` instead of 15, as required by the decoder task (3x3=9 spatial bins). The bin edges at 25 and 50 cm coincide with the physical partition walls.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous 2D position is discretized into 9 categories using `class = 3 * y_bin + x_bin`, where each axis is divided into 3 equal bins of 25 cm. Values at the boundary (75 cm) are clipped into the last bin.

ii.
```python
bin_down = (ARENA_SIZE + BUFFER) / N_SPATIAL_BINS  # ~25.0 cm
xy = np.floor(position_binned_xy / bin_down).astype(int)
xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. The 3x3 discretization is required by the task ("Mouse position discretized into 3 x 3 = 9 spatial bins"). The bin edges at 25 cm match the physical partition walls in the arena.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are acquired on the same DAQ at 30 Hz and are frame-aligned in the raw data. Both undergo the same temporal binning (3-frame average pooling) and the same speed-based bin exclusion, so they remain aligned throughout processing. Trials are cut from the same time indices.

ii.
```python
pos_binned = bin_frames(position_day)           # (2, nbins)
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
```

iii. The AI verifies alignment through processing plots and by reproducing the published decoding error to within 0.15 cm.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- All-NaN neurons (unregistered cells) are removed
- Low-activity cells (≤5 events while moving) are removed
- Immobile timepoints (speed ≤ 5 cm/s) are excluded
- Trailing partial minutes are dropped
- Trials with < 30 moving bins are dropped
- Sessions with < 2 trials are dropped (none actually are)
- First-frame speed is set to 0 (undefined from differencing)
- Position at exactly 75 cm is clipped into the last bin
- Partially NaN cells are asserted to not exist
- Tiny fractions of timepoints in blocked partitions (DeepLabCut tracking jitter) are left as-is

ii.
```python
nan_all = np.isnan(trace_day).all(axis=1)
registered = ~nan_all
assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
...
keep_cells = registered & (events_moving > CELL_THRESHOLD)
...
if m.sum() < MIN_TRIAL_BINS:
    continue
...
if len(neural_trials) < 2:
    continue
...
speed[1:] = gaussian_filter1d(d, sigma=V_FILT_SIGMA)  # speed[0] = 0
...
xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
```

iii. The AI documents these edge cases in CONVERSION_NOTES.md Step 10 Check 5 and explains the rationale for each.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies file loading as the dominant cost: 9-16 seconds per animal file (joblib), while per-session processing takes only 0.3-1 second. Total conversion is ~220 seconds for all 207 sessions.

ii. N/A (timing is printed, not a code decision)

iii. CONVERSION_NOTES.md Step 7 documents: "Each animal file is loaded exactly once (9-16 s each); the per-session processing is 0.3-1 s."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized all binning and discretization operations using numpy reshape/mean rather than per-frame loops. The trial-splitting loop is inherently sequential. No obvious remaining vectorization opportunities.

ii.
```python
def bin_frames(x, nframes_per_bin=TEMPORAL_BIN_FRAMES):
    T = x.shape[-1]
    nbins = T // nframes_per_bin
    x = x[..., :nbins * nframes_per_bin]
    return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)
```

iii. The AI notes this is ~100x faster than reference-style per-frame loops.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each animal file is loaded once, each session processed once. The speed computation, neural binning, and position discretization are each done once per session.

ii. N/A

iii. No redundant computation identified.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores detailed `session_info` metadata (environment name, blocked partitions, neuron counts, etc.) and auxiliary data for plotting. The plotting data is not saved to the pickle file. The `envs` field is loaded but only stored in metadata, not used for the geometry input (which correctly uses `blocked`).

ii.
```python
session_info.append(dict(animal=animal, day=int(d), env=str(envs[d]),
                         blocked=np.where(stats['geometry'] > 0)[0].tolist(),
                         n_registered=stats['n_registered'],
                         n_neurons=stats['n_kept_cells'],
                         ...))
```

iii. The extra metadata is useful for debugging and documentation but not needed by the decoder. Processing cost is negligible.
