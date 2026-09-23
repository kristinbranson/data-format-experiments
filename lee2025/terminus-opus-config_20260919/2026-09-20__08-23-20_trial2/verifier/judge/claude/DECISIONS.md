# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib files (not the `.mat` files) using `joblib.load()`. Each file contains a dictionary keyed by animal name, with fields `trace`, `position`, `envs`, and `blocked`. The AI iterates over a hardcoded list of 7 animal names (`ANIMALS`), loading each file from `/app/data/<animal_name>`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
envs = np.asarray(dat['envs']).ravel()
n_days = dat['trace'].shape[0]
```

iii. The AI chose joblib over `.mat`/`h5py` because the reference code's `load_dat()` function uses joblib format, the files are identical in content, and joblib loads ~5x faster. This is documented in CONVERSION_NOTES.md Step 2.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to a separate joblib file. The AI iterates over the hardcoded `ANIMALS` list, and each animal becomes a separate subject in the output.

ii.
```python
animals = ANIMALS[3:4] if sample else ANIMALS   # sample: the smallest animal
for ai, animal in enumerate(animals):
    ...
    subject_idx = len(data['subjects'])
    data['subjects'].append(animal)
```

iii. The subject split follows from the data organization: one file per animal, each containing all recording sessions for that animal.

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple recording days (sessions). The AI iterates over the first axis of `dat['trace']` (shape `(n_days, n_cells, n_frames)`), treating each day as a separate session.

ii.
```python
n_days = dat['trace'].shape[0]
...
for d in range(n_days):
    res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
```

iii. Each day corresponds to a single recording session in the paper ("one session was recorded per day"), yielding 207 total sessions.

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into non-overlapping 1-minute trials. The AI works in 100ms bins (600 bins per trial) after temporal rebinning. A trailing partial trial is kept if it is >= 30 seconds (300 bins), otherwise discarded.

ii.
```python
TRIAL_BINS = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES       # 600 bins
MIN_PARTIAL_TRIAL_BINS = TRIAL_BINS // 2               # keep a trailing trial if >= 30 s

def trial_slices(n_bins):
    slices = []
    n_full = n_bins // TRIAL_BINS
    for t in range(n_full):
        slices.append((t * TRIAL_BINS, (t + 1) * TRIAL_BINS))
    rem = n_bins - n_full * TRIAL_BINS
    if rem >= MIN_PARTIAL_TRIAL_BINS:
        slices.append((n_full * TRIAL_BINS, n_bins))
    return slices
```

iii. The AI noted that all sessions are 39.9-40.1 minutes, yielding 39 full 60s trials plus a final partial trial of 55.5-60s, which always exceeds the 30s threshold. This results in 40 trials per session (8,280 total).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All trials from all sessions are kept. The only filtering is discarding trailing partial trials shorter than 30 seconds (which never occurs in practice).

ii. N/A (no filtering code beyond the partial trial threshold shown in 1-d).

iii. The paper has no trial structure (continuous free foraging), so there are no quality criteria for trials. The AI documented this in CONVERSION_NOTES.md Step 5.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field, which contains binarized rising-phase calcium event vectors (0/1 values) at 30 Hz. Shape is `(n_cells, n_frames)` per session.

ii.
```python
tr = trace_day[registered].astype(np.float32)
assert np.all(np.isin(tr, (0.0, 1.0))), "trace must be binary for registered cells"
```

iii. The AI correctly identified that `trace` contains already-preprocessed binary calcium events, citing the paper: "This binary vector was treated as the firing rate in all further analyses."

## 2-b. How is the `neural` data processed?

i. The AI applies temporal processing matching the reference code's `fit_decoder` function: (1) Gaussian smoothing with sigma=3 frames along the time axis, (2) 3-frame non-overlapping average pooling (equivalent to `AvgPool1d(kernel_size=3, stride=3)`), (3) multiplication by 30 (the FPS) to convert to events/second. Smoothing is applied to the entire continuous session before trial splitting to avoid edge artifacts.

ii.
```python
SMOOTH_SIGMA_FRAMES = 3
TEMPORAL_BIN_FRAMES = 3

tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (bin_time_series(tr_s) * FPS).astype(np.float32)      # events / s

def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)
```

iii. The AI explicitly followed the reference code's `fit_decoder` (utils.py:1776) which uses `gaussian_filter1d(sigma=temporal_bin_size)` followed by `AvgPool1d(kernel_size=3, stride=3)`. This is documented extensively in CONVERSION_NOTES.md Steps 1, 5, and 10.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells registered in a given session (non-NaN trace) are included. The AI checks the first frame for NaN to determine registration status, then asserts that unregistered cells are NaN for the entire session.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
assert np.all(np.isnan(trace_day[~registered]).all(axis=1)), \
    "unregistered cells must be NaN for the entire session"
tr = trace_day[registered].astype(np.float32)
```

iii. The AI chose to include all registered cells (69,744 cell-sessions), matching the paper's statement that "these results motivated the inclusion of all cells in subsequent analyses." The AI explicitly did not apply the `cell_threshold=5` event criterion from `decode_position_within`, noting it would only drop 1.3% of cell-sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous free foraging with no discrete task events. Trials are artificial 1-minute segments starting from the beginning of each session. The temporal alignment event is the start of each trial.

ii.
```python
for (a, b) in trial_slices(n_bins):
    neural.append(np.ascontiguousarray(rates[:, a:b]))
```

iii. The AI documented: "there is no explicit task event - free foraging" and set `temporal_alignment_event` to describe alignment to the start of each 1-minute trial, with `off_start=0.0` and `off_end=60.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins from 30 Hz (33.33 ms) to 10 Hz (100 ms) using 3-frame average pooling, matching the reference code's `fit_decoder` temporal binning. This yields 600 time bins per 60-second trial.

ii.
```python
FPS = 30
TEMPORAL_BIN_FRAMES = 3       # reference `temporal_bin_size` -> 100 ms bins
TRIAL_BINS = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES       # 600 bins
# time_bin_size in metadata:
'time_bin_size': 1000.0 * TEMPORAL_BIN_FRAMES / FPS,   # 100.0 ms
```

iii. The AI followed the reference `fit_decoder` code which uses `temporal_bin_size=3` (3 frames at 30 Hz = 100 ms). This is documented in CONVERSION_NOTES.md Steps 1 and 5.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from two sources: (1) the `envs` field (environment name strings like "square", "o", "t", etc.) which is mapped to a 3x3 binary matrix via hardcoded `ENV_MATS`, and (2) the `blocked` field (indices of blocked partitions) which is used as a consistency check.

ii.
```python
envs = np.asarray(dat['envs']).ravel()
...
blocked_vec = env_blocked_vector(env_name)

# consistency check against the dataset's own `blocked` field
bl = np.atleast_1d(np.asarray(blocked_field[0]).ravel()).astype(int)
from_field = np.zeros(9, dtype=np.float32)
if not (bl.size == 1 and bl[0] == -1):
    from_field[bl] = 1
assert np.array_equal(from_field, blocked_vec)
```

iii. The AI used the environment name -> matrix mapping from the reference code's `get_env_mat()` and verified it against the raw `blocked` field for all 207 sessions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a 3x3 binary matrix (from `ENV_MATS`), flipped vertically (`np.flipud`), raveled to a 9-element vector, and inverted (1 = blocked, 0 = open). The result is a static length-9 float32 vector per trial.

ii.
```python
ENV_MATS = {
    'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'o':         [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    ...
}

def env_blocked_vector(env_name):
    mat = np.array(ENV_MATS[env_name], dtype=float)
    open_flat = np.flipud(mat).ravel()      # index p -> 1 if open
    return (open_flat == 0).astype(np.float32)
```

iii. The AI derived the encoding from the reference code's `get_env_mat()` and the partition indexing convention `p = 3*floor(y/25) + floor(x/25)`, which requires `np.flipud` of the matrix. This was verified against the `blocked` field for all sessions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the `position` field, which contains 2D (x, y) head position in cm at 30 Hz, tracked by DeepLabCut. Shape is `(2, n_frames)` per session.

ii.
```python
pos_b = bin_time_series(position_day.astype(np.float64))       # (2, n_bins)
```

iii. The paper describes position tracking using DeepLabCut in the 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first average-pools the raw 30 Hz position into 100 ms bins (matching the neural temporal binning), then discretizes into a 3x3 grid using floor division by 25 cm (the partition size). The partition index is `p = 3*floor(y/25) + floor(x/25)`, clipped to [0, 2] on each axis.

ii.
```python
pos_b = bin_time_series(position_day.astype(np.float64))       # (2, n_bins)
...
def position_to_partition(pos_binned, blocked_vec):
    col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
    row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
    p = (N_PART * row + col).astype(np.int64)
```

iii. The AI followed the reference code's spatial binning convention and verified the partition indexing against the `blocked` field for all 207 sessions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid) by dividing each axis by 25 cm and taking the floor. Values at the arena edge (75 cm) are clipped to bin 2. Positions that fall in blocked partitions (0.003% of bins) are snapped to the nearest open partition.

ii.
```python
PART_CM = ARENA_CM / N_PART   # 25 cm
col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
p = (N_PART * row + col).astype(np.int64)
bad = blocked_vec[p] > 0
if np.any(bad):
    open_ids = np.flatnonzero(blocked_vec == 0)
    d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None], axis=2)
    p[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. The snapping of blocked-partition positions mirrors the reference decoder's approach (`decode_position_within`) which snaps predictions to valid spatial bins. The AI noted this affects only 0.003% of frames.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same DAQ frame index at 30 Hz. Both are temporally binned using the same 3-frame average pooling, then split into trials using the same bin indices, ensuring frame-for-frame alignment.

ii.
```python
rates = (bin_time_series(tr_s) * FPS).astype(np.float32)
pos_b = bin_time_series(position_day.astype(np.float64))
n_bins = rates.shape[1]
assert pos_b.shape[1] == n_bins
for (a, b) in trial_slices(n_bins):
    neural.append(np.ascontiguousarray(rates[:, a:b]))
    output.append(part[a:b][None, :].astype(np.int64))
```

iii. The AI verified alignment by asserting that neural and position have the same number of bins after pooling, and by visual inspection in the processing plots.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of data issues are handled: (1) Unregistered neurons (NaN for the entire session) are removed. (2) Trailing frames that don't fill a complete 100ms bin are dropped (0-2 frames per session). (3) Positions tracked inside blocked partitions (0.003% of bins) are snapped to the nearest open partition.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
assert np.all(np.isnan(trace_day[~registered]).all(axis=1))
tr = trace_day[registered].astype(np.float32)

# In bin_time_series:
n = (x.shape[-1] // kernel) * kernel  # drops trailing frames

# In position_to_partition:
bad = blocked_vec[p] > 0
if np.any(bad):
    ...  # snap to nearest open partition
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Step 10, Check 5, noting that position at the arena edge (x or y == 75.0) is handled by `np.clip`.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified data loading (joblib.load) as the most time-consuming step (~89s for all 7 animals), followed by session processing (~93s total, 0.45s/session), and pickle writing (~5s). Total runtime was 187s.

ii. N/A (timing data from CONVERSION_NOTES.md Step 9).

iii. The AI profiled the conversion and documented timing in Steps 7 and 9.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main computational loops. The `bin_time_series` function uses reshape+mean instead of a loop. Gaussian smoothing operates on the full `(n_cells, n_frames)` matrix at once via scipy. The main remaining loop is over sessions within each animal, which cannot easily be vectorized due to variable session lengths.

ii.
```python
def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)
```

iii. The AI documented in CONVERSION_NOTES.md Step 6 that it replaced the reference code's per-frame loops with vectorized operations, achieving ~100x speedup on the neural processing step.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. The `env_blocked_vector()` function is called once per session and the result is reused for all trials. The Gaussian smoothing and average pooling are each applied once per session before trial splitting.

ii. N/A

iii. The AI's design processes each data stream once per session and then slices into trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores diagnostic information (`rates`, `pos_b`, `part`, `tr`) in the session result dictionary for plotting purposes, but these are deleted after use (`del res`). The blocked-partition snapping computation runs for all sessions even though only 0.003% of bins are affected.

ii.
```python
return {
    'neural': neural, 'output': output, 'input': inputs,
    'n_neurons': int(registered.sum()),
    'n_bins': n_bins,
    'n_snapped': n_snapped,
    'blocked_vec': blocked_vec,
    'rates': rates, 'pos_b': pos_b, 'part': part, 'tr': tr,
}
...
del res
```

iii. The diagnostic data is useful for `--show-processing` mode but is always computed even when not plotting. The overhead is minor since the data is deleted promptly.
