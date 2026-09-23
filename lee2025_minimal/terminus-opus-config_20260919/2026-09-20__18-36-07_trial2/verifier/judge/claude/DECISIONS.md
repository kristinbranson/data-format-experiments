# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject corresponds to a joblib file in the data directory (`/app/data/QLAK-CA1-*`). The joblib files contain nested dictionaries keyed by animal ID, with fields `trace`, `position`, `envs`, `blocked`, etc. Data is loaded using `joblib.load()`. The 7 animal IDs are hardcoded in the `ANIMALS` list.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
traces, positions = dat['trace'], dat['position']
envs = [str(e[0]) for e in dat['envs']]
blocked = dat['blocked']
```

iii. The agent inspected the data directory (step 2), found both `.mat` and joblib files, then loaded a sample joblib file (step 15) to understand the structure. The agent chose joblib over `.mat` because the paper's own code (`load_dat` in `utils.py`) uses joblib as the default format. The agent verified data dimensions match the paper (207 sessions, 5413 cells, 69744 rate maps).

## 1-b. How are the data split into subjects?

i. Each joblib file in the data directory corresponds to one subject (mouse). The subject name is the animal ID string (e.g., `'QLAK-CA1-08'`). The 7 subjects are listed in the hardcoded `ANIMALS` list.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for ai, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
```

iii. The agent identified the 7 animal files by listing the data directory (step 2) and confirmed the structure by loading one file (step 15). Each file's top-level key is the animal ID.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days. The `trace` array has shape `(n_days, n_cells, T)` and `position` has shape `(n_days, 2, T)`. Each day becomes a separate session.

ii.
```python
n_days = traces.shape[0]
for day in range(n_days):
    ...
    tr = np.asarray(traces[day])       # (n_cells, T)
    pos = np.asarray(positions[day], dtype=np.float64)  # (2, T)
```

iii. The agent inspected the data structure (steps 15-16) and found that position has shape `(n_days, 2, T)` and trace has shape `(n_days, n_cells, T)`. Each day is a separate recording session (40 min at 30 Hz). The agent confirmed 207 total sessions across all 7 animals (step 21).

## 1-d. How are the data split into trials?

i. Each 40-minute recording session is split into consecutive 1-minute segments (600 time bins at 100 ms per bin after temporal pooling). Within each segment, only time bins where the animal was running (speed > 5 cm/s after binning) are kept. Trials with fewer than 30 running bins are discarded. The last partial segment (if shorter than 600 bins) is also included if it has enough running bins.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN))  # 600 bins = 1 min
MIN_BINS_PER_TRIAL = 30
...
for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    if keep.sum() < MIN_BINS_PER_TRIAL:
        continue
    sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
    sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
    sess_input.append(inp.copy())
```

iii. The instructions specify 1-minute trials. The agent applied the running-speed filter from the paper's `decode_position_within` function, keeping only running time bins within each trial. The minimum of 30 bins (~3 seconds of running) ensures trials have sufficient data.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 30 running time bins are discarded. Sessions with fewer than 2 usable trials are dropped entirely (as required by the decoder format for cross-validation).

ii.
```python
if keep.sum() < MIN_BINS_PER_TRIAL:
    continue
...
if len(sess_neural) < 2:
    print(f'  day {day} ({env}): < 2 usable trials, skipped')
    continue
```

iii. The agent imposed a minimum trial size to ensure sufficient data per trial for decoding, and a minimum of 2 trials per session to enable train/test splitting in the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib file, which contains binarized calcium-transient rising phases (binary events: 0 or 1) with shape `(n_days, n_cells, T)`.

ii.
```python
traces = dat['trace']
...
tr = np.asarray(traces[day])  # (n_cells, T)
```

iii. The agent examined the data (step 16) and confirmed traces contain binary values (0 and 1) matching the paper's description of binarized rising-phase events. Unregistered cells appear as NaN for the entire day.

## 2-b. How is the `neural` data processed?

i. Processing follows the paper's `fit_decoder` and `decode_position_within` functions:
1. Only cells registered on that day are kept (NaN rows removed).
2. Cells with <= 5 events during running frames are excluded.
3. Traces are Gaussian-smoothed with sigma=3 frames.
4. Smoothed traces are average-pooled over 3 frames into 100 ms time bins.
5. Only time bins where the animal was running are kept.

ii.
```python
# keep only registered cells
registered = ~np.isnan(tr[:, 0])
tr = tr[registered].astype(np.float32)

# exclude poorly active cells
active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]

# temporal smoothing + 100 ms binning
tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
```

iii. The agent read the paper's `fit_decoder` function (step 8, lines 1776-1806) which shows Gaussian smoothing (sigma=temporal_bin_size=3) followed by average pooling (AvgPool1d with kernel_size=3). The agent also read `decode_position_within` (step 9, lines 1845-1935) which shows the velocity filter and cell activity threshold. The agent replicated these steps exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied per session:
1. Cells not registered (tracked) on a given day (all-NaN rows) are removed.
2. Cells with <= 5 total events during running frames are excluded.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered].astype(np.float32)
...
active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]
```

iii. The agent identified both filters from the paper's `decode_position_within` function (step 9): `cell_threshold=5` parameter and the line `cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold`. The NaN check follows from the data structure where unregistered cells are NaN.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The recording is continuous, and trials are consecutive 1-minute segments starting from the beginning of the recording session. The temporal alignment event is described as "start of the recording session."

ii.
```python
'temporal_alignment_event': (
    'start of the recording session; each session is cut into consecutive '
    '1-minute trials'),
'off_start': 0.0,
'off_end': 60.0,
```

iii. There is no stimulus onset or behavioral event to align to in this free-foraging paradigm. The agent correctly identified that trials are artificial time segments of the continuous recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned from the native 30 Hz (33.3 ms) to 100 ms bins by averaging 3 consecutive frames, following the paper's `fit_decoder` function (`temporal_bin_size=3`). Additionally, traces are Gaussian-smoothed (sigma=3 frames) before pooling.

ii.
```python
TEMPORAL_BIN = 3      # frames per time bin -> 100 ms
...
tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
...
'time_bin_size': 100.0,
```

iii. The agent read the `fit_decoder` function (step 8) which uses `AvgPool1d(kernel_size=temporal_bin_size, stride=temporal_bin_size)` with `temporal_bin_size=3` and `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)`. The agent replicated this as `pool_mean` and `gaussian_filter1d`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field in the dataset and cross-checked against the `envs` field and the paper's `get_env_mat` function.

ii.
```python
envs = [str(e[0]) for e in dat['envs']]
blocked = dat['blocked']
...
inp = blocked_vector(env, blocked[day])
```

iii. The agent inspected the data structure (step 15) and found the `blocked` field contains indices of blocked positions. The agent verified the mapping between `blocked` indices and the spatial partition layout using occupancy analysis (step 19) and cross-checked against the `get_env_mat` function from `utils.py`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked partition indices are converted to a 9-dimensional binary vector (1 = blocked, 0 = open). A value of `[-1]` in the blocked field indicates no partitions are blocked (all zeros). The vector is cross-checked against the flipped `get_env_mat` matrix. The blocked vector is static per session (same for all trials within a session).

ii.
```python
def blocked_vector(env_name, blocked_field):
    b = np.atleast_1d(np.asarray(blocked_field, dtype=float).ravel())
    vec = np.zeros(9, dtype=np.float32)
    if not (b.size == 1 and b[0] == -1):
        vec[b.astype(int)] = 1.0
    from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
    assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
    return vec
...
sess_input.append(inp.copy())  # same blocked vector for all trials
```

iii. The agent discovered a convention mismatch between `get_env_mat` (rows run north-to-south) and the dataset's `blocked` field (partition index = 3*y_bin + x_bin). This was identified when the initial assertion failed (step 31) and fixed by using `flipud` to reconcile the two conventions (step 32).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The mouse position output is derived from the `position` field in the dataset, which contains 2D (x, y) coordinates in centimeters within the 75x75 cm arena, at the native 30 Hz frame rate.

ii.
```python
positions = dat['position']
...
pos = np.asarray(positions[day], dtype=np.float64)  # (2, T)
```

iii. The agent inspected the position data (step 16) and confirmed it ranges from 0-75 cm in both x and y dimensions with no NaN values.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first average-pooled over 3 frames (same as neural data) to obtain 100 ms bins, then discretized into the 3x3 partition grid (9 spatial bins). The partition index is computed as `3 * y_bin + x_bin` where bins are obtained by `floor(coordinate / 25)`, clipped to [0, 2].

ii.
```python
pos_b = pool_mean(pos, TEMPORAL_BIN)  # (2, nbins)
out_bins = position_to_bin(pos_b)[np.newaxis, :]  # (1, nbins)
...
def position_to_bin(pos_xy):
    edge = ARENA_SIZE / N_SPACE_BINS  # 25 cm
    xb = np.clip(np.floor(pos_xy[0] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
    yb = np.clip(np.floor(pos_xy[1] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
    return 3 * yb + xb
```

iii. The agent verified the partition convention empirically by computing occupancy in each 3x3 bin for multiple environment geometries (step 19) and confirming zero occupancy in blocked partitions when using `3*y_bin + x_bin`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (0-8) by dividing each axis into 3 equal 25-cm bins. The partition index = `3 * y_bin + x_bin` gives a unique category for each spatial bin.

ii.
```python
edge = ARENA_SIZE / N_SPACE_BINS  # 75 / 3 = 25 cm
xb = np.clip(np.floor(pos_xy[0] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
yb = np.clip(np.floor(pos_xy[1] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
return 3 * yb + xb
```

iii. The 3x3 grid follows directly from the task instructions ("Mouse position discretized into 3 x 3 = 9 spatial bins") and the paper's environment design. The agent verified the convention against the blocked partition data.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are recorded at the same 30 Hz frame rate. Both are average-pooled over the same 3-frame bins, then the same running-speed mask is applied, and both are split using the same trial indices. This ensures frame-for-frame alignment.

ii.
```python
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)  # (n_cells, nbins)
pos_b = pool_mean(pos, TEMPORAL_BIN)                        # (2, nbins)
...
sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
```

iii. The agent applied the same temporal pooling and masking operations to both neural and position data, preserving their alignment throughout the pipeline.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled:
1. Cells not registered on a given day (NaN rows) are removed.
2. Cells with insufficient activity during running (<= 5 events) are excluded.
3. Trials with too few running time bins (< 30) are discarded.
4. Sessions with fewer than 2 usable trials are dropped.
5. An assertion verifies the animal is essentially never found in blocked partitions (< 2% occupancy).

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered].astype(np.float32)
...
active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]
if tr.shape[0] == 0:
    continue
...
if keep.sum() < MIN_BINS_PER_TRIAL:
    continue
...
if len(sess_neural) < 2:
    continue
...
assert occ[inp.astype(bool)].sum() < 0.02
```

iii. The agent combined the paper's quality controls (NaN filtering, cell activity threshold, speed threshold) with decoder-specific requirements (minimum trials per session) and data consistency checks (occupancy in blocked partitions).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading the large joblib files (I/O bound, files range from 70 MB to 150 MB) and the Gaussian smoothing + average pooling of traces (~72,000 frames x hundreds of cells per session).

ii. N/A

iii. The agent observed that loading took significant time during execution (steps 34-35, conversion of all 7 animals took several minutes).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over 1-minute segments and applies boolean indexing. This could potentially be vectorized using array operations, though the variable-length output (different numbers of running bins per trial) makes full vectorization difficult.

ii.
```python
for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    ...
```

iii. The loop is relatively lightweight (only ~40 iterations per session) and the main computation (Gaussian filtering, pooling) is already vectorized using NumPy/SciPy operations.

## 6-c. What processing does the code repeat multiple times?

i. The speed computation and Gaussian smoothing of speed is done once per session at the raw frame level, then the speed threshold is re-applied after temporal binning. This is a minor redundancy but ensures correct filtering at both levels.

ii.
```python
speed = gaussian_filter1d(speed, sigma=V_FILT_SIGMA)
running_frames = speed > V_THRESH
...
speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]
run_b = speed_b > V_THRESH
```

iii. The speed filter is applied at two levels: first at the raw frame level (to determine cell activity during running) and then at the binned level (to select running time bins for trials). Both are necessary for different purposes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `get_env_mat` function and the assertion cross-check against flipped env matrices is computed for every session but is only used for validation (not in the final output). The occupancy sanity check is similarly a debug/validation step that doesn't contribute to the output.

ii.
```python
from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
...
occ = np.bincount(out_bins[0], minlength=9) / nbins
assert occ[inp.astype(bool)].sum() < 0.02
```

iii. These checks ensure data integrity but add minimal computational cost. They are good practice for data conversion scripts.
