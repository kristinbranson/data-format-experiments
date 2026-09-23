# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from MATLAB v7.3 `.mat` files (HDF5 format) using `h5py`. It opens each animal's file with `h5py.File()`, reads session-level arrays (`trace`, `position`, `blocked`, `envs`) by following HDF5 object references. It processes each session lazily (one at a time) to keep memory low. A hardcoded list of 7 animal IDs (`ANIMALS`) is used rather than globbing the data directory.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def open_animal(animal):
    return h5py.File(os.path.join(DATA_DIR, f'{animal}.mat'), 'r')

def read_meta(f):
    envs = [''.join(chr(c) for c in f[r][:].ravel()) for r in f['envs'][0]]
    blocked = [np.atleast_1d(f[r][:].ravel()).astype(int) for r in f['blocked'][0]]
    return envs, blocked

def read_session(f, day):
    trace = f[f['trace'][day, 0]][:]
    position = f[f['position'][day, 0]][:]
    return trace, position
```

iii. The AI chose `h5py` over `joblib` for lazy per-session reads, reducing peak memory from >10 GB to ~1 GB. It also reads the `envs` field (environment geometry name per session), which the reference code does not, to independently derive the blocked-partition vector and verify it against the stored `blocked` field. The AI documented verifying that the h5py-loaded data was byte-identical to the joblib contents.

## 1-b. How are the data split into subjects?

i. Each `.mat` file corresponds to one subject. The AI uses a hardcoded list of 7 animal IDs rather than globbing. The subject index into `data['subjects']` is tracked via `ANIMALS.index(animal)`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for ai, animal in enumerate(animals):
    subj_idx = ANIMALS.index(animal)
```

iii. The AI hardcoded the animal list to match the paper's 7 subjects. This ensures all 7 subjects are always in `data['subjects']` even in sample mode, which uses only 2.

## 1-c. How are the data split into sessions?

i. Each `.mat` file contains multiple recording sessions (days). The AI reads `envs` to get the number of sessions and iterates over days within each file. Each recording day becomes a separate session in the output.

ii.
```python
envs, blocked = read_meta(f)
n_days = len(envs)
for day in days:
    trace, position = read_session(f, day)
    ...
```

iii. Each day is a distinct recording session with its own geometry, neurons, and behavioral data. The file structure stores sessions as indexed arrays.

## 1-d. How are the data split into trials?

i. The AI first temporally bins the data from 30 Hz to 100 ms bins (3-frame average pooling), giving 600 bins per 60-second trial. It then cuts the binned session into consecutive 60-second blocks. The final partial block is kept if it contains at least 50% of a full trial (300 bins). Within each trial, only time bins where the animal's smoothed speed exceeds 5 cm/s are retained, making trial lengths variable. Trials with fewer than 30 bins of movement data (~3 seconds) are dropped entirely.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN_FRAMES))   # 600
MIN_TRIAL_FRAC = 0.5
MIN_BINS_PER_TRIAL = 30

start = 0
while start < n_bins:
    stop = min(start + BINS_PER_TRIAL, n_bins)
    if (stop - start) < MIN_TRIAL_FRAC * BINS_PER_TRIAL:
        break
    sel = np.where(moving_binned[start:stop])[0] + start
    if sel.size >= MIN_BINS_PER_TRIAL:
        neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
        input_trials.append(input_vec.copy())
        output_trials.append(out_class[sel][None, :].copy())
    start = stop
```

iii. The AI justified this by following `decode_position_within` from the reference code, which filters out non-locomotion periods. The 50% threshold for partial trials keeps ~55-second remainders (avoiding 2.5% data loss). The 30-bin minimum ensures enough data for meaningful decoding.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies speed-based filtering: only time bins where the animal's smoothed speed exceeds 5 cm/s are kept. Trials where fewer than 30 bins (3 seconds) of movement remain after filtering are dropped entirely. Sessions with fewer than 2 usable trials are skipped (though none fall below this threshold in practice).

ii.
```python
speed = compute_speed(position)
moving = speed > V_THRESH  # V_THRESH = 5.0 cm/s
...
moving_binned = speed_binned > V_THRESH
...
sel = np.where(moving_binned[start:stop])[0] + start
if sel.size >= MIN_BINS_PER_TRIAL:  # MIN_BINS_PER_TRIAL = 30
    ...
```

iii. The AI followed `decode_position_within` which uses `v_thresh=5` cm/s and `v_filt_size=5` frames of Gaussian smoothing. The rationale is that CA1 place coding is only well-defined during locomotion, and immobility periods contain replay/SWR activity unrelated to current position.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binarized rising-phase calcium transient events at 30 Hz.

ii.
```python
trace = f[f['trace'][day, 0]][:]  # (T, n_cells) float64
```

iii. The paper states: "This binary vector was treated as the firing rate in all further analyses." No dF/F computation is needed as the authors already performed the extraction.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps from the reference code's `fit_decoder` function: (1) Gaussian smoothing with sigma=3 frames using `gaussian_filter1d`, (2) average pooling into 100 ms bins (3-frame non-overlapping mean), and (3) scaling by 30 Hz to convert to events/second.

ii.
```python
TRACE_SMOOTH_SIGMA = 3
TEMPORAL_BIN_FRAMES = 3

tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SMOOTH_SIGMA, axis=0)
neural = pool_mean(tr_smooth).astype(np.float32) * FPS  # -> events/s

def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    n = (x.shape[0] // k) * k
    x = x[:n]
    return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)
```

iii. The AI documented that this exactly matches the reference `fit_decoder`/`test_decoder` preprocessing: `gaussian_filter1d(sigma=3)` then `AvgPool1d(kernel_size=3, stride=3)`. The FPS scaling converts to events/s and keeps values O(0.1-3). The `pool_mean` function is a numpy reimplementation of PyTorch's `AvgPool1d`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two filters from `decode_position_within`: (1) cells must be registered on that day (non-NaN at frame 0), and (2) cells must have more than 5 calcium events during locomotion periods (when smoothed speed > 5 cm/s). No place-cell selection is applied.

ii.
```python
CELL_THRESHOLD = 5

registered = ~np.isnan(trace[0, :])
events_moving = np.nansum(trace[moving, :], axis=0)
keep_cells = registered & (events_moving > CELL_THRESHOLD)
tr = trace[:, keep_cells]
```

iii. The AI followed the reference code's `decode_position_within` which uses `cell_threshold=5`. The paper explicitly states results "motivated the inclusion of all cells in subsequent analyses" (meaning no place-cell selection, but the event threshold is still applied). This results in 68,862 of 69,744 registered cell-sessions being kept (98.7%).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous free exploration with no discrete task events. Neural and behavioral data are acquired by the same DAQ at 30 Hz and are frame-aligned. Both streams are subjected to the same temporal binning (3-frame average pooling).

ii.
```python
neural = pool_mean(tr_smooth).astype(np.float32) * FPS
pos_binned = pool_mean(position)
speed_binned = pool_mean(speed[:, None])[:, 0]
assert pos_binned.shape[0] == n_bins == speed_binned.shape[0]
```

iii. The AI verified alignment by reproducing the dataset's own rate maps from `trace` + `position` (exact match), confirming zero lag between the streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins from 30 Hz (33.33 ms) to 100 ms bins using 3-frame non-overlapping average pooling, matching `fit_decoder` in the reference code.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100 ms
...
neural = pool_mean(tr_smooth).astype(np.float32) * FPS
```

iii. The `time_bin_size` in metadata is set to 100.0 ms. The AI documented this matches `AvgPool1d(kernel_size=3, stride=3)` in `fit_decoder`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the blocked-partition vector from the `envs` field (geometry name per session) using a hardcoded dictionary of environment matrices (`ENV_MATS`), following `get_env_mat` from the reference code. It also reads the `blocked` field and verifies consistency between the two sources for all 207 sessions.

ii.
```python
ENV_MATS = {
    'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'o':         [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    ...
}

def env_blocked_vector(env_name):
    m = np.array(ENV_MATS[env_name], dtype=float)
    return (np.flipud(m).ravel() == 0).astype(np.float32)

# Verification in main loop:
expected = np.where(env_blocked_vector(envs[day]))[0]
got = np.sort(blocked[day][blocked[day] >= 0])
assert np.array_equal(expected, got)
```

iii. The AI derived blocked partitions from the environment name rather than directly from the `blocked` field, then verified both sources agree. This provides additional robustness against potential data inconsistencies.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is looked up in `ENV_MATS` to get a 3x3 binary matrix (1=open, 0=blocked). The matrix is flipped vertically (`np.flipud`) and flattened, then inverted (0->1 for blocked). The result is a 9-dimensional binary vector where 1 indicates a blocked partition. This is static per trial.

ii.
```python
def env_blocked_vector(env_name):
    m = np.array(ENV_MATS[env_name], dtype=float)
    return (np.flipud(m).ravel() == 0).astype(np.float32)
...
input_vec = env_blocked_vector(env_name)
input_trials.append(input_vec.copy())
```

iii. The `flipud` operation maps the visual matrix representation to the partition index convention (3*ybin + xbin), matching `clean_rate_maps` in the reference code. The AI verified this against the dataset's `blocked` field for all 207 sessions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D (x, y) coordinates in cm in the range [0, 75].

ii.
```python
position = f[f['position'][day, 0]][:]  # (T, 2) float64, cm in [0, 75]
```

iii. Position data comes from DeepLabCut head tracking, as described in the methods.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is first averaged into 100 ms bins (same pooling as neural data), then discretized into a 3x3 grid. Bin assignment is `floor(position / (75/3))` clipped to [0, 2], with class label = `3 * ybin + xbin`. Samples that land in blocked partitions (0.005% due to tracking noise) are snapped to the nearest open partition.

ii.
```python
def position_to_class(position, blocked_vec=None):
    part = ARENA_SIZE / N_PART  # 25 cm
    bins = np.clip((position / part).astype(int), 0, N_PART - 1)
    cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
    if blocked_vec is not None and blocked_vec.any():
        blocked_ids = np.where(blocked_vec > 0)[0]
        bad = np.isin(cls, blocked_ids)
        if bad.any():
            open_ids = np.where(blocked_vec == 0)[0]
            centres = np.stack([(open_ids % N_PART) * part + part / 2,
                                (open_ids // N_PART) * part + part / 2], axis=1)
            d = np.linalg.norm(position[bad][:, None, :] - centres[None], axis=2)
            cls[bad] = open_ids[np.argmin(d, axis=1)]
    return cls
```

iii. The blocked-partition snapping is the 3x3 analogue of the reference code's `decode_position_within`, which snaps both true and predicted positions to the nearest bin belonging to the environment.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by dividing by the partition size (25 cm), taking the floor, and clipping to [0, 2]. This gives 3 bins per axis, combined into 9 categories via `3 * ybin + xbin`.

ii.
```python
part = ARENA_SIZE / N_PART  # 25 cm
bins = np.clip((position / part).astype(int), 0, N_PART - 1)
cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
```

iii. The partition convention (3*ybin + xbin) matches the dataset's `blocked` field indexing, verified for all 207 sessions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is subjected to the same 3-frame average pooling (`pool_mean`) as the neural data, ensuring frame-for-frame alignment in the 100 ms binned space. Both are then split into trials and speed-filtered using the same indices.

ii.
```python
pos_binned = pool_mean(position)
speed_binned = pool_mean(speed[:, None])[:, 0]
n_bins = neural.shape[0]
assert pos_binned.shape[0] == n_bins == speed_binned.shape[0]
...
out_class = position_to_class(pos_binned, blocked_vec=input_vec)
...
sel = np.where(moving_binned[start:stop])[0] + start
neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
output_trials.append(out_class[sel][None, :].copy())
```

iii. The `assert` statement explicitly checks that neural, position, and speed have identical time dimensions after pooling.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data quality issues: (1) Unregistered neurons (all-NaN) are excluded by the `registered` check; (2) low-activity neurons (<=5 events during locomotion) are excluded by the cell threshold; (3) the ~0.005% of position samples that fall inside blocked partitions (DeepLabCut tracking noise) are snapped to the nearest open partition; (4) trailing frames that don't fill a complete temporal bin are dropped by `pool_mean`; (5) partial trials at the end of a session are kept if >=50% of a full trial, otherwise dropped.

ii.
```python
registered = ~np.isnan(trace[0, :])
events_moving = np.nansum(trace[moving, :], axis=0)
keep_cells = registered & (events_moving > CELL_THRESHOLD)
...
# In pool_mean: n = (x.shape[0] // k) * k  -> drops trailing frames
# In position_to_class: snaps blocked-partition samples
# In trial loop: MIN_TRIAL_FRAC and MIN_BINS_PER_TRIAL thresholds
```

iii. The AI documented finding 127 out of 2,573,752 samples (0.005%) in blocked partitions across 4 sessions. This was identified in Step 10 (Critical Review) edge-case checks and fixed by adding the snapping logic.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified that loading `.mat` files via `h5py` is the dominant cost (~0.43s per session). Processing (smoothing, binning, filtering) takes ~0.46s per session. Total runtime for 207 sessions is ~4 minutes.

ii. N/A (timing output, not code logic)

iii. The AI explicitly printed per-session timing and estimated total runtime. They replaced the initial joblib loading approach (6-40s per animal, >10 GB RAM) with lazy HDF5 reads.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is largely vectorized. The main loop is over sessions (unavoidable since each session is loaded from disk independently). Within a session, the trial-cutting loop iterates over ~40 trial boundaries, which is trivial. All neural processing (smoothing, pooling, speed computation) is vectorized with numpy/scipy.

ii.
```python
# Vectorized smoothing and pooling:
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SMOOTH_SIGMA, axis=0)
neural = pool_mean(tr_smooth).astype(np.float32) * FPS
```

iii. The AI noted that the reference code's spatial binning uses Python loops over frames, but this was unnecessary for the 3x3 discretization, which is fully vectorized.

## 6-c. What processing does the code repeat multiple times?

i. No significant redundant computation was identified. Each session is processed once. The speed computation and temporal binning are each done once per session.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores detailed `session_info` metadata (environment name, sequence number, raw frame count, etc.) that goes beyond what the decoder uses. The blocked-partition snapping for the 0.005% of samples is arguably unnecessary for decoder accuracy (confirmed negligible impact). The AI also derives the blocked vector independently from `envs` when it could directly use the `blocked` field.

ii.
```python
session_info.append({
    'session_id': sid, 'animal': animal, 'day': day,
    'environment': envs[day], 'sequence': day // 10,
    'n_neurons': int(keep.sum()),
    'n_cells_registered': int((~np.isnan(trace[0, :])).sum()),
    'n_cells_file': int(trace.shape[1]),
    'n_trials': len(ntr),
    'n_frames_raw': int(trace.shape[0]),
    'n_bins_kept': int(sum(t.shape[1] for t in ntr)),
})
```

iii. The extra metadata is useful for documentation and debugging but not consumed by the decoder. The consistency assertion between `envs`-derived and stored `blocked` fields is a sanity check, not a processing step.
