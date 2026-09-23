# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from joblib-serialized files (one per animal) in the `/app/data` directory, rather than the `.mat` files. Each joblib file is a dictionary keyed by animal ID, containing fields `trace`, `position`, `blocked`, `envs`, `maps`, `SFPs`, and `centroids`. The AI iterates over sorted animal files, loads each with `joblib.load`, then iterates over days within each animal.

ii.
```python
def animal_files():
    """Return primary joblib animal files, excluding larger .mat duplicates."""
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.is_file() and p.name.startswith('QLAK-CA1-') and p.suffix != '.mat')

# In convert():
files = animal_files(); subjects = [p.name for p in files]
for si, f in enumerate(files):
    raw = joblib.load(f)[f.name]
    for day in range(raw['trace'].shape[0]):
        ntr, itr, otr, info, payload = process_session(
            raw['position'][day], raw['trace'][day], raw['blocked'][day], ...)
```

iii. The AI identified that the joblib files are what the reference notebooks load (Figure 1 uses `joblib.load`), and they contain the same aligned data as the `.mat` files but are more compact. The AI documented this choice in CONVERSION_NOTES.md Steps 1-2, noting the joblib files are the "primary conversion source" because "these are what the reference notebooks load and they contain all required aligned streams."

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (mouse). The subject name is the filename of the joblib file. Seven subjects are identified: QLAK-CA1-08, -30, -50, -51, -56, -74, -75.

ii.
```python
files = animal_files(); subjects = [p.name for p in files]
for si, f in enumerate(files):
    raw = joblib.load(f)[f.name]
    ...
    subject_idx.append(si)
```

iii. Each animal file contains all recording sessions for one mouse. The AI uses the filename as the subject identifier and assigns an integer index per subject.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject's data constitutes one session. The AI iterates over the day axis of the `trace` array (first dimension) within each animal file.

ii.
```python
for day in range(raw['trace'].shape[0]):
    ntr, itr, otr, info, payload = process_session(
        raw['position'][day], raw['trace'][day], raw['blocked'][day],
        f.name, day, raw['envs'].ravel()[day])
```

iii. The data contains 207 total sessions (31 per animal for six animals, 21 for QLAK-CA1-51). Each day is one continuous ~40-minute recording session.

## 1-d. How are the data split into trials?

i. Per instructions, trials are defined as 60-second non-overlapping segments. The AI applies temporal pooling first (3-frame mean pooling, converting from 30 Hz to 10 Hz / 100 ms bins), then splits into trials of 600 time bins (= 60 seconds). Incomplete final segments are discarded.

ii.
```python
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)  # 600
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800

n_trials = position.shape[1] // RAW_TRIAL_FRAMES
used_raw = n_trials * RAW_TRIAL_FRAMES
...
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, ...)
...
for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], ...))
```

iii. The AI documented that trials are 60-second segments from continuous recordings, yielding 39 trials/session for subjects with ~71,866 frames and 40 trials/session for subjects with ~72,000+ frames (8,187 total trials).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Sessions with fewer than 2 complete 60-second trials are rejected (though none are in practice). Only incomplete remainder frames at the end of each session are discarded.

ii.
```python
if n_trials < 2:
    raise ValueError(f'{subject} day {source_day}: fewer than two complete trials')
```

iii. The AI noted that no rewarded/unrewarded or failed trials exist since mice freely explored. The only filtering is discarding incomplete final segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable in the joblib file, which contains binary rising-phase calcium event traces with shape `(n_days, n_registered_cells, n_frames)`.

ii.
```python
raw = joblib.load(f)[f.name]
...
trace = raw['trace'][day]  # shape: (n_registered_cells, n_frames)
```

iii. The AI identified that `trace` contains pre-processed binary calcium events (binarized at z > 2.5 from derivative of median-subtracted calcium), not raw fluorescence. The CONVERSION_NOTES state: "No reference function computes dF/F during analysis; therefore conversion should not recompute dF/F."

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps matching the reference paper's decoder code (`fit_decoder`/`test_decoder`):
1. **Cell selection**: Removes NaN neurons (absent cells) and applies activity-based filtering (see 2-c).
2. **Gaussian temporal smoothing**: `gaussian_filter1d(selected, sigma=3, axis=1)` on the full session before splitting.
3. **3-frame average pooling**: Non-overlapping mean over groups of 3 source frames, producing 100 ms bins.

ii.
```python
selected = np.asarray(trace[keep], dtype=np.float32)
smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, dtype=np.float32)
```

iii. The AI justified this by matching the reference `fit_decoder` code (utils.py:1776) which applies Gaussian trace smoothing with sigma=3 and non-overlapping 3-frame average pooling. The smoothing is applied over the continuous session before trial splitting to avoid minute-boundary edge artifacts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two levels of filtering are applied:
1. **Presence filter**: Neurons with all-NaN traces (not recorded on that day) are excluded.
2. **Activity filter** (from reference decoder code): Compute displacement speed at 30 Hz, Gaussian smooth with sigma=5, identify "moving" frames (speed > 5 cm/s), then keep only neurons whose summed trace activity during moving frames exceeds 5.

ii.
```python
def select_cells(position, trace):
    present = np.isfinite(trace).any(axis=1)
    displacement_speed = np.linalg.norm(np.diff(position.T, axis=0) * FPS, axis=1)
    moving = np.zeros(position.shape[1], dtype=bool)
    moving[1:] = gaussian_filter1d(displacement_speed, sigma=5, axis=0) > 5.0
    event_sum = np.nansum(trace[:, moving], axis=1)
    keep = present & (event_sum > 5)
    return keep, moving, event_sum
```

iii. The AI justified the activity filter by matching `decode_position_within` (utils.py:1845) in the reference code. This removed 882 of 69,744 day-cell recordings (1.26%), leaving 68,862 retained neurons across all sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous free exploration with no stimulus events. Trials are non-overlapping 60-second segments starting from time zero of each session.

ii. N/A (alignment is implicit from session start)

iii. The AI noted there is no stimulus event to align to. The `temporal_alignment_event` metadata is set to "Start of each non-overlapping 1-minute segment within a continuous animal-day recording."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 3-frame non-overlapping average pooling converts the native 30 Hz data (~33.33 ms bins) to 10 Hz (100 ms bins). Each trial has 600 time bins instead of 1800.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)  # 600

smooth = gaussian_filter1d(selected, sigma=POOL, axis=1)
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, ...)
```

iii. The AI justified this by matching the reference `fit_decoder`/`test_decoder` temporal processing which uses 3-frame bins. CONVERSION_NOTES Step 4 states: "The reference code temporally smooths/pools three source frames (100 ms) for decoding. This is the most directly applicable common neural/behavior time bin."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the data file, which contains indices of blocked reward locations for each recording session.

ii.
```python
raw['blocked'][day]  # per-day blocked indices
```

iii. The `blocked` variable stores which of the 9 possible arena partitions (in row-major 3x3 order) are blocked during each session. A value of `-1` indicates no positions are blocked (open square geometry).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary vector (one-hot encoding over the 3x3 grid). If no positions are blocked (index is -1), the vector is all zeros. The vector is static per session and repeated for all trials.

ii.
```python
def blocked_mask(raw):
    idx = np.asarray(raw).ravel().astype(np.int64)
    mask = np.zeros(9, dtype=np.float32)
    idx = idx[idx >= 0]  # -1 denotes no blocked partition
    if idx.size:
        if np.any(idx > 8):
            raise ValueError(f'Invalid blocked indices: {idx}')
        mask[idx] = 1.0
    return mask

# Per trial:
input_trials.append(geometry.copy())
```

iii. One-hot encoding allows the decoder to treat each blocked position independently. The encoding is static per session (constant across trials) as blocked positions don't change within a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the data file, which contains 2D coordinates (x, y) of the animal in the arena at each frame.

ii.
```python
raw['position'][day]  # shape: (2, n_frames) - x,y coordinates in cm
```

iii. The `position` variable records the animal's head location (tracked with DeepLabCut) in a 75x75 cm open-field arena at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first pools the continuous position coordinates using the same 3-frame average pooling applied to neural data, then discretizes into a 3x3 grid (9 classes) using fixed 25 cm physical boundaries.

ii.
```python
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
```

iii. The AI justified pooling position before discretization to match the reference `fit_decoder` which pools behavior in the same 3-frame bins. Fixed 25 cm boundaries come from the experimental 3x3 partition of the 75 cm arena. The `floor` division with clipping handles the exact 75 cm boundary.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using `floor(coord / 25.0)` clipped to [0, 2] on each axis, producing bins at [0,25), [25,50), [50,75] cm. The class label is computed as `y_bin * 3 + x_bin` (row-major), giving 9 classes (0-8) labeled NW, N, NE, W, center, E, SW, S, SE.

ii.
```python
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
```

iii. The fixed 25 cm boundaries match the paper's experimental 3x3 partition. Clipping ensures coordinates exactly at 75 cm are assigned to the last bin.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are from the same source frames at 30 Hz. Both are pooled over identical 3-frame windows, then split into trials using the same indices (600 bins per trial).

ii.
```python
# Both use the same used_raw frames and POOL size:
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, ...)
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
# Split with same trial indices:
for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], ...))
    output_trials.append(np.ascontiguousarray(labels[sl][None, :], ...))
```

iii. Exact frame alignment is guaranteed because both arrays share the same source frames, same pooling windows, and same trial slicing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of missing/edge data are handled:
1. **NaN neurons**: Neurons not recorded in a session (all-NaN traces) are removed by the presence filter in `select_cells`.
2. **Incomplete trials**: Remainder frames that don't fill a complete 60-second segment are discarded.
3. **Boundary positions**: Coordinates exactly at 75 cm are clipped into the last bin rather than creating an out-of-range bin.

ii.
```python
present = np.isfinite(trace).any(axis=1)
...
n_trials = position.shape[1] // RAW_TRIAL_FRAMES  # remainder implicitly dropped
...
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
```

iii. The AI verified that position is always finite (no NaN positions) and that NaN filtering of traces correctly identifies absent neurons. The CONVERSION_NOTES document exhaustive assertions including checks that all remaining data is finite and nonnegative.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and decompressing the joblib animal files (8-16 seconds each, ~88 seconds total for 7 files). Per-session processing is fast (~0.3-1.0 seconds per session). Total conversion time was 219 seconds.

ii.
```python
load_t = time.time(); raw = joblib.load(f)[f.name]
print(f'Loaded {f.name} in {time.time()-load_t:.2f}s', flush=True)
```

iii. The conversion output confirms load times: QLAK-CA1-08 in 8.83s, QLAK-CA1-30 in 14.63s, etc. Processing per session is dominated by Gaussian filtering and array operations (~0.3-1.0s).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop (iterating over trials to create sliced arrays) could potentially be vectorized using reshape, but the list-of-arrays output format inherently requires individual array creation.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
    input_trials.append(geometry.copy())
    output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The AI notes in CONVERSION_NOTES that vectorized Gaussian filtering and reshape-based pooling replace per-frame loops. The trial splitting loop is the remaining non-vectorized component, but is inherent to the output format.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session's data is processed once. The blocked mask is computed once per session and copied for each trial.

ii.
```python
geometry = blocked_mask(blocked)
input_trials.append(geometry.copy())  # copy per trial, not recomputed
```

iii. The AI designed the code to smooth/pool the full continuous session once before splitting into trials, avoiding redundant computation at trial boundaries.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The Gaussian smoothing and 3-frame average pooling are additional processing steps not present in the reference human solution. Whether they are "unnecessary" depends on perspective: the AI considered them necessary to match the paper's decoder preprocessing, but the downstream `train_decoder.py` may perform its own processing, making these redundant.

ii.
```python
smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, ...)
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
```

iii. The AI justified these steps by matching the reference `fit_decoder` code. However, since the downstream decoder trains its own model, pre-smoothing and temporal pooling may not be needed and reduce the temporal resolution available to the downstream decoder from 1800 to 600 timepoints per trial.
