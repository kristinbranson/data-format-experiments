# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `joblib.load()` from extensionless compressed files in `/app/data/`. Each file is keyed by the animal ID and contains a dictionary payload with fields `trace`, `position`, `blocked`, `envs`, etc. The AI hardcodes the list of 7 animal IDs (`ANIMALS`) and iterates over them, loading one file per animal.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
DATA_DIR = Path("/app/data")
...
payload = joblib.load(DATA_DIR / animal)[animal]
trace = np.asarray(payload["trace"])
position = np.asarray(payload["position"])
envs = np.asarray(payload["envs"]).squeeze()
```

iii. The AI chose joblib because the reference code's `load_dat` function also uses `joblib.load()`. The data exists in both `.mat` (HDF5) and joblib formats; the AI used the format matching the reference code pipeline.

## 1-b. How are the data split into subjects?

i. Each file in the data directory corresponds to one subject (mouse). The AI uses a hardcoded list of 7 animal IDs and iterates over them. Each animal gets a sequential subject index.

ii.
```python
for animal_idx, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(animal_idx)
```

iii. Each data file contains all recording sessions for one animal. The seven animal IDs are taken from the reference code's `main.py`.

## 1-c. How are the data split into sessions?

i. Each animal's data contains multiple recording days stored along axis 0 of the `trace` and `position` arrays. Each day becomes a separate session in the output. The AI iterates over all days for each animal.

ii.
```python
selected_days = range(trace.shape[0]) if targets[animal] is None else sorted(targets[animal])
for day in selected_days:
    ...
    neural.append(session_neural)
```

iii. The reference code treats each recording day as a separate analysis unit. Six animals have 31 days and one (QLAK-CA1-51) has 21 days, giving 207 total sessions.

## 1-d. How are the data split into trials?

i. Each continuous recording session (~40 minutes) is split into non-overlapping 60-second windows. After 3-frame pooling (100ms bins), each trial has 600 timepoints. Frames that don't fill a complete 60-second window are discarded. The splitting is done after Gaussian smoothing and mean-pooling.

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS  # 1800
TIMEPOINTS_PER_TRIAL = RAW_TRIAL_FRAMES // POOL_FRAMES  # 600
...
n_raw_frames = int(position.shape[2])
n_trials = n_raw_frames // RAW_TRIAL_FRAMES
n_used_frames = n_trials * RAW_TRIAL_FRAMES
...
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
    session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
```

iii. Per the task instructions, trials are defined as 1-minute segments. The AI first processes the entire session (smoothing + pooling), then slices into 600-bin trials. Remainder frames are discarded (1666, 219, 91, 60, or 71 frames depending on the animal).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied beyond discarding incomplete tails. The AI checks that each session has at least 2 complete trials and validates that all neural/position values are finite within retained frames.

ii.
```python
if n_trials < 2:
    raise ValueError(f"{animal} day {day} has fewer than two complete trials")
```

iii. The paper describes no trial rejection criteria. Sessions are continuous free exploration with no behavioral events that would define good/bad trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary rising-phase calcium event data with shape `(days, neurons, timepoints)`.

ii.
```python
trace = np.asarray(payload["trace"])
...
selected = np.asarray(raw_trace[present, :], dtype=np.float32)
```

iii. The `trace` variable stores pre-processed binary calcium events (0/1). The upstream processing (motion correction, cell segmentation, calcium derivative, Gaussian smoothing, z-scoring, thresholding) was already performed by the original authors.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps matching the reference code's position decoder: (1) Gaussian smoothing with sigma=3 native frames along the time axis, and (2) non-overlapping 3-frame mean pooling. Smoothing is applied to the entire continuous session before trial slicing to avoid filter boundary artifacts.

ii.
```python
def process_neural(raw_trace, present, n_used_frames):
    selected = np.asarray(raw_trace[present, :], dtype=np.float32)
    gaussian_filter1d(selected, sigma=POOL_FRAMES, axis=1, output=selected)
    selected = selected[:, :n_used_frames]
    return selected.reshape(selected.shape[0], -1, POOL_FRAMES).mean(axis=2)
```

iii. The AI followed the reference code's `fit_decoder`/`test_decoder` functions, which apply Gaussian smoothing (sigma=3 frames) and AvgPool1d with kernel_size=3. This converts the binary event series into a continuous smoothed firing rate at 100ms resolution, matching the paper's within-day position decoder methodology.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons not registered in a given session (all-NaN traces for that day) are excluded. Registered neurons (finite traces) are all retained regardless of activity level. No place-cell filtering or minimum-event threshold is applied.

ii.
```python
present = np.isfinite(trace[day, :, 0])
...
selected = np.asarray(raw_trace[present, :], dtype=np.float32)
if not np.isfinite(selected).all():
    raise ValueError("Registered neural traces contain intermittent NaN/Inf")
```

iii. The paper explicitly states that all cells (not just place cells) are included in subsequent analyses. The reference code's >5-event filter is specific to the speed-masked decoder and would conflict with continuous trial requirements. The AI retains all 69,744 session-neuron instances matching the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous free exploration with no stimulus events. Trials begin at the start of each contiguous 60-second window within the session. Neural and position data share the same frame indices and are sliced identically.

ii.
```python
# metadata:
"temporal_alignment_event": "start of each contiguous one-minute window within the recording session",
"off_start": 0.0,
"off_end": 60.0,
```

iii. There are no discrete stimulus events to align to. The 30 Hz calcium imaging and position tracking are simultaneously acquired and timestamp-aligned, requiring no lag correction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data from the native 30 Hz (~33.33 ms per frame) to 100 ms bins by mean-pooling every 3 consecutive frames. Each trial has 600 timepoints at 100 ms resolution.

ii.
```python
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
TIMEPOINTS_PER_TRIAL = RAW_TRIAL_FRAMES // POOL_FRAMES  # 600
...
return selected.reshape(selected.shape[0], -1, POOL_FRAMES).mean(axis=2)
```

iii. The AI matched the reference code's `temporal_bin_size=3` (3-frame pooling) used in the position decoder. This is documented in the metadata as `time_bin_size: 100.0`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field in the data payload, which contains indices of blocked reward positions for each recording day.

ii.
```python
geometry = blocked_vector(payload["blocked"][day][0])
```

iii. The `blocked` field stores which of the 9 possible partition positions were blocked (walls inserted) in each session's arena geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary vector (1=blocked, 0=accessible). Crucially, the AI transposes the 3x3 matrix from native (y,x) order to (x,y) order to align with the position class convention (`x*3+y`). If the only index is -1 (no blocks), all zeros are returned. The vector is static per session (same for all trials).

ii.
```python
def blocked_vector(raw_blocked):
    values = np.asarray(raw_blocked).reshape(-1)
    native_yx = np.zeros(9, dtype=np.float32)
    if values.size == 1 and float(values[0]) == -1.0:
        return native_yx
    indices = values.astype(np.int64)
    native_yx[indices] = 1.0
    return np.ascontiguousarray(native_yx.reshape(3, 3).T.ravel())
```

iii. The AI determined through empirical testing that native blocked indices flatten in (y,x) order while position coordinates use (x,y). Transposing the 3x3 matrix before flattening reduced position/block overlap from 17.14% to 0.003%, confirming correct alignment. Input names are `blocked_x{x}_y{y}`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` variable, which contains 2D (x,y) coordinates of the animal at each timepoint, stored as `(days, 2, frames)`.

ii.
```python
position = np.asarray(payload["position"])
...
binned_position = pool_position(position[day], n_used_frames)
labels = position_classes(binned_position)
```

iii. The position is recorded via DeepLabCut head tracking at 30 Hz in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first mean-pooled over the same 3-frame windows as neural data (to maintain temporal alignment at 100ms resolution), then discretized into a 3x3 grid.

ii.
```python
def pool_position(raw_position, n_used_frames):
    selected = np.asarray(raw_position[:, :n_used_frames], dtype=np.float32)
    return selected.reshape(2, -1, POOL_FRAMES).mean(axis=2)
```

iii. Position pooling matches the neural data pooling to ensure temporal alignment at 100ms resolution. The pooled position is then discretized.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 2D position is discretized into a 3x3 grid (9 classes). Each axis is divided into 3 equal 25cm bins. The class label is computed as `x_bin * 3 + y_bin` (x-first, row-major). Values at the exact arena boundary (75 cm) are clipped into bin 2.

ii.
```python
def position_classes(position_binned):
    upper = np.nextafter(np.float32(ARENA_CM), np.float32(0.0))
    xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
    labels = xy[0] * SPATIAL_BINS + xy[1]
    return labels
```

iii. The AI uses `floor(clip(pos, 0, ~74.999) / 25)` for binning, which creates bins [0,25), [25,50), [50,75]. The x-first convention (`x*3+y`) is consistent with the transposed blocked vector.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are mean-pooled over the same 3-frame windows, ensuring exact temporal alignment at 100ms resolution. Both are then sliced into trials using the same indices.

ii.
```python
binned_position = pool_position(position[day], n_used_frames)
labels = position_classes(binned_position)
binned_neural = process_neural(trace[day], present, n_used_frames)
...
# same slicing for both:
sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. Both neural and position are acquired at 30 Hz with aligned frames. The identical 3-frame pooling and trial slicing preserves this alignment at 100ms resolution.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons not registered in a session (all-NaN traces) are excluded via the `present` mask. Registered neurons are validated to contain no intermittent NaN/Inf. Position is checked for finiteness. Incomplete tail frames at the end of sessions are discarded. The blocked field uses -1 to indicate no blocked positions, which is handled explicitly. 45 zero-event registered neurons are retained intentionally.

ii.
```python
present = np.isfinite(trace[day, :, 0])
...
if not np.isfinite(selected).all():
    raise ValueError("Registered neural traces contain intermittent NaN/Inf")
...
if not np.isfinite(selected).all():
    raise ValueError("Position contains NaN/Inf in a retained complete trial")
```

iii. The AI documented that there are no intermittent NaN values in registered traces and no missing position data, so most error handling is defensive validation rather than active correction.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the data files via `joblib.load()`, which takes 6-16 seconds per animal (total ~88 seconds). Per-session processing (smoothing, pooling, discretization) is 0.12-0.59 seconds each. Saving the 6.2 GiB pickle takes ~5.6 seconds.

ii.
```python
load_start = time.perf_counter()
payload = joblib.load(DATA_DIR / animal)[animal]
print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f} s", flush=True)
```

iii. The conversion output shows I/O is the dominant cost. Total conversion time was ~161 seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-slicing loop (iterating over trials to create per-trial arrays) could potentially be replaced with a single reshape + split operation. However, the loop is over at most 40 trials and is not the bottleneck.

ii.
```python
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
    session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
```

iii. The AI already vectorized the heavy operations (smoothing uses scipy, pooling uses reshape+mean). The remaining loop is lightweight.

## 6-c. What processing does the code repeat multiple times?

i. The animal payload is loaded once per animal and reused across all days, avoiding redundant I/O. Within each session, smoothing and pooling are done once on the full session before trial slicing. No significant repeated computation was identified.

ii.
```python
payload = joblib.load(DATA_DIR / animal)[animal]
trace = np.asarray(payload["trace"])
position = np.asarray(payload["position"])
# Used for all days of this animal
```

iii. The AI designed the code to minimize redundant work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes diagnostic statistics (position class counts, blocked-position overlap counts, per-session metadata) that are not used by the decoder but are useful for validation. The Gaussian smoothing processes the full session including the tail frames that are later discarded, though this is intentional to avoid filter boundary effects.

ii.
```python
# Diagnostic statistics computed but not used by decoder:
occupied_geometry = geometry[labels]
blocked_position_count += int(occupied_geometry.sum())
total_position_count += int(labels.size)
class_counts += np.bincount(labels, minlength=9)
```

iii. These diagnostics are minimal overhead and were valuable during development for catching the geometry axis convention bug.
