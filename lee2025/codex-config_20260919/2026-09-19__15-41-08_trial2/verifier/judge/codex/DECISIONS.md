# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads one preprocessed joblib file per mouse from `/app/data`, using a hard-coded animal list rather than globbing `.mat` files. For each loaded payload it reads `trace`, `position`, `envs`, and `blocked`, then iterates through all day indices and later slices each day into one-minute trials.

ii.
```python
DATA_DIR = Path("/app/data")
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_idx, animal in enumerate(ANIMALS):
    ...
    payload = joblib.load(DATA_DIR / animal)[animal]
    ...
    trace = np.asarray(payload["trace"])
    position = np.asarray(payload["position"])
    envs = np.asarray(payload["envs"]).squeeze()
```

iii. In `CONVERSION_NOTES.md`, the agent says the extensionless joblib files are the primary data objects and that each is `{animal_id: payload}` with the needed fields. In its notes it also says this matches the reference code path that uses joblib-backed animal payloads.

## 1-b. How are the data split into subjects (mice)?

i. Each hard-coded animal/file is treated as one subject. The final `subjects` field is simply the fixed `ANIMALS` list, and each session gets the current animal index as `subject_idx`.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_idx, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(animal_idx)
...
"subjects": ANIMALS.copy(),
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The notes say there are seven animal files, one per mouse, and that sessions remain in animal-major order with the seven IDs preserved directly from the dataset.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day within each mouse payload as one session. It iterates over `day` across `trace.shape[0]` and appends one session entry per day to `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx`.

ii.
```python
selected_days = range(trace.shape[0]) if targets[animal] is None else sorted(targets[animal])

for day in selected_days:
    ...
    neural.append(session_neural)
    decoder_input.append(session_input)
    output.append(session_output)
    subject_idx.append(animal_idx)
    brain_region_idx.append(np.zeros(int(present.sum()), dtype=np.int64))
```

iii. The notes repeatedly describe “each recording day is one target session” and justify that as the natural session boundary in the source dataset.

## 1-d. How are the data split into trials?

i. Within each day/session, the agent defines trials as contiguous, non-overlapping 60-second windows. At 30 Hz this is 1,800 raw frames. Because the agent rebins to 100 ms, each retained trial becomes 600 pooled time bins. Any incomplete final window is dropped.

ii.
```python
FPS = 30
POOL_FRAMES = 3
TRIAL_SECONDS = 60
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
TIMEPOINTS_PER_TRIAL = RAW_TRIAL_FRAMES // POOL_FRAMES
...
n_trials = n_raw_frames // RAW_TRIAL_FRAMES
n_used_frames = n_trials * RAW_TRIAL_FRAMES
...
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
    session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
    session_input.append(geometry.copy())
    session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. In the notes the agent explicitly states that “consecutive non-overlapping 1,800-frame windows are exactly 60 s and become trials,” with only the incomplete tail discarded.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply a per-trial quality-control filter. It only requires that each session have at least two complete 60-second windows and discards the incomplete final tail. It does not reject individual retained trials for behavior, missingness, or neural quality.

ii.
```python
n_trials = n_raw_frames // RAW_TRIAL_FRAMES
n_used_frames = n_trials * RAW_TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day} has fewer than two complete trials")
```

iii. In the notes the agent argues against velocity-based frame filtering because it would break the requested contiguous one-minute trial structure. It treats the tail drop as a windowing policy, not a QC rejection rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data is derived from the per-day `trace` array in each animal payload.

ii.
```python
trace = np.asarray(payload["trace"])
...
binned_neural = process_neural(trace[day], present, n_used_frames)
```

iii. The notes say the native neural stream is `trace`, already containing binary rising-phase calcium-event data sampled at 30 Hz, so the agent does not try to derive neural data from any other raw variable.

## 2-b. How is the `neural` data processed?

i. The agent selects cells present on that day, casts to `float32`, Gaussian-smooths each cell’s full-session trace with `sigma=3` raw frames, crops to the complete-trial portion, and mean-pools over non-overlapping groups of 3 frames to produce 100 ms bins.

ii.
```python
def process_neural(raw_trace: np.ndarray, present: np.ndarray, n_used_frames: int) -> np.ndarray:
    selected = np.asarray(raw_trace[present, :], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Registered neural traces contain intermittent NaN/Inf")
    gaussian_filter1d(selected, sigma=POOL_FRAMES, axis=1, output=selected)
    selected = selected[:, :n_used_frames]
    return selected.reshape(selected.shape[0], -1, POOL_FRAMES).mean(axis=2)
```

iii. The notes justify this by claiming the conversion should follow the paper’s position-decoder temporal processing: “Gaussian smoothing (sigma 3 native frames) followed by 3-frame average pooling.” The agent also notes it intentionally smooths the full physical session before trimming the incomplete tail.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent removes cells not registered in the current day/session by keeping only neurons whose first sample is finite, which it assumes identifies day-present cells. It keeps all finite registered cells, including zero-event cells, and raises an error if any retained trace contains intermittent NaN/Inf values.

ii.
```python
present = np.isfinite(trace[day, :, 0])
if not present.any():
    raise ValueError(f"{animal} day {day} has no registered neurons")
...
selected = np.asarray(raw_trace[present, :], dtype=np.float32)
if not np.isfinite(selected).all():
    raise ValueError("Registered neural traces contain intermittent NaN/Inf")
```

iii. The notes say absent cells are stored as all-NaN traces and that all manually curated, day-registered cells should be retained, with no place-cell or activity threshold. The agent explicitly says the 45 zero-event session-neurons are retained.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. The agent does not use any experimental event. It defines the alignment event as the start of each contiguous one-minute trial window within the continuous session, with trial offsets `0.0` to `60.0` seconds.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each contiguous one-minute window within the recording session",
    "off_start": 0.0,
    "off_end": 60.0,
    ...
}
```

iii. The notes justify this by saying the recording is continuous and that temporal binning begins at session frame 0, so the natural artificial alignment point is the start of each minute-long window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. The agent rebins both neural activity and position by mean-pooling non-overlapping groups of 3 native 30 Hz frames after applying Gaussian smoothing to neural traces.

ii.
```python
FPS = 30
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
...
"time_bin_size": TIME_BIN_MS,
"neural_processing": "binary rising-phase events; Gaussian sigma=3 source frames; non-overlapping 3-frame mean pooling",
"position_processing": "non-overlapping 3-frame mean pooling; clipped to arena; floor-divided into 25 cm bins; class=x_bin*3+y_bin",
```

iii. The notes explicitly describe this as the reference decoder’s `temporal_bin_size=3` processing and state that the resulting trials have 600 time bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the per-day `blocked` field in each animal payload.

ii.
```python
geometry = blocked_vector(payload["blocked"][day][0])
```

iii. The notes say `blocked[day]` is the authoritative source for geometry, rather than reconstructing geometry from the environment-name labels alone.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent converts blocked partition indices into a length-9 binary vector with `1` for blocked and `0` for accessible. If the day has `[-1]`, it returns all zeros. It interprets the native blocked layout as `(y, x)`, transposes the 3x3 matrix, flattens it in an x-first order, and then repeats the same static geometry vector for every trial in the session.

ii.
```python
def blocked_vector(raw_blocked: object) -> np.ndarray:
    values = np.asarray(raw_blocked).reshape(-1)
    native_yx = np.zeros(9, dtype=np.float32)
    if values.size == 1 and float(values[0]) == -1.0:
        return native_yx
    indices = values.astype(np.int64)
    native_yx[indices] = 1.0
    return np.ascontiguousarray(native_yx.reshape(3, 3).T.ravel())
...
session_input.append(geometry.copy())
```

iii. The notes justify the transpose by saying the native blocked indices flatten `(y, x)` while the position/output convention flattens `(x, y)`, and report that transpose reduced geometry/position overlap from 17.14% to 0.0031%.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the per-day `position` array in each animal payload.

ii.
```python
position = np.asarray(payload["position"])
...
binned_position = pool_position(position[day], n_used_frames)
labels = position_classes(binned_position)
```

iii. The notes describe `position` as the aligned 2D behavioral stream and treat it as the direct source for the decoder target.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first mean-pools the 2D position stream over non-overlapping 3-frame windows to get 100 ms bins. It then clips coordinates into the arena bounds, divides each axis by 25 cm, floors to integer bin indices, and turns the result into a single 0-8 class label per time bin.

ii.
```python
def pool_position(raw_position: np.ndarray, n_used_frames: int) -> np.ndarray:
    selected = np.asarray(raw_position[:, :n_used_frames], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Position contains NaN/Inf in a retained complete trial")
    return selected.reshape(2, -1, POOL_FRAMES).mean(axis=2)

def position_classes(position_binned: np.ndarray) -> np.ndarray:
    upper = np.nextafter(np.float32(ARENA_CM), np.float32(0.0))
    xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
    labels = xy[0] * SPATIAL_BINS + xy[1]
    return labels
```

iii. The notes justify the pooling as part of the same 100 ms decoder-style processing applied to the neural data, and say the coarse 3x3 discretization is the requested analogue of the paper’s finer position decoder.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The agent uses a 3 x 3 grid over the 75 cm arena. Each axis is divided into bins `[0,25)`, `[25,50)`, and `[50,75)`, after clipping exact upper-bound values just inside the arena. The final class is computed as `x_bin * 3 + y_bin`, yielding labels 0 through 8.

ii.
```python
SPATIAL_BINS = 3
SPATIAL_BIN_CM = ARENA_CM / SPATIAL_BINS
...
upper = np.nextafter(np.float32(ARENA_CM), np.float32(0.0))
xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
labels = xy[0] * SPATIAL_BINS + xy[1]
...
"output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
```

iii. The notes say the class convention is intentionally x-first and that clipping exact 75 cm values avoids generating an out-of-range class.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The agent assumes the source `trace` and `position` streams are already sample-aligned. It rebins them over the same retained raw frames, checks that neural and output bin counts are identical, and slices both into trials with the same trial boundaries.

ii.
```python
binned_position = pool_position(position[day], n_used_frames)
labels = position_classes(binned_position)
binned_neural = process_neural(trace[day], present, n_used_frames)

if binned_neural.shape[1] != labels.size:
    raise AssertionError("Neural and position bins are temporally misaligned")
...
sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The notes say the streams are already one-to-one aligned in the source data, so no lag correction or interpolation is applied; instead, identical pooling and slicing are used to preserve alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent excludes unregistered cells by removing non-finite day entries, errors out if retained neural or position data contain intermittent NaN/Inf, clips exact arena-edge positions into the valid final bin, and drops incomplete final trial tails. It does not impute missing values.

ii.
```python
present = np.isfinite(trace[day, :, 0])
...
if not np.isfinite(selected).all():
    raise ValueError("Registered neural traces contain intermittent NaN/Inf")
...
if not np.isfinite(selected).all():
    raise ValueError("Position contains NaN/Inf in a retained complete trial")
...
upper = np.nextafter(np.float32(ARENA_CM), np.float32(0.0))
xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
...
"tail_policy": "discard incomplete final 60-second window",
```

iii. The notes say all-NaN traces represent absent registrations, there are no intermittent missing samples in the inspected data, and exact 75 cm coordinates are clipped into the final spatial bin rather than causing invalid labels.

## 6-a. What are the most time-consuming steps of the code?

i. The heaviest steps are loading each animal payload from disk and per-session neural smoothing/pooling. Optional processing plots also add cost when enabled.

ii.
```python
payload = joblib.load(DATA_DIR / animal)[animal]
...
gaussian_filter1d(selected, sigma=POOL_FRAMES, axis=1, output=selected)
...
if show_processing and plot_count < 2:
    save_processing_plot(...)
```

iii. The notes explicitly separate “native load” time from “session conversion,” estimate total full-run time from those components, and describe one-animal loading plus in-place smoothing as the main performance concerns.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining vectorizable loop is the per-trial Python loop that slices already-binned session arrays and repeatedly appends to lists. The code also repeatedly appends identical static geometry vectors one trial at a time.

ii.
```python
session_neural = []
session_input = []
session_output = []
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
    session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
    session_input.append(geometry.copy())
    session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The agent did not call this out explicitly in the final notes, but the code structure makes this the clearest leftover Python-loop hotspot after the earlier vectorized smoothing and reshape/mean pooling.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats static geometry copying for every trial within a session, redoes per-trial slicing in Python after already constructing full-session binned arrays, and recomputes the same shape/value checks across every trial during validation.

ii.
```python
for trial in range(n_trials):
    ...
    session_input.append(geometry.copy())
...
for n, i, o in zip(data["neural"][s], data["input"][s], data["output"][s]):
    if n.shape != (nneurons, TIMEPOINTS_PER_TRIAL):
        ...
    if i.shape != (9,) or o.shape != (1, TIMEPOINTS_PER_TRIAL):
        ...
```

iii. The notes focus more on reducing repeated large-array work, but the surviving repeated work is the trial-by-trial packaging and repeated validation traversal over already-constructed data.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints several diagnostics that are not used by downstream decoder training: global class counts, blocked-bin overlap counts, detailed `session_info` provenance, load/runtime accounting, and optional six-panel processing plots.

ii.
```python
class_counts = np.zeros(9, dtype=np.int64)
blocked_position_count = 0
total_position_count = 0
session_info: list[dict] = []
...
occupied_geometry = geometry[labels]
blocked_position_count += int(occupied_geometry.sum())
total_position_count += int(labels.size)
class_counts += np.bincount(labels, minlength=9)
...
"session_info": session_info,
...
if show_processing and plot_count < 2:
    save_processing_plot(...)
```

iii. The notes frame these as sanity checks and provenance rather than core decoder inputs/outputs. They are useful for auditing but not consumed by the downstream model itself.
