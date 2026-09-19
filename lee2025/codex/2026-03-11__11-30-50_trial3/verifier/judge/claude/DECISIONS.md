# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the primary joblib animal files (not the `.mat` files). Each animal file is loaded via `joblib.load()`, keyed by subject ID. The data is organized as a dictionary containing `trace`, `position`, `blocked`, `envs`, and `maps` fields. Sessions are enumerated by iterating over the `envs` array, and an animal cache is used to avoid reloading the same file for each session.

ii.
```python
def load_animal_dataset(data_dir: str, animal: str) -> dict:
    return joblib.load(os.path.join(data_dir, animal))[animal]

def iter_session_refs(data_dir: str, sample: bool) -> tuple[list[str], list[SessionRef]]:
    animals = get_animal_ids(data_dir)
    session_refs: list[SessionRef] = []
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
            ...
```

iii. The AI chose joblib because it matches the reference code's default loading path in `load_dat(..., format="joblib")` and avoids inheriting any downstream analysis assumptions from cached results (CONVERSION_NOTES Step 5, Key Decision 1).

## 1-b. How are the data split into subjects?

i. Each joblib file in the data directory (files starting with `QLAK-CA1-` without a `.` extension) corresponds to one subject. Subjects are sorted alphabetically.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )
```

iii. The raw data is organized as one file per animal. The filename filter excludes `.mat` variants and other non-primary files.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple recording days. Sessions are enumerated by iterating over `dat["envs"].shape[0]` (number of days), where each day becomes a separate session in the output.

ii.
```python
for day_index in range(dat["envs"].shape[0]):
    session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. Each original recording day becomes one session. The session count per subject matches the data (31 for 6 animals, 21 for QLAK-CA1-51, totaling 207 sessions).

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute session is split into non-overlapping 1-minute trials of exactly 1800 frames (at 30 Hz). Remaining frames at the end of a session that don't fill a complete 60-second window are discarded.

ii.
```python
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)  # 1800

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. The 1-minute trial duration is specified in the decoder task instructions. The reference experiment does not define native trials; sessions are continuous 40-minute recordings.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 full 1-minute trials are rejected (raises an error). No other trial-level quality filtering is applied.

ii.
```python
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." In practice, all sessions have 39-40 trials so this check never triggers.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` field in each animal's joblib dataset, which contains rise-extracted calcium event traces where 1 indicates a significant event. Shape is `(n_sessions, n_registered_cells, n_frames)`.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The `trace` data is already the binary rising-phase representation used in the paper. The README describes it as rise-extracted calcium traces. No delta F/F computation is needed.

## 2-b. How is the `neural` data processed?

i. Present cells are selected (cells not NaN at the first timepoint), the trace is cast to float16 for memory efficiency, and then sliced into trial windows. No additional processing (smoothing, normalization, etc.) is applied.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    neural_trials.append(neural_trial)
```

iii. The paper states that all cells were included in subsequent analyses. The released traces are already preprocessed (motion correction, cell segmentation, transient extraction). Using float16 reduces memory from ~18 GB to ~9 GB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are NaN at the first timepoint (not recorded in that session) are excluded. No place-cell filtering or activity threshold is applied.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
```

iii. Cells absent on a given day appear as NaN in per-day traces. The paper states all cells were included in analyses, and the reference decoder in `main.py` does not pre-filter to place cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous and trials are simply non-overlapping 1-minute windows from the start of the session. Neural and behavioral streams are already synchronized at 30 Hz.

ii. N/A (alignment is implicit via the common frame index used for all data streams)

iii. There is no stimulus onset or behavioral event to align to. The paper describes continuous 40-minute sessions with simultaneously acquired behavioral and cellular imaging streams at 30 Hz.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz frame rate is preserved, giving a time bin size of ~33.33 ms. No temporal rebinning is applied.

ii.
```python
FPS = 30.0
# in metadata:
"time_bin_size": 1000.0 / FPS,  # ~33.33 ms
```

iii. Both behavioral and cellular imaging streams were acquired at 30 Hz. The reference code uses the same frame rate directly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment geometry input is derived from the `blocked` field in each animal's dataset, which stores blocked partition indices in a 3x3 arena indexing scheme. Additionally, the `maps["smoothed"]` field is used to cross-validate the geometry.

ii.
```python
geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
```

iii. The `blocked` field stores which of the 9 possible reward locations were blocked during each session. A value of `[-1]` indicates no positions were blocked. The cross-validation against `maps["smoothed"]` ensures the geometry encoding is spatially correct.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-dimensional binary vector where 1 = open (accessible) and 0 = blocked. The vector is then reshaped to a 3x3 grid and transposed to align with the position/map coordinate frame, then flattened back to 9 dimensions. The resulting geometry vector is static per-trial (same for all trials within a session).

ii.
```python
def blocked_to_geometry_vector(blocked_list: list, day_index: int) -> tuple[np.ndarray, np.ndarray]:
    blocked = extract_day_blocked_entry(blocked_list, day_index)
    geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
    if not (blocked.size == 1 and blocked[0] == -1):
        geometry[blocked] = 0.0
    geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
    return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)
```

iii. The transpose is required because the raw `blocked` indices use a 3x3 numbering scheme that is transposed relative to the position/map coordinate frame. This was verified by comparing against the valid spatial support of `maps["smoothed"]` for all 207 sessions (CONVERSION_NOTES Steps 4, 5, 10).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` field in each animal's dataset, which contains continuous 2D coordinates (x, y) of the animal in the arena at 30 Hz. Shape is `(n_sessions, 2, n_frames)`.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
```

iii. Position was obtained from DeepLabCut head tracking at 30 Hz, synchronized with calcium imaging.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2D position is discretized into a 3x3 grid (9 classes) using session-wide position normalization. For each session, the maximum position along each axis (plus a small buffer of 1e-5) is divided by 3 to get the bin scale. Each frame's position is then floor-divided by this scale and clipped to [0, 2].

ii.
```python
def compute_position_bins(position_day: np.ndarray, n_bins: int = POSITION_BINS) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(position_day, dtype=np.float64).T
    scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
    binned = np.floor(coords / scale).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
    return binned, class_idx
```

iii. Session-wide normalization matches the reference code's approach in `fit_decoder` and `get_rate_maps`, which use session-wide position maxima for binning. The buffer ensures the maximum position falls within the last bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 3x3 = 9 spatial bins using floor division after session-wide normalization. The class index is computed as `dim0_bin * 3 + dim1_bin`, producing integer labels 0-8.

ii.
```python
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The 3x3 grid was specified in the decoder task instructions. The linearization formula produces a unique class for each (x, y) bin combination.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are stored at the same 30 Hz frame rate in the original data and are already temporally aligned. Both are sliced into trials using the same frame indices.

ii.
```python
position_bins, output_class = compute_position_bins(position_day)
# ...
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. Both streams share the same time base and trial slicing, ensuring frame-for-frame alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple types of data issues are handled: (1) Neurons with NaN at the first timepoint are excluded from that session. (2) Tail frames that don't fill a complete 1-minute trial are discarded. (3) Sessions with fewer than 2 trials would be rejected (though this never occurs). (4) The `blocked` field entries have inconsistent formats (sometimes nested lists), handled by `extract_day_blocked_entry`. (5) Geometry is validated against smoothed maps for every session.

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)
```

iii. NaN filtering ensures only recorded neurons are included. The blocked entry normalization handles inconsistent Python list formatting in the raw data. The geometry validation ensures no environment encoding errors.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib animal files is the dominant cost. Each file is 68-145 MB containing all sessions and registered cells for one subject. The actual per-session processing (cell filtering, position binning, trial slicing) is fast by comparison.

ii. N/A

iii. The AI implemented an animal cache to avoid reloading the same file for each session: one file is loaded per animal and reused across all of that animal's sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial slicing loop iterates over trial slices and creates individual arrays. This could potentially be replaced with array reshaping for a minor speedup, but the impact is minimal since slicing is fast.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
    neural_trials.append(neural_trial)
```

iii. The loop is simple and the bottleneck is I/O, not computation. The AI noted this in CONVERSION_NOTES Step 6.

## 6-c. What processing does the code repeat multiple times?

i. The code loads `maps["smoothed"]` for each session to perform geometry validation. This is an additional I/O cost that could be avoided after initial validation, but it provides a useful sanity check.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
valid_grid = aggregate_valid_map(smoothed_day)
```

iii. This is a deliberate choice to validate geometry alignment for every session, ensuring no encoding errors.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and processes `maps["smoothed"]` data solely for geometry validation. This data is not included in the output and is discarded after the check. Additionally, the code computes 2D position bins (`binned`) which are only used to derive the 1D class index and for plotting.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
valid_grid = aggregate_valid_map(smoothed_day)
# valid_grid is only used for the assertion check, not in the output
```

iii. The smoothed map loading adds processing time but ensures correctness. The 2D bins are a natural intermediate step in computing the 1D class index and are also useful for diagnostic plots.
