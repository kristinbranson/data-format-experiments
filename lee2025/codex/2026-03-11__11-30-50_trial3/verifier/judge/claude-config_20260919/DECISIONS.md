# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the **joblib** per-animal files in `/app/data` (`QLAK-CA1-08`, ... , i.e. the extension-less files), not the `.mat` files. It selects them by name (`startswith("QLAK-CA1-")` and no `.` in the filename), so the `.mat` twins, `behav_dict` and `precomputed_results/` are excluded. Each file is a dict keyed by the animal ID containing `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`. The conversion makes **two passes**: `iter_session_refs()` loads every animal file once only to enumerate `(animal, day_index)` session references, and then the main loop re-loads each animal file (caching one animal at a time) to process its sessions. Result: 7 subjects, 207 sessions, 8,187 trials, 69,744 session-present neurons.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("QLAK-CA1-") and "." not in filename
    )

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
        del dat
        gc.collect()
    return animals, session_refs

    for session_index, session_ref in enumerate(session_refs):
        if session_ref.animal not in animal_cache:
            animal_cache.clear()
            gc.collect()
            animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
```

iii. From CONVERSION_NOTES.md Step 5, Key Decision 1: *"Use primary joblib animal files, not cached analysis results: This matches the reference loading path in `load_dat(..., format="joblib")` and avoids inheriting any downstream analysis assumptions."* The reference `utils.load_dat()` indeed defaults to `format="joblib"` and the repo ships `mat2joblib()` to produce exactly these files, so the joblib files are the reference code's canonical entry point. The AI verified the totals against the paper: 5,413 registered cells, 207 sessions, 69,744 session-present cell-maps ("5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps").

## 1-b. How are the data split into subjects?

i. One subject per animal file; the subject ID is the filename (`QLAK-CA1-08`, ...). `subjects` is the sorted list of the 7 animal IDs and `subject_idx` is built from a name→index lookup, one entry per session.

ii.
```python
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}
...
"subject_idx": subject_lookup[session_ref.animal],
...
converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 2/Step 5: each joblib file is a dict keyed by the animal ID and contains all of that animal's recording days, so the file/key name is the natural subject identifier. Verified in `verification_full_out.txt`: 7 subjects with 31/31/31/21/31/31/31 sessions.

## 1-c. How are the data split into sessions?

i. One converted session per **recording day** within an animal file. The number of days is read from `dat["envs"].shape[0]`, and day `d` indexes `position[d]`, `trace[d]`, `blocked[d]`, `maps["smoothed"][:,:,:,d]`. Session IDs are `f"{animal}_day{day:02d}"`. This yields 31 sessions for six animals and 21 for `QLAK-CA1-51` → 207 sessions.

ii.
```python
@dataclass(frozen=True)
class SessionRef:
    animal: str
    day_index: int
    @property
    def session_id(self) -> str:
        return f"{self.animal}_day{self.day_index:02d}"
...
for day_index in range(dat["envs"].shape[0]):
    session_refs.append(SessionRef(animal=animal, day_index=day_index))
...
day = session_ref.day_index
env_name = str(dat["envs"][day, 0])
position_day = np.asarray(dat["position"][day], dtype=np.float64)
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 2: *"Keep one target session per original recording day: The raw data are organized by day/session, and the target format supports multiple trials within each session."* The paper states one 40-min session per day, and the per-day environment/geometry changes, so a day is the correct session unit.

## 1-d. How are the data split into trials?

i. Each continuous ~40-min session is cut into **non-overlapping 60-s segments of exactly 1800 frames** (30 Hz). The trailing partial minute is discarded. Slices are computed once per session from the frame count and applied identically to neural and output streams. Sessions with fewer than 2 complete trials raise an error (never triggered: all sessions give 39 or 40 trials, 8,187 total).

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)   # 1800

def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
...
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 3: *"Create trials by splitting each continuous 40-minute session into non-overlapping 1-minute windows: This satisfies the decoder task while preserving within-session context. Trial length will be exactly 1800 frames at 30 Hz; any trailing partial minute will be discarded."* Step 4 records that this trialization is task-imposed, not part of the original acquisition (the paper analyses continuous sessions). The `>= 2 trials` guard implements the format requirement that every session must contain at least two trials.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality filtering.** The only trial-level rules are structural: incomplete trailing segments are dropped (`n_frames // 1800`), and a session must yield ≥2 trials. Notably, the reference decoder's **velocity filter** (`v_thresh=5` in `decode_position_within`) is *not* applied — the AI identified it in Step 1 but did not port it, since dropping low-speed frames would destroy the fixed-length 1800-frame trial structure required by the target format.

ii.
```python
n_trials = n_frames // trial_frames          # incomplete tail dropped
...
if len(slices) < 2:
    raise ValueError(...)
...
"discarded_tail_frames": int(n_frames - len(slices) * TRIAL_FRAMES),
```

iii. CONVERSION_NOTES.md Step 3 ("Trial curation rules") states: *"The reference experiment does not define trialized behavior; sessions are continuous 40-minute recordings"*, so there is no reference trial-curation rule to reproduce. Step 10 Check 5 reports the discarded remainders explicitly (`[60, 71, 91, 219, 1666]` frames) and that all trials are 1800 frames and all sessions have ≥2 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively `dat[animal]["trace"][day]`, shape `(n_registered_cells, n_frames)`, the released **binary rising-phase calcium-event** matrix (values ∈ {0, 1}, with whole rows of `NaN` for cells not registered on that day). `SFPs`, `centroids` and `maps` are not used as neural signals (`maps["smoothed"]` is only read as a cross-check for the geometry input).

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
...
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
```

iii. CONVERSION_NOTES.md Step 1: *"The neural signal is not raw fluorescence and does not require delta F/F computation... the README explicitly describes `trace` as rise-extracted calcium traces where `1` indicates a significant event"*, and the reference decoder is called as `decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, ...)`, i.e. it consumes `trace` directly.

## 2-b. How is the `neural` data processed?

i. Essentially **no processing**: select the cells present on that day, keep the native binary 0/1 event values at 30 Hz, orient as `(n_neurons, n_timepoints)` (the raw array is already in that orientation), slice into trials, and store as **float16**. No ΔF/F, no smoothing, no z-scoring, no rebinning, no normalization.

ii.
```python
present_mask = session_present_cell_mask(trace_day)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 6: *"Do not compute new calcium features: The released `trace` is already the binary rising-phase representation used in the paper/code."* Step 3 records the paper's statement that the thresholded (z > 2.5) binary rising-phase vector *"is treated as the firing rate in all further analyses."* The float16 cast is justified in Step 6/Step 10 as a storage optimization (*"reduced `sample_data.pkl` from about 92 MB to 46 MB"*); it is lossless for binary data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The **only** neuron filter is removal of cells not registered on that day, detected as `NaN` in the trace. The mask is computed from the **first frame only** (`~np.isnan(trace_day[:, 0])`). An error is raised if a session has no present cells. No place-cell selection and no event-count (`cell_threshold=5`) filter are applied. Mean 336.93 neurons/session (min 113, max 564); 69,744 session-present neurons total.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
...
present_mask = session_present_cell_mask(trace_day)
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. CONVERSION_NOTES.md Step 5, Key Decisions 4 and 5: *"Use all session-present cells: This matches the paper's statement that all cells were included in subsequent analyses. Cells absent on a day are removed by excluding `NaN` rows for that session"* and *"Do not pre-filter to place cells: The paper's position decoder is not place-cell-restricted, and place-cell status is an analysis label rather than a required curation step for decoding."* Step 10 verifies that 69,744 session-present cells exactly reproduce the paper's "69,744 rate maps" and that no NaN/Inf remain in any converted array.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is **no experimental alignment event** — the recordings are continuous. Alignment is therefore to the start of each 1-minute segment: trial `i` spans frames `[i*1800, (i+1)*1800)` of the session. Neural, input, and output streams are cut with the *same* slice objects, so they are aligned frame-for-frame at 30 Hz. The metadata records this explicitly.

ii.
```python
"temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
...
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice]...
    output_trial = output_class[trial_slice][np.newaxis, :]...
```

iii. CONVERSION_NOTES.md Step 3/Step 10(c): *"Behavioral and cellular imaging streams were simultaneously acquired at 30 Hz and timestamped for post-hoc alignment"* and *"converter preserves the raw 30 Hz synchronized time base and uses the released aligned `position` and `trace` streams directly."* Step 10 Check 2 confirms with `np.allclose()` that the concatenated converted trials reproduce the raw `trace[present, :n_trials*1800]` slice exactly for sessions 0, 93 and 206.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native acquisition resolution is kept: **33.33 ms bins (30 Hz)**, 1800 bins per trial for every trial and session. **No temporal rebinning, pooling or smoothing is applied**, even though the reference `fit_decoder`/`decode_position_within` pool 3 frames (→10 Hz) before fitting the Gaussian Naive Bayes decoder.

ii.
```python
FPS = 30.0
...
"time_bin_size": 1000.0 / FPS,
"raw_fps": FPS,
"trial_length_frames": TRIAL_FRAMES,
"trial_length_seconds": TRIAL_SECONDS,
```

iii. CONVERSION_NOTES.md Step 3 records both streams are acquired at 30 Hz; Step 1 notes the reference decoder's 3-frame pooling as a *decoder-side* step (`fit_decoder` "Temporal-bins behavior and traces (3-frame pooling by default)"), i.e. part of the reference model rather than of the dataset. Keeping the native resolution preserves maximum information for the new downstream decoder, which does its own temporal modelling. `verification_full_out.txt` confirms T = 1800 for every trial (mean = median = min = max).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. `dat[animal]["blocked"][day]` — a per-day list of blocked partition IDs in the README's 3×3 numbering `[[0,1,2],[3,4,5],[6,7,8]]`, with `-1` meaning "nothing blocked". Entries appear as nested one-element lists of float arrays and are normalized with `np.atleast_1d(...).astype(int)`. `dat["envs"]` (the geometry name) is carried only as per-session metadata, and `dat["maps"]["smoothed"]` is used purely as an independent cross-check of the derived geometry.

ii.
```python
def extract_day_blocked_entry(blocked_list: list, day_index: int) -> np.ndarray:
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)
```

iii. CONVERSION_NOTES.md Step 4: *"Raw data also contain an explicit `blocked` list with partition IDs in the stated 3x3 indexing scheme... For decoder input, use `blocked` as authoritative, transpose the 3x3 matrix to align with `position`, and use `envs` as a metadata cross-check."* `blocked` is preferred over `envs` + `get_env_mat()` because it is the raw per-session record rather than a label-to-matrix lookup that the reference code sometimes flips.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked IDs are converted to a 9-D **binary occupancy/availability vector with 1 = open, 0 = blocked** (the polarity of the reference `get_env_mat()`), reshaped to 3×3, **transposed** to match the `(x, y)` frame used by `position` and the rate maps, and flattened row-major so that element `k = 3*x + y` describes the partition containing position bin `k`. The same static `(9,)` float32 vector is copied into every trial of the session. Before use, the AI asserts per session that this grid equals the valid (non-NaN) support of `maps["smoothed"]` collapsed from 15×15 to 3×3; the full 207-session run completed without raising, so the check passed everywhere.

ii.
```python
def blocked_to_geometry_vector(blocked_list, day_index):
    blocked = extract_day_blocked_entry(blocked_list, day_index)
    geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
    if not (blocked.size == 1 and blocked[0] == -1):
        geometry[blocked] = 0.0
    # Raw blocked indices use the README 3x3 numbering. Transpose to match
    # the coordinate frame used by position and the 15x15 spatial maps.
    geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
    return geometry_grid, geometry_grid.reshape(-1).astype(np.float32)

def aggregate_valid_map(smoothed_maps_day):
    valid_mask = np.any(~np.isnan(smoothed_maps_day), axis=2).astype(np.float32)
    return valid_mask.reshape(3, 5, 3, 5).max(axis=(1, 3))
...
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask")
...
input_trials.append(geometry_vector.copy())
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 7: *"Align geometry input to the map/position frame by transposing the 3x3 blocked matrix: This is required for asymmetric geometries and was verified against the non-NaN support of `maps['smoothed']` for every session."* Step 10 lists "Environment orientation ambiguity" as an issue found and resolved before the full run. Geometry is static per session because the arena configuration does not change within a day.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `dat[animal]["position"][day]`, shape `(2, n_frames)` — the DeepLabCut head-tracking x/y coordinates in cm within the 75 × 75 cm arena, sampled at 30 Hz and already timestamp-aligned to `trace`.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
...
position_bins, output_class = compute_position_bins(position_day)
```

iii. CONVERSION_NOTES.md Step 3: *"Position was obtained from DeepLabCut head tracking"*; Step 5 maps `dat[animal]['position'][day, :, frame_start:frame_end]` → `output[trial]`. This is the same variable the reference passes to `get_rate_maps` and `decode_position_within`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. No filtering, smoothing, interpolation or velocity gating of the trajectory. The `(2, T)` array is transposed to `(T, 2)`, cast to float64, and passed straight to the discretizer. A single 9-class integer label per frame is produced (rather than two separate x/y outputs) and stored as `int8` with shape `(1, 1800)` per trial.

ii.
```python
def compute_position_bins(position_day, n_bins=POSITION_BINS):
    coords = np.asarray(position_day, dtype=np.float64).T
    scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
    binned = np.floor(coords / scale).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
    return binned, class_idx
...
"output_names": ["position_bin_3x3"],
"output_values": [[f"x{x}_y{y}" for x in range(POSITION_BINS) for y in range(POSITION_BINS)]],
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 9: *"Represent output as one 9-class time-varying variable instead of separate x/y outputs: The task explicitly requests 3x3=9 spatial bins, so a single categorical output variable is the most direct representation."* The reference code likewise converts binned x-y position into a single class label before fitting the Bayesian decoder.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The **reference code's binning rule** is reproduced exactly, with `n_bins` changed from 15 to 3: divide each coordinate by `(session-wide per-axis nanmax + 1e-5) / 3`, take the floor, clip to `[0, 2]`, and flatten as `class = x_bin * 3 + y_bin` → 9 classes. Bin edges are computed once per session (not per trial), so all trials of a session share one spatial partition. Empirically, the per-session maxima are 72.4–75.0 cm across the 207 sessions, so these edges sit within ~3.5% of the fixed 25/50 cm thirds of the 75 cm arena.

ii.
```python
POSITION_BINS = 3
POSITION_BUFFER = 1e-5
...
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.clip(np.floor(coords / scale).astype(np.int64), 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```
Reference `utils.get_rate_maps` for comparison:
```python
position_binned = (position // ((np.nanmax(position, axis=0) + buffer) / n_bins)).astype(int)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 8: *"Use session-wide position normalization when binning to 3x3 outputs: The reference code bins position using maxima from the full session/day, not from smaller windows. This keeps spatial bins consistent across all trials within a session."* Step 10(d): *"converter uses the same session-wide position floor-division rule as the reference code, but with 3 x 3 bins instead of 15 x 15 because the decoder task requires 9 spatial classes."* The resulting class distribution (`[0.099, 0.075, 0.116, 0.098, 0.056, 0.142, 0.135, 0.077, 0.202]`) is reported in Step 9.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame. `position` and `trace` are released already co-registered at 30 Hz with identical frame counts, so the class-label time series is cut with the *same* `slice` objects as the neural matrix; no shifting, lag, or resampling is introduced. Each output trial is `(1, 1800)`, matching the `(n_neurons, 1800)` neural trial.

ii.
```python
slices = trial_slices(n_frames)          # n_frames from position_day.shape[1]
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. CONVERSION_NOTES.md Step 10 Check 2 verifies alignment against the raw files (not through the conversion code) for sessions 0, 93 and 206: the concatenated converted outputs equal the raw-position-derived 3×3 labels, and the concatenated converted neural data equals the raw present-cell trace slice, with `np.allclose()`. The `--show-processing` figures additionally plot the label time series with trial boundaries and the first trial's event raster.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four issue classes are handled:
- **Cells not registered on a day** appear as all-`NaN` trace rows → dropped per session (2-c). Sessions with no present cell raise an error.
- **Ragged `blocked` entries** (nested one-element lists, scalar `array(-1.)`, or arrays of several floats) are normalized by `extract_day_blocked_entry`; the `-1` sentinel maps to "nothing blocked" (all-ones geometry).
- **Session lengths that are not multiples of 1800** (71,866–72,219 frames): the trailing partial minute is discarded and the number of dropped frames is recorded per session in `metadata['session_info']['discarded_tail_frames']`.
- **Silent inconsistencies** are turned into hard failures: geometry-vs-map mismatch, <2 trials, or zero present cells all raise `ValueError`.
Positions are additionally clipped into `[0, 2]` so boundary samples cannot produce out-of-range classes (the data contain no NaN positions, so `np.nanmax` never hides a dropout).

ii.
```python
def extract_day_blocked_entry(blocked_list, day_index):
    entry = blocked_list[day_index]
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    return np.atleast_1d(np.asarray(entry)).astype(int)
...
if not (blocked.size == 1 and blocked[0] == -1):
    geometry[blocked] = 0.0
...
if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask")
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
...
"discarded_tail_frames": int(n_frames - len(slices) * TRIAL_FRAMES),
```

iii. CONVERSION_NOTES.md Step 10 Check 5 (edge cases): *"`discarded_tail_frames` values were exactly the raw remainders `[60, 71, 91, 219, 1666]`; all converted trials had length 1800; all sessions had at least 2 trials; no NaN/Inf values remained in neural, input, or output arrays."* The `verification_full_out.txt` log reports "Data format is valid, no errors or warnings."

## 6-a. What are the most time-consuming steps of the code?

i. Ranked by measured cost:
1. **Reading the joblib animal files** — the first session of each animal costs ~10 s versus ~0.05 s for subsequent sessions of the same animal (`conversion_full_out.txt`), and the enumeration pass in `iter_session_refs()` pays this cost a *second* time for all 7 files before the timer even starts.
2. **Pickling the 9.3 GB output** at the end of `convert_dataset()`.
3. **Per-session materialization of the neural matrix**: `trace_day[present_mask]` is a fancy-index copy of the full float64 session (up to ~300 MB for a 564-neuron session) before the float16 cast; for the largest animal this dominates the ~3.8 s/session steady-state cost.
Total wall time for the full run was 258.98 s (1.25 s/session) excluding the enumeration pass.

ii.
```python
total_start = time.time()   # NOTE: set after iter_session_refs(), so the enumeration pass is untimed
for session_index, session_ref in enumerate(session_refs):
    session_start = time.time()
    ...
    print(f"Processed {session_ref.session_id}: ... in {elapsed:.2f}s")
...
with open(outpicklefile, "wb") as f:
    pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Elapsed: {total_elapsed:.2f}s ({total_elapsed / max(len(session_refs), 1):.2f}s/session)")
```

iii. CONVERSION_NOTES.md Step 6: *"Loading the full animal files is the main runtime cost because each file contains all sessions and registered cells for one subject."* Mitigations claimed: *"Process sessions animal-by-animal so only one large subject file is kept in memory at a time"*, *"Reuse one loaded animal dataset across all of its sessions before releasing it"*, and storing float16/int8 to shrink the output. The notes do not acknowledge the extra enumeration-pass load.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- `plot_processing_figure()` accumulates the 3×3 occupancy map with a Python `for` loop over all ~72,000 frames; `np.add.at` / `np.bincount` would be far faster. (Only runs under `--show-processing`.)
- The per-trial loop in `process_session()` could be a single `reshape(n_neurons, n_trials, 1800)` + `np.moveaxis`, avoiding 8,187 Python iterations and per-trial `astype` calls (the slices are views, so the cost is small).
- `trial_slices()` builds the slice list in a comprehension; it could be index arithmetic, but this is negligible.
Nothing that affects the output values is loop-bound; the dominant costs (6-a) are I/O and one big fancy-index copy, not Python loops.

ii.
```python
    occupancy = np.zeros((POSITION_BINS, POSITION_BINS), dtype=np.float32)
    for xbin, ybin in position_bins:          # ~72k Python iterations
        occupancy[xbin, ybin] += 1.0
...
    for trial_slice in slices:                # could be one reshape
        neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
        output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
        neural_trials.append(neural_trial)
        input_trials.append(geometry_vector.copy())
        output_trials.append(output_trial)
```

iii. CONVERSION_NOTES.md Step 6 claims the vectorization-relevant speedups that were made: *"Slice full-session binned outputs and present-cell traces directly without redundant recomputation inside trials"* — i.e. position binning and cell selection are done once per session and only sliced per trial. The remaining loops are not discussed; the run finished in ~4 min, comfortably under the 15-minute budget, so no further optimization was pursued.

## 6-c. What processing does the code repeat multiple times?

i. The clearest repetition is the **double load of every animal file**: `iter_session_refs()` loads all 7 joblib files in full just to read `dat["envs"].shape[0]`, discards them, and the main loop then re-loads each file to process it. Only `envs` (also available in the much smaller `data/behav_dict`, or as a cached count) was needed. Secondarily, `np.float16` casting is applied twice per trial (once to the whole session matrix, once per slice — the second is a no-op), and `geometry_vector.copy()` is stored 39–40 times per session for a value that is constant across the session.

ii.
```python
def iter_session_refs(data_dir: str, sample: bool):
    for animal in animals:
        dat = load_animal_dataset(data_dir, animal)   # full ~GB load, only envs used
        for day_index in range(dat["envs"].shape[0]):
            session_refs.append(SessionRef(animal=animal, day_index=day_index))
        del dat
...
    animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)  # loaded again
...
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)  # already float16
    input_trials.append(geometry_vector.copy())                                  # 39-40 copies
```

iii. CONVERSION_NOTES.md does not flag the double load; it only claims the positive side (*"Reuse one loaded animal dataset across all of its sessions before releasing it"*, *"Avoids reloading the same 68-145 MB joblib file for every session"*). The repeated per-trial geometry copy is an intentional consequence of the target schema, which requires one input entry per trial.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work that does not reach the output file:
- **`maps["smoothed"]` is loaded and reduced for every session** (`np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)` is a strided gather over a `(15, 15, n_cells, n_days)` array, then `aggregate_valid_map`) purely to assert geometry orientation. It is a valuable sanity check but is recomputed for all 207 sessions rather than done once as a separate validation, and nothing derived from it is saved.
- `compute_position_bins()` returns the `(T, 2)` `binned` array in addition to the class labels; only the class labels are stored — `binned` is used solely by the optional plot.
- `position_day`/`trace_day` are forced to float64 before being reduced to int8/float16.
- The full-session `trace_day[present_mask]` float64 copy is created and then immediately down-cast.
- `env_name` is read per session (cheap; kept in metadata, unused by the decoder).
- The enumeration pass (6-c) is entirely discarded work.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
...
valid_grid = aggregate_valid_map(smoothed_day)
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(...)
...
def compute_position_bins(position_day, n_bins=POSITION_BINS):
    ...
    return binned, class_idx      # `binned` only used for plotting
...
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
```

iii. The map-based check is deliberate: CONVERSION_NOTES.md Step 5 Key Decision 7 and Step 10 justify it as the evidence that resolved the "Environment orientation ambiguity" (*"verified against the non-NaN support of `maps['smoothed']` for every session"*), and the task instructions explicitly ask for sanity checks. The float64 intermediates are not discussed in the notes; the float16/int8 down-casts are justified there as storage optimizations.
