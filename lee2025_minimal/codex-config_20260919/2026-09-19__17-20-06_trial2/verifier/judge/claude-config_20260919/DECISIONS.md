# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the **joblib** copies of the Zenodo dataset (`/app/data/QLAK-CA1-XX`, the extension-less files) rather than the `.mat` copies. Each file unpickles to a one-key dict `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}`. For each animal it takes only `trace` (n_days, n_cells, n_frames), `position` (n_days, 2, n_frames), `envs` (n_days,) and `blocked` (list of length n_days); `SFPs`, `centroids` and `maps` are ignored. The list of 7 animals is **hard-coded**. Shape/consistency assertions are run per animal (days match between trace/position, frames match, metadata lengths match), and after the whole loop the code asserts that the total number of session-neurons equals the paper's 69,744. Memory is released per animal with `del` + `gc.collect()`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal_index, animal in enumerate(ANIMALS):
    source_path = data_dir / animal
    wrapped = joblib.load(source_path)
    source = wrapped[animal]

    traces = np.asarray(source["trace"])
    positions = np.asarray(source["position"])
    envs = np.asarray(source["envs"]).reshape(-1)
    blocked = source["blocked"]

    if traces.shape[0] != positions.shape[0] or traces.shape[2] != positions.shape[2]:
        raise ValueError(f"Neural/position alignment mismatch for {animal}")
    if traces.shape[0] != len(envs) or traces.shape[0] != len(blocked):
        raise ValueError(f"Session metadata length mismatch for {animal}")
    ...
    del wrapped, source, traces, positions, envs, blocked
    gc.collect()

total_session_neurons = sum(len(x) for x in brain_region_idx)
if total_session_neurons != 69_744:
    raise ValueError(...)
```

iii. From the trajectory (steps 9, 12–13, 29): the AI inspected both the `.mat` and joblib versions, noted the joblib files are the repo's native Python format containing "the paper's final, manually curated rise-extracted calcium events and frame-aligned DeepLabCut positions", and therefore chose not to re-derive anything from raw video. It cross-checked the loaded contents against the paper: 207 recording days and exactly 69,744 registered session-cells, which it then froze into an assertion as a "strong reference check".

## 1-b. How are the data split into subjects?

i. One source file = one mouse. The seven animal IDs (`QLAK-CA1-08/30/50/51/56/74/75`) are written out as a constant list; the list index is used directly as `subject_idx` for every session of that animal, and `subjects` is that same list. No globbing of the data directory is performed.

ii.
```python
ANIMALS = ["QLAK-CA1-08", ..., "QLAK-CA1-75"]
for animal_index, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(animal_index)
...
"subjects": ANIMALS.copy(),
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The dataset README states each file is named after an animal ID from the original study, and the top-level key inside each joblib file is that same animal ID (the AI verified this in step 12). Hard-coding the list also lets the loader index `wrapped[animal]` without guessing the dict key.

## 1-c. How are the data split into sessions?

i. One **recording day** (one 40-min session in one geometry) is one decoder session. The AI iterates over the first axis of `trace` for each animal, giving 31 + 31 + 31 + 21 + 31 + 31 + 31 = 207 sessions, matching the paper. The environment name (`envs[day]`) and the source day index are recorded in `metadata['session_info']`.

ii.
```python
for day in range(traces.shape[0]):
    day_trace = traces[day]
    ...
    session_info.append({
        "subject": animal,
        "source_day_index": day,
        "environment": str(envs[day]),
        "n_source_frames": int(day_trace.shape[1]),
        "n_neurons": int(present.sum()), ...})
...
"session_definition": "one recording day in one environment",
```

iii. Methods/README: "All sessions were 40 min, and one session was recorded per day"; each day is a distinct geometry with its own cell registration and its own blocked-partition configuration, so a day is the natural session unit. The AI confirmed the resulting count (207) equals the 207 sessions reported in the paper (trajectory step 49).

## 1-d. How are the data split into trials?

i. Each day is split into exactly **40 contiguous, nearly equal windows** with `np.array_split`, so that every recorded frame is kept. Because the recordings are 71,866–72,219 frames (not exactly 40 × 1800), trials end up 1,796–1,806 frames long (59.87–60.2 s at 30 Hz) instead of a fixed 1,800. Total: 8,280 trials. Each chunk is copied to a contiguous array.

ii.
```python
FPS = 30.0
N_TRIALS = 40

def split_trials(array: np.ndarray, axis: int) -> list[np.ndarray]:
    """Split a full recording into the nominal 40 contiguous minute windows."""
    return [np.ascontiguousarray(x) for x in np.array_split(array, N_TRIALS, axis=axis)]
...
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
```

iii. Trajectory step 25: "The session arrays differ slightly around the nominal 40 minutes (for example, 71,866 versus 72,219 frames). I'll split each recording into exactly 40 contiguous, nearly equal windows with `array_split`; this preserves every aligned frame and avoids inventing a spurious 7-second '41st trial' or discarding almost a minute from shorter recordings. Each resulting trial is about 60 seconds at 30 Hz." The docstring adds that `array_split` "keeps every frame despite small recording-length differences around 40 minutes."

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level filtering at all.** Every one of the 40 windows of every one of the 207 sessions is emitted (8,280 trials). In particular the AI deliberately did not drop stationary periods or apply the paper's running-speed / event-count screens used in its within-session Bayesian decoder. It does apply session-level *sanity* guards that abort the run rather than filter data: a partially-NaN neuron trace, a session with no registered neurons, a non-binary trace, or >5% of position samples falling in nominally blocked bins all raise.

ii.
```python
if not np.array_equal(finite_any, finite_all):
    raise ValueError(f"Partially finite neuron trace in {animal}, day {day}")
if not np.any(present):
    raise ValueError(f"No registered neurons in {animal}, day {day}")
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
    raise ValueError(f"Non-binary rise-extracted trace in {animal}, day {day}")
if blocked_position_samples / session_output_full.shape[1] > 0.05:
    raise ValueError(f"More than 5% of positions enter blocked bins in {animal}, day {day}; "
                     "check the x/y convention")
```

iii. Trajectory steps 22/28/29: the AI measured how many session-cells survive alternative screens (69,744 present / 69,632 with >5 events / 68,862 with >5 running events) and concluded the ">5-event screen in `decode_position_within` is a decoder-model feature screen, not part of the source neural preprocessing" (script docstring). It also states "I'll also retain stationary frames so the 40 contiguous trials cover the full recordings." The data as distributed are already the authors' manually curated final product, so no further curation was deemed warranted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely the `trace` field: the authors' rise-extracted, binarized calcium event vectors, shape `(n_days, n_cells, n_frames)` in the joblib files. No use of `SFPs`, `centroids` or `maps`.

ii.
```python
traces = np.asarray(source["trace"])
...
day_trace = traces[day]
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
```

iii. README: "**trace**: rise-extracted calcium traces, where '1' indicates a significant event"; Methods: "This binary vector was treated as the firing rate in all further analyses." The AI's docstring states it "deliberately does not redo calcium extraction" because the supplied traces are already the paper's final signal.

## 2-b. How is the `neural` data processed?

i. Essentially no processing: select the rows (neurons) registered on that day, cast to `float32`, keep the native `(n_neurons, n_timepoints)` orientation (joblib is already neurons × time, so no transpose needed), and verify the values are still strictly 0/1. No smoothing, deconvolution, z-scoring, normalization or rebinning is applied.

ii.
```python
# The source is binary, but float32 avoids thousands of downstream
# dtype warnings and is the trainer's native representation.
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
    raise ValueError(f"Non-binary rise-extracted trace in {animal}, day {day}")
...
"neural_representation": (
    "binary rising phase of calcium transients supplied by the authors; "
    "thresholded at 2.5 noise-standard-deviations in the source pipeline"),
```

iii. Trajectory step 9: "The source files already contain the paper's final rise-extracted binary calcium events and frame-aligned position, so I won't re-derive signals from raw video." The paper treats the binary rising-phase vector as the firing rate, so reproducing the paper's processing means passing it through unchanged; `float32` is chosen only for the trainer's dtype expectations.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cross-day registration filtering: a neuron is kept for a day if its whole trace on that day is finite. Cells not registered on a day are all-NaN and are dropped, so the neuron count varies per session (113–564, mean 337; total 69,744 session-neurons). The code first checks that "finite" is all-or-nothing per neuron (no partially-NaN traces). No activity, place-field, SNR or running-speed screen is applied.

ii.
```python
finite_any = np.any(np.isfinite(day_trace), axis=1)
finite_all = np.all(np.isfinite(day_trace), axis=1)
if not np.array_equal(finite_any, finite_all):
    raise ValueError(f"Partially finite neuron trace in {animal}, day {day}")
present = finite_all
...
brain_region_idx.append(np.zeros(int(present.sum()), dtype=np.int64))
...
total_session_neurons = sum(len(x) for x in brain_region_idx)
if total_session_neurons != 69_744:
    raise ValueError("Registered-neuron total differs from the paper's 69,744 rate maps: ...")
```

iii. README: "If cell is not registered on given day, will appear as nan the same shape." Script docstring: "Across the supplied data this recovers all 69,744 curated session-specific rate maps reported in the paper. No additional place-cell or activity screen is imposed: the >5-event screen in `decode_position_within` is a decoder-model feature screen, not part of the source neural preprocessing." Trajectory step 29 records the explicit comparison with the alternative screens (69,632 / 68,862) before choosing to keep all registered cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the recording is continuous free foraging. The AI defines the trial-onset (the start of each contiguous ~1-minute window) as the alignment event and reports `off_start = 0.0`, `off_end = 60.0` s in metadata. Neural and behavioral streams are cut with the identical `array_split` boundaries, so within a trial frame *t* of `neural`, `output` refers to the same acquisition frame.

ii.
```python
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
...
"temporal_alignment_event": "start of each contiguous nominal one-minute window",
"off_start": 0.0,
"off_end": 60.0,
"trialization": ("each approximately 40-minute session split into 40 contiguous, "
                 "nearly equal windows; all source frames retained"),
```

iii. The paradigm has no trial structure or stimulus onset, so the only meaningful alignment is the artificial window onset; the AI still fills the required metadata fields rather than leaving them empty. The DAQ "simultaneously acquired behavioral and cellular imaging streams at 30 Hz ... timestamped for post-hoc alignment" (Methods), so the supplied streams are already frame-aligned and no shift is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native acquisition resolution is kept: 30 Hz, i.e. `time_bin_size = 1000/30 = 33.33 ms`. **No** rebinning, downsampling, smoothing or averaging is applied, and no frames are dropped, which is why the resulting pickle is ~18.8 GiB (69,744 session-neurons × ~72,000 frames × 4 bytes).

ii.
```python
FPS = 30.0
...
"time_bin_size": 1000.0 / FPS,
"source_sampling_rate_hz": FPS,
```

iii. Trajectory step 14: the AI explicitly flagged the scale problem ("roughly five billion neural values and an impractically large training object") and checked whether the supplied `decoder.py` does any temporal binning before deciding. Finding none required, it kept every 30 Hz frame so as not to alter the paper's signal ("the converter will still retain every complete minute and all curated cells present in each session"), and verified the trainer handled the resulting size on the available GPU.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field: a per-day MATLAB cell giving the IDs of the occluded partitions of the 3×3 grid, e.g. `[array([1., 2., 4., 5.])]`, with the sentinel `-1` for the open square. `envs` (the geometry's string name) is loaded too but only stored in `metadata['session_info']`, not used as a decoder input.

ii.
```python
blocked = source["blocked"]
...
geometry = geometry_vector(blocked[day])
...
raw = blocked_entry[0] if isinstance(blocked_entry, list) else blocked_entry
blocked = np.asarray(raw).reshape(-1).astype(np.int64)
```

iii. README: "**blocked**: location of blocked (occluded) partitions in 3x3 design of environment ... organized in the following way – [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If no partitions are blocked, value is -1." This is exactly the "which parts of the arena are blocked" information the instructions ask for as decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A static 9-element `float32` multi-hot vector per session: 1 = blocked, 0 = accessible, indexed row-major (`y*3+x`) to match the README's grid layout. The `-1` sentinel yields an all-zero vector (open square). Out-of-range IDs raise. The same vector is repeated (as 40 independent copies) for every trial of the session, with names `blocked_row_r_col_c`.

ii.
```python
def geometry_vector(blocked_entry: object) -> np.ndarray:
    """Return the source geometry as 1=blocked, 0=accessible."""
    raw = blocked_entry[0] if isinstance(blocked_entry, list) else blocked_entry
    blocked = np.asarray(raw).reshape(-1).astype(np.int64)
    geometry = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    blocked = blocked[blocked >= 0]  # -1 is the source sentinel for the square.
    if blocked.size:
        if blocked.max() >= geometry.size:
            raise ValueError(f"Invalid blocked-cell ID(s): {blocked.tolist()}")
        geometry[blocked] = 1.0
    return geometry
...
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
cell_names = [f"row_{row}_col_{col}" for row in range(3) for col in range(3)]
"input_names": [f"blocked_{name}" for name in cell_names],
```

iii. Docstring: "Geometry is a static nine-element float vector with 1 for blocked and 0 for accessible. The square's sentinel blocked ID (-1) produces all zeros." The geometry is fixed for a whole recording day, so a per-trial constant vector (allowed shape `(n_input,)`) carries all the contextual information. The row-major indexing was not assumed but *verified against the data* (trajectory steps 16, 20, 41): with `y*3+x`, occupancy in nominally blocked cells is ~0%, whereas `x*3+y` or axis flips put 11–66% of tracking samples inside walls.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field only: DeepLabCut head-tracking `(x, y)` in cm, shape `(n_days, 2, n_frames)`, range 0–75 cm.

ii.
```python
positions = np.asarray(source["position"])
...
session_output_full = discretize_position(positions[day])
```

iii. README: "**position**: x-y position data for all days ... x-y position in first dimension, and number of temporal bins / frames in second dimension"; Methods: "Position data were generated from tracking the head with DeepLabCut." The AI verified the global range is exactly [0, 75] on both axes and that all samples are finite (trajectory step 13).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. None beyond discretization — no smoothing, interpolation, speed filtering or re-scaling of the raw coordinates. Continuous cm coordinates are converted directly to a single categorical time series of shape `(1, n_timepoints)` at the native 30 Hz, then split into the same 40 windows as the neural data. Shape is validated (`(2, time)`).

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    """Map source (x, y) positions to one row-major 3x3 class time series."""
    if position.ndim != 2 or position.shape[0] != 2:
        raise ValueError(f"Expected position shape (2, time), got {position.shape}")
    ...
    location = xy[1] * GRID_SIZE + xy[0]
    return location[np.newaxis, :]
```

iii. The positions supplied are already the paper's final, frame-aligned tracking output, so the AI's docstring states it "deliberately does not redo calcium extraction or interpolate either stream". The instructions require a categorical, time-varying output, so the only transformation needed is the 3×3 discretization.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Fixed 25 cm bins on the physical 75 cm arena: `bin = floor(coord / 25)`, clipped to `[0, 2]` on each axis (so the boundary value of exactly 75.0 cm falls into the last bin rather than a spurious fourth bin), then flattened row-major as `class = y_bin*3 + x_bin`, giving 9 classes named `row_0_col_0 … row_2_col_2`. Bins are absolute (arena-referenced), not quantiles of occupancy, so class frequencies are unequal (0.057–0.201 of samples).

ii.
```python
ARENA_SIZE_CM = 75.0
GRID_SIZE = 3
...
bin_width = ARENA_SIZE_CM / GRID_SIZE
xy = np.floor(position / bin_width).astype(np.int64)
# Source values can be exactly 75 cm; those belong to the last, not a fourth,
# spatial bin.  Clipping also protects against tiny tracking overshoots.
xy = np.clip(xy, 0, GRID_SIZE - 1)
location = xy[1] * GRID_SIZE + xy[0]
...
"output_names": ["mouse_position_3x3"],
"output_values": [cell_names],
"position_binning": "fixed 25 cm bins over the 75x75 cm arena; class = y_bin*3+x_bin",
```

iii. The paper "partitioned an open square (75 × 75 cm) into a 3 × 3 grid space", and the blocked-partition IDs are defined on exactly that grid, so using the same 25 cm partition makes the output classes and the geometry input share one coordinate system. The row-major convention was chosen empirically (steps 16/20/41) because it is the only flattening under which the animal essentially never appears inside a declared wall.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, with no shift or resampling: both streams come from the same 30 Hz DAQ and have identical frame counts, which the code asserts per animal, and both are cut by the same `np.array_split(..., 40)` boundaries, so trial *k* of `output` and trial *k* of `neural` have equal `n_timepoints`. An additional consistency audit counts position samples that land in blocked cells (0 for nearly all sessions; max 0.59%) and stores that count per session instead of editing frames.

ii.
```python
if traces.shape[0] != positions.shape[0] or traces.shape[2] != positions.shape[2]:
    raise ValueError(f"Neural/position alignment mismatch for {animal}")
...
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
...
blocked_position_samples = int(np.count_nonzero(geometry[session_output_full[0]] != 0))
```

iii. Methods: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz ... and all recorded frames were timestamped for post-hoc alignment", so the delivered arrays are already aligned; the AI's docstring notes it does not interpolate either stream. The blocked-cell audit is used as an independent check that the alignment and coordinate convention are right (trajectory steps 38, 43): a wrong convention would put 11–19% of samples behind walls.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled explicitly:
- **Unregistered cells (NaN traces)** — dropped for that day; the code additionally verifies NaN-ness is whole-trace rather than partial, and cross-checks the surviving total against the paper's 69,744.
- **Ragged recording lengths (71,866–72,219 frames)** — absorbed by `array_split` into 40 windows rather than truncated, so no frames are lost.
- **Positions exactly at 75.0 cm / tiny tracking overshoots** — clipped into the last spatial bin instead of creating a fourth bin.
- **Tracking samples inside nominally blocked partitions** (2 frames in one animal, up to 0.59% in one "bit donut" session) — *retained unchanged* and counted per session in `metadata['session_info']['blocked_position_samples']`; only a gross error (>5%) aborts the conversion.
No NaN/Inf reaches the output (the supplied verifier reported no errors or warnings).

ii.
```python
present = finite_all
...
xy = np.clip(xy, 0, GRID_SIZE - 1)
...
# Count (but do not silently delete or relocate) source tracking
# samples in an omitted partition.  A handful lie just across a 25 cm
# boundary.  Retaining them preserves exact neural/behavioral frame
# alignment; the count makes this source-data edge case auditable.
blocked_position_samples = int(np.count_nonzero(geometry[session_output_full[0]] != 0))
if blocked_position_samples / session_output_full.shape[1] > 0.05:
    raise ValueError(...)
session_info.append({..., "blocked_position_samples": blocked_position_samples,
                     "trial_lengths": [int(x.shape[1]) for x in neural_trials]})
```

iii. Trajectory steps 35–38, 43: after the first guard fired, the AI quantified the anomaly ("only 2 of 2,228,000 frames ... both immediately beside 25 cm boundaries") and compared alternative coordinate conventions before deciding: "isolated boundary/tracking samples should be retained and reported rather than silently deleted, while a substantial fraction would indicate the coordinate convention is wrong." It relaxed the guard to 5% and kept the per-session audit so the edge case stays visible downstream.

## 6-a. What are the most time-consuming steps of the code?

i. The run is dominated by I/O and bulk memory traffic, not by computation:
1. `joblib.load` of the seven source files (~9–32 s each, measured in the trajectory) — decompressing ~5 GB of float64 arrays.
2. Writing the 18.8 GiB pickle to disk at the end.
3. The full-array passes over each day's `(n_cells, ~72,000)` trace: two `np.isfinite` reductions, the `float32` cast/copy, the 0/1 validity check, and the `np.ascontiguousarray` copy of each of the 40 chunks.
Total wall time was roughly 3 minutes; the discretization, geometry encoding and bookkeeping are negligible.

ii.
```python
wrapped = joblib.load(source_path)          # dominant per-animal cost
...
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)   # full copy
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):  # full scan
...
return [np.ascontiguousarray(x) for x in np.array_split(array, N_TRIALS, axis=axis)]  # full copy
with temporary.open("wb") as handle:
    pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)     # 18.8 GiB write
```

iii. Not discussed as an optimization target in the trajectory; the AI's only performance-related measures were per-animal `del`/`gc.collect()` to bound peak memory (step 12 measured an ~9 s load for one file) and an atomic `.tmp` + `os.replace` write.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Very little: the per-animal and per-day loops are inherent to the ragged structure (different cell sets and geometries per day) and each already operates on whole arrays. Two minor cases exist:
- `input_trials = [geometry.copy() for _ in range(N_TRIALS)]` makes 40 separate 9-element copies per session (8,280 tiny arrays) where one shared array (or a repeat) would do; negligible cost, but it is a loop that need not exist.
- `finite_any`/`finite_all` are two separate full reductions over the same array; one pass (e.g. `np.isfinite(...).sum(axis=1)` compared to 0 and n_frames) would halve that traffic.
Neither changes the runtime materially, since the cost is dominated by load/write.

ii.
```python
finite_any = np.any(np.isfinite(day_trace), axis=1)
finite_all = np.all(np.isfinite(day_trace), axis=1)
...
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
...
"trial_lengths": [int(x.shape[1]) for x in neural_trials],
```

iii. No justification is given; the AI never raised vectorization as a concern, having judged the conversion cheap relative to the decoder run.

## 6-c. What processing does the code repeat multiple times?

i. Redundant repeated work, all of it cheap-but-non-zero full passes over the largest arrays:
- `np.isfinite(day_trace)` is computed **twice** per session (once for `any`, once for `all`).
- The 0/1 validity check `np.all((x == 0) | (x == 1))` re-scans the whole (already-copied) session trace, creating two temporary boolean arrays the size of the data.
- The day trace is materialized at least twice: the `float32` fancy-index copy, then again as 40 `ascontiguousarray` chunks (`array_split` views would have sufficed for pickling).
- `geometry` is re-copied 40 times per session, and `present.sum()` is computed twice (once for `brain_region_idx`, once for `session_info`).

ii.
```python
finite_any = np.any(np.isfinite(day_trace), axis=1)
finite_all = np.all(np.isfinite(day_trace), axis=1)
...
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
...
brain_region_idx.append(np.zeros(int(present.sum()), dtype=np.int64))
session_info.append({..., "n_neurons": int(present.sum()), ...})
```

iii. These repeats exist because the AI deliberately layered defensive assertions on top of the conversion ("verify that no discretized positions enter blocked cells, and enforce the paper's 69,744-neuron reference total", step 32). They are validation rather than analysis, and the AI accepted their cost.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Everything the guards and the audit compute is thrown away after the run:
- the binary-value check and the partial-NaN check over the full dataset;
- `blocked_position_samples` (a full lookup + count per session) and the 5% guard;
- the 69,744-neuron assertion;
- `session_info` bookkeeping (`trial_lengths` for all 8,280 trials, `environment`, `n_source_frames`) — useful provenance, but unused by the decoder;
- `np.ascontiguousarray` on every chunk, which duplicates the entire 18.8 GiB dataset in RAM without changing what gets pickled.
The largest "discarded" cost is arguably not in this script at all: keeping all 30 Hz frames yields sessions of ~57,000 training timepoints, of which the supplied trainer uses only 50,000 for its SVD initialization.

ii.
```python
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
    raise ValueError(...)
blocked_position_samples = int(np.count_nonzero(geometry[session_output_full[0]] != 0))
...
"trial_lengths": [int(x.shape[1]) for x in neural_trials],
...
total_session_neurons = sum(len(x) for x in brain_region_idx)
if total_session_neurons != 69_744:
    raise ValueError(...)
```

iii. The AI justified the audits as protection against the one genuinely ambiguous choice in the conversion — the x/y flattening convention — and against silent data corruption: "Gross convention mistakes place tens of percent of samples behind walls. A 5% guard catches those" (code comment), and "the count makes this source-data edge case auditable". The neuron-count assertion is framed as a reproducibility check against the paper (step 29).
