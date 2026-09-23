# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from pre-saved joblib files (one per subject) in the `/app/data/` directory. It hardcodes the list of 7 animal IDs (`ANIMALS`) and loads each using `joblib.load()`. Each loaded dictionary contains keys like `trace`, `position`, `blocked`, and `envs` already converted to numpy arrays.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
source_path = data_dir / animal
wrapped = joblib.load(source_path)
source = wrapped[animal]
traces = np.asarray(source["trace"])
positions = np.asarray(source["position"])
envs = np.asarray(source["envs"]).reshape(-1)
blocked = source["blocked"]
```

iii. From the trajectory: "The source files already contain the paper's final rise-extracted binary calcium events and frame-aligned position, so I won't re-derive signals from raw video." The AI explored both the `.mat` and joblib file formats and chose joblib as it contained the same data in a more convenient Python-native format.

## 1-b. How are the data split into subjects?

i. Each joblib file corresponds to one subject (animal). The AI iterates over a hardcoded list of 7 animal names and loads each file separately. The animal name serves as the subject identifier.

ii.
```python
for animal_index, animal in enumerate(ANIMALS):
    source_path = data_dir / animal
    wrapped = joblib.load(source_path)
    source = wrapped[animal]
```

iii. The AI identified the 7 animals from the paper's data organization and the README documentation describing the dataset structure.

## 1-c. How are the data split into sessions?

i. Each subject file contains multi-day recording data as 3D arrays where the first dimension indexes recording days. Each day becomes a separate session in the output. The AI iterates `for day in range(traces.shape[0])`.

ii.
```python
traces = np.asarray(source["trace"])  # shape: (n_days, n_neurons, n_timepoints)
positions = np.asarray(source["position"])  # shape: (n_days, 2, n_timepoints)
...
for day in range(traces.shape[0]):
    day_trace = traces[day]
```

iii. From the trajectory: the AI noted "207 daily sessions" across all 7 animals, consistent with the paper's reported dataset size.

## 1-d. How are the data split into trials?

i. The AI splits each ~40-minute recording session into exactly 40 contiguous, nearly equal windows using `np.array_split`. This preserves every recorded frame but produces trials with slightly varying lengths (approximately 1796-1806 frames depending on the session's total frame count).

ii.
```python
N_TRIALS = 40
...
def split_trials(array: np.ndarray, axis: int) -> list[np.ndarray]:
    """Split a full recording into the nominal 40 contiguous minute windows."""
    return [np.ascontiguousarray(x) for x in np.array_split(array, N_TRIALS, axis=axis)]
```

iii. From the trajectory: "The session arrays differ slightly around the nominal 40 minutes (for example, 71,866 versus 72,219 frames). I'll split each recording into exactly 40 contiguous, nearly equal windows with `array_split`; this preserves every aligned frame and avoids inventing a spurious 7-second '41st trial' or discarding almost a minute from shorter recordings."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All 40 trials from each session are retained.

ii. N/A

iii. The AI did not mention any trial-level filtering criteria. All trials derived from the split are included.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary rising-phase calcium event vectors (1 = significant event, 0 = no event), as described in the paper's methods.

ii.
```python
traces = np.asarray(source["trace"])
...
day_trace = traces[day]
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
```

iii. From the trajectory: "The source files already contain the paper's final rise-extracted binary calcium events." The AI verified that trace values are exclusively 0 and 1 with explicit validation code.

## 2-b. How is the `neural` data processed?

i. The only processing is selecting neurons present on that day (finite traces), casting to float32, and validating that values are binary (0 or 1). No additional smoothing, deconvolution, or normalization is applied. The data is already in (neurons, timepoints) format from the source.

ii.
```python
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
    raise ValueError(f"Non-binary rise-extracted trace in {animal}, day {day}")
```

iii. From the trajectory: "Neural values are the supplied binary rising-phase event vectors, stored as float32 as required by the downstream trainer. Frames are retained at the simultaneous acquisition rate of 30 Hz."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are not registered on a given day (all-NaN traces) are removed. The AI checks that each neuron's trace is entirely finite for the day, and validates that there are no partially-finite neurons (i.e., a neuron is either fully present or fully NaN).

ii.
```python
finite_any = np.any(np.isfinite(day_trace), axis=1)
finite_all = np.all(np.isfinite(day_trace), axis=1)
if not np.array_equal(finite_any, finite_all):
    raise ValueError(f"Partially finite neuron trace in {animal}, day {day}")
present = finite_all
```

iii. From the trajectory: "A neuron is included in a day only when its trace is finite for the whole day. NaN traces denote a registered cell absent on that day." The AI also verified that this recovers all 69,744 curated session-specific rate maps reported in the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous, and trials are contiguous windows of the full session. Each trial is aligned to its own start.

ii. N/A — trials are simply contiguous segments of the full recording.

iii. From the metadata: `"temporal_alignment_event": "start of each contiguous nominal one-minute window"`. There is no stimulus event to align to in this freely-exploring paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native 30 Hz frame rate (~33.33 ms per bin). No temporal rebinning is applied.

ii.
```python
FPS = 30.0
...
"time_bin_size": 1000.0 / FPS,  # ~33.33 ms
```

iii. From the trajectory: "Frames are retained at the simultaneous acquisition rate of 30 Hz."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The geometry input is derived from the `blocked` variable in the source data, which contains indices of blocked positions in the 3x3 grid for each recording session.

ii.
```python
blocked = source["blocked"]
...
geometry = geometry_vector(blocked[day])
```

iii. From the trajectory: "Geometry is a static nine-element float vector with 1 for blocked and 0 for accessible. The square's sentinel blocked ID (-1) produces all zeros."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary vector (1=blocked, 0=accessible). The sentinel value -1 (indicating no blocked positions, i.e., the square environment) results in an all-zeros vector. The vector is static per trial and replicated across all 40 trials in a session.

ii.
```python
def geometry_vector(blocked_entry: object) -> np.ndarray:
    raw = blocked_entry[0] if isinstance(blocked_entry, list) else blocked_entry
    blocked = np.asarray(raw).reshape(-1).astype(np.int64)
    geometry = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    blocked = blocked[blocked >= 0]  # -1 is sentinel for the square
    if blocked.size:
        if blocked.max() >= geometry.size:
            raise ValueError(f"Invalid blocked-cell ID(s): {blocked.tolist()}")
        geometry[blocked] = 1.0
    return geometry
...
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
```

iii. The AI verified the blocked indices convention against the actual position data to ensure consistency.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the `position` variable, which contains 2D (x, y) coordinates from DeepLabCut pose estimation at each timepoint.

ii.
```python
positions = np.asarray(source["position"])  # shape: (n_days, 2, n_timepoints)
...
session_output_full = discretize_position(positions[day])
```

iii. From the methods: "Position data were generated from tracking the head with DeepLabCut pose-estimation software."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous (x, y) position is discretized into a 3x3 grid (9 categories). Each axis is divided into 3 bins of 25 cm width (75 cm / 3). Values at the boundaries (exactly 75 cm) are clipped to the last bin.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    bin_width = ARENA_SIZE_CM / GRID_SIZE  # 25 cm
    xy = np.floor(position / bin_width).astype(np.int64)
    xy = np.clip(xy, 0, GRID_SIZE - 1)
    location = xy[1] * GRID_SIZE + xy[0]  # row-major: y * 3 + x
    return location[np.newaxis, :]
```

iii. From the trajectory: "Position uses the physical 75 cm arena grid: x and y are clipped to bins 0--2 after division by 25 cm. The categorical ID is row-major `y * 3 + x`. This is the convention of the supplied `blocked` IDs."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized using `np.floor(position / 25)` to assign each coordinate to one of 3 bins (0, 1, 2). Values are clipped to [0, 2]. The 2D bins are combined into a single category via `y_bin * 3 + x_bin` (row-major order), producing 9 categories (0-8).

ii.
```python
bin_width = ARENA_SIZE_CM / GRID_SIZE  # 25.0
xy = np.floor(position / bin_width).astype(np.int64)
xy = np.clip(xy, 0, GRID_SIZE - 1)
location = xy[1] * GRID_SIZE + xy[0]
```

iii. The AI verified this convention by checking that positions don't land in blocked cells (>95% compliance), confirming the row-major `y*3+x` convention matches the `blocked` indices.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are frame-aligned in the source data (both recorded at 30 Hz). They are split into trials using the same `array_split` call on the same axis, preserving alignment.

ii.
```python
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
```

iii. From the trajectory: the AI verified alignment between neural and position arrays before splitting.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with NaN traces (not registered on that day) are removed. A small number of position samples that land in nominally blocked cells (<1% in the worst case) are retained to preserve frame alignment; the AI audits and reports these rather than silently dropping them. The AI enforces that the total session-neuron count matches the paper's 69,744.

ii.
```python
blocked_position_samples = int(
    np.count_nonzero(geometry[session_output_full[0]] != 0)
)
if blocked_position_samples / session_output_full.shape[1] > 0.05:
    raise ValueError(...)
...
if total_session_neurons != 69_744:
    raise ValueError(...)
```

iii. From the trajectory: "A 'bit donut' session has 0.59% of frames in nominal blocked cells... This confirms the convention and indicates a source tracking/arena-boundary effect. I've kept the per-session audit count and relaxed only the gross-error guard to 5%; no frames are altered or dropped."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the joblib files via `joblib.load()`, which involves decompressing and deserializing large numpy arrays for each animal. The AI observed load times of 16-28 seconds per animal.

ii. N/A

iii. From the trajectory: load+calc times ranged from ~17-28 seconds per animal, dominated by I/O.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop iterates over days within each animal, but each day requires different neuron subsets, making full vectorization impractical. The `split_trials` function uses `np.array_split` which is already vectorized. No significant vectorization opportunities exist.

ii. N/A

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat significant processing. Each animal is loaded once, each day processed once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI validates that neural traces are binary (0 or 1) and counts positions that fall in blocked cells. These validation checks are defensive and don't affect the output data, but they are not strictly necessary for the conversion. The `session_info` metadata tracks per-session statistics that the downstream decoder does not use.

ii.
```python
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
    raise ValueError(...)
blocked_position_samples = int(
    np.count_nonzero(geometry[session_output_full[0]] != 0)
)
```

iii. The AI chose to include these validations as safety checks to catch convention errors early.
