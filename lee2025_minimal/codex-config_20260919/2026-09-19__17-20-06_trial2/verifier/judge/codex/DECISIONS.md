# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the HDF5 `.mat` files directly. Instead, it hard-codes the seven animal IDs, loads the sidecar `joblib` file for each animal from `/app/data/<animal>`, and reads `trace`, `position`, `envs`, and `blocked` from the loaded dictionary. Trials are not stored explicitly in the source; they are created later by splitting each session.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]
...
for animal_index, animal in enumerate(ANIMALS):
    source_path = data_dir / animal
    wrapped = joblib.load(source_path)
    source = wrapped[animal]

    traces = np.asarray(source["trace"])
    positions = np.asarray(source["position"])
    envs = np.asarray(source["envs"]).reshape(-1)
    blocked = source["blocked"]
```

iii. In the trajectory, the AI said the "source files already contain the paper's final rise-extracted binary calcium events and frame-aligned position" (step 9), so it chose the prepackaged `joblib` objects rather than reloading the `.mat` files. It also treated the data as 207 daily sessions across the seven hard-coded animals (steps 14, 32, 49).

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. Each list entry becomes one subject, and `subject_idx` stores the index of that animal for every session appended from that subject.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]
...
for animal_index, animal in enumerate(ANIMALS):
    ...
    for day in range(traces.shape[0]):
        ...
        subject_idx.append(animal_index)
```

iii. The trajectory repeatedly refers to "7 subjects" and processes fixed animal names rather than discovering subjects from the filesystem (steps 32, 49, 54).

## 1-c. How are the data split into sessions?

i. Each day / recording index along the first axis of the `trace`, `position`, `envs`, and `blocked` arrays is treated as one session. The AI calls these "recording day" sessions.

ii.
```python
traces = np.asarray(source["trace"])
positions = np.asarray(source["position"])
envs = np.asarray(source["envs"]).reshape(-1)
blocked = source["blocked"]
...
for day in range(traces.shape[0]):
    day_trace = traces[day]
    ...
    session_info.append(
        {
            "subject": animal,
            "source_day_index": day,
            "environment": str(envs[day]),
```

iii. The AI explicitly described the files as containing "207 daily sessions" and said it would preserve those session boundaries (steps 14, 32, 49). It also wrote `"session_definition": "one recording day in one environment"` into metadata.

## 1-d. How are the data split into trials?

i. Trials are created by splitting every session into exactly 40 contiguous, nearly equal segments with `np.array_split`. This keeps every source frame, so trial lengths vary slightly instead of being exactly 1800 frames. The code treats these as the "nominal 40 one-minute trials."

ii.
```python
FPS = 30.0
N_TRIALS = 40
...
def split_trials(array: np.ndarray, axis: int) -> list[np.ndarray]:
    """Split a full recording into the nominal 40 contiguous minute windows."""
    return [np.ascontiguousarray(x) for x in np.array_split(array, N_TRIALS, axis=axis)]
...
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
```

iii. The trajectory gives the rationale directly: session lengths vary slightly around 40 minutes, so the AI chose `array_split` to "preserve every aligned frame" and avoid either discarding almost a minute or creating a short extra trial; it noted resulting trial lengths of 1,796-1,806 frames (steps 25 and 54).

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter or drop individual trials after splitting. Every one of the 40 per-session trial windows is kept. The only checks are session-level assertions, such as rejecting partially finite neuron traces or sessions with too many samples in blocked bins.

ii.
```python
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
...
if not np.array_equal(finite_any, finite_all):
    raise ValueError(f"Partially finite neuron trace in {animal}, day {day}")
...
if blocked_position_samples / session_output_full.shape[1] > 0.05:
    raise ValueError(
        f"More than 5% of positions enter blocked bins in {animal}, day {day}; "
        "check the x/y convention"
    )
```

iii. The AI said it would "retain stationary frames so the 40 contiguous trials cover the full recordings" and keep all curated cells rather than apply an extra activity screen (step 29). It did not describe any trial-level curation step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` array loaded from each animal's `joblib` dictionary.

ii.
```python
source = wrapped[animal]
traces = np.asarray(source["trace"])
...
day_trace = traces[day]
```

iii. The trajectory says the source files already contain the paper's final neural representation, so the AI did not attempt to derive neural data from other variables or from video (step 9).

## 2-b. How is the `neural` data processed?

i. For each day/session, the AI selects only neurons with fully finite traces, casts the resulting session matrix to `float32`, checks that the values are binary `0/1`, and leaves the 30 Hz frames unchanged. It does not smooth, rebin, deconvolve, or otherwise transform the traces before trial splitting.

ii.
```python
day_trace = traces[day]
finite_any = np.any(np.isfinite(day_trace), axis=1)
finite_all = np.all(np.isfinite(day_trace), axis=1)
...
present = finite_all
...
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
if not np.all((session_neural_full == 0) | (session_neural_full == 1)):
    raise ValueError(f"Non-binary rise-extracted trace in {animal}, day {day}")
...
neural_trials = split_trials(session_neural_full, axis=1)
```

iii. The AI justified this by saying the files "already contain the paper's final rise-extracted binary calcium events" and therefore it would not redo calcium extraction (step 9). It also said keeping all finite registered cells exactly reproduced the paper's 69,744 session-specific neurons (step 29).

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is kept only if its entire day/session trace is finite. All-NaN rows are treated as registered cells absent on that day and removed. The AI also checks for the unexpected case of a neuron being partly finite and partly non-finite, and raises an error if that happens. It explicitly does not apply the paper code's downstream ">5 event" feature screen.

ii.
```python
finite_any = np.any(np.isfinite(day_trace), axis=1)
finite_all = np.all(np.isfinite(day_trace), axis=1)
if not np.array_equal(finite_any, finite_all):
    raise ValueError(f"Partially finite neuron trace in {animal}, day {day}")
present = finite_all
if not np.any(present):
    raise ValueError(f"No registered neurons in {animal}, day {day}")
...
session_neural_full = np.asarray(day_trace[present], dtype=np.float32)
...
if total_session_neurons != 69_744:
    raise ValueError(
        "Registered-neuron total differs from the paper's 69,744 rate maps: "
        f"got {total_session_neurons}"
    )
```

iii. In step 29, the AI said the 69,744-neuron total was a "strong reference check" and that it would keep all finite registered cells per day. It also explicitly rejected the ">5 running events" filter as a decoder-model feature screen rather than source preprocessing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological or stimulus event alignment. The AI keeps the continuous 30 Hz frames as-is and defines alignment implicitly by the start of each contiguous trial window. Its metadata names the alignment event as the "start of each contiguous nominal one-minute window."

ii.
```python
neural_trials = split_trials(session_neural_full, axis=1)
...
"metadata": {
    ...
    "temporal_alignment_event": "start of each contiguous nominal one-minute window",
    "off_start": 0.0,
    "off_end": 60.0,
```

iii. The trajectory emphasizes that the recordings are continuous and already frame-aligned with position (step 9), then explains that trials are contiguous windows carved from those sessions (step 25). The alignment event was therefore introduced by the conversion rather than taken from the experiment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keeps the native 30 Hz sampling. The time bin size is `1000 / 30 = 33.33...` ms, and the AI does not apply any temporal rebinning or resampling.

ii.
```python
FPS = 30.0
...
"metadata": {
    ...
    "time_bin_size": 1000.0 / FPS,
    ...
    "source_sampling_rate_hz": FPS,
```

iii. The AI said it would preserve the paper's "30 Hz frame alignment" and keep all frames rather than temporally coarsen them (steps 9, 14, 25).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the raw `blocked` entry for each day/session.

ii.
```python
blocked = source["blocked"]
...
geometry = geometry_vector(blocked[day])
```

iii. The trajectory explains that the geometry convention was resolved by comparing blocked-cell IDs against the observed positions, so the AI chose to treat `blocked` as the source of the geometry input (step 20).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts each session's `blocked` entry into a 9-element static row-major vector with `1` for blocked cells and `0` for accessible cells. The special value `-1` means the square environment and produces an all-zero vector. That vector is copied for every trial in the session.

ii.
```python
def geometry_vector(blocked_entry: object) -> np.ndarray:
    raw = blocked_entry[0] if isinstance(blocked_entry, list) else blocked_entry
    blocked = np.asarray(raw).reshape(-1).astype(np.int64)
    geometry = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    blocked = blocked[blocked >= 0]
    if blocked.size:
        if blocked.max() >= geometry.size:
            raise ValueError(f"Invalid blocked-cell ID(s): {blocked.tolist()}")
        geometry[blocked] = 1.0
    return geometry
...
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
```

iii. In step 20, the AI said it had resolved the geometry indexing as row-major `y_bin * 3 + x_bin` and would use that same convention for the nine geometry inputs. The top-of-file docstring also records the `-1` sentinel rule.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` mouse position is derived from the raw `position` array for each day/session.

ii.
```python
positions = np.asarray(source["position"])
...
session_output_full = discretize_position(positions[day])
```

iii. The trajectory says the source files already contain frame-aligned position, so the AI used `position` directly as the behavioral stream to discretize (step 9).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI converts `(x, y)` coordinates into one categorical time series over a 3x3 grid. It divides the 75 cm arena into 25 cm bins with `floor(position / 25)`, clips both axes to `[0, 2]`, and then encodes the spatial bin as the row-major label `y * 3 + x`.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    if position.ndim != 2 or position.shape[0] != 2:
        raise ValueError(f"Expected position shape (2, time), got {position.shape}")
    bin_width = ARENA_SIZE_CM / GRID_SIZE
    xy = np.floor(position / bin_width).astype(np.int64)
    xy = np.clip(xy, 0, GRID_SIZE - 1)
    location = xy[1] * GRID_SIZE + xy[0]
    return location[np.newaxis, :]
```

iii. The trajectory says the row-major convention was chosen because it matched the blocked-cell IDs and minimized impossible visits into blocked cells (step 20).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded by fixed 25 cm boundaries on each axis, creating three bins per axis and therefore nine categories total. Values exactly at 75 cm, or slight tracking overshoots beyond the arena, are clipped into the outermost bin instead of being dropped.

ii.
```python
bin_width = ARENA_SIZE_CM / GRID_SIZE
xy = np.floor(position / bin_width).astype(np.int64)
# Source values can be exactly 75 cm; those belong to the last, not a fourth,
# spatial bin.  Clipping also protects against tiny tracking overshoots.
xy = np.clip(xy, 0, GRID_SIZE - 1)
location = xy[1] * GRID_SIZE + xy[0]
```

iii. The AI justified the clipping in the code comments themselves: exact boundary values and small tracking overshoots should stay in the outer bin rather than create invalid categories.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes neural and position streams are already aligned frame-by-frame within each day/session. It validates equal session counts and equal time-axis lengths, then splits neural and position with the same 40-window function so they remain synchronized.

ii.
```python
if traces.shape[0] != positions.shape[0] or traces.shape[2] != positions.shape[2]:
    raise ValueError(f"Neural/position alignment mismatch for {animal}")
...
session_output_full = discretize_position(positions[day])
...
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
```

iii. In step 9, the AI said the files already contained "frame-aligned position." Step 25 then explains that the same contiguous-window segmentation is applied to preserve that alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several small-data issues defensively rather than imputing values. All-NaN neurons are removed as absent registrations, partially finite neuron traces trigger an error, exact-boundary or slight out-of-bounds positions are clipped into valid bins, and a small number of samples falling into blocked bins is tolerated and audited. The AI does not interpolate missing neural or behavioral data, and it keeps all recording frames by splitting sessions into 40 nearly equal windows.

ii.
```python
finite_any = np.any(np.isfinite(day_trace), axis=1)
finite_all = np.all(np.isfinite(day_trace), axis=1)
if not np.array_equal(finite_any, finite_all):
    raise ValueError(f"Partially finite neuron trace in {animal}, day {day}")
present = finite_all
...
xy = np.floor(position / bin_width).astype(np.int64)
xy = np.clip(xy, 0, GRID_SIZE - 1)
...
blocked_position_samples = int(
    np.count_nonzero(geometry[session_output_full[0]] != 0)
)
if blocked_position_samples / session_output_full.shape[1] > 0.05:
    raise ValueError(
        f"More than 5% of positions enter blocked bins in {animal}, day {day}; "
        "check the x/y convention"
    )
```

iii. The trajectory says the AI wanted to preserve exact neural/behavioral frame alignment and keep all frames, while still making boundary excursions and indexing mistakes auditable (steps 20, 25, 29, 32).

## 6-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading each large `joblib` animal file, materializing and storing all 207 sessions at 30 Hz, and writing the very large output pickle. The code also calls `gc.collect()` after each subject to manage memory pressure.

ii.
```python
for animal_index, animal in enumerate(ANIMALS):
    source_path = data_dir / animal
    print(f"Loading {source_path} ...", flush=True)
    wrapped = joblib.load(source_path)
    ...
    del wrapped, source, traces, positions, envs, blocked
    gc.collect()
...
with temporary.open("wb") as handle:
    pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory explicitly flagged the scale of the full conversion: 207 sessions, 69,744 session-neurons, roughly five billion neural values, and an 18.82 GiB pickle (steps 14 and 49).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most numeric work is already vectorized with NumPy. The remaining Python-level loops are mainly structural loops over animals and sessions, plus the per-trial duplication of the static geometry vector. The latter could be replaced with a broadcasted/shared representation if downstream code allowed it, but the converter keeps a Python list of 40 separate trial inputs.

ii.
```python
for animal_index, animal in enumerate(ANIMALS):
    ...
    for day in range(traces.shape[0]):
        ...
        input_trials = [geometry.copy() for _ in range(N_TRIALS)]
```

iii. The trajectory does not discuss vectorization explicitly. This is inferred from the final code structure.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same trial split separately for neural and output arrays, duplicates the same geometry vector for every trial in a session, and iterates again to record per-trial lengths into metadata. It also reloads per-animal arrays and reruns the same validation logic for every session.

ii.
```python
neural_trials = split_trials(session_neural_full, axis=1)
output_trials = split_trials(session_output_full, axis=1)
input_trials = [geometry.copy() for _ in range(N_TRIALS)]
...
"trial_lengths": [int(x.shape[1]) for x in neural_trials],
```

iii. The trajectory does not call these repetitions out explicitly; this is visible from the final implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are used only for validation or metadata, not for the decoder inputs/outputs themselves: loading `envs`, counting `blocked_position_samples`, checking the global 69,744-neuron invariant, storing verbose `session_info`, and repeatedly copying the static geometry vector for each trial. These steps make the conversion more auditable but are not required by downstream decoding.

ii.
```python
envs = np.asarray(source["envs"]).reshape(-1)
...
blocked_position_samples = int(
    np.count_nonzero(geometry[session_output_full[0]] != 0)
)
...
session_info.append(
    {
        "subject": animal,
        "source_day_index": day,
        "environment": str(envs[day]),
        "n_source_frames": int(day_trace.shape[1]),
        "n_neurons": int(present.sum()),
        "blocked_position_samples": blocked_position_samples,
        "trial_lengths": [int(x.shape[1]) for x in neural_trials],
    }
)
...
if total_session_neurons != 69_744:
    raise ValueError(...)
```

iii. The trajectory shows these as audit checks: the AI wanted to verify its neuron count against the paper and ensure the discretized positions were consistent with the geometry mapping (steps 20, 29, 32, 54).
