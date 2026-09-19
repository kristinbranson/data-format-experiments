# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the raw `.mat` files. It hard-codes the seven animal IDs, then loads one joblib file per animal from `/app/data/<animal>`. From each loaded dictionary it reads `envs`, `trace`, `position`, and `blocked`, then iterates over the first axis of `trace`/`position` to process all day-sessions and later split them into trials.

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

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    envs = flatten_envs(dat["envs"])
    trace = np.asarray(dat["trace"], dtype=np.float32)
    position = np.asarray(dat["position"], dtype=np.float32)
```

iii. In the trajectory, the AI explicitly said that "the repo exposes exactly the fields we need: binary rise-event traces, per-day positions, environment labels, and 3x3 geometry masks." In its later notes it justified the choice as using the provided `data/<animal>` structures directly while preserving the paper/code preprocessing.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hard-coded animal IDs in `ANIMALS`. Each loaded animal file becomes one subject, and `subject_id` is assigned by the order in that list.

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

for subject_id, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(subject_id)

data = {
    ...
    "subjects": list(ANIMALS),
    "subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. The trajectory says the AI treated "the seven animal IDs" as the subjects and later reported them explicitly in `CONVERSION_NOTES.md`. It justified this as matching the paper's per-animal organization.

## 1-c. How are the data split into sessions?

i. The AI treats each day in the loaded `trace` and `position` arrays as one session. It iterates over `range(trace.shape[0])`, where the first dimension is assumed to be recording day/session.

ii.
```python
if trace.ndim != 3:
    raise ValueError(f"{animal}: trace shape should be (n_days, n_cells, T), got {trace.shape}")
if position.ndim != 3:
    raise ValueError(f"{animal}: position shape should be (n_days, 2, T), got {position.shape}")

for day in range(trace.shape[0]):
    ...
    neural.append(session_trials_neural)
    decoder_input.append(session_trials_input)
    decoder_output.append(session_trials_output)
```

iii. The AI stated in the trajectory that it would "preserve the paper’s day/session structure," and later wrote that it treated "each original recording day as one decoder session."

## 1-d. How are the data split into trials?

i. Within each session/day, the AI splits the continuous recording into contiguous non-overlapping 60 second windows. At 30 Hz, each trial is 1800 frames. Any trailing remainder shorter than 1800 frames is dropped.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)

n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
dropped = T - n_trials * TRIAL_FRAMES

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The trajectory says the AI added "one explicit adaptation for this task: contiguous 60 s trial windows," and later described the source recordings as "continuous long sessions rather than pre-segmented trials."

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter based on behavior or signal quality. Trials are only constrained indirectly: incomplete trailing windows are discarded, and a session must have at least two full 60 second trials or the script raises an error.

ii.
```python
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
dropped = T - n_trials * TRIAL_FRAMES
stats.total_frames_dropped += int(dropped)
```

iii. The trajectory does not mention any trial-quality curation beyond the task requirement to split into 1-minute trials. The closest explicit justification is that the AI wanted contiguous full 60 second windows and noted that incomplete remainders were discarded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays are derived from the `trace` field in each animal's loaded joblib dictionary.

ii.
```python
dat = load_animal(data_dir, animal)
trace = np.asarray(dat["trace"], dtype=np.float32)
...
day_trace = trace[day]
```

iii. The trajectory repeatedly describes these as the paper's "binary rise-event traces" and says the repo already exposes the needed neural field directly.

## 2-b. How is the `neural` data processed?

i. The AI largely uses the neural signal as-is. It selects the neurons registered on the current day, casts to `float32`, preserves the `(neurons, time)` orientation already present in the loaded joblib data, and slices that array into trial windows. It does not smooth, deconvolve, threshold again, or rebin in time.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
...
registered_today = ~np.isnan(day_trace[:, 0])
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
...
session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. The AI explicitly justified this in both the trajectory and its notes: it said it would use the "binary rise-event traces" directly, and wrote that it "did not re-deconvolve, smooth, or re-threshold the calcium traces" because the paper analyses already use the binarized rising phase of calcium transients.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neural data by keeping only neurons that are registered on the current day. It infers registration from whether the neuron's first frame is non-NaN, and then enforces that registered neurons contain no NaNs while unregistered neurons are entirely NaN.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The trajectory says the implementation would "drop only neurons that are unregistered for a given day," and the conversion notes justify this as removing per-day all-NaN/unregistered cells so the exported decoder arrays contain no NaNs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological or task event alignment. The AI uses an artificial alignment event: the start of each contiguous 60 second trial window within a continuous session.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
    ...
}
```

iii. The trajectory frames trialization as a decoder-task adaptation layered on top of continuous sessions. Its notes explicitly call the recordings "continuous long sessions rather than pre-segmented trials."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz sampling rate, so each time bin is `1000 / 30` ms. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"metadata": {
    ...
    "time_bin_size": TIME_BIN_MS,
    "fps": FPS,
```

iii. The trajectory consistently ties trial length to "30 Hz sampling" and 1800-frame minute-long windows. It does not mention any additional temporal aggregation.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment input is derived primarily from the raw `blocked` field. The code also reads `envs`, but only for bookkeeping and sanity summaries; it does not derive the actual decoder input from `env_name`.

ii.
```python
envs = flatten_envs(dat["envs"])
...
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_name = str(envs[day])
env_mask = mask_from_blocked(blocked_entry)
```

iii. The AI's later notes are explicit: "I derived this from the `blocked` metadata, not only from the environment name," because it believed `blocked` was the "authoritative geometry/orientation information" for the decoder task.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI normalizes the `blocked` entry, converts it into a `3 x 3` mask where `1` means open and `0` means blocked, flattens that mask to a length-9 vector, and reuses the same static vector for every trial in the session.

ii.
```python
def normalize_blocked_entry(entry) -> tuple[int, ...]:
    ...
    if arr.size == 1 and arr[0] == -1:
        return ()
    return tuple(sorted(arr.tolist()))

def mask_from_blocked(blocked_bins: tuple[int, ...]) -> np.ndarray:
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask

...
input_vec = env_mask.reshape(-1).astype(np.float32)
...
session_trials_input.append(input_vec.copy())
```

iii. The trajectory justifies this as a decoder-specific adaptation: a static `3 x 3` geometry input per trial, using `blocked` rather than `env` because environment names alone were considered insufficiently specific.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw `position` field in each animal dictionary.

ii.
```python
position = np.asarray(dat["position"], dtype=np.float32)
...
day_pos = position[day]
```

iii. The trajectory describes the repo as exposing "per-day positions" directly and later states that the decoder output is built from those position traces.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first bins the continuous 2D positions into a `3 x 3` grid using each session's per-axis maximum coordinate rather than a fixed 75 cm arena edge. It then projects any bins that land in blocked partitions to the nearest valid open bin using a nearest-neighbor lookup over the session's geometry mask.

ii.
```python
def build_nearest_valid_lookup(env_mask: np.ndarray) -> np.ndarray:
    valid = np.argwhere(env_mask > 0)
    lookup = np.zeros((3, 3, 2), dtype=np.int64)
    for x in range(3):
        for y in range(3):
            if env_mask[x, y] > 0:
                lookup[x, y] = np.array([x, y], dtype=np.int64)
                continue
            dists = np.sum((valid - np.array([x, y])) ** 2, axis=1)
            lookup[x, y] = valid[np.argmin(dists)]
    return lookup

def bin_position_to_grid(position_xy: np.ndarray, n_bins: int = 3) -> np.ndarray:
    max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
    bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)

...
binned_xy = bin_position_to_grid(day_pos, n_bins=3)
lookup = build_nearest_valid_lookup(env_mask)
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. The trajectory and conversion notes give two explicit justifications. First, the AI said it was matching the paper code's "max-based spatial scaling" used before rate-map construction. Second, it called the nearest-open-bin reassignment a decoder-task adaptation "in the spirit" of geometry-aware cleanup in the paper's decoder analysis.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. After obtaining `x` and `y` grid bins in `0..2`, the AI encodes the categorical label as `x_bin * 3 + y_bin`, producing nine classes `0..8`. It names those classes `x0_y0` through `x2_y2`.

ii.
```python
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)

data = {
    ...
    "output_names": ["position_bin"],
    "output_values": [[f"x{x}_y{y}" for x in range(3) for y in range(3)]],
    ...
}
```

iii. The AI explicitly documented this in its notes: "Label definition is `x_bin * 3 + y_bin`." The trajectory does not show an additional deeper justification beyond making the requested 9-way categorical position output.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position output is aligned frame-for-frame with the neural trace within each session/day, and both are split into trials using the same `start:stop` frame boundaries. After discretization, the output is stored as shape `(1, time)` for each trial.

ii.
```python
day_trace = trace[day]
day_pos = position[day]
...
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The trajectory treats both `trace` and `position` as synchronized 30 Hz per-day streams from the same source and describes the trialization as a common 60 second windowing step applied to the session.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several irregularities explicitly. It normalizes nested `blocked` entries and treats `-1` as no blocked bins; it raises errors if position contains NaNs or if the registered/unregistered neuron NaN pattern is inconsistent; it drops incomplete trailing trial fragments; and it reassigns coarse position bins that fall into blocked partitions to the nearest valid open bin.

ii.
```python
arr = np.asarray(entry, dtype=float).reshape(-1)
arr = arr[~np.isnan(arr)]
if arr.size == 0:
    return ()
...
if arr.size == 1 and arr[0] == -1:
    return ()

if np.isnan(day_pos).any():
    raise ValueError(f"{animal} day {day}: position contains NaNs")

if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(f"{animal} day {day}: registered neurons contain NaNs")
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(f"{animal} day {day}: unregistered neurons are not consistently NaN")

dropped = T - n_trials * TRIAL_FRAMES

reassigned = np.any(projected_xy != binned_xy, axis=0)
stats.total_frames_reassigned += int(np.sum(reassigned))
```

iii. The trajectory justifies the NaN handling by saying unregistered neurons should be dropped per day. It justifies the blocked-bin reassignment as a geometry-aware cleanup step for the decoder task. It does not give an explicit justification for failing hard on position NaNs, other than checking dataset consistency.

## 6-a. What are the most time-consuming steps of the code?

i. The code is likely dominated by repeatedly loading large per-animal joblib files, traversing every session/day, binning every position frame, and copying trial slices into Python lists. The later summary pass over all output trials also touches the full dataset again.

ii.
```python
def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    ...
    for day in range(trace.shape[0]):
        ...
        binned_xy = bin_position_to_grid(day_pos, n_bins=3)
        lookup = build_nearest_valid_lookup(env_mask)
        ...
        for trial_idx in range(n_trials):
            ...
            session_trials_neural.append(session_neural[:, start:stop].copy())
            session_trials_output.append(output_position[np.newaxis, start:stop].copy())

def summarize_dataset(data: dict) -> dict:
    ...
    for session in data["output"]:
        for trial in session:
            vals, counts = np.unique(trial[0], return_counts=True)
```

iii. The trajectory does not explicitly discuss runtime hotspots. The closest evidence is that the AI added `gc.collect()`, saved extensive sanity statistics, and ran full-dataset verification after conversion, implying it expected dataset-scale I/O and full-array processing to matter.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main avoidable Python loops are the nested `x/y` loop used to build the nearest-valid-bin lookup, the per-trial slicing loop that appends copies one trial at a time, and the repeated scan over `subject_idx` when constructing the sample subset. These work correctly, but they are not especially vectorized.

ii.
```python
for x in range(3):
    for y in range(3):
        if env_mask[x, y] > 0:
            lookup[x, y] = np.array([x, y], dtype=np.int64)
            continue
        dists = np.sum((valid - np.array([x, y])) ** 2, axis=1)
        lookup[x, y] = valid[np.argmin(dists)]

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
    session_trials_input.append(input_vec.copy())
    session_trials_output.append(output_position[np.newaxis, start:stop].copy())

for subject_id in range(min(args.sample_subject_count, len(ANIMALS))):
    sessions_for_subject = []
    for sess_idx, subj in enumerate(data["subject_idx"]):
        if int(subj) == subject_id:
            sessions_for_subject.append(sess_idx)
```

iii. The trajectory does not contain an explicit justification for keeping these loops. The code and notes suggest the AI prioritized transparent, session-oriented logic over optimization.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several kinds of processing: it reconstructs a nearest-valid lookup for every session even though there are only a small number of unique geometries; it copies the same static input vector once per trial; it scans `subject_idx` again to build the sample subset; and it recomputes dataset summaries after already collecting many statistics during conversion.

ii.
```python
lookup = build_nearest_valid_lookup(env_mask)
...
session_trials_input.append(input_vec.copy())
...
for subject_id in range(min(args.sample_subject_count, len(ANIMALS))):
    sessions_for_subject = []
    for sess_idx, subj in enumerate(data["subject_idx"]):
        if int(subj) == subject_id:
            sessions_for_subject.append(sess_idx)
...
"summary": summarize_dataset(data),
...
print(json.dumps(summarize_dataset(sample_data), indent=2))
```

iii. There is no explicit trajectory justification for these repeated computations. The closest implicit rationale is that the AI wanted extensive sanity checking and a second sample dataset in addition to the required full dataset.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several parts of the script are not needed for downstream decoder training on the required full dataset: it builds a sample dataset and sample pickle, tracks environment and frame-length statistics, compares observed counts to expected reference counts, stores large metadata summaries, performs explicit garbage collection, and defines an `ENV_TO_MASK` constant that is never used in the conversion path.

ii.
```python
ENV_TO_MASK = {
    "square": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
    ...
}

env_counts = Counter()
blocked_patterns_by_env: dict[str, set[tuple[int, ...]]] = {}
frame_lengths = Counter()
dropped_seconds_by_session = []

"expected_reference_counts": {
    "sessions": EXPECTED_TOTAL_SESSIONS,
    "unique_neurons": EXPECTED_TOTAL_UNIQUE_NEURONS,
    "rate_maps": EXPECTED_TOTAL_RATE_MAPS,
},
"observed_reference_counts": {
    "sessions": stats.total_sessions,
    "unique_neurons": stats.total_unique_neurons,
    "rate_maps": stats.total_session_neurons,
},

sample_data = subset_sessions(data, sample_session_indices)
with open(args.sample_out, "wb") as f:
    pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory indicates these were added as sanity checks and deliverables for the agent's own validation workflow, not because the downstream decoder required them. There is no evidence that the extra sample dataset or most metadata fields were needed by the target format.
