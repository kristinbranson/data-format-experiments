# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses a fixed list of seven animal IDs and loads each extensionless per-animal joblib file. From the nested animal dictionary it reads `envs`, `trace`, `position`, and later `blocked`; it processes every day of every listed animal.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]

for subject_id, animal in enumerate(ANIMALS):
    dat = load_animal(data_dir, animal)
    envs = flatten_envs(dat["envs"])
    trace = np.asarray(dat["trace"], dtype=np.float32)
    position = np.asarray(dat["position"], dtype=np.float32)
```

iii. The trajectory says the AI traced the repository's loading code and treated these repository-native joblib files as authoritative. It checked the resulting totals against the paper (207 sessions, 5,413 unique neurons, and 69,744 session-neuron maps).

## 1-b. How are the data split into subjects?

i. Each entry in the fixed `ANIMALS` list is one subject. Its integer list position becomes `subject_idx`; the output subject names are copied from the same list.

ii.
```python
for subject_id, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(subject_id)
...
"subjects": list(ANIMALS),
"subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. The AI documented that these are the seven animal IDs in the source data and verified paper-level subject/session counts.

## 1-c. How are the data split into sessions?

i. A day on the first axis of an animal's arrays is treated as one session. Each day's trials and metadata are appended once to the session-level output lists.

ii.
```python
for day in range(trace.shape[0]):
    day_trace = trace[day]
    day_pos = position[day]
    ...
    neural.append(session_trials_neural)
    decoder_input.append(session_trials_input)
    decoder_output.append(session_trials_output)
```

iii. The AI said this matches the paper code, which iterates over the day/session axes of `trace`, `position`, and `envs`.

## 1-d. How are the data split into trials?

i. Each continuous session is split into non-overlapping, contiguous 60-second windows. At 30 Hz this is 1,800 frames. Only complete windows are kept and the trailing remainder is discarded; the code also requires at least two complete trials.

ii.
```python
TRIAL_FRAMES = int(FPS * TRIAL_SECONDS)
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(...)
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_FRAMES
    stop = start + TRIAL_FRAMES
    session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. The trajectory explains that the source contains continuous recordings, so artificial one-minute trials were required by the task. It reports 39 or 40 complete trials per session and explicitly documents dropping incomplete tails.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level behavioral or neural-quality exclusion. A session is rejected if it has fewer than two full trials; otherwise all complete trials are retained and only an incomplete tail is dropped.

ii.
```python
n_trials = T // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day}: only {n_trials} full 1-minute trials")
```

iii. No trajectory justification was given for additional trial filtering because none was applied. The two-trial check directly enforces the downstream format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from each animal dictionary's `trace` array.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
day_trace = trace[day]
```

iii. The AI identified `trace` as the provided binary rising-phase calcium-event signal used by the paper.

## 2-b. How is the `neural` data processed?

i. The already processed trace is cast to `float32`, sliced by day, restricted to neurons registered that day, copied, and then sliced into `(neurons, 1800)` trial arrays. It is not smoothed, deconvolved, rebinned, or re-thresholded.

ii.
```python
trace = np.asarray(dat["trace"], dtype=np.float32)
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
session_trials_neural.append(session_neural[:, start:stop].copy())
```

iii. The trajectory states that the source traces already encode the binarized rising phase of calcium transients used in the paper, so further signal processing would depart from the reference pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained for a day when its first sample is not NaN. The code then verifies that retained neurons contain no NaNs anywhere and excluded neurons are entirely NaN. No place-cell or activity threshold is applied.

ii.
```python
registered_today = ~np.isnan(day_trace[:, 0])
if np.any(np.isnan(day_trace[registered_today])):
    raise ValueError(...)
if np.any(~np.isnan(day_trace[~registered_today])):
    raise ValueError(...)
session_neural = day_trace[registered_today].astype(np.float32, copy=True)
```

iii. The AI reasoned that all-NaN rows represent neurons not registered on that day, while additional activity/place-cell filtering would remove valid recorded content. The consistency checks establish that the first-frame test is equivalent to all-NaN filtering for this dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Each artificial trial is aligned to the start of its contiguous one-minute window, with offsets 0 to 60 seconds.

ii.
```python
"temporal_alignment_event": "trial start of each contiguous 60 second window within a recording session",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

iii. The trajectory notes that the task concerns continuous exploration rather than stimulus-locked trials, making window start the applicable alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz resolution is preserved, giving 33.333 ms per bin. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30.0
TIME_BIN_MS = 1000.0 / FPS
...
"time_bin_size": TIME_BIN_MS,
```

iii. The AI chose the native frame rate because neural activity and position are already sampled together at 30 Hz.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the session's `blocked` entry. `envs` is loaded and used for summaries, but not to construct the decoder input.

ii.
```python
blocked_entry = normalize_blocked_entry(dat["blocked"][day])
env_name = str(envs[day])
env_mask = mask_from_blocked(blocked_entry)
```

iii. The AI explicitly argued that `blocked` is authoritative because it captures orientation and geometry details that an environment-name label alone may not encode.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Nested/NaN blocked values are normalized; `-1` becomes no blocked bins. The code starts with a 3-by-3 matrix of ones and sets blocked indices to zero, flattens it to a static 9-vector, and copies it into every trial in the session. Thus its polarity is `1=open, 0=blocked`.

ii.
```python
def mask_from_blocked(blocked_bins):
    mask = np.ones((3, 3), dtype=np.float32)
    for idx in blocked_bins:
        mask[idx // 3, idx % 3] = 0.0
    return mask
...
input_vec = env_mask.reshape(-1).astype(np.float32)
session_trials_input.append(input_vec.copy())
```

iii. The AI described this as a flattened open-bin mask and chose `blocked` rather than canonical environment names to preserve session-specific geometry/orientation.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the two-coordinate, time-varying `position` array for each day.

ii.
```python
position = np.asarray(dat["position"], dtype=np.float32)
day_pos = position[day]
```

iii. The AI identified this as the mouse's frame-aligned x/y location.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. For each session, each coordinate is divided by one third of that session's per-axis maximum, floored, and clipped to 0–2. Positions landing in blocked coarse bins are then projected to the nearest open bin. Finally x/y indices are flattened as `x * 3 + y` into one integer time series.

ii.
```python
max_per_axis = np.nanmax(position_xy, axis=1) + 1e-12
bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
...
projected_xy = lookup[binned_xy[0], binned_xy[1]].T
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. The AI said max-based scaling without minimum subtraction matches spatial binning in the paper code. It characterized nearest-valid-bin projection as a decoder-specific, geometry-aware cleanup analogous to the paper's decoding error calculation.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Two thresholds per coordinate are implicit at one-third and two-thirds of that session's coordinate maximum. The resulting x/y categories are clipped to 0–2, blocked categories are reassigned to the nearest open category, and the pair is encoded into labels 0–8 as `x_bin * 3 + y_bin`.

ii.
```python
bins = np.floor(position_xy / (max_per_axis[:, None] / n_bins)).astype(np.int64)
return np.clip(bins, 0, n_bins - 1)
...
output_position = (projected_xy[0] * 3 + projected_xy[1]).astype(np.int64)
```

iii. The AI justified the 3-by-3 categorization from the decoder task and the scaling/projection choices from its reading of paper code.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural arrays are checked to have the same session and frame dimensions, processed without temporal shifts, and sliced with identical `start:stop` indices for each trial.

ii.
```python
if trace.shape[2] != position.shape[2]:
    raise ValueError(...)
...
session_trials_neural.append(session_neural[:, start:stop].copy())
session_trials_output.append(output_position[np.newaxis, start:stop].copy())
```

iii. The AI relied on the source streams being frame-synchronous and used common trial boundaries to preserve that alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. NaNs in `blocked` are removed and `-1` is normalized to no blocked bins. All-NaN unregistered neural rows are excluded. Unexpected partial neural NaNs, any position NaN, dimension disagreements, or sessions with fewer than two trials cause explicit errors rather than imputation. Incomplete trailing frames are discarded.

ii.
```python
arr = arr[~np.isnan(arr)]
...
if np.isnan(day_pos).any():
    raise ValueError(...)
registered_today = ~np.isnan(day_trace[:, 0])
...
dropped = T - n_trials * TRIAL_FRAMES
```

iii. The trajectory says NaN neural rows encode non-registration, not missing samples. It favored strict checks for unexpected corruption and documented the intentional loss of incomplete tails.

## 6-a. What are the most time-consuming steps of the code?

i. Within conversion, loading and materializing the large per-animal joblib arrays, copying every neural trial, and serializing the full pickle dominate. Position projection and summary scans also traverse all frames but are simpler operations.

ii.
```python
return joblib.load(os.path.join(path, animal))[animal]
session_trials_neural.append(session_neural[:, start:stop].copy())
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory does not provide a conversion profiler. It describes conversion/full-data generation as expensive and separately identifies full decoder training as the last expensive workflow step; the conversion bottlenecks above are inferred from code and data volume.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The 3-by-3 nearest-valid lookup loop is vectorizable, though tiny. Trial slicing could use reshape/split operations, and sample-session selection could use NumPy indexing. The outer subject/session loops are appropriate because neuron counts and trial lists are ragged.

ii.
```python
for x in range(3):
    for y in range(3):
        ...
for trial_idx in range(n_trials):
    ...
for sess_idx, subj in enumerate(data["subject_idx"]):
```

iii. The trajectory gives no explicit vectorization rationale. The AI prioritized clear construction of the required nested, ragged Python structure.

## 6-c. What processing does the code repeat multiple times?

i. It copies the same static geometry vector for every trial, repeatedly slices/copies each stream per trial, scans outputs again to summarize class counts, and scans subject indices again when building the sample dataset.

ii.
```python
session_trials_input.append(input_vec.copy())
session_trials_output.append(output_position[np.newaxis, start:stop].copy())
...
for session in data["output"]:
    for trial in session:
        vals, counts = np.unique(trial[0], return_counts=True)
```

iii. No explicit trajectory justification was given. Some repetition follows from the required per-trial nested format; the extra scans support sanity reporting and sample generation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/flattens `envs` only for diagnostics, defines an unused `ENV_TO_MASK`, computes extensive counters/statistics and a sample dataset not needed for `converted_data.pkl`, and calculates `reassigned` solely to count changed frames. These diagnostics do not affect downstream full-data decoding.

ii.
```python
ENV_TO_MASK = {...}
envs = flatten_envs(dat["envs"])
reassigned = np.any(projected_xy != binned_xy, axis=0)
stats.total_frames_reassigned += int(np.sum(reassigned))
...
sample_data = subset_sessions(data, sample_session_indices)
```

iii. The trajectory shows these were deliberate sanity checks and deliverables: the AI used them to compare paper-level counts, inspect geometry patterns, and validate a smaller dataset before full training.
