# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads each of seven explicitly named subjects from the repository's extensionless, joblib-serialized processed file. It retrieves that subject's `trace`, `position`, `envs`, and `blocked` arrays, then iterates through every recording day. Full mode is unconditional; the code asserts the expected 207 sessions and 69,744 session-neurons.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
source = joblib.load(DATA_DIR / subject)[subject]
traces = source['trace']
positions = source['position']
envs = source['envs']
blocked = source['blocked']
...
if len(neural) != 207 or total_valid != 69744:
    raise RuntimeError(...)
```

iii. The trajectory says the agent inspected both the raw MATLAB files and repository-produced processed files. It chose the smaller joblib files because they already contained the paper's aligned, preprocessed 30-Hz binary traces and DLC positions, and it verified the totals against the paper.

## 1-b. How are the data split into subjects?

i. A fixed list defines the seven mice. Each file is keyed by its subject ID, and the loop index becomes that subject's index for every session.

ii.
```python
for subj_i, subject in enumerate(SUBJECTS):
    source = joblib.load(DATA_DIR / subject)[subject]
...
subject_idx.append(subj_i)
```

iii. The agent found exactly seven animal files and used their QLAK identifiers. The fixed ordering makes `subjects` and `subject_idx` deterministic.

## 1-c. How are the data split into sessions?

i. Each recording day (the first dimension of `trace`) is one output session. Sessions are not combined across days or environments.

ii.
```python
for day in range(traces.shape[0]):
    tr = np.asarray(traces[day])
    pos = np.asarray(positions[day])
...
neural.append([counts[t] for t in range(n_trials)])
```

iii. The trajectory identified 207 recording days and concluded that a recording day is the natural session boundary used by the source repository and paper.

## 1-d. How are the data split into trials?

i. Each session is divided from its start into complete, non-overlapping 60-second windows: 1,800 source frames at 30 Hz. Any trailing partial minute is dropped, and sessions with fewer than two complete trials raise an error.

ii.
```python
TRIAL_FRAMES = FS * TRIAL_SECONDS
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(...)
used = n_trials * TRIAL_FRAMES
```

iii. The task explicitly defines one-minute trials. The agent chose complete equal-length windows to satisfy the decoder format and reported 8,187 trials after dropping incomplete tails.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral or trial-quality filter. Only incomplete trailing windows are omitted; every complete minute is retained. The code enforces the requirement of at least two trials per session.

ii.
```python
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f'too few complete trials: {subject} day {day}')
used = n_trials * TRIAL_FRAMES
```

iii. The trajectory found no instructed trial-level exclusion rule. Equal duration and the decoder's minimum-trial constraint motivated the checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the processed source's `trace` variable. These are paper-preprocessed binary rising-phase calcium-transient events at 30 Hz.

ii.
```python
traces = source['trace']
...
tr = np.asarray(traces[day])
```

iii. From the methods and value inspection, the agent determined that `trace` was already binary and resulted from the authors' derivative smoothing, noise z-scoring, and thresholding pipeline, so it did not reprocess calcium video.

## 2-b. How is the `neural` data processed?

i. After defensive orientation checks and neuron filtering, the agent truncates to complete trials, reshapes each minute into 60 groups of 30 frames, and sums binary events in each group. The result is float32 event counts with shape `(neurons, 60)` per trial.

ii.
```python
if tr.shape[1] != pos.shape[1] and tr.shape[0] == pos.shape[1]:
    tr = tr.T
...
counts = tr[:, :used].reshape(
    n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
```

iii. The agent reasoned that summing preserves binary-event mass while reducing a very large 30-Hz decoder dataset to a tractable 1-Hz representation. This was a practical choice, not processing specified by the paper/reference conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Within each session, only neuron rows finite at every frame are retained. The code verifies that no neuron is partially missing and errors if one is; all-NaN rows, interpreted as tracked neurons absent that day, are removed.

ii.
```python
finite = np.isfinite(tr)
valid = finite.all(axis=1)
if np.any(finite.any(axis=1) != valid):
    raise ValueError(f'partially missing neuron: {subject} day {day}')
tr = tr[valid]
```

iii. The trajectory audited all animals and found each neuron/session row wholly finite or wholly NaN. Filtering absent rows reproduced the paper's 69,744 session-specific neuron maps exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Time zero is the start of each artificial, non-overlapping one-minute window; neural and position data use identical frame boundaries. Metadata records offsets of 0 to 60 seconds.

ii.
```python
'temporal_alignment_event': 'start of each non-overlapping one-minute window',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The experiment is a continuous open-field recording with no instructed trial event, so the agent used the constructed trial start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 1,000 ms. Thirty native 30-Hz frames are rebinned into each second, producing 60 time bins per trial.

ii.
```python
FS = 30
BIN_FRAMES = 30
BINS_PER_TRIAL = TRIAL_SECONDS
...
'time_bin_size': 1000.0,
```

iii. The agent explicitly chose 1-second bins to reduce storage and decoder runtime. It acknowledged that the paper uses spatial rather than temporal binning and treated this as a pragmatic decoder choice.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the per-session `blocked` list, whose values are flattened 3×3 grid indices. Environment labels are retained only as session metadata.

ii.
```python
blocked = source['blocked']
...
block_values = np.asarray(blocked[day], dtype=float).reshape(-1)
env = scalar_env(envs[day])
```

iii. Inspection showed that `blocked` explicitly represents the arena's unavailable grid cells, including `-1` for the unblocked square, making it the direct decoder input requested.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent creates nine float32 binary indicators. Each finite integer in `[0, 8]` sets its corresponding indicator to one; `-1` and invalid values are ignored. The same static vector is copied for every trial in that session.

ii.
```python
geometry = np.zeros(9, dtype=np.float32)
for b in block_values:
    if np.isfinite(b) and 0 <= int(b) < 9:
        geometry[int(b)] = 1.0
...
inputs.append([geometry.copy() for _ in range(n_trials)])
```

iii. The agent inferred a row-major flattened grid from the source and used independent indicators so multiple blocked cells can be represented; ignoring `-1` naturally produces an all-zero vector.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position comes from the processed source's `position` stream, containing synchronized DLC x/y coordinates for every imaging frame.

ii.
```python
positions = source['position']
...
pos = np.asarray(positions[day])
if pos.shape[0] != 2 and pos.shape[1] == 2:
    pos = pos.T
```

iii. The agent verified that positions were finite, aligned with `trace`, and scaled from 0 to 75 cm as described by the paper.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first averages x and y over each 30-frame, one-second neural bin, then converts the averaged coordinate to a spatial class. Outputs have shape `(1, 60)` per trial and integer dtype.

ii.
```python
mean_pos = pos[:, :used].reshape(
    2, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
...
outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
```

iii. Averaging was chosen so position would share the agent's new 1-second neural resolution. The trajectory frames it as synchronized aggregation needed for tractability.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis of the 75-cm arena is divided at 25 and 50 cm. Bins are computed by `floor(coordinate / 25)`, clipped to 0–2. The class is row-major: `y_bin * 3 + x_bin`, yielding labels 0–8.

ii.
```python
xbin = np.clip(np.floor(mean_pos[0] / 25.0), 0, 2).astype(np.int64)
ybin = np.clip(np.floor(mean_pos[1] / 25.0), 0, 2).astype(np.int64)
spatial_class = ybin * 3 + xbin
```

iii. These thresholds exactly reflect the experiment's 3×3 construction in a 75-cm square. Clipping safely handles coordinates on or marginally beyond arena boundaries.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The code checks equal source-frame counts, truncates both streams to the same complete-minute boundary, reshapes both with identical trial/second/frame dimensions, sums neural events and averages position over the same groups of 30 frames.

ii.
```python
if tr.shape[1] != pos.shape[1]:
    raise ValueError(f'unaligned streams: {subject} day {day}')
...
counts = tr[:, :used].reshape(..., BIN_FRAMES).sum(axis=3, ...)
mean_pos = pos[:, :used].reshape(..., BIN_FRAMES).mean(axis=3)
```

iii. The agent established that source streams were already frame-aligned and deliberately used identical aggregation windows to preserve synchronization after rebinning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Expected all-NaN absent-neuron rows are removed. Partially missing neural rows, mismatched stream lengths, sessions with too few trials, and unexpected global totals cause explicit errors. Incomplete trailing time is safely discarded; geometry ignores non-finite or out-of-range entries.

ii.
```python
if np.any(finite.any(axis=1) != valid):
    raise ValueError(...)
if tr.shape[1] != pos.shape[1]:
    raise ValueError(...)
if n_trials < 2:
    raise ValueError(...)
if np.isfinite(b) and 0 <= int(b) < 9:
    geometry[int(b)] = 1.0
```

iii. The audit showed only the documented all-NaN absence markers. The agent preferred fail-fast checks for unobserved anomalies so silent corruption would not enter the decoder dataset.

## 6-a. What are the most time-consuming steps of the code?

i. Loading each large compressed joblib subject, scanning the full neural arrays for finite values, aggregating traces/positions, and serializing the large nested pickle are the main costs. The agent also explicitly invokes garbage collection after each subject.

ii.
```python
source = joblib.load(DATA_DIR / subject)[subject]
finite = np.isfinite(tr)
counts = tr[:, :used].reshape(...).sum(...)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the trajectory, multi-subject audits and conversion required repeated waits; it described loading as large-data I/O and selected 1-second aggregation partly to control downstream storage and runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Subject and session loops are structurally necessary because shapes differ. The short loop over blocked indices and list constructions over trials could be replaced with indexed assignment or array-oriented conversion, though these are negligible beside I/O and neural processing.

ii.
```python
for b in block_values:
    if np.isfinite(b) and 0 <= int(b) < 9:
        geometry[int(b)] = 1.0
...
neural.append([counts[t] for t in range(n_trials)])
```

iii. The trajectory did not discuss these micro-optimizations. Its major numerical work was already vectorized with reshape, sum, and mean.

## 6-c. What processing does the code repeat multiple times?

i. Per session it repeatedly checks orientation/alignment, constructs a full boolean finiteness array, reshapes synchronized streams, copies the same geometry vector for every trial, and builds trial lists. These repetitions are required by session-specific shapes except for the geometry copies, which enforce independent trial objects.

ii.
```python
finite = np.isfinite(tr)
...
inputs.append([geometry.copy() for _ in range(n_trials)])
outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
```

iii. The trajectory emphasizes session-wise filtering because cell presence changes by day. It gives no separate efficiency justification for repeated list conversion or geometry copying.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `envs`, normalizes each environment string, and stores extensive `session_info`; the decoder does not use those values. It also creates a full `finite` matrix to derive/validate neuron masks and explicitly calls garbage collection. The 1-second position averaging is not discarded, but it is extra processing absent from the reference.

ii.
```python
envs = source['envs']
...
env = scalar_env(envs[day])
session_info.append({'environment': env, ...})
...
gc.collect()
```

iii. Environment/session metadata was retained for provenance and auditing, not decoder features. The trajectory used totals and metadata to validate correspondence with the paper.
