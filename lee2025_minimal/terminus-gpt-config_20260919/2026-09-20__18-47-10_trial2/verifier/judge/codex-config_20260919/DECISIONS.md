# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, loads each extensionless processed animal file with `joblib.load`, unwraps the animal-keyed dictionary, and reads its per-session `trace`, `position`, and `envs` collections. It checks that those collections have equal session counts and processes every session.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
    'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]
...
outer = joblib.load(DATA_DIR / animal)
dat = outer[animal]
traces = dat['trace']
positions = dat['position']
envs = np.asarray(dat['envs']).reshape(-1)
if not (len(traces) == len(positions) == len(envs)):
    raise ValueError(f'inconsistent session count for {animal}')
```

iii. The trajectory says the agent identified the extensionless files as the repository's Python/joblib processed datasets and preferred their already processed binary rise-event traces. It verified that the seven animals comprise 207 sessions and ultimately included all 207.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is one subject. The outer joblib object is indexed by that same ID, and `subj_i` is appended once for every session belonging to the animal.

ii.
```python
for subj_i, animal in enumerate(ANIMALS):
    outer = joblib.load(DATA_DIR / animal)
    dat = outer[animal]
    ...
    subject_idx.append(subj_i)
...
'subjects': ANIMALS,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. The agent found seven named mice in the supplied files and treated the per-animal objects as the natural subject boundary.

## 1-c. How are the data split into sessions?

i. Each element of `envs` is treated as a recording day/session; the same index selects its trace and position. One output session is appended per day.

ii.
```python
for day, env_raw in enumerate(envs):
    env = str(env_raw)
    counts, labels, n_cells = aggregate_session(traces[day], positions[day])
    ...
    neural.append(sess_neural)
    inputs.append(sess_input)
    outputs.append(sess_output)
```

iii. The trajectory reports 31 sessions for six animals and 21 for one animal, totaling the paper's 207 sessions, and notes that environment order varies by mouse and repeats across sequences.

## 1-d. How are the data split into trials?

i. After converting each nominal 40-minute session to exactly 2,400 one-second bins, the agent slices it into 40 contiguous, non-overlapping 60-bin (one-minute) trials.

ii.
```python
N_SECONDS = 40 * 60
TRIAL_SECONDS = 60
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. The agent wanted exactly forty one-minute trials per nominal 40-minute recording while retaining every aligned source frame, even though frame counts differ slightly from 72,000.

## 1-e. How are trials filtered based on quality controls?

i. No trials or sessions are filtered. Alignment and minimum-length conditions are validated before aggregation; every valid session yields 40 trials.

ii.
```python
if position.shape != (2, n_frames):
    raise ValueError(f'alignment mismatch: trace {trace.shape}, position {position.shape}')
...
if np.any(widths <= 0):
    raise ValueError('recording too short for one-second aggregation')
```

iii. The trajectory emphasizes full coverage (207 sessions and 8,280 trials) and identifies no paper-specified trial-level exclusion criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each session's released `trace` array in the processed joblib animal object.

ii.
```python
traces = dat['trace']
...
counts, labels, n_cells = aggregate_session(traces[day], positions[day])
```

iii. From the methods and repository, the agent concluded that `trace` contains manually curated binary rising phases of significant calcium transients (z > 2.5), already representing the paper's neural events.

## 2-b. How is the `neural` data processed?

i. Session-absent neurons are removed, then binary framewise events are summed into 2,400 variable-width one-second bins. Counts are cast to `float32`, sliced into trials, and made contiguous. No calcium extraction, smoothing, or place-cell selection is redone.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
starts, widths = edges[:-1], np.diff(edges)
counts = np.add.reduceat(tr, starts, axis=1)
counts = counts[:, :N_SECONDS].astype(np.float32, copy=False)
```

iii. The agent regarded the released trace as already paper-processed. It chose one-second event counts to keep the complete 207-session decoder dataset tractable, while preserving event totals and the nominal 40-minute duration.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained for a session if any trace sample is finite. The code then requires every retained sample to be finite; partially non-finite retained neurons cause an error. No place-cell or activity-rate filter is applied.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
```

iii. Inspection showed all-NaN rows represent globally registered cells absent on a given day. The agent explicitly decided not to select only place cells because that was a separate analysis rather than the released population definition.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Trials are aligned to the start of each contiguous one-minute segment; neural and position use identical one-second bin edges and trial slices.

ii.
```python
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_output.append(labels[None, start:stop].copy())
...
'temporal_alignment_event': 'start of each contiguous one-minute segment of a 40-minute free-exploration session',
```

iii. The task defines artificial one-minute trials in continuous free exploration, so the agent found no stimulus event and documented segment start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 1,000 ms. Approximately 30 native 30 Hz frames are rebinned into each second using proportional integer edges across all aligned frames.

ii.
```python
N_SECONDS = 40 * 60
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
...
'time_bin_size': 1000.0,
'source_sampling_rate_hz': 30.0,
```

iii. The agent initially investigated the repository decoder's temporal processing, then chose one-second bins for tractable full-dataset size and exact 2,400-bin sessions. It reasoned that proportional edges avoid dropping or inventing a final minute when aligned frame counts differ slightly.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the raw session environment name in `envs`, looked up in a hard-coded transcription of the repository's `get_env_mat` definitions.

ii.
```python
envs = np.asarray(dat['envs']).reshape(-1)
...
env = str(env_raw)
geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)
```

iii. The trajectory says the behavior dictionary supplies environment labels but not blocked indices, so the agent used the repository's exact geometry matrices keyed by those labels.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3×3 repository mask is converted to `float32` and flattened in x-major order to nine static indicators, where 1 means accessible and 0 means blocked. A copy is attached to every trial.

ii.
```python
geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)
...
sess_input.append(geom.copy())
...
'input_names': [f'accessible_x{x}_y{y}' for x in range(3) for y in range(3)],
```

iii. The agent sought consistency with the paper repository's rate-map axis convention and interpreted “environment geometry” as accessibility context static within a session/trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each session's two-row `position` array in the processed joblib object.

ii.
```python
positions = dat['position']
...
counts, labels, n_cells = aggregate_session(traces[day], positions[day])
```

iii. The agent identified these as timestamp-aligned x/y coordinates in centimeters produced by the paper's tracking pipeline.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position samples are summed over exactly the same variable-width bins as neural activity and divided by each bin's frame count to obtain mean x/y position per second. Those means are then discretized into one categorical label.

ii.
```python
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
pos_mean = pos_sum / widths[None, :]
...
labels = (3 * xy_bin[0] + xy_bin[1]).astype(np.int64)
```

iii. The agent chose averaging so each rebinned output represents position across the same one-second interval as its neural event count.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Mean x and y coordinates are divided by 25 cm and floored, then clipped to bins 0–2. The class is x-major: `3*x_bin + y_bin`, yielding 0–8.

ii.
```python
xy_bin = np.floor(pos_mean / 25.0).astype(np.int64)
np.clip(xy_bin, 0, 2, out=xy_bin)
labels = (3 * xy_bin[0] + xy_bin[1]).astype(np.int64)
```

iii. This follows the physical 75×75 cm arena's 3×3, 25 cm partitions and what the agent identified as the repository's x-major `[x, y]` convention. Clipping handles exact wall coordinates.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The code first asserts frame-level alignment, applies the same proportional bin starts and widths to traces and positions, and applies identical one-minute slice boundaries to counts and labels.

ii.
```python
n_frames = trace.shape[1]
if position.shape != (2, n_frames):
    raise ValueError(f'alignment mismatch: trace {trace.shape}, position {position.shape}')
...
counts = np.add.reduceat(tr, starts, axis=1)
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
```

iii. The methods describe timestamp-aligned 30 Hz streams; the agent preserved this correspondence during aggregation and trial splitting.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Expected all-NaN rows for absent neurons are removed. Unexpected partial non-finite traces, neural/position length mismatches, inconsistent session counts, or recordings too short for aggregation raise errors. Slight frame-count differences are handled by exhaustive proportional bin edges rather than truncation.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
```

iii. The agent distinguished structural NaN padding from unexpected corruption and explicitly designed proportional edges to preserve all source frames despite small alignment-related duration differences.

## 6-a. What are the most time-consuming steps of the code?

i. Loading each large joblib animal object, aggregating all session traces, and serializing the roughly 644 MiB float32 pickle are the dominant operations. Garbage collection is forced after each animal.

ii.
```python
outer = joblib.load(DATA_DIR / animal)
...
counts = np.add.reduceat(tr, starts, axis=1)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory records long waits while scanning/loading animals, converting, and writing the full dataset. It also notes that native 30 Hz output would be impractically large.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The animal and variable-sized session loops are structurally appropriate. The 40-iteration trial-slicing loop could be replaced by reshaping/transposing the 2,400-bin arrays, although copies would still be needed for the required list-of-trials format. The agent already vectorizes time aggregation with `np.add.reduceat`.

ii.
```python
for day, env_raw in enumerate(envs):
    ...
    for start in range(0, N_SECONDS, TRIAL_SECONDS):
        stop = start + TRIAL_SECONDS
        sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
```

iii. The trajectory does not explicitly discuss loop vectorization; its implementation instead uses vectorized reduction for the expensive framewise work.

## 6-c. What processing does the code repeat multiple times?

i. For every trial it copies the same static geometry vector and copies/slices neural and output arrays. For every session it reconstructs identical proportional edges for sessions with the same frame length and repeatedly looks up/converts one of ten geometry matrices.

ii.
```python
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
...
geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)
...
sess_input.append(geom.copy())
```

iii. No explicit justification is given for caching. Copies are justified in a code comment as keeping the pickle independent of full-session buffers.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little scientific processing is discarded. The code computes `widths` for both position averaging and validation, and full-session buffers are intentionally discarded after trial copies. It does, however, make 40 duplicate geometry arrays per session and converts compact integer event counts to float32, increasing serialization and I/O without changing event values.

ii.
```python
counts = counts[:, :N_SECONDS].astype(np.float32, copy=False)
...
sess_input.append(geom.copy())
...
del outer, dat, traces, positions
gc.collect()
```

iii. The trajectory shows that counts were initially `uint8`, then changed to `float32` solely to eliminate a validator dtype warning. Thus the extra storage is intentional validator compatibility rather than analysis necessity.
