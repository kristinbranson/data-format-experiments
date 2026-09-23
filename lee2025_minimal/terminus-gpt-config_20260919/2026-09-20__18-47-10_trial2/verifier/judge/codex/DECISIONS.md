# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the `.mat` files used by the human reference. It hard-codes the seven animal IDs, then loads one extensionless per-animal `joblib` file from `/app/data`. For each animal, it reads the nested dictionary fields `trace`, `position`, and `envs`, then iterates through the sessions in those arrays.

ii.
```python
DATA_DIR = Path('/app/data')
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
    'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]
...
for subj_i, animal in enumerate(ANIMALS):
    outer = joblib.load(DATA_DIR / animal)
    dat = outer[animal]
    traces = dat['trace']
    positions = dat['position']
    envs = np.asarray(dat['envs']).reshape(-1)
```

iii. In the trajectory, the AI says it inspected the repository README and concluded the extensionless files were the released processed datasets and that `trace` already contained the paper’s curated neural representation (steps 4 and 6). It then decided to use those files directly rather than the `.mat` files.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the fixed `ANIMALS` list. Each entry corresponds to one animal and becomes one subject name in the output.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
    'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]
...
for subj_i, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(subj_i)
...
'subjects': ANIMALS,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. The trajectory states that the dataset contains seven mice with those exact IDs and treats each per-animal file as one subject (steps 3, 7, and 9).

## 1-c. How are the data split into sessions?

i. For each animal, the AI treats each element of `trace`, `position`, and `envs` as one session. It iterates over `envs` with `enumerate`, and each index `day` becomes one output session.

ii.
```python
traces = dat['trace']
positions = dat['position']
envs = np.asarray(dat['envs']).reshape(-1)
if not (len(traces) == len(positions) == len(envs)):
    raise ValueError(f'inconsistent session count for {animal}')

for day, env_raw in enumerate(envs):
    env = str(env_raw)
    counts, labels, n_cells = aggregate_session(traces[day], positions[day])
```

iii. In the trajectory, the AI reports that each animal has multiple sessions stored in aligned per-session arrays, and that the total dataset contains 207 sessions across the seven mice (steps 7, 9, and 12).

## 1-d. How are the data split into trials?

i. The AI first aggregates each full session into exactly 2,400 one-second bins, then splits those bins into 40 contiguous one-minute trials of 60 bins each. It does not split the original frame-level data directly into 1,800-frame chunks.

ii.
```python
N_SECONDS = 40 * 60
TRIAL_SECONDS = 60
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. The trajectory says the aligned session lengths vary slightly around 72,000 frames, so the AI chose proportional integer edges to preserve all aligned frames while still forcing each nominal 40-minute session into 2,400 one-second bins and then 40 one-minute trials (steps 7 and 12).

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. Every session is deterministically split into 40 trials once the per-second aggregation is complete.

ii.
```python
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. The trajectory does not mention any trial rejection criteria. It emphasizes preserving all 207 source sessions and all 40 nominal trials per session (steps 12, 15, and 22).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data is derived from the `trace` arrays loaded from each animal’s processed dataset.

ii.
```python
dat = outer[animal]
traces = dat['trace']
...
counts, labels, n_cells = aggregate_session(traces[day], positions[day])
```

iii. In the trajectory, the AI says the methods show that the supplied `trace` is already the paper’s manually curated binary rising-phase neural representation and therefore should be used directly (steps 4, 6, and 12).

## 2-b. How is the `neural` data processed?

i. The AI treats each session’s `trace` as binary event data, removes absent neurons, aggregates the full session into one-second event counts with `np.add.reduceat`, casts to `float32`, and then slices those counts into one-minute trials.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
starts, widths = edges[:-1], np.diff(edges)
...
counts = np.add.reduceat(tr, starts, axis=1)
counts = counts[:, :N_SECONDS].astype(np.float32, copy=False)
```

iii. The trajectory says the AI wanted a tractable full-dataset representation and interpreted the released traces as 0/1 calcium-transient events, so it summed them into per-second event counts rather than keeping the native 30 Hz stream (steps 7, 10, and 12).

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is kept only if it has at least one finite value in that session. All-NaN neurons are treated as absent on that day and removed. If a kept neuron still has a non-finite value anywhere, the code raises an error.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
```

iii. The trajectory says inspection showed that session-specific missing neurons appear as all-NaN rows, so they must be removed for the decoder, while partially non-finite included neurons would indicate an unexpected data problem (steps 7, 11, and 12).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to an experimental event from the original task. Instead, it treats the start of each contiguous one-minute segment as the alignment event and records that choice in metadata.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'start of each contiguous one-minute segment of a 40-minute free-exploration session',
    'off_start': 0.0,
    'off_end': 60.0,
    ...
}
```

iii. The trajectory says the task is continuous free exploration with no stimulus-triggered event, so the AI adopted trial-start alignment for the artificial one-minute segments it created (steps 12 and 22).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 1,000 ms bins. Yes: the code rebins each session from its original frame-level stream into 2,400 one-second bins using proportional integer edges.

ii.
```python
N_SECONDS = 40 * 60
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
...
'metadata': {
    ...
    'time_bin_size': 1000.0,
    'source_sampling_rate_hz': 30.0,
    ...
}
```

iii. In the trajectory, the AI first considered the repository’s 100 ms position-decoding binning, but then chose 1-second bins to keep the full converted dataset practical while preserving nominal session duration across slightly mismatched frame counts (steps 10 and 12).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the per-session `envs` labels, not from the raw `blocked` variable. It uses the environment name to look up a hard-coded 3×3 template in `ENV_MATS`.

ii.
```python
ENV_MATS = {
    'square':    [[1,1,1], [1,1,1], [1,1,1]],
    'o':         [[1,1,1], [1,0,1], [1,1,1]],
    ...
}
...
envs = np.asarray(dat['envs']).reshape(-1)
...
env = str(env_raw)
geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)
```

iii. The trajectory says the behavior dictionary did not expose blocked indices in the way the AI wanted, while the repository code defined geometry templates by environment name, so it chose to reconstruct geometry from `envs` via `get_env_mat`-style definitions (steps 8 and 12).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts each environment label into a 3×3 binary accessibility mask, flattens it to length 9 in x-major order, and duplicates that same static vector for every trial in the session. The vector uses `1` for accessible cells and `0` for blocked cells.

ii.
```python
geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    ...
    sess_input.append(geom.copy())
...
'input_names': [f'accessible_x{x}_y{y}' for x in range(3) for y in range(3)],
```

iii. The trajectory says the AI wanted the decoder input to reflect the full session geometry, and considered the environment templates from the repository to be the cleanest way to encode that geometry as static contextual input (steps 8 and 12).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `output` position labels are derived from the per-session `position` arrays.

ii.
```python
positions = dat['position']
...
counts, labels, n_cells = aggregate_session(traces[day], positions[day])
```

iii. The trajectory says the released datasets contain aligned x/y position streams for each session and that those should be used directly rather than reconstructed from another source (steps 4, 6, and 12).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI sums neural data but averages position within each one-second bin. It then converts the per-second mean x/y coordinates into a single categorical position label for each second.

ii.
```python
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
pos_mean = pos_sum / widths[None, :]
...
labels = (3 * xy_bin[0] + xy_bin[1]).astype(np.int64)
```

iii. The trajectory says the AI wanted the behavioral target aligned to the same one-second bins as the neural counts, so it averaged position over the same rebinned intervals before discretizing (step 12).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI divides the 75 cm arena into 25 cm bins along x and y, floors the averaged coordinates to obtain bin indices 0, 1, or 2, clips them to range, and converts them to a single label with the x-major formula `3*x_bin + y_bin`.

ii.
```python
xy_bin = np.floor(pos_mean / 25.0).astype(np.int64)
np.clip(xy_bin, 0, 2, out=xy_bin)
labels = (3 * xy_bin[0] + xy_bin[1]).astype(np.int64)
...
'output_values': [[f'x{x}_y{y}' for x in range(3) for y in range(3)]],
```

iii. The trajectory says the AI wanted to match the physical 3×3 arena partition and to use the repository’s x-major convention for flattening the grid (steps 7 and 12).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural counts and position labels are aligned by applying the same proportional rebinned edges to the session-long `trace` and `position` arrays, then slicing both rebinned streams into the same 60-second trial boundaries.

ii.
```python
n_frames = trace.shape[1]
if position.shape != (2, n_frames):
    raise ValueError(f'alignment mismatch: trace {trace.shape}, position {position.shape}')
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
...
counts = np.add.reduceat(tr, starts, axis=1)
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
...
sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
sess_output.append(labels[None, start:stop].copy())
```

iii. The trajectory says the position and neural streams are already aligned in the released data, so it used the same rebinned edges for both streams to keep them synchronized after temporal aggregation (steps 6 and 12).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes all-NaN neurons, errors out on partially non-finite kept neurons, and handles slight session-length deviations by redistributing all frames across exactly 2,400 one-second bins instead of dropping a remainder.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
starts, widths = edges[:-1], np.diff(edges)
if np.any(widths <= 0):
    raise ValueError('recording too short for one-second aggregation')
```

iii. The trajectory says the AI observed slight animal-specific differences in aligned recording length and chose proportional edges to avoid inventing or dropping a final fragment of each nominal 40-minute session, while still removing unrecorded neurons stored as all NaNs (steps 7 and 12).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading each large per-animal `joblib` object, aggregating every full session with `np.add.reduceat`, and copying 40 trial slices per session into standalone arrays.

ii.
```python
outer = joblib.load(DATA_DIR / animal)
...
counts = np.add.reduceat(tr, starts, axis=1)
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    ...
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. The trajectory repeatedly frames the conversion as a tractability problem for a very large dataset and focuses on memory-conscious full-dataset processing. It does not explicitly benchmark costs, but it identifies loading the large animal files and session-wise aggregation as the central workload (steps 7, 9, 10, and 12).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over the 40 trial windows could have been vectorized by reshaping the 2,400 rebinned time bins into `(40, 60)` blocks instead of appending one trial at a time. The repeated per-trial copies of `geom` and `labels` are also scalar Python-loop work.

ii.
```python
sess_neural, sess_input, sess_output = [], [], []
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. The trajectory does not explicitly discuss vectorization opportunities. Its optimization rationale is mostly about reducing dataset size and avoiding repeated full-animal loads, not about eliminating Python loops (steps 8, 9, and 12).

## 6-c. What processing does the code repeat multiple times?

i. The code repeats per-trial copying within every session: it copies the same static geometry vector 40 times, slices and copies every 60-second neural block separately, and slices and copies every 60-second label block separately.

ii.
```python
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. The trajectory justifies copies as keeping the pickle independent of full-session buffers and emphasizes memory-conscious conversion, but it does not explicitly discuss the repeated duplication of identical per-trial inputs (step 12 and the inline code comment).

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes per-second mean x/y positions only to immediately collapse them into categorical labels, duplicates the same static geometry vector for every trial rather than storing it once per session, and builds detailed `session_info` metadata that the downstream decoder does not require.

ii.
```python
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
pos_mean = pos_sum / widths[None, :]
...
sess_input.append(geom.copy())
...
session_info.append({
    'subject': animal, 'day_index': int(day), 'environment': env,
    'source_frames': int(traces.shape[-1]),
    'n_neurons': n_cells, 'n_trials': 40,
})
```

iii. The trajectory’s justification is validator compatibility and richer metadata rather than minimality. It does not claim these extra computations are needed by the downstream decoder itself (steps 12, 15, and 22).
