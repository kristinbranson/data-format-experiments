# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven curated subject IDs and loads each extensionless `/app/data/<subject>` joblib file one at a time. It extracts `trace`, `position`, `envs`, and `blocked`, processes every daily recording, then frees each subject before loading the next. This produced 207 sessions.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
for subj_i, subject in enumerate(SUBJECTS):
    source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
    traces = source['trace']
    positions = source['position']
    envs = np.asarray(source['envs']).reshape(-1)
    blocked = source['blocked']
```

iii. The trajectory says the AI inspected both the HDF5 `.mat` files and the smaller extensionless files, found the latter loadable by joblib and containing the same structured data, and regarded them as the intended reference format. Loading one animal at a time was chosen to limit memory.

## 1-b. How are the data split into subjects?

i. Each hard-coded subject ID identifies one joblib file and one top-level dictionary entry. `subject_idx` records the enumerated subject index for each daily recording.

ii.
```python
for subj_i, subject in enumerate(SUBJECTS):
    source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
    ...
    subject_idx.append(subj_i)
```

iii. The AI found exactly seven animal files and identified the seven IDs during data inspection.

## 1-c. How are the data split into sessions?

i. Every daily recording (the first dimension of `trace`) becomes a separate decoder session. There are six animals with 31 recordings and one with 21, for 207 sessions.

ii.
```python
for day in range(traces.shape[0]):
    tr = traces[day]
    pos = positions[day]
    ...
    neural.append(ses_neural)
    inputs.append(ses_input)
    outputs.append(ses_output)
```

iii. The trajectory states that neuron availability differs by recording, making a daily recording the natural session unit, and notes that the resulting 207 sessions match the paper.

## 1-d. How are the data split into trials?

i. Each recording is first forced into 2,400 equal-duration bins. Those bins are sliced into 40 consecutive, non-overlapping trials of 60 bins, nominally one minute each. Unlike the reference, this does not use native 1,800-frame windows or discard a remainder.

ii.
```python
N_SECONDS = 40 * 60
TRIAL_SECONDS = 60
...
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    ses_neural.append(counts[:, start:stop].copy())
```

iii. The AI reasoned that all recordings nominally lasted 40 minutes but had slightly different frame counts after timestamp alignment. It chose equal partitions to preserve every frame and always obtain exactly 40 trials instead of losing data near the end.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied; all 40 constructed trials from every recording are retained.

ii.
```python
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    ...
    ses_neural.append(counts[:, start:stop].copy())
    ses_input.append(geometry_ts.copy())
    ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The trajectory identified the provided subject files as already curated and did not find a basis for excluding individual trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `source['trace']`, a session × tracked-cell × aligned-frame array.

ii.
```python
traces = source['trace']
...
tr = traces[day]
```

iii. Inspection showed that valid trace values are binary 0/1 events and NaN rows denote cells absent from a recording. The AI interpreted the trace as the paper's thresholded rising-phase calcium-event vector.

## 2-b. How is the `neural` data processed?

i. Absent-neuron rows are removed. The remaining binary events are summed within 2,400 equal bins per recording, converted to `uint8`, and sliced into neuron × 60-bin trials.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
...
ses_neural.append(counts[:, start:stop].copy())
```

iii. The AI wanted a compact, tractable decoder dataset and treated summed binary events as one-second event counts. It also argued that equal bins accommodate the slight variation around 30 Hz while preserving all frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained only if its recording-specific row contains no NaN. Empty sessions raise an error. No activity-rate or other cell-quality filter is applied.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
```

iii. The AI checked that NaN masks were constant across time within each recording, supporting the conclusion that NaN rows are absent tracked neurons rather than intermittent missing samples.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Alignment is to the start of each artificial consecutive one-minute segment; metadata records offsets of 0 to 60 seconds.

ii.
```python
'temporal_alignment_event': 'start of each consecutive one-minute segment',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The recording is continuous and the requested trials are artificial one-minute segments, so the AI used segment start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has nominal 1,000 ms bins. Temporal rebinning is applied: every recording's approximately 72,000 frames are partitioned into 2,400 groups, and events are summed per group.

ii.
```python
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
...
'time_bin_size': 1000.0,
```

iii. The trajectory says raw 30 Hz data would make a multi-gigabyte output and expensive decoder workload. The AI chose one-second aggregation for compactness and to normalize slightly differing frame counts.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from `source['blocked']` for each daily recording. `envs` is saved only as session metadata and is not a decoder input.

ii.
```python
envs = np.asarray(source['envs']).reshape(-1)
blocked = source['blocked']
...
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
```

iii. Data inspection showed that `blocked` directly identifies inaccessible cells in the arena's row-major 3×3 grid, including `-1` as the open-square sentinel.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Negative sentinel values are removed and the remaining indices set entries in a nine-element blocked/not-blocked vector. The vector is repeated across 60 time bins and copied into every trial of that session.

ii.
```python
geometry = np.zeros(9, dtype=np.float32)
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
blocked_idx = blocked_idx[blocked_idx >= 0]
geometry[blocked_idx] = 1.0
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
...
ses_input.append(geometry_ts.copy())
```

iii. The AI chose nine independent binary channels because each source index directly names a blocked spatial compartment. It repeated the static vector to share the decoder's trial time axis.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position comes from `source['position']`, a session × (x, y) × aligned-frame array.

ii.
```python
positions = source['position']
...
pos = positions[day]
```

iii. The AI observed complete positions scaled to the 75 cm arena and aligned frame-for-frame with trace data.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. X and y coordinates are averaged over the same 2,400 temporal bins used for neural aggregation. The averaged coordinates are then converted to a single row-major 3×3 spatial class and sliced into 60-bin trials.

ii.
```python
xy = aggregate_equal_bins(pos, edges, 'mean')
col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
```

iii. The AI averaged position to put it on the same one-second time base as neural counts, then used the arena's designed 25 cm grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is floor-divided by 25 cm and clipped to 0–2. The category is `row * 3 + column`, yielding labels 0–8.

ii.
```python
col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
```

iii. The 75 cm square was designed as a 3×3 grid. Clipping handles occasional exact-boundary coordinates such as 75 cm.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Trace events and positions use identical frame boundaries for each of the 2,400 temporal bins, then identical 60-bin trial boundaries. Neural activity is summed while position is averaged.

ii.
```python
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
ses_neural.append(counts[:, start:stop].copy())
ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The AI verified that source streams were already aligned frame-for-frame and deliberately reused the same aggregation and trial boundaries.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Recording-specific NaN neuron rows are removed. A recording with no neurons or any NaN position raises `ValueError`; missing positions are not imputed. Slight frame-count differences are absorbed by equal-width temporal partitioning, so no remainder frames are dropped.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
if np.isnan(pos).any():
    raise ValueError(f'NaN position: {subject}, day {day}')
```

iii. The AI established that neural NaNs consistently mean an absent cell, while position had no NaNs. It viewed variable frame counts as timestamp/acquisition variation and preserved all samples through equal partitioning.

## 6-a. What are the most time-consuming steps of the code?

i. Loading/decompressing each large joblib subject array is the main I/O cost. The repeated 2,400-bin aggregation of every session, especially neural traces, and final pickle serialization are the main added computation and output costs.

ii.
```python
source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
...
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly notes the large subject files, waits during per-animal loading/conversion, and processes one animal at a time for memory control. It also chose compact one-second data partly to reduce decoder and serialization costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. `aggregate_equal_bins` loops over 2,400 bins for both traces and positions in every session. Because frame counts are not always exactly divisible by 2,400, this could be implemented with indexed/bin reductions; the regular trial loop could also be replaced with reshape/splitting operations.

ii.
```python
return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                 for i in range(len(edges)-1)], axis=-1)
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
```

iii. The code comment explicitly says an explicit loop was selected because `reduceat` can be unsafe with repeated final boundaries and the loop was considered memory-efficient and unambiguous. In these data, bins are about 30 frames, so repeated boundaries should not occur.

## 6-c. What processing does the code repeat multiple times?

i. It performs nearly identical temporal slice-and-reduce passes separately for neural traces and positions in every session. It also copies the same static 9×60 geometry array into all 40 trials and copies every sliced neural and output trial.

ii.
```python
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
ses_neural.append(counts[:, start:stop].copy())
ses_input.append(geometry_ts.copy())
ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The trajectory does not explicitly discuss these repetitions. They follow from applying common alignment boundaries to different streams and making each stored trial self-contained.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The major unnecessary transformation is temporal aggregation to one-second bins, which discards native temporal resolution and is not requested by the reference solution. `envs` and several `session_info` fields are loaded/constructed only as metadata, and repeated copies of static geometry add redundant storage.

ii.
```python
envs = np.asarray(source['envs']).reshape(-1)
...
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
ses_input.append(geometry_ts.copy())
```

iii. The AI intentionally aggregated for tractability and file size, so it did not regard that work as unnecessary. Relative to the human solution, however, it is an extra lossy operation; environment labels and detailed metadata do not enter decoder training.
