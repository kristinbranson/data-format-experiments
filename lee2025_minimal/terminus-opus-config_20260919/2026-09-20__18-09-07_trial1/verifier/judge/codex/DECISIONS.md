# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs and loads each extensionless per-animal joblib file. It indexes the returned dictionary by animal ID and processes every recording day in `trace`, yielding all 207 sessions.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
pos_all, trace_all = dat['position'], dat['trace']
n_days = trace_all.shape[0]
for day in range(n_days):
```

iii. The trajectory says the agent inspected both file forms and the repository, successfully loaded a joblib file, verified the field shapes, and confirmed that the seven files contain 207 days, matching the paper.

## 1-b. How are the data split into subjects?

i. One extensionless joblib file is one subject. Subject order is the fixed `ANIMALS` order, and each emitted session receives that animal's index.

ii.
```python
data = {'subjects': list(animals), 'subject_idx': [], ...}
for ai, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    ...
    data['subject_idx'].append(ai)
```

iii. The agent established that the data are stored per animal and explicitly checked all seven animal datasets.

## 1-c. How are the data split into sessions?

i. Each recording day is a session. The code iterates the first dimension of `trace` and uses the corresponding `position`, `envs`, and `blocked` entry.

ii.
```python
n_days = trace_all.shape[0]
for day in range(n_days):
    env = envs[day]
    blocked = np.array(dat['blocked'][day]).ravel().astype(int)
    pos = pos_all[day].T.astype(np.float64)
    trace = trace_all[day]
```

iii. The trajectory identifies each day as a roughly 40-minute session and confirms 207 total days/sessions, as reported in the paper.

## 1-d. How are the data split into trials?

i. After temporal pooling, each session is divided into consecutive, non-overlapping 600-bin (60-second) trials. The trailing partial minute is discarded; sessions with fewer than two complete trials are skipped.

ii.
```python
TRIAL_BINS = int(round(TRIAL_SEC * 1000.0 / BIN_MS))
...
n_bins = min(neural.shape[1], labels.shape[0])
n_trials = n_bins // TRIAL_BINS
if n_trials < 2:
    continue
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
```

iii. The one-minute segmentation is required by the task. Using integer division gives uniform trials and satisfies the minimum-two-trials decoder constraint.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality filter. Only full one-minute trials are retained, and a session would be skipped if it had fewer than two. The paper's scattered speed-based frame filter is deliberately not used.

ii.
```python
n_trials = n_bins // TRIAL_BINS
if n_trials < 2:
    print(f'  skipping {animal} day {day}: < 2 full trials')
    continue
```

iii. The agent states that removing frames below 5 cm/s would destroy contiguous one-minute trials and the common neural/behavioral time base; position remains meaningful during immobility.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace` array: binarized rising phases of calcium transients.

ii.
```python
pos_all, trace_all = dat['position'], dat['trace']
...
trace = trace_all[day]
```

iii. The trajectory records that repository inspection identified `trace` as binary transient events at 30 Hz, with NaN rows for cells not registered on a day.

## 2-b. How is the `neural` data processed?

i. Registered, sufficiently active cells are selected; their float32 traces are Gaussian-smoothed along time with sigma 3 frames and mean-pooled in non-overlapping groups of three frames, producing neuron-by-time matrices at 100 ms resolution.

ii.
```python
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
```

iii. The agent inspected `utils.fit_decoder`/`test_decoder` and chose the paper repository's decoding pipeline: sigma-3 smoothing followed by three-frame temporal binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained only if it is registered that day (not an all-NaN row) and has more than five transients over the session. A session with no passing cells is skipped.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=1)
cell_ids = np.where(registered)[0]
tr = trace[cell_ids]
enough = np.nansum(tr, axis=1) > CELL_EVENT_THRESHOLD
cell_ids, tr = cell_ids[enough], tr[enough]
if tr.shape[0] == 0:
    continue
```

iii. The threshold comes from `utils.decode_position_within` (`cell_threshold=5`). Because speed filtering is omitted, the threshold is evaluated over the whole session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Each trial is aligned to the arbitrary start of a consecutive one-minute segment of continuous exploration; metadata records offsets 0 to 60 seconds.

ii.
```python
'temporal_alignment_event': (
    'start of each 1-min trial, i.e. arbitrary segmentation of the '
    'continuous 40-min free-exploration session (no task events)'),
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. The experiment is continuous free exploration, so the trial boundary is the only applicable alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output bin is 100 ms. Native 30 Hz frames are average-pooled three at a time after neural smoothing; position is pooled over the same groups.

ii.
```python
FPS = 30.0
TEMPORAL_BIN = 3
BIN_MS = 1000.0 * TEMPORAL_BIN / FPS
...
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T
```

iii. The agent matched the temporal bin size used by the paper's position-decoding functions.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the per-day `blocked` partition indices. Negative `-1` entries mean that no partition is blocked and are removed.

ii.
```python
blocked = np.array(dat['blocked'][day]).ravel().astype(int)
blocked = blocked[blocked >= 0]
```

iii. Repository/data inspection showed that `blocked` names occluded cells of the arena's 3×3 partition grid.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The indices become a static nine-element float32 multi-hot vector: 1 for blocked and 0 for accessible. A copy is stored for every trial.

ii.
```python
geom = np.zeros(NGRID * NGRID, dtype=np.float32)
geom[blocked] = 1.0
...
input_trials.append(geom.copy())
```

iii. The representation exposes each grid partition independently and directly follows the dataset's blocked-index convention.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the per-day `position` field, containing simultaneous x-y DeepLabCut coordinates in centimeters.

ii.
```python
pos_all, trace_all = dat['position'], dat['trace']
...
pos = pos_all[day].T.astype(np.float64)
```

iii. The agent inspected the data and verified coordinates span the 0–75 cm arena and share the 30 Hz acquisition rate.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. X-y coordinates are mean-pooled over each three-frame temporal bin, discretized into the 3×3 arena partitions, and rare samples assigned to a blocked partition are snapped to the nearest accessible partition center.

ii.
```python
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T
labels = partition_labels(pos_b, open_mask).astype(np.int64)
...
bad = ~open_mask[labels]
labels[bad] = open_idx[np.argmin(d, axis=1)]
```

iii. The trajectory empirically verified the grid indexing against occupancy. Snapping handles the observed sub-1% wall/tracking leakage and mirrors the paper code's restriction to visitable rate-map bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each 75 cm axis is divided at 25 and 50 cm. Coordinates are floored into bins 0–2 and clipped at arena boundaries; the class is `3*y_bin + x_bin`, yielding labels 0–8.

ii.
```python
xb = np.clip(np.floor(pos_binned_xy[:, 0] / PART_SIZE).astype(int), 0, NGRID - 1)
yb = np.clip(np.floor(pos_binned_xy[:, 1] / PART_SIZE).astype(int), 0, NGRID - 1)
labels = NGRID * yb + xb
```

iii. The agent verified this mapping over all animals against the `blocked` field's `[[0,1,2],[3,4,5],[6,7,8]]` indexing.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are simultaneously sampled at 30 Hz, pooled with the same three-frame stride, truncated to their common binned length, and sliced with identical trial boundaries.

ii.
```python
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T
n_bins = min(neural.shape[1], labels.shape[0])
...
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
output_trials.append(labels[sl][None, :].copy())
```

iii. The paper states that imaging and behavior were simultaneously acquired; identical pooling and slices preserve frame correspondence.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN unregistered cells are removed; negative blocked sentinels are ignored; rare tracking samples inside blocked geometry are reassigned to the nearest open partition; stream lengths are limited to their common minimum; partial final trials are dropped. No general interpolation is performed.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=1)
blocked = blocked[blocked >= 0]
...
n_bins = min(neural.shape[1], labels.shape[0])
n_trials = n_bins // TRIAL_BINS
```

iii. The trajectory explicitly investigated NaN structure, blocked-bin leakage, and lengths. These operations address known dataset conventions and small tracking artifacts without inventing neural observations.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant work is loading large joblib arrays, Gaussian-filtering every retained neuron's full session trace, constructing the large nested output, and serializing the approximately 6.65 GB pickle. The agent did not provide a timed profile.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
...
pickle.dump(data, f, protocol=4)
```

iii. The trajectory monitored runtime and final size but offered no benchmark separating I/O, filtering, and serialization costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python trial loop could be replaced by a reshape/move-axis operation to form trial views before converting to the required lists. Animal/day loops are appropriate because cell counts, blocked geometry, and retained lengths vary by session.

ii.
```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(labels[sl][None, :].copy())
```

iii. The trajectory contains no explicit vectorization analysis. The loop is simple and also creates the list structure required downstream.

## 6-c. What processing does the code repeat multiple times?

i. For every day it recomputes registered/activity masks, temporal smoothing/pooling, open-grid centers when needed, and trial copies. The static geometry vector is copied once per trial, although its value is unchanged within a session.

ii.
```python
for day in range(n_days):
    ...
    tr_s = gaussian_filter1d(...)
    neural = pool_mean(...)
    pos_b = pool_mean(...)
    ...
    input_trials.append(geom.copy())
```

iii. The repeated session-level transformations are necessary because each day has different cells and geometry. The trajectory gives no separate justification for per-trial geometry copies; they prevent aliasing and satisfy the target representation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No major derived data are immediately discarded. `env`, `cell_ids`, and detailed session information are retained only as metadata and are not used by decoder training. The trailing incomplete bins are intentionally discarded, and `.copy()`/contiguity operations increase memory work but produce independent, decoder-ready arrays.

ii.
```python
session_info.append({
    'subject': animal, 'day': int(day), 'environment': env,
    ... 'cell_ids': [int(c) for c in cell_ids],
})
...
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. The trajectory does not identify unnecessary processing. It did retain provenance metadata to make the conversion auditable and patched it to be JSON-serializable.
