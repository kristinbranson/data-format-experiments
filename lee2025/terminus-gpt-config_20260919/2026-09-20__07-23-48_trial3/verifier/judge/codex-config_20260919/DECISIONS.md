# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The code uses a hard-coded list of seven animals, loads each animal's extensionless compressed joblib file, unwraps the animal-keyed dictionary, and iterates over aligned `trace`, `position`, `blocked`, and `envs` session arrays. Full mode selects all seven; sample mode selects the first two.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
selected=ANIMALS[:2] if sample else ANIMALS
for subj,animal in enumerate(selected):
    root=joblib.load('/app/data/'+animal); dat=root[animal]
    for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
```

iii. The notes say joblib is the representation preferred by the repository's `load_dat`, avoids duplicate MATLAB data, and permits one-animal-at-a-time loading to bound memory. The animal order is the reference order.

## 1-b. How are the data split into subjects?

i. Each named joblib file is one mouse. The hard-coded animal list becomes `subjects`, and the enumeration index is appended once for every session.

ii.
```python
for subj,animal in enumerate(selected):
    ...
    subject_idx.append(subj)
...
'subjects':selected,'subject_idx':np.asarray(subject_idx,dtype=np.int64),
```

iii. The notes identify the same seven animals as the paper/repository and preserve their reference ordering.

## 1-c. How are the data split into sessions?

i. Each native recording day is an output session. The code zips four per-day fields and appends one nested list of trials for each day.

ii.
```python
for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
    nt,it,ot,ncells,nfixed,tail=process_session(tr,pos,blk,animal,day)
    neural.append(nt); inputs.append(it); outputs.append(ot)
```

iii. The notes state that sessions are one approximately 40-minute recording per animal-day and report 207 sessions, matching the released data and paper counts.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second windows. At 30 Hz this is 1,800 source frames; after pooling by three, every trial has 600 bins. An incomplete final window is discarded, and sessions with fewer than two complete trials raise an error.

ii.
```python
TRIAL_FRAMES = 60 * FPS
TRIAL_BINS = TRIAL_FRAMES // POOL
ntrials = trace.shape[1] // TRIAL_FRAMES
nframes = ntrials * TRIAL_FRAMES
if ntrials < 2:
    raise ValueError(...)
neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. This directly implements the requested artificial one-minute trials. The notes document 8,187 complete trials and explicitly justify dropping only incomplete tails.

## 1-e. How are trials filtered based on quality controls?

i. No complete trial is removed for behavior, speed, event count, or place-cell quality. Only the incomplete session tail is discarded; source dimensions and values are validated, and fewer than two complete trials is fatal.

ii.
```python
ntrials = trace.shape[1] // TRIAL_FRAMES
nframes = ntrials * TRIAL_FRAMES
if ntrials < 2:
    raise ValueError(...)
```

iii. The agent says speed and event-count restrictions belong to a specific reference decoder analysis and would disrupt the fixed continuous one-minute target trials, so all complete windows are retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace` array in the released joblib data. These are already extracted binary rising-phase calcium events, not raw fluorescence.

ii.
```python
for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
    nt,it,ot,ncells,nfixed,tail=process_session(tr,pos,blk,animal,day)
...
trace = np.asarray(trace)
```

iii. The notes cite the repository and methods: finite `trace` values are 0/1 after upstream event extraction, so delta-F/F is not recomputed.

## 2-b. How is the `neural` data processed?

i. All-NaN cell rows are removed, finite binary events are cast to float32, Gaussian-smoothed along time with sigma three source frames, then averaged in non-overlapping groups of three. The resulting session array is sliced into contiguous neuron-by-600 trial matrices.

ii.
```python
raw = trace[finite_all]
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
pooled = pooled.astype(np.float32)
```

iii. The agent explicitly chose the preprocessing in the paper repository's position decoder (`sigma=3`, then `AvgPool1d(3,3)`), believing “same processing” applied to the target decoder. It rejected additional fluorescence processing because the released events are already processed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained only when every sample in that session is finite. Entirely missing rows are removed; partially missing rows, non-binary finite data, and malformed dimensions cause errors. No place-cell, speed, or minimum-event filter is imposed.

ii.
```python
finite_all = np.all(np.isfinite(trace), axis=1)
finite_any = np.any(np.isfinite(trace), axis=1)
if np.any(finite_any != finite_all):
    raise ValueError(...)
raw = trace[finite_all]
vals = np.unique(raw)
if not np.all(np.isin(vals, [0, 1])):
    raise ValueError(...)
```

iii. The notes report that absent longitudinal identities are represented by wholly NaN rows and that the paper's total 69,744 session-neurons is recovered by keeping all finite rows. Analysis-specific place-cell/event filters were intentionally not used.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Each trial aligns to the start of its consecutive one-minute segment. Neural and behavior are processed from corresponding source frames, and metadata records offsets of 0 to 60 seconds.

ii.
```python
'temporal_alignment_event':'start of each consecutive non-overlapping 60-second segment within a recording session',
'off_start':0.0,'off_end':60.0,
```

iii. The recordings are continuous and have no native trials or stimulus onset. The agent validated equal trace/position frame counts for all sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Converted data use 100 ms bins. Both streams are rebinned from 30 Hz by non-overlapping three-frame averaging; neural activity is smoothed before pooling.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
...
'time_bin_size':BIN_MS
```

iii. The agent adopted the reference repository decoder's default three-frame temporal binning and describes this as matching its neural and behavioral operations.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the per-session `blocked` indices. The simultaneously loaded `envs` value is retained only as metadata, not used to construct the input.

ii.
```python
bmask = blocked_vector(blocked)
...
env_name=str(np.asarray(env).reshape(-1)[0])
```

iii. The notes say `blocked` directly stores omitted row-major 3×3 partitions and that repeated named geometries have canonical matching masks.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Nonnegative blocked indices are converted to a nine-element float32 multi-hot vector, where 1 means blocked and 0 means accessible. `-1` therefore produces all zeros. A copy of the static session mask is stored for every trial.

ii.
```python
out = np.zeros(9, dtype=np.float32)
idx = np.asarray(blocked).reshape(-1).astype(int)
idx = idx[idx >= 0]
out[idx] = 1
...
input_trials = [bmask.copy() for _ in range(ntrials)]
```

iii. The agent chose direct source encoding, documented the polarity, and used float32 because uint8 inputs produced validator warnings.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from each session's two-row `position` array: x in row zero and y in row one.

ii.
```python
position = np.asarray(position)
if trace.ndim != 2 or position.ndim != 2 or position.shape[0] != 2:
    raise ValueError(...)
```

iii. The notes identify this as framewise DeepLabCut head location, simultaneously recorded and matched in length to `trace`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is truncated to complete trials and averaged within the same non-overlapping three-frame groups as neural data. The pooled x/y points are discretized into a 3×3 grid. Rare labels falling in blocked partitions are reassigned to the nearest open-cell center.

ii.
```python
return p.reshape(2, -1, POOL).mean(axis=2)
...
labels = row * 3 + col
invalid = blocked_mask[labels].astype(bool)
...
nearest = np.argmin(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
labels[invalid] = open_labels[nearest]
```

iii. The agent says temporal averaging matches the paper's decoder. It found 152 of 4,912,200 pooled points in blocked cells and treated these as tracking/interpolation artifacts, applying what it viewed as an analogue of the reference decoder's nearest-valid-bin cleanup.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is floor-divided by 25 cm and clipped to 0–2. The class is row-major `y_bin * 3 + x_bin`, yielding categories 0–8. Blocked categories are subsequently snapped to a nearest accessible class.

ii.
```python
col = np.clip(np.floor(xy[0] / 25.0), 0, 2).astype(np.int64)
row = np.clip(np.floor(xy[1] / 25.0), 0, 2).astype(np.int64)
labels = row * 3 + col
```

iii. The 75 cm arena naturally gives 25 cm bins. The notes report that the opposite orientation put 17.1% of frames in blocked cells, supporting row-major `y*3+x`; clipping handles arena boundaries.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The code first requires identical source frame counts, then independently mean-pools both streams over identical three-frame groups, truncates both to the same number of complete one-minute trials, and slices both using identical 600-bin boundaries.

ii.
```python
if trace.shape[1] != position.shape[1]:
    raise ValueError(...)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
xy = pool_position(position, nframes)
output_trials = [np.ascontiguousarray(labels[None, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. The agent relied on simultaneous acquisition/frame timestamps and independently checked converted trials against raw source calculations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Whole-session NaN cell rows are interpreted as unregistered cells and removed. Partial neural missingness, nonfinite position, nonbinary traces, length mismatch, invalid blocked indices, or malformed dimensions raise errors. Incomplete terminal frames are dropped. Rare pooled positions in blocked areas are reassigned to the nearest open partition.

ii.
```python
if np.any(finite_any != finite_all): raise ValueError(...)
if p.shape != (2, nframes) or not np.isfinite(p).all(): raise ValueError(...)
if np.any(idx > 8): raise ValueError(...)
labels[invalid] = open_labels[nearest]
```

iii. The agent distinguished expected all-NaN registration padding from unexpected corruption, favored fail-fast validation, and documented tail loss and the 0.0031% blocked-label correction.

## 6-a. What are the most time-consuming steps of the code?

i. Loading/decompressing the large per-animal joblib files, Gaussian filtering the very large neuron-by-frame arrays, and writing the roughly 6.17 GiB pickle dominate. Optional plotting and garbage collection add smaller costs.

ii.
```python
root=joblib.load('/app/data/'+animal)
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
with open(outfile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes emphasize 15 GB of source data and about 1.674 billion converted neural values, report a 203.7-second full conversion, and identify whole-session vectorization and one-animal loading as speed/memory measures.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Core numeric operations are already vectorized. The remaining Python loops over animals and sessions are needed for ragged files/arrays; trial list construction and invariant checks could be reduced with reshaping/batched validation, but the required nested-list output still needs per-trial objects.

ii.
```python
neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
for sidx in range(ns):
    for n,x,y in zip(neural[sidx],inputs[sidx],outputs[sidx]):
```

iii. The notes claim Gaussian filtering, pooling, discretization, and blocked-bin correction were vectorized and acknowledge that per-trial copies are driven by the target nested-list structure.

## 6-c. What processing does the code repeat multiple times?

i. `blocked_vector(blk)` is recomputed when creating session metadata and again for optional plots after already being computed inside `process_session`. Each trial also receives a duplicate geometry array, and invariant checks revisit every trial after construction.

ii.
```python
nt,it,ot,...=process_session(tr,pos,blk,animal,day)
...
'blocked_indices':np.flatnonzero(blocked_vector(blk)).tolist(),
...
plot_processing(...,blocked_vector(blk))
```

iii. No explicit justification is given for recomputing the tiny nine-value mask. Copies per trial are justified by the target representation, and the second pass supplies structural validation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `envs` only to create metadata, computes timing/progress statistics, runs extensive validation passes, calls garbage collection, and optionally builds plots; these do not feed decoder arrays. It also creates `nfixed`, tail, and provenance summaries used only for logging/metadata.

ii.
```python
env_name=str(np.asarray(env).reshape(-1)[0])
session_info.append({...})
if show_processing and plot_count < 2:
    plot_processing(...)
del root,dat; gc.collect()
```

iii. These steps are intended for provenance, sanity checks, diagnostics, and memory control rather than decoder computation. The agent considered them useful verification rather than scientific preprocessing.
