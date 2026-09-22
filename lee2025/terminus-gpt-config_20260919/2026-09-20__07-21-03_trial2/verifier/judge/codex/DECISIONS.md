# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers the seven extensionless `QLAK-CA1-*` animal files in `/app/data`, loads each compressed joblib object, selects its animal-keyed dictionary, and iterates over every recording day. It loads one animal at a time and frees it afterward.

ii.
```python
def animal_files():
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.name.startswith('QLAK-CA1-') and not p.suffix and '_' not in p.name)
...
for si, f in enumerate(files):
    d=joblib.load(f)[f.name]
    for day in range(d['trace'].shape[0]):
        ...
    del d; gc.collect()
```

iii. The notes say these are the authors' compressed files used by reference `load_dat`, contain the same seven animals and 207 days, and are loaded sequentially to bound memory and avoid repeatedly decompressing an animal for each day.

## 1-b. How are the data split into subjects?

i. Each discovered joblib file is one mouse. Its filename is the subject ID; the sorted file order defines `subject_idx`. The full subject vocabulary is retained even in sample mode.

ii.
```python
files=animal_files()
subjects=[p.name for p in files]
...
for si, f in enumerate(files):
    ...
    subject_idx.append(si)
```

iii. The agent found seven animal-keyed deposits and documented that each joblib root is `{animal_id: dataset}`.

## 1-c. How are the data split into sessions?

i. Every recording day along axis 0 of `trace` becomes one output session, with the corresponding day from `position`, `blocked`, and `envs`.

ii.
```python
for day in range(d['trace'].shape[0]):
    nmat, out, geom, present, ncorrect = process_day(
        d['trace'][day], d['position'][day], d['blocked'][day])
    ...
    neural.append(ns); outputs.append(os); inputs.append(ins)
```

iii. The notes justify a day as a session because it preserves a constant simultaneously recorded cell set and matches the paper's one-session-per-day organization.

## 1-d. How are the data split into trials?

i. After temporal pooling, each session is divided into contiguous, non-overlapping 600-bin segments (60 seconds at 100 ms/bin). Only complete minutes are retained; the tail is dropped.

ii.
```python
ntrials=nmat.shape[1]//TRIAL_BINS
keep=ntrials*TRIAL_BINS
ns=[np.ascontiguousarray(x, dtype=np.float32)
    for x in np.split(nmat[:,:keep],ntrials,axis=1)]
os=[np.ascontiguousarray(x, dtype=np.int8)
    for x in np.split(out[:,:keep],ntrials,axis=1)]
```

iii. The instructions require one-minute trials. The notes explain that incomplete tails are neither padded nor represented as short trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial behavioral or signal-quality filter. Complete minutes are retained, tails are discarded, and an entire session would be skipped if it had fewer than two full trials.

ii.
```python
if ntrials < 2:
    print(f'WARNING skipping {f.name} day {day}: only {ntrials} full trials')
    continue
```

iii. The two-trial minimum implements the target-format requirement. In the full data every session has 39 or 40 trials, so this check removes nothing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's `trace`, described as already curated binary calcium rising-event traces. Cells absent that day are represented by NaNs.

ii.
```python
nmat, out, geom, present, ncorrect = process_day(
    d['trace'][day], d['position'][day], d['blocked'][day])
```

iii. The agent concluded from the repository and data inspection that `trace` is already rise-extracted, so dF/F or deconvolution should not be recomputed.

## 2-b. How is the `neural` data processed?

i. Day-present traces are transposed to time-by-neuron, cast to float32, Gaussian-smoothed over time with sigma 3 native frames, averaged in non-overlapping groups of three frames, and finally returned neuron-by-time.

ii.
```python
raw = np.asarray(trace_day[present].T, dtype=np.float32)
smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
pooled_neural = smooth[:n_used].reshape(
    n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
...
return pooled_neural.T, classes[None, :], geom, present, n_corrected
```

iii. The notes say this matches the paper repository's `fit_decoder`/`test_decoder` temporal aggregation: sigma-3 smoothing followed by stride-3 average pooling. Whole-session processing avoids filter artifacts at artificial minute boundaries.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained if its first sample is finite, with an assertion that this presence mask agrees at the final sample. Any non-finite value among retained traces raises an error. No place-cell or activity threshold is applied.

ii.
```python
present = np.isfinite(trace_day[:, 0])
if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
    raise ValueError('Cell registration mask changes within a session')
raw = np.asarray(trace_day[present].T, dtype=np.float32)
if not np.isfinite(raw).all():
    raise ValueError('Non-finite value in a day-present neural trace')
```

iii. NaNs encode cross-day registered cells absent from a particular day. The agent retained all curated day-present CA1 cells, reasoning that place-cell filtering belongs to map analyses and the paper includes all cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. The alignment event is the start of each artificial contiguous minute; neural and position streams use identical pooling and minute boundaries.

ii.
```python
temporal_alignment_event='Start of each contiguous non-overlapping 1-minute segment within a recording day',
off_start=0.0, off_end=60.0
```

iii. The source is continuous free navigation without stimulus-defined trials, so the notes define artificial minute starts as the only applicable event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 100 ms. Native 30 Hz samples are smoothed and then rebinned by averaging each non-overlapping group of three frames.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
...
pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
```

iii. The agent chose this to mirror temporal processing in the repository's position-decoder functions.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the per-day `blocked` list; `envs` is retained only as descriptive metadata.

ii.
```python
def blocked_indices(raw):
    vals = np.asarray(raw[0]).ravel().astype(int)
    return vals[vals >= 0]
...
geom = geometry_vector(blocked_raw)
```

iii. The notes identify `blocked` as row-major 3x3 partition indices, with `-1` meaning no blocked partition, and say it agrees with the repository's named geometry matrices.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A nine-element float32 accessibility vector is initialized to ones and blocked indices are set to zero. A copy of this static vector is stored for every trial.

ii.
```python
def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom
...
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The accessibility convention (`1=accessible`, `0=blocked`) matches the paper repository's `get_env_mat`; explicit accessibility names avoid ambiguity.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each day's two-coordinate `position` array.

ii.
```python
pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
```

iii. The agent verified that position and trace have matching native frame counts and are simultaneously sampled at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. X and y are averaged over the same non-overlapping three-frame groups as neural data, discretized into a 3x3 grid, and rare labels falling in blocked partitions are reassigned to the nearest accessible grid center.

ii.
```python
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
classes = (ybin * 3 + xbin).astype(np.int8)
classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
```

iii. Pooling was chosen to match the reference decoder's temporal aggregation. The agent says snapping 152 rare pooled tracking artifacts follows the spirit of the repository's valid-map snapping and prevents geometry/output contradictions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each 75 cm axis is divided into three 25 cm bins. Coordinates divided by 25 are truncated and clipped to 0--2, then flattened row-major as `y_bin * 3 + x_bin`, producing classes 0--8.

ii.
```python
xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
classes = (ybin * 3 + xbin).astype(np.int8)
```

iii. The 3x3 task and 75 cm arena imply 25 cm cells. The agent tested orientation against blocked masks and found row-major `y*3+x` was consistent with the data.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is truncated to the same native-frame count, pooled over the identical groups of three frames, and sliced at the same minute boundaries as neural data.

ii.
```python
n_pool = raw.shape[0] // POOL
n_used = n_pool * POOL
...
pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
...
ns=[... for x in np.split(nmat[:,:keep],ntrials,axis=1)]
os=[... for x in np.split(out[:,:keep],ntrials,axis=1)]
```

iii. The notes report equal source lengths for every session and independent exact reconstruction checks of the aligned outputs.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN absent cells are removed; inconsistent or partial non-finite neural data causes a hard error. Incomplete temporal pooling groups and sub-minute tails are discarded. Coordinates are clipped to valid grid indices, and labels in blocked cells are snapped to a nearest accessible cell.

ii.
```python
present = np.isfinite(trace_day[:, 0])
...
n_pool = raw.shape[0] // POOL
n_used = n_pool * POOL
...
keep=ntrials*TRIAL_BINS
...
classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
```

iii. The agent treats NaNs as registration absence, refuses unexpected non-finite retained data, avoids padding incomplete trials, and regards blocked-bin observations as rare boundary/tracking artifacts.

## 6-a. What are the most time-consuming steps of the code?

i. Decompressing/loading the large joblib animal files and serializing the roughly 6.17 GiB converted pickle dominate; vectorized per-session conversion is comparatively fast.

ii.
```python
t_load=time.time(); print(f'Loading {f.name}...', flush=True)
d=joblib.load(f)[f.name]
...
with open(outfile,'wb') as fh: pickle.dump(data,fh,protocol=5)
```

iii. The notes measured roughly 9--23 seconds per animal load and about 0.2 seconds per session conversion, and anticipated saving as another major cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Core numerical operations are already vectorized. The remaining list comprehensions that create trial arrays and repeat geometry could be replaced by reshape/view-based construction only if the required nested-list output and independent arrays were handled carefully; subject/day iteration is appropriate for variable neuron counts and bounded memory.

ii.
```python
ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The notes explicitly credit vectorized Gaussian filtering, reshape pooling, discretization, and blocked-bin correction; they do not identify a materially expensive loop left to vectorize.

## 6-c. What processing does the code repeat multiple times?

i. Geometry is copied once per trial, and per-trial contiguous casts/copies are repeated while constructing the required nested representation. Validation iterates through every trial after conversion. Expensive smoothing and pooling occur only once per full session.

ii.
```python
ins=[geom.copy() for _ in range(ntrials)]
...
for sidx in range(len(neural)):
    ...
    for n,i,o in zip(neural[sidx],inputs[sidx],outputs[sidx]):
        assert ...
```

iii. The agent deliberately performs session-wide filtering before slicing, noting that this avoids repeated trial computations and filter-boundary artifacts. Repeated copies/checks support the specified structure and validation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes provenance and diagnostics such as `present`, correction counts, environment strings, timing, output-class summaries, and optional plots; these do not enter decoder features. It also computes `used_subjects` but never uses it.

ii.
```python
used_subjects=sorted(set(subject_idx))
...
info=dict(... corrected_blocked_pooled_bins=int(ncorrect))
session_info.append(info)
...
if show_processing and plot_count < 2:
    plot_processing(...)
```

iii. Most diagnostics were intentionally retained for provenance, sanity checks, and auditability. `used_subjects` is genuinely dead computation; plotting is optional and limited to two sessions.
