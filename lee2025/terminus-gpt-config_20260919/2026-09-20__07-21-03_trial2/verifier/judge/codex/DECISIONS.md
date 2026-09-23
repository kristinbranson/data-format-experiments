# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the extensionless per-animal `joblib` files in `/app/data`, not the `.mat` files. It discovers files whose names start with `QLAK-CA1-`, loads each full animal dictionary with `joblib.load`, then iterates over recording days inside that dictionary.

ii.
```python
def animal_files():
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.name.startswith('QLAK-CA1-') and not p.suffix and '_' not in p.name)

for si, f in enumerate(files):
    d=joblib.load(f)[f.name]
    for day in range(d['trace'].shape[0]):
        nmat, out, geom, present, ncorrect = process_day(
            d['trace'][day], d['position'][day], d['blocked'][day])
```

iii. In `CONVERSION_NOTES.md`, the AI says the reference repository’s `load_dat` function uses the preprocessed joblib format, and that the joblib and `.mat` files are duplicates. It justified this as matching the reference code path more directly and keeping the deposited preprocessing intact.

## 1-b. How are the data split into subjects?

i. Each extensionless `QLAK-CA1-*` file is treated as one mouse. The subject ID is the filename.

ii.
```python
subjects=[p.name for p in files]
...
for si, f in enumerate(files):
    ...
    subject_idx.append(si)
```

iii. The notes say each deposited animal file corresponds to one mouse and that the target `subjects` list should be the sorted animal IDs from those files.

## 1-c. How are the data split into sessions?

i. Each recording day inside an animal dictionary is treated as one session. The AI iterates over `range(d['trace'].shape[0])` and appends one session entry per day.

ii.
```python
for day in range(d['trace'].shape[0]):
    ...
    neural.append(ns); outputs.append(os); inputs.append(ins)
    subject_idx.append(si); region_idx.append(np.zeros(nmat.shape[0],dtype=np.int8))
    info=dict(session_id=f'{f.name}_day{day:02d}', subject=f.name, source_day=day, ...)
    session_info.append(info)
```

iii. The notes state that “each original recording day is one target session” because that preserves the simultaneously recorded cell set and matches the paper’s one-session-per-day framing.

## 1-d. How are the data split into trials?

i. The AI first temporally pools the continuous day-long recording into 100 ms bins, then splits the pooled arrays into contiguous non-overlapping 60 s trials of 600 bins each. It discards any leftover tail shorter than one minute.

ii.
```python
BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)
...
ntrials=nmat.shape[1]//TRIAL_BINS
keep=ntrials*TRIAL_BINS
ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The AI’s notes say the target format requires one-minute trials and that it chose to apply the reference decoder’s 3-frame temporal aggregation before trialization so there would be 600 time bins per minute instead of 1800 native frames.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level behavioral quality-control filter. The only filtering is structural: sessions with fewer than two complete one-minute trials are skipped, and incomplete trailing trial fragments are dropped.

ii.
```python
ntrials=nmat.shape[1]//TRIAL_BINS
if ntrials < 2:
    print(f'WARNING skipping {f.name} day {day}: only {ntrials} full trials')
    continue
keep=ntrials*TRIAL_BINS
```

iii. The notes say the source recordings are continuous and already well aligned, with no invalid-time mask described in the paper or data, so only the downstream decoder’s minimum-trial requirement and exact-minute formatting were enforced.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `d['trace'][day]`, the deposited binary calcium-event traces for one recording day.

ii.
```python
nmat, out, geom, present, ncorrect = process_day(
    d['trace'][day], d['position'][day], d['blocked'][day])
...
raw = np.asarray(trace_day[present].T, dtype=np.float32)  # time x neurons
```

iii. The notes say `trace` is already the authors’ binarized rising-phase event representation, so the AI intentionally did not recompute dF/F or deconvolution.

## 2-b. How is the `neural` data processed?

i. The AI keeps day-present cells, transposes to time-by-neuron, applies Gaussian smoothing with `sigma=3` native frames, then averages non-overlapping groups of 3 frames and finally transposes back to neuron-by-time.

ii.
```python
present = np.isfinite(trace_day[:, 0])
raw = np.asarray(trace_day[present].T, dtype=np.float32)  # time x neurons
n_pool = raw.shape[0] // POOL
n_used = n_pool * POOL
smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
return pooled_neural.T, classes[None, :], geom, present, n_corrected
```

iii. The notes repeatedly justify this as matching the temporal preprocessing used by the reference decoder functions `fit_decoder`/`test_decoder`: Gaussian smoothing followed by 3-frame pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removes cells that are absent on a given day, using finite values in the day’s trace array as the registration mask. It also raises an error if that mask is inconsistent within the session or if kept traces contain non-finite values.

ii.
```python
present = np.isfinite(trace_day[:, 0])
if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
    raise ValueError('Cell registration mask changes within a session')
raw = np.asarray(trace_day[present].T, dtype=np.float32)
if not np.isfinite(raw).all():
    raise ValueError('Non-finite value in a day-present neural trace')
```

iii. The notes say absent cross-day registered cells appear as all-NaN for that day and should be removed, while all day-present curated cells should be retained rather than place-cell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. The AI treats the start of each contiguous 1-minute segment within a recording day as the alignment point and slices neural and position with identical windows.

ii.
```python
metadata=dict(
    ...
    temporal_alignment_event='Start of each contiguous non-overlapping 1-minute segment within a recording day',
    off_start=0.0, off_end=60.0,
    ...
)
```

iii. The notes say the recordings are continuous, not naturally trialized, and that no stimulus-like event exists, so artificial one-minute segments are the only alignment structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms time bins. Yes: the AI rebins from 30 Hz native sampling by smoothing and averaging each non-overlapping block of 3 frames.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
...
smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
```

iii. The notes say this was chosen to match the reference decoder’s temporal aggregation and reduce dataset size while keeping neural and position streams identically processed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. It is derived from `d['blocked'][day]`, with `envs` only used for metadata and cross-checking rather than direct construction.

ii.
```python
def blocked_indices(raw):
    vals = np.asarray(raw[0]).ravel().astype(int)
    return vals[vals >= 0]

def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom
...
nmat, out, geom, present, ncorrect = process_day(
    d['trace'][day], d['position'][day], d['blocked'][day])
```

iii. The notes say the `blocked` indices exactly match the named geometries and are sufficient to reconstruct the 3x3 arena geometry directly.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI builds a 9-element row-major accessibility vector, initialized to ones and setting blocked indices to zero. That static vector is copied for every trial in the session.

ii.
```python
def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom
...
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The notes justify this as matching `get_env_mat` semantics from the reference code: `1=accessible`, `0=blocked`. The AI explicitly says it chose accessibility rather than blocked-position one-hot to encode full environment geometry and avoid semantic ambiguity.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from `d['position'][day]`, the x-y trajectory for that recording day.

ii.
```python
nmat, out, geom, present, ncorrect = process_day(
    d['trace'][day], d['position'][day], d['blocked'][day])
...
pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
```

iii. The notes identify `position` as the aligned DeepLabCut head-location stream recorded simultaneously with calcium activity.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI averages position over the same non-overlapping 3-frame windows used for neural data, bins x and y into 25 cm arena partitions, combines them as `y_bin*3+x_bin`, and then corrects pooled samples that fall into blocked bins by snapping them to the nearest accessible bin.

ii.
```python
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
classes = (ybin * 3 + xbin).astype(np.int8)
geom = geometry_vector(blocked_raw)
classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
```

iii. The notes say this follows the reference decoder’s temporal pooling, applies the task-required 3x3 discretization, and uses nearest-valid correction to handle rare tracking points in blocked partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is thresholded into 3 bins by dividing the 75 cm arena into 25 cm segments, clipping to `[0, 2]`, then converting to a single row-major class `0..8` via `y_bin*3 + x_bin`.

ii.
```python
xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
classes = (ybin * 3 + xbin).astype(np.int8)
```

iii. The notes justify the row-major orientation with an explicit blocked-bin sanity check and say the 3x3 grid is the task-required coarse spatial categorization.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned by applying the same 3-frame pooling windows to both streams before trial slicing, then splitting both with the same minute boundaries.

ii.
```python
smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
...
ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
```

iii. The notes say the raw streams already have matched frame counts, so no interpolation is needed; identical pooling and slicing is sufficient for alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or absent cells are removed via the finite-value registration mask. If the day’s cell mask changes within a session or if retained traces contain non-finite values, the script raises an error. Short tails that do not fill a complete minute are discarded. Rare pooled position samples that land in blocked bins are snapped to the nearest accessible class.

ii.
```python
present = np.isfinite(trace_day[:, 0])
if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
    raise ValueError('Cell registration mask changes within a session')
...
if not np.isfinite(raw).all():
    raise ValueError('Non-finite value in a day-present neural trace')
...
classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
...
ntrials=nmat.shape[1]//TRIAL_BINS
keep=ntrials*TRIAL_BINS
```

iii. The notes justify these choices as preserving only day-present curated neurons, documenting the deliberate loss of sub-minute tails, and resolving very rare blocked-bin artifacts caused by boundary/tracking noise.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies per-animal `joblib` loading/decompression as the main bottleneck, with saving the very large output pickle also nontrivial. Session processing itself is relatively fast.

ii.
```python
for si, f in enumerate(files):
    t_load=time.time(); print(f'Loading {f.name}...', flush=True)
    d=joblib.load(f)[f.name]
    print(f'  loaded in {time.time()-t_load:.2f}s', flush=True)
...
t_save=time.time()
with open(outfile,'wb') as fh: pickle.dump(data,fh,protocol=5)
print(f'Saved in {time.time()-t_save:.2f}s; total {time.time()-t_all:.2f}s; ...',flush=True)
```

iii. In the notes, the AI explicitly says compressed source loading is slower than per-session processing and gives runtime estimates showing data loading dominates conversion time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most heavy numerical work is already vectorized. The remaining non-vectorized pieces are mostly Python-level trial list construction and repeated copying of the static geometry vector for each trial.

ii.
```python
ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The notes emphasize that Gaussian filtering, pooling, discretization, and blocked-bin correction were already rewritten in vectorized form, leaving only lightweight list assembly around trial packaging.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats trial packaging work for every session: splitting arrays into Python lists of per-trial matrices, casting each split array to contiguous arrays, and copying the same static geometry vector once per trial.

ii.
```python
keep=ntrials*TRIAL_BINS
ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The notes say session-wide filtering and pooling are done once to avoid repeated computations, implying that the repeated work that remains is mostly packaging for the required output structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extra provenance and QC information that downstream decoder training does not need, such as `environment`, `blocked`, `native_frames`, `discarded_native_equivalent_frames`, and `corrected_blocked_pooled_bins` in `metadata['session_info']`. It also computes `used_subjects` and never uses it.

ii.
```python
env=str(np.asarray(d['envs']).ravel()[day])
info=dict(session_id=f'{f.name}_day{day:02d}', subject=f.name, source_day=day,
          environment=env, blocked=blocked_indices(d['blocked'][day]).tolist(),
          native_frames=int(d['trace'].shape[2]), present_neurons=int(present.sum()),
          n_trials=int(ntrials), retained_time_bins=int(keep),
          discarded_native_equivalent_frames=int(d['trace'].shape[2]-keep*POOL),
          corrected_blocked_pooled_bins=int(ncorrect))
session_info.append(info)
...
used_subjects=sorted(set(subject_idx))
```

iii. The AI’s notes frame these as provenance and validation aids rather than decoder inputs. The trajectory and notes also show substantial plotting and verification machinery whose outputs are useful for auditing but not consumed in downstream training.
