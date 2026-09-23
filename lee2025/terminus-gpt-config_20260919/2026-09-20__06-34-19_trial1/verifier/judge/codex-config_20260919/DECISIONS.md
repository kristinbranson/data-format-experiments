# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers the seven extensionless `QLAK-CA1-*` joblib animal files in `/app/data`, deliberately excluding the duplicate `.mat` files. It loads each complete animal object once with `joblib.load`, extracts the dictionary keyed by the filename, and iterates through every day in its `trace` array. Full mode processes every file and day; sample mode stops after two sessions.

ii.
```python
def animal_files():
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.is_file() and p.name.startswith('QLAK-CA1-') and p.suffix != '.mat')

for si,f in enumerate(files):
    raw=joblib.load(f)[f.name]
    for day in range(raw['trace'].shape[0]):
        ntr,itr,otr,info,payload=process_session(
            raw['position'][day],raw['trace'][day],raw['blocked'][day],
            f.name,day,raw['envs'].ravel()[day])
```

iii. The agent says the joblib files are the primary compact animal files used by the repository notebooks, contain all required aligned streams, and duplicate the larger MATLAB files. Loading once per animal also avoids repeated decompression. Its checks reproduced 7 animals, 207 sessions, 5,413 registered identities, and 69,744 day-level cell records.

## 1-b. How are the data split into subjects?

i. Each discovered joblib file is one subject. Its complete filename (for example, `QLAK-CA1-08`) is the subject ID, and the file's sorted index is appended to `subject_idx` for every day/session from that animal.

ii.
```python
files=animal_files(); subjects=[p.name for p in files]
...
for si,f in enumerate(files):
    ...
    subject_idx.append(si)
```

iii. The notes establish that each animal-level file contains all sessions for exactly one of the seven mice and that these IDs agree with the released dataset and paper totals.

## 1-c. How are the data split into sessions?

i. Every animal-day is emitted as one target session. Days are traversed along axis 0 of `raw['trace']`; position, trace, blocked geometry, and environment at the same day index are passed together to `process_session`.

ii.
```python
for day in range(raw['trace'].shape[0]):
    ntr,itr,otr,info,payload=process_session(
        raw['position'][day],raw['trace'][day],raw['blocked'][day],
        f.name,day,raw['envs'].ravel()[day])
    neural.append(ntr); inputs.append(itr); outputs.append(otr)
```

iii. The agent found that the source has 207 continuous animal-day recordings and no native trials. Treating each day as a session preserves the reference repository's day/session organization.

## 1-d. How are the data split into trials?

i. A continuous day is divided from time zero into non-overlapping, complete 60-second trials. At 30 source frames/s this is 1,800 raw frames; after 3-frame pooling each trial has 600 samples. An incomplete tail is dropped, never padded or wrapped.

ii.
```python
n_trials = position.shape[1] // RAW_TRIAL_FRAMES
used_raw = n_trials * RAW_TRIAL_FRAMES
...
for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
    input_trials.append(geometry.copy())
    output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. This follows the requested artificial one-minute trial definition. The agent records source, used, and discarded frame counts in metadata and reports 8,187 complete trials, with 39 or 40 per session depending on source length.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral or trial-quality exclusion. All complete one-minute windows are retained; only the incomplete final remainder is discarded. A session is rejected with an error if it has fewer than two complete trials, but no source session triggers that condition.

ii.
```python
n_trials = position.shape[1] // RAW_TRIAL_FRAMES
used_raw = n_trials * RAW_TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f'{subject} day {source_day}: fewer than two complete trials')
```

iii. The task requires at least two trials per session, while the free-exploration data have no success/failure trials. The notes therefore preserve every complete contiguous window and only remove unusable tails.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the released per-day `trace` calcium-event array. For a day, the joblib layout is neuron-by-frame.

ii.
```python
raw['trace'][day]
...
selected = np.asarray(trace[keep], dtype=np.float32)
```

iii. The notes say `trace` contains the authors' already processed, binary rising-phase calcium events. It is not raw fluorescence, so the agent does not recompute dF/F or perform deconvolution.

## 2-b. How is the `neural` data processed?

i. After neuron selection, traces are cast to float32, Gaussian-smoothed continuously along time with sigma 3 source frames, cropped to complete minutes, and averaged in non-overlapping groups of three frames. This produces neuron-by-time arrays at 100 ms resolution. Smoothing occurs before trial slicing to avoid artificial boundary effects.

ii.
```python
smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
neural_pooled = smooth[:, :used_raw].reshape(
    selected.shape[0], -1, POOL).mean(axis=2, dtype=np.float32)
...
neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
```

iii. The agent chose this to match `fit_decoder`/`test_decoder` in the paper repository, which smooth trace vectors and average-pool three 30 Hz frames. It viewed that decoder-specific temporal processing as directly applicable and also noted its storage benefit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, cells absent on a day are removed using whether any trace value is finite. The agent then reproduces the repository decoder's activity filter: calculate framewise displacement speed, smooth it with sigma 5 frames, define moving frames as speed greater than 5 cm/s, and retain only present cells whose event sum on moving frames is greater than 5. No timepoints themselves are removed.

ii.
```python
present = np.isfinite(trace).any(axis=1)
displacement_speed = np.linalg.norm(np.diff(position.T, axis=0) * FPS, axis=1)
moving = np.zeros(position.shape[1], dtype=bool)
moving[1:] = gaussian_filter1d(displacement_speed, sigma=5, axis=0) > 5.0
event_sum = np.nansum(trace[:, moving], axis=1)
keep = present & (event_sum > 5)
```

iii. The notes justify this as matching `decode_position_within` while retaining all timepoints needed for fixed-duration trials. It removes 882 of 69,744 day-cell records and retains 68,862; the agent explicitly avoids place-cell-only filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. The alignment event is defined as the start of each artificial non-overlapping one-minute segment. Neural trials begin at pooled index `tr * 600`, corresponding to raw frame `tr * 1800`.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
...
'temporal_alignment_event':'Start of each non-overlapping 1-minute segment within a continuous animal-day recording',
'off_start':0.0,
'off_end':60.0,
```

iii. The source is continuous free exploration with no stimulus or trial-onset event. The agent therefore uses the mandated synthetic trial boundary and documents it explicitly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 100 ms. The native 30 Hz data are rebinned by non-overlapping mean pooling of three source frames, after Gaussian smoothing of neural activity.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)
...
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                               dtype=np.float32)
```

iii. The agent cites the reference repository decoder's three-frame pooling as the reason for using a common 100 ms neural/behavior bin instead of retaining native 33.33 ms frames.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from `raw['blocked'][day]`, the source list of row-major indices for blocked cells in the 3×3 arena. The environment name is retained only as session metadata.

ii.
```python
raw['blocked'][day]
...
geometry = blocked_mask(blocked)
```

iii. The repository documentation defines `blocked` as the actual per-day geometry, including `-1` for an open arena. The agent uses it rather than inferring geometry from names.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are expanded into a static length-9 float32 binary mask in row-major order, where 1 means blocked. Negative values (notably `-1`) are ignored, yielding all zeros for an open arena. A copy of the mask is stored for every trial.

ii.
```python
idx = np.asarray(raw).ravel().astype(np.int64)
mask = np.zeros(9, dtype=np.float32)
idx = idx[idx >= 0]
if idx.size:
    if np.any(idx > 8):
        raise ValueError(f'Invalid blocked indices: {idx}')
    mask[idx] = 1.0
...
input_trials.append(geometry.copy())
```

iii. The agent says a fixed binary vector is explicit, decoder-compatible, and matches the requirement that environment geometry be static per trial. It validated geometry frequencies and mask sums against raw blocked indices.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from `raw['position'][day]`, a 2-by-frame array of aligned x/y coordinates in centimeters.

ii.
```python
raw['position'][day]
...
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
```

iii. The agent found position to be fully finite, span the 0–75 cm arena, and have exact frame alignment with `trace`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. X and y coordinates are averaged over each same three-frame temporal group used for neural pooling. The pooled coordinates are divided into fixed 25 cm arena partitions, converted to integer x/y bins, clipped to 0–2, and combined as the row-major class `y_bin * 3 + x_bin`. A singleton output dimension is added.

ii.
```python
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
...
output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The agent chose fixed physical partitions rather than session-specific ranges. Pooling continuous position before classification mirrors the reference decoder's behavior pooling and keeps it synchronized with pooled neural data.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate uses boundaries at 25 and 50 cm: values in `[0,25)`, `[25,50)`, and `[50,75]` map to bins 0, 1, and 2. Values at or beyond endpoints are clipped into valid bins. Nine classes are assigned row-major as `y_bin * 3 + x_bin`, labeled NW, N, NE, W, center, E, SW, S, and SE.

ii.
```python
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
OUTPUT_VALUES = ['NW','N','NE','W','center','E','SW','S','SE']
```

iii. The thresholds directly represent the experimental 75×75 cm arena's 3×3 geometry. Clipping is used only for exact 75 cm endpoints or small boundary excursions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace are source-frame aligned. Both streams are cropped to `used_raw`; each position sample and neural sample is computed from the identical non-overlapping three-frame group; then both use the identical 600-bin trial slice.

ii.
```python
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                               dtype=np.float32)
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
...
sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The agent emphasizes that there is no interpolation, shift, or independently chosen truncation. Independent spot checks regenerated first, middle, and last sessions from raw data and found exact alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN-padded neurons absent from a day are removed before smoothing. Nonfinite position, shape/alignment mismatches, no selected cells, invalid blocked indices, or too few trials cause explicit errors rather than silent repair. Incomplete final minutes are dropped and recorded. Position classes are clipped for exact arena endpoints. The output is structurally asserted before saving.

ii.
```python
present = np.isfinite(trace).any(axis=1)
...
if not np.isfinite(position).all():
    raise ValueError(f'{subject} day {source_day}: nonfinite position')
if selected.shape[0] < 1 or not np.isfinite(selected).all():
    raise ValueError(f'{subject} day {source_day}: no valid selected cells')
...
'discarded_tail_frames': int(position.shape[1] - used_raw),
...
validate_complete(data)
```

iii. Exploration showed that trace NaNs specifically mark unrecorded registered cells, while position is finite. The agent therefore removes absent cells, refuses malformed data, avoids padding, and retains provenance in `session_info`.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant step is loading and decompressing each large joblib animal file. Neural Gaussian smoothing/pooling and copying trials consume additional per-session time; serializing the roughly 6.1 GiB pickle also costs several seconds. The code prints load, session, save, and total timings.

ii.
```python
load_t=time.time(); raw=joblib.load(f)[f.name]
print(f'Loaded {f.name} in {time.time()-load_t:.2f}s',flush=True)
...
smooth = gaussian_filter1d(selected, sigma=POOL, axis=1)
...
with open(tmp,'wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify compressed-source decompression as unavoidable and report animal loads around 6–16 seconds, session conversion around 0.3–1 second, and 219 seconds total for the full conversion.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-frame numerical work is already vectorized: speed, filtering, smoothing, pooling, discretization, and class counts use NumPy/SciPy. The animal/day loops are needed for variable session shapes and incremental loading. The per-trial loop could mechanically be replaced by reshaping into a trial axis, but the required output is a list of separate arrays and copies, so this offers little benefit.

ii.
```python
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                               dtype=np.float32)
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
```

iii. The agent explicitly says vectorized session smoothing, reshape-based pooling, and label construction replace per-frame loops. It accepts trial iteration because the target schema necessarily holds individual trial arrays.

## 6-c. What processing does the code repeat multiple times?

i. It does not repeat expensive neural processing per trial: each session is filtered, smoothed, pooled, and discretized once, then sliced. Small repeated work includes copying the same static geometry for each trial, making contiguous trial arrays, and concatenating trial outputs to update diagnostic class counts.

ii.
```python
smooth = gaussian_filter1d(selected, sigma=POOL, axis=1)
...
for tr in range(n_trials):
    ...
    input_trials.append(geometry.copy())
...
class_counts += np.bincount(np.concatenate([x.ravel() for x in otr]),minlength=9)
```

iii. The notes highlight that smoothing is deliberately performed once on the continuous session before slicing, both for efficiency and to prevent minute-boundary artifacts. Static-mask copying is required by the requested nested trial representation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full conversion, little computed data is discarded beyond incomplete tail frames and temporary intermediates. `moving` and `event_sum` exist only to select cells; a first-trial `plot_payload` is constructed for every session even when `--show-processing` is false and is then unused. Diagnostic class counts and timing/info computations do not affect decoder arrays. Optional plotting is also validation-only.

ii.
```python
plot_payload = (pos_pooled[:, :TRIAL_BINS], labels[:TRIAL_BINS],
                neural_pooled[:min(30, selected.shape[0]), :TRIAL_BINS], geometry)
return neural_trials, input_trials, output_trials, info, plot_payload
...
ntr,itr,otr,info,payload=process_session(...)
if show_processing and plots < 2:
    make_processing_plot(payload,sid)
```

iii. The agent describes plots, aggregate counts, and raw-source spot checks as sanity validation. They are not inputs to decoder training. It intentionally drops incomplete tails to guarantee equal-duration trials and frees each loaded animal after processing.
