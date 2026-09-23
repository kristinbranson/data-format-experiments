# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses a fixed list of seven animals and loads one extensionless, joblib-serialized file per animal from `/app/data`. Each loaded object contains all days and arrays for that animal. Full mode processes every animal, optionally in parallel; sample mode selects two days.

ii.
```python
DATA_DIR = '/app/data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
jobs = [(a, None) for a in ANIMALS]
```

iii. The notes say each large animal file is loaded exactly once, both to avoid repeated 6–16 second loads and to process all its sessions from the same in-memory object. Seven-way animal parallelism was used for speed.

## 1-b. How are the data split into subjects?

i. Each named joblib file/dictionary key is one mouse. The output always lists the seven fixed IDs, and each output session receives the corresponding index in that list.

ii.
```python
data = {'subjects': list(ANIMALS), 'subject_idx': [], ...}
data['subject_idx'].append(ANIMALS.index(s['animal']))
```

iii. The AI reports that the source contains seven mice and that its counts reproduce the paper’s seven animals and 5,413 unique neurons.

## 1-c. How are the data split into sessions?

i. The first axis of each animal's `trace`, `position`, `envs`, and `blocked` arrays is treated as recording day/session. Every day is processed separately and normally becomes one output session.

ii.
```python
if days is None:
    days = list(range(trace.shape[0]))
for day in days:
    res = process_session(trace[day], position[day], blocked[day], str(envs[day]), ...)
```

iii. The notes identify one 40-minute recording per day, yielding 207 sessions: 31 for six animals and 21 for QLAK-CA1-51. They state there were no reference session-level exclusions.

## 1-d. How are the data split into trials?

i. Sessions are first reduced to 100 ms bins. The AI defines consecutive 600-bin windows in original session time (one minute), retains only moving bins inside each window, keeps the final partial window if usable, and therefore produces variable-length trials.

ii.
```python
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TBIN))
for start in range(0, nb, TRIAL_BINS):
    idx = np.arange(start, min(start + TRIAL_BINS, nb))
    sel = idx[moving[idx]]
    trials_neural.append(np.ascontiguousarray(neural[:, sel]))
    trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. The task required one-minute trials. The AI kept the last partial block to avoid discarding data and noted that movement filtering makes trial lengths variable, which the target representation permits.

## 1-e. How are trials filtered based on quality controls?

i. Within every minute, bins at or below 5 cm/s are removed. A block is dropped if fewer than 30 moving 100 ms bins (three seconds) remain. During assembly, any session with fewer than two usable trials is skipped.

ii.
```python
moving = speed_b > V_THRESH
if len(sel) < MIN_TRIAL_BINS:
    n_dropped += 1
    continue
...
if len(s['neural']) < 2:
    continue
```

iii. The speed threshold is attributed to `decode_position_within`. The three-second minimum is justified as removing unusable trials, and the two-trial session check enforces the target format’s decoder requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the per-day `trace` array. The AI interprets it as the supplied binarized rising phases of calcium transients, with rows for animal-level cells and columns for imaging frames.

ii.
```python
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
res = process_session(trace[day], position[day], ...)
```

iii. The methods and source-code exploration led the AI to conclude that `trace` is already a rising-phase binary event signal, so neither dF/F nor deconvolution can or should be recomputed.

## 2-b. How is the `neural` data processed?

i. Unregistered rows are removed, binary traces are cast to float32, Gaussian-smoothed along continuous session time with sigma three frames, mean-pooled over non-overlapping three-frame bins, multiplied by 30 to express events/s, filtered to curated cells and moving bins, and copied into trials.

ii.
```python
raw = trace[registered][:, :nb * TBIN].astype(np.float32)
smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
neural = neural[active]
trials_neural.append(np.ascontiguousarray(neural[:, sel]))
```

iii. The AI says sigma-3 smoothing plus three-frame average pooling follows `fit_decoder`. It deliberately smooths the continuous trace before selecting moving bins, arguing that smoothing after concatenating noncontiguous moving samples would mix distant times. Events/s was chosen to match rate-map units and give the downstream optimizer a useful scale.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is registered if its first sample is non-NaN. Among registered cells, it is retained only if it has more than five raw events in the 100 ms bins classified as moving.

ii.
```python
registered = ~np.isnan(trace[:, 0])
events_b = bin_time(raw, nb, how='sum')
active = events_b[:, moving].sum(axis=1) > CELL_THRESH
neural = neural[active]
```

iii. The notes say NaNs are all-or-none for a cell/day and cite the paper decoder’s `cell_threshold=5`. The intent is to include all adequately active registered cells, not select place cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Alignment is to the start of each artificial one-minute block. Neural and behavior frames are asserted to have equal lengths, pooled with identical three-frame boundaries, and selected with the same movement-bin indices.

ii.
```python
assert position.shape[1] == n_frames
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
pos_b = bin_time(position, nb)
trials_neural.append(neural[:, sel])
trials_output.append(part[sel][np.newaxis, :])
```

iii. The notes state that trace frame i and position frame i are timestamp-aligned. Metadata names trial start as the alignment event with offsets 0 and 60 seconds.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms. The native 30 Hz streams are rebinned in non-overlapping groups of three frames; incomplete trailing frames are omitted.

ii.
```python
TBIN = 3
nb = n_frames // TBIN
x = x[..., :nbins * TBIN]
x = x.reshape(x.shape[:-1] + (nbins, TBIN))
'time_bin_size': 1000.0 * TBIN / FPS
```

iii. The AI attributes the three-frame temporal bin to the paper’s position decoder and applies the same boundaries to neural activity, position, and speed.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. It is derived from the per-session `blocked` field, rather than from the environment name.

ii.
```python
blocked_vec, blocked_ids = blocked_vector(blocked_entry)
trials_input.append(blocked_vec.copy())
```

iii. The AI verified that each geometry has one stable blocked set and preferred the session’s direct `blocked` value after finding orientation conventions in geometry matrices potentially confusing.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Nonnegative blocked partition indices set entries in a nine-element float32 binary vector; `-1` is removed and therefore produces all zeros. A copy of the same static vector is stored for every trial in the session.

ii.
```python
idx = np.atleast_1d(np.asarray(e, dtype=float)).ravel()
idx = idx[idx >= 0].astype(int)
vec = np.zeros(N_GRID * N_GRID, dtype=np.float32)
vec[idx] = 1.0
```

iii. This directly encodes which of the nine arena partitions are unavailable in decoder-usable form. Validation reportedly found exactly ten unique vectors matching the ten geometries.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each day’s two-row `position` array, containing x and y head position in centimeters at the imaging frame rate.

ii.
```python
res = process_session(trace[day], position[day], ...)
pos_b = bin_time(position, nb)
```

iii. The methods and data inspection indicated that behavior and imaging were both sampled at 30 Hz and timestamp-aligned.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. X and y are averaged within each three-frame bin, converted to a single integer partition label, restricted to moving bins, and stored as an int64 array of shape `(1, T)`. Rare labels landing in physically blocked partitions are changed to the nearest open partition center.

ii.
```python
pos_b = bin_time(position, nb)
part, n_fixed = discretize_position(pos_b, blocked_ids)
trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. The AI sought to mirror average pooling in the reference decoder. It characterizes blocked-bin cases as wall-tracking noise (0.0003%) and treats nearest-open reassignment as analogous to the reference decoder’s cleaning to valid bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. A fixed 75 cm square is divided at 25 and 50 cm on each axis. Column is `floor(x/25)`, row is `floor(y/25)`, both clipped to 0–2, and the category is `3*row + column` (0–8). Blocked results may then be reassigned to the nearest open category.

ii.
```python
xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
part = N_GRID * yb + xb
part[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. Fixed physical boundaries preserve category identity across sessions and agree with the `blocked` indexing. The axis/order convention was checked against occupancy maps and blocked regions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position have equal native frame counts. Both are pooled over the same three-frame intervals and indexed by the same per-trial `sel` array after movement filtering.

ii.
```python
assert position.shape[1] == n_frames
pos_b = bin_time(position, nb)
neural = bin_time(smoothed, nb) * FPS
trials_neural.append(neural[:, sel])
trials_output.append(part[sel][np.newaxis, :])
```

iii. The AI reports visual and numerical checks showing no lag and identical bin edges, with frame i in one stream corresponding to frame i in the other.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN/unregistered cell-days are removed using the first-frame NaN flag. Incomplete three-frame tails are truncated. Rare positions categorized into blocked regions are reassigned to the nearest open region. Low-information trial blocks and sessions are dropped; the final partial one-minute block is otherwise retained.

ii.
```python
registered = ~np.isnan(trace[:, 0])
nb = n_frames // TBIN
x = x[..., :nbins * TBIN]
part[bad] = open_ids[np.argmin(d, axis=1)]
if len(sel) < MIN_TRIAL_BINS:
    continue
```

iii. The AI found NaNs to be an intentional marker for cells not registered that day. It calls blocked labels tracking noise, retains partial blocks to minimize loss, and documents that discarded low-movement data are a small fraction of otherwise retained bins.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies joblib loading/decompression, Gaussian smoothing of large trace arrays, and writing the roughly 3.4 GB pickle as the main costs. Its measured full run was about 41 seconds with animal-level parallelism.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Notes report 6–16 seconds to load an animal, about 17 seconds for parallel per-animal processing, and about 20 seconds to assemble/write the output. Smoothing is described as the dominant compute step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI says time-bin loops were the obvious vectorization target and implements them with reshape/reduce. It retains loops over animals, sessions, and approximately 40 trial windows because these create heterogeneous nested outputs. A small loop remains when constructing open partition IDs.

ii.
```python
x = x.reshape(x.shape[:-1] + (nbins, TBIN))
return x.mean(axis=-1) if how == 'mean' else x.sum(axis=-1)
for start in range(0, nb, TRIAL_BINS):
    ...
```

iii. The notes estimate vectorized binning is about 50 times faster than per-frame loops. They do not identify another material loop that should be vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The same speed computation, position binning/discretization, neural smoothing/binning, and curation are performed once for each session. The same blocked vector is copied per trial. At the end, the nested results are traversed again to compute summary statistics. The expensive animal file load is intentionally not repeated per session.

ii.
```python
for day in days:
    res = process_session(trace[day], position[day], blocked[day], ...)
trials_input.append(blocked_vec.copy())
allout = np.concatenate([t.ravel() for sess in data['output'] for t in sess])
```

iii. The AI specifically optimized away repeated file loading: one decompression supplies all days for an animal. Per-session transforms are necessary because traces, movement, positions, and geometry differ by day.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full conversion, no major scientific transform is computed solely to be discarded. Some bookkeeping (`trial_bin_idx`, fixed/dropped counts, environment strings) supports plots, logging, or metadata rather than decoder arrays, and the final summary concatenates all labels only to print their distribution. With `--show-processing`, extensive diagnostic plotting is intentionally extra and is not used by training.

ii.
```python
trial_bin_idx.append(sel)
allout = np.concatenate([t.ravel() for sess in data['output'] for t in sess])
if show_processing:
    _plot_processing(...)
```

iii. The notes present the optional plots and summaries as sanity checks. Intermediate event counts are needed for cell curation, speed for movement filtering, and binned positions for labels, so those computations are not unnecessary even though only final arrays are serialized.
