# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses a fixed list of seven animals and loads one extensionless joblib file per animal from `/app/data`. Each loaded animal dictionary supplies all days' `trace`, `position`, `envs`, and `blocked` arrays; full mode processes every listed animal and day.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
animals = ANIMALS[3:4] if sample else ANIMALS
for ai, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    envs = np.asarray(dat['envs']).ravel()
    n_days = dat['trace'].shape[0]
```

iii. The notes say the joblib files are the repository's normal `load_dat` representation and are identical to the `.mat` twins, while loading much faster. The fixed list was checked against the paper's seven mice and 207 sessions.

## 1-b. How are the data split into subjects?

i. Each joblib file is one mouse. The animal ID is appended once to `subjects`, and the current index is attached to every day/session from that file.

ii.
```python
subject_idx = len(data['subjects'])
data['subjects'].append(animal)
...
data['subject_idx'].append(subject_idx)
```

iii. The agent states that the seven files correspond to the seven paper animals and verifies the expected per-animal cell and session counts.

## 1-c. How are the data split into sessions?

i. One animal-day is one output session. The code iterates the first axis of `trace`, processes that day's neural and behavioral streams together, and appends one nested session to each target field.

ii.
```python
n_days = dat['trace'].shape[0]
for d in range(n_days):
    res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
```

iii. The notes identify an animal-day as the paper's recording-session unit and report 207 such sessions.

## 1-d. How are the data split into trials?

i. After conversion to 100 ms bins, sessions are cut into non-overlapping 600-bin (60-second) slices. Unlike the human reference, a final partial slice is retained when it is at least 300 bins/30 seconds.

ii.
```python
TRIAL_BINS = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES
MIN_PARTIAL_TRIAL_BINS = TRIAL_BINS // 2
...
for t in range(n_full):
    slices.append((t * TRIAL_BINS, (t + 1) * TRIAL_BINS))
if rem >= MIN_PARTIAL_TRIAL_BINS:
    slices.append((n_full * TRIAL_BINS, n_bins))
```

iii. The agent argues that all recordings are about 40 minutes and keeping the 55.5–60 second tails preserves nearly all data while retaining substantial trials. It acknowledges that its initial plan said to drop partial minutes and later corrected that documentation.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level behavioral or quality exclusion. A session must yield at least two trials; all full trials and qualifying trailing partial trials are kept. No speed filter is applied.

ii.
```python
for (a, b) in trial_slices(n_bins):
    neural.append(np.ascontiguousarray(rates[:, a:b]))
...
assert len(data['neural'][s]) >= 2, f'session {s} has < 2 trials'
```

iii. The notes say speed filtering would remove about 40% of frames, break contiguous one-minute trials, and conflict with the desired time-varying output, so all frames are retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each animal dictionary's per-day `trace`, containing author-preprocessed binary calcium-event rising phases.

ii.
```python
res = process_session(dat['trace'][d], ...)
...
tr = trace_day[registered].astype(np.float32)
assert np.all(np.isin(tr, (0.0, 1.0)))
```

iii. The paper and repository describe `trace` as binarized rising-phase events treated as firing rate, so the agent concludes no dF/F extraction is needed.

## 2-b. How is the `neural` data processed?

i. Registered binary traces are Gaussian-smoothed across the entire continuous session with sigma 3 frames, average-pooled in non-overlapping 3-frame windows, multiplied by 30 to express events/second, cast to float32, then sliced into trials.

ii.
```python
tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (bin_time_series(tr_s) * FPS).astype(np.float32)
...
neural.append(np.ascontiguousarray(rates[:, a:b]))
```

iii. The agent chose the processing in the repository's `fit_decoder`/`test_decoder`, including whole-session smoothing to avoid artificial filter boundaries. It also says scaling by 30 gives interpretable Hz and avoids a validator warning for binary neural data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells whose first sample is NaN are treated as unregistered and removed for that session; the code asserts those rows are entirely NaN. All registered cells are kept, with no event-count or place-cell threshold.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
assert np.all(np.isnan(trace_day[~registered]).all(axis=1))
tr = trace_day[registered].astype(np.float32)
```

iii. The agent ties this to the paper's 69,744 registered cell-sessions and statement motivating inclusion of all cells. It deliberately rejects the decoder's `>5` moving-frame event filter because it is whole-session-specific and would remove 1.3% of cell-sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trial start is the alignment event: trial `k` begins 60×`k` seconds after continuous-session start. Neural smoothing/binning occurs first and the same slice boundaries are then used for neural and position labels.

ii.
```python
for (a, b) in trial_slices(n_bins):
    neural.append(np.ascontiguousarray(rates[:, a:b]))
    output.append(part[a:b][None, :].astype(np.int64))
...
'temporal_alignment_event': 'start of each 1-minute trial; ...'
```

iii. The notes say trace and position share a 30 Hz clock, confirmed by reproducing stored rate maps, so no lag correction is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Output resolution is 100 ms. Three native 30 Hz frames are average-pooled per bin after neural smoothing; position is averaged over the identical frame windows.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
...
return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)
...
'time_bin_size': 1000.0 * TEMPORAL_BIN_FRAMES / FPS
```

iii. The agent selected 100 ms because the paper repository's decoder uses `temporal_bin_size=3` and `AvgPool1d(3,3)`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The actual input vector is derived from the session's `envs` geometry name via a hard-coded reproduction of `get_env_mat`. The raw `blocked` field is independently decoded and asserted equal, but is not the returned input source.

ii.
```python
blocked_vec = env_blocked_vector(env_name)
...
bl = np.atleast_1d(np.asarray(blocked_field[0]).ravel()).astype(int)
...
assert np.array_equal(from_field, blocked_vec)
```

iii. The agent says the environment matrices come from reference `get_env_mat`, and it empirically verified their flipped indexing against every session's `blocked` field.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The named 3×3 open-space matrix is vertically flipped, flattened, inverted to a length-nine float32 blocked indicator (1 = blocked), copied once for every trial, and cross-checked against raw blocked indices.

ii.
```python
mat = np.array(ENV_MATS[env_name], dtype=float)
open_flat = np.flipud(mat).ravel()
return (open_flat == 0).astype(np.float32)
...
inputs.append(blocked_vec.copy())
```

iii. The flip makes partition IDs agree with `p = 3*floor(y/25)+floor(x/25)`. A static nine-vector directly represents the task-specified geometry.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each day's two-channel `position` array containing x,y head coordinates in centimeters.

ii.
```python
res = process_session(..., dat['position'][d], ...)
...
pos_b = bin_time_series(position_day.astype(np.float64))
```

iii. The notes identify this as DeepLabCut head position sampled on the same 30 Hz clock as calcium activity.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are averaged in each 3-frame/100 ms window, converted into a 3×3 partition ID, and rare labels falling in blocked partitions are reassigned to the nearest open partition center. Labels are int64 with shape `(1,time)`.

ii.
```python
pos_b = bin_time_series(position_day.astype(np.float64))
part, n_snapped = position_to_partition(pos_b, blocked_vec)
...
output.append(part[a:b][None, :].astype(np.int64))
```

iii. Averaging matches the selected temporal bins. The agent characterizes blocked-compartment points (0.003%) as tracking noise and says snapping mirrors `decode_position_within`'s valid-bin correction.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is floor-divided by 25 cm and clipped to 0–2. Category is `3*y_bin + x_bin`, producing 0–8. If this category is blocked, Euclidean distance to the nine partition centers is used to choose the nearest open one.

ii.
```python
col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
p = (N_PART * row + col).astype(np.int64)
...
p[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. The 25 cm thresholds divide the specified 75 cm arena evenly and use the same partition numbering as geometry input. Clipping handles boundaries; snapping prevents tiny impossible classes.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural traces use corresponding native frames. Both are reduced to equal-length three-frame bins, asserted to have the same bin count, and cut with identical `(a,b)` trial slices.

ii.
```python
rates = (bin_time_series(tr_s) * FPS).astype(np.float32)
pos_b = bin_time_series(position_day.astype(np.float64))
...
assert pos_b.shape[1] == n_bins
```

iii. The agent validated shared frame alignment by reproducing stored maps and by raw-versus-processed spot checks.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Whole-session NaN neuron rows are asserted and removed; registered traces and final neural arrays must be finite. Up to two native frames that cannot form a 3-frame bin are truncated. Long trial remainders are retained, and blocked-partition tracking anomalies are snapped to an open partition. Position is reported to contain no NaNs.

ii.
```python
n = (x.shape[-1] // kernel) * kernel
x = x[..., :n]
...
assert np.isfinite(nrl).all()
...
if np.any(bad):
    ...
    p[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. The notes distinguish structural NaNs for unregistered cells from partial missingness, preserve nearly all recording time, and interpret blocked-area coordinates as rare tracking noise.

## 6-a. What are the most time-consuming steps of the code?

i. Per-animal joblib deserialization, Gaussian filtering/pooling all neural matrices, serialization of the 6.73 GB pickle, and later validation/training dominate. The code explicitly times animal loading, per-session processing, and pickle writing.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Notes report roughly 89 seconds loading, 93 seconds processing, and 5 seconds writing in the full 187-second conversion, and identify the large trace arrays as the main workload.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural smoothing, temporal pooling, coordinate categorization, and blocked-point reassignment are already vectorized. Remaining Python loops over animals, sessions, trial slices, validation trials, occupancy accumulation, and diagnostic concatenation could partly be consolidated, though nested variable neuron/trial shapes limit useful global vectorization.

ii.
```python
for d in range(n_days):
    res = process_session(...)
...
for (a, b) in trial_slices(n_bins):
    neural.append(...)
...
for s in range(ns):
    for o in data['output'][s]:
        occ += np.bincount(o[0], minlength=9)
```

iii. The agent emphasizes that it replaced reference per-frame loops with matrix-wide SciPy filtering and reshape/mean pooling, estimating a large speedup. It does not claim the small organizational loops are a bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The geometry vector and consistency check are recomputed per session; trial slicing is performed in a loop; after construction, every trial is scanned for assertions and outputs are scanned again for occupancy. Neural values are scanned/concatenated again for summary statistics.

ii.
```python
blocked_vec = env_blocked_vector(env_name)
...
for tr_i in range(len(data['neural'][s])):
    ...
for s in range(ns):
    for o in data['output'][s]:
        occ += np.bincount(o[0], minlength=9)
allr = np.concatenate([data['neural'][s][0].ravel() for s in range(ns)])
```

iii. These repetitions are primarily sanity checks and reporting. The agent intentionally pays this cost to validate geometry, shapes, labels, and ranges.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `process_session` returns full-session intermediate arrays (`rates`, binned positions, partitions, raw registered traces) used only for optional plots/diagnostics and then deletes the result. It also computes extensive assertions and summaries. Notably, the neural summary concatenates only trial 0 from each session, so it is diagnostic and incomplete. Environment geometry is rebuilt from names despite an available raw blocked field.

ii.
```python
return {
    ... 'rates': rates, 'pos_b': pos_b, 'part': part, 'tr': tr,
}
...
if args.show_processing and n_plotted < 2:
    plot_processing(...)
del res
...
allr = np.concatenate([data['neural'][s][0].ravel() for s in range(ns)])
```

iii. The agent justifies the intermediates for `--show-processing` and the checks for conversion validation. They do not enter the saved dataset, so in ordinary full runs they add temporary memory/references and reporting work without downstream analytical value.
