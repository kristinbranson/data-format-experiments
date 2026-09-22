# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses a fixed list of seven animal IDs and opens each animal's MATLAB v7.3/HDF5 `.mat` file with `h5py`. In full mode it processes every day referenced by `trace`; animal files are processed in parallel and the resulting sessions are sorted by animal and day.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
with h5py.File(os.path.join(DATA_DIR, f"{animal}.mat"), 'r') as f:
    n_days = f['trace'].shape[0]
    day_list = range(n_days) if days is None else days
...
with Pool(min(args.nproc, len(tasks))) as pool:
    results = pool.map(process_animal, tasks)
```

iii. The notes say the `.mat` files permit lazy per-day access, are equivalent to the joblib copies, and avoid materializing 9–25 GB per animal. The fixed list matches the seven animals used by the paper, and the full run was checked to contain 207 sessions.

## 1-b. How are the data split into subjects?

i. One `.mat` file/entry in `ANIMALS` is treated as one mouse. Subject names are the fixed animal IDs, sorted during assembly, and each session receives the corresponding index.

ii.
```python
subjects = sorted({s['animal'] for s in all_sessions})
...
'subject_idx': np.array([subjects.index(s['animal']) for s in all_sessions], dtype=np.int64),
```

iii. The agent states that the source provides one file per animal and reports seven subjects, consistent with the paper and dataset organization.

## 1-c. How are the data split into sessions?

i. Each recording day referenced by an animal file's HDF5 arrays is one output session. Sessions are identified as `<animal>_dayXX`; a session would be omitted if it produced fewer than two usable trials.

ii.
```python
n_days = f['trace'].shape[0]
day_list = range(n_days) if days is None else days
for day in day_list:
    position, trace, env, blocked = read_session(f, day)
    sess = process_session(position, trace, env, blocked,
                           session_id=f"{animal}_day{day:02d}", ...)
    if sess is None:
        continue
```

iii. The notes justify recording day as the session because geometry changes by day and neuron count may vary between days. Although a minimum-two-trials guard exists, the full run retained all 207 source sessions.

## 1-d. How are the data split into trials?

i. The agent first makes 100 ms bins, defines nominal contiguous 60-second blocks of 600 bins, and also retains a final partial block if it spans at least 10 seconds. Within each block, non-running bins are removed, so saved trials are variable-length rather than contiguous 60-second sequences. Trials with fewer than 30 retained bins are dropped.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / BIN_FRAMES))
MIN_TAIL_BINS = 100
MIN_TRIAL_BINS = 30
...
starts = list(range(0, n_bins - BINS_PER_TRIAL + 1, BINS_PER_TRIAL))
bounds = [(s, s + BINS_PER_TRIAL) for s in starts]
if n_bins - tail_start >= MIN_TAIL_BINS:
    bounds.append((tail_start, n_bins))
for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    if idx.size < MIN_TRIAL_BINS:
        continue
```

iii. The notes cite the requested one-minute trial definition, but add that a substantial tail preserves data and that removing immobility follows the paper's decoder. They explicitly acknowledge that retained trials have unequal lengths.

## 1-e. How are trials filtered based on quality controls?

i. Time bins with speed at or below 5 cm/s are removed. A nominal trial is discarded if fewer than 30 moving 100 ms bins remain, and a session is discarded if fewer than two trials survive. A trailing fragment shorter than 100 bins is also discarded.

ii.
```python
moving = speed > V_THRESH
moving_bins = moving[:n_use].reshape(n_bins, BIN_FRAMES).mean(axis=1) >= 0.5
...
if idx.size < MIN_TRIAL_BINS:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The agent says the speed filter mirrors `decode_position_within`, removes immobility-associated replay/SWR activity, and that very short trials are unsuitable for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the per-day `trace` dataset. The agent interprets it as already-binarized calcium-transient rising phases, with columns corresponding to registered cells and all-NaN columns marking unregistered cells.

ii.
```python
trace = f[f['trace'][day, 0]][()].astype(np.float32)
...
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
```

iii. The notes report that trace values are exactly 0/1 or NaN and that the original preprocessing was performed upstream, so no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. After neuron filtering, traces are Gaussian-smoothed along time with sigma 3 frames, mean-pooled in non-overlapping groups of three frames, transposed to neuron-by-time, cast to float32, and finally indexed to retain moving bins inside each nominal trial.

ii.
```python
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
...
neural_trials.append(np.ascontiguousarray(neural[:, idx]))
```

iii. The agent attributes smoothing and three-frame pooling to `fit_decoder`. It deliberately smooths the intact session before speed masking, arguing this avoids convolution across discontinuous retained frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It removes all-NaN/unregistered columns, asserts that remaining columns contain no partial NaNs, and then keeps only cells with more than five summed binary events during running frames.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.any(np.isnan(trace))
events_running = trace[moving].sum(axis=0)
keep_cell = events_running > CELL_THRESH
trace = trace[:, keep_cell]
```

iii. The notes say these are the registration and `cell_threshold=5` criteria used by the paper's within-session position decoder, with no place-cell selection.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Nominal trials align to each successive 60-second boundary from session start; after binning, the same moving-bin indices select neural and position arrays.

ii.
```python
starts = list(range(0, n_bins - BINS_PER_TRIAL + 1, BINS_PER_TRIAL))
...
idx = np.flatnonzero(moving_bins[s:e]) + s
neural_trials.append(np.ascontiguousarray(neural[:, idx]))
output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

iii. The metadata explains that free foraging has no stimulus or behavioral alignment event, and that imaging and behavior were synchronously acquired at 30 Hz.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved resolution is 100 ms (10 Hz). The code Gaussian-smooths the 30 Hz trace and averages each non-overlapping group of three frames; position and the motion mask are downsampled on the same groups.

ii.
```python
BIN_FRAMES = 3
TIME_BIN_MS = 1000.0 * BIN_FRAMES / FPS
...
smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1)
```

iii. The agent chose the paper decoder's `temporal_bin_size=3`, also noting that continuous smoothed values satisfy the downstream validator better than binary values.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from each day's `blocked` references. The `envs` geometry name is separately decoded and used to cross-check the blocked indices.

ii.
```python
env = _h5_str(f, f['envs'][0, day])
blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
blocked = blocked[blocked >= 0].astype(int)
...
assert np.array_equal(np.sort(blocked), expected)
```

iii. The agent says `blocked` directly stores the unavailable 3×3 partitions and verified it against the reference geometry matrices for every session.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Negative sentinel values are removed, then a static nine-element float32 vector is created with 1 for each blocked partition and 0 for each open partition. A copy is stored for every trial.

ii.
```python
geometry = np.zeros(NBINS * NBINS, dtype=np.float32)
geometry[blocked] = 1.0
...
input_trials.append(geometry.copy())
```

iii. The notes say this representation directly satisfies the requested environment-geometry input and makes corresponding unreachable output classes explicit.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the per-day `position` dataset, read as time-by-two coordinates in centimeters.

ii.
```python
position = f[f['position'][day, 0]][()].astype(np.float64)
...
assert position.shape == (n_frames_raw, 2)
```

iii. The notes identify the two columns as x and y head position in the 75×75 cm arena, synchronously sampled with imaging.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are averaged over each three-frame temporal bin, converted to an integer 3×3 class, and rare classes that fall in a blocked partition are snapped to the nearest open grid cell. Only moving bins are saved.

ii.
```python
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
pos_bin = position_to_bin(position_binned_cm).astype(np.int64)
pos_bin = snap_to_open_bins(pos_bin, blocked)
...
output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

iii. Temporal averaging follows the referenced decoder. The notes justify snapping as treatment of rare tracking/wall-boundary errors and an analogue of the reference decoder's nearest-valid-rate-map-bin scoring step.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided at 25 and 50 cm using floor division by 25 cm, clipped into indices 0–2. The output category is `3*y_bin + x_bin`; blocked results are subsequently remapped to the nearest open category.

ii.
```python
b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
return NBINS * b[:, 1] + b[:, 0]
```

iii. The fixed 25 cm thresholds come from the 75 cm arena rather than session extrema. The agent reports validating this convention against stored occupancy maps and the dataset's blocked-index layout.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural trace and position are asserted to have equal raw frame counts, both are pooled over identical three-frame groups, and both are selected with the same per-trial moving-bin indices.

ii.
```python
assert position.shape == (n_frames_raw, 2)
...
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
...
neural_trials.append(np.ascontiguousarray(neural[:, idx]))
output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

iii. The agent says the streams are frame-aligned at acquisition and reports spot checks and plots showing no lag.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Expected all-NaN cell columns are removed; partial NaNs and neural/position length mismatches trigger assertions. The `-1` blocked sentinel is removed. Coordinates are clipped to valid bins, rare blocked-bin positions are snapped to the nearest open bin, incomplete frame groups are dropped, short tails/trials are dropped, and sessions with fewer than two trials would be skipped.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
assert not np.any(np.isnan(trace))
...
blocked = blocked[blocked >= 0].astype(int)
...
b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
pos_bin = snap_to_open_bins(pos_bin, blocked)
n_bins = n_frames_raw // BIN_FRAMES
```

iii. The notes distinguish expected NaN padding from unexpected corruption, and describe snapping 152 samples (0.006%) as correcting small tracking/wall errors. Data-loss thresholds were chosen to preserve substantial tails while maintaining usable trials.

## 6-a. What are the most time-consuming steps of the code?

i. Per-session HDF5 reads and Gaussian smoothing of large trace matrices dominate processing; serial processing was estimated at roughly two seconds per session. Pickling the approximately 3.44 GB result is another material step. Parallel processing reduced full conversion to about 44 seconds.

ii.
```python
trace = f[f['trace'][day, 0]][()].astype(np.float32)
...
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes benchmark read/process time at 1.7–2.1 seconds per session and identify lazy HDF5 reads, float32 processing, seven workers, and reshape-based pooling as the main optimizations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most high-volume work is vectorized. Remaining Python loops iterate over animals, sessions, trial bounds, and the small set of blocked bins; dataset assembly and diagnostic summaries also loop over nested sessions/trials. Trial extraction could be partially vectorized only before speed filtering, because retained trial lengths vary.

ii.
```python
for day in day_list:
    ...
for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
...
for b in blocked:
    d = np.linalg.norm(coords[open_bins] - coords[b], axis=1)
```

iii. The agent specifically says it avoided naive per-frame loops by vectorizing position classification and reshape/mean pooling. It considered the remaining structural loops small or necessary for variable-length outputs.

## 6-c. What processing does the code repeat multiple times?

i. It computes motion and temporal transforms independently for every session, copies the same geometry vector for every trial, and traverses all trials again during summary/metadata construction. With `--show-processing`, some arrays are also revisited for plotting. These repetitions are mostly session-specific or required by the target nested format.

ii.
```python
for (s, e) in bounds:
    ...
    input_trials.append(geometry.copy())
...
for s in all_sessions:
    data['neural'].append(s['neural'])
...
allout = np.concatenate([t.ravel() for s in data['output'] for t in s])
```

iii. The notes focus on avoiding repeated whole-animal joblib loading. They do not identify a major redundant core transform; the repeated geometry copies are needed to conform to per-trial input storage.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads and decodes environment names primarily for validation/metadata, derives expected blocked bins for an assertion, and computes extensive bookkeeping statistics. Optional plotting builds diagnostic arrays that are removed before serialization. More importantly relative to the human conversion, speed estimation/filtering, neuron event filtering, smoothing, snapping, and related bookkeeping are extra transformations not required by the target's minimal conversion.

ii.
```python
expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected)
...
if show_processing:
    result['_plot'] = dict(...)
...
del session['_plot']
```

iii. The agent considers the checks and optional plots valuable validation rather than wasted work, and considers the extra signal processing necessary to reproduce the paper decoder. The human reference, however, does not use those extra transformations.
