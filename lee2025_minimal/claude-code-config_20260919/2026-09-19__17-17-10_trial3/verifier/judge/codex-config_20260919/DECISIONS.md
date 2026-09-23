# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses a fixed list of seven animal IDs and loads each animal's extensionless joblib file. It extracts `trace`, `position`, `envs`, and `blocked`; the joblib files are converted copies of the supplied MATLAB data. Unless `--max-sessions` is supplied, every day in every listed animal is processed.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position = dat["trace"], dat["position"]
envs = [str(e) for e in np.asarray(dat["envs"]).ravel()]
blocked_all = dat["blocked"]
```

iii. The trajectory says the agent inspected both formats and regarded the joblib files as outputs of the repository's `mat2joblib` conversion with content identical to the `.mat` files. It chose them for direct NumPy-style access and explicitly intended to retain all 7 mice and all 207 sessions.

## 1-b. How are the data split into subjects?

i. Each named joblib file is one subject. Its position in `animals` is used as the integer subject index, and the supplied animal IDs become `subjects`.

ii.
```python
for subject, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
"subjects": list(animals),
"subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. The agent inferred from the file layout and dictionary keys that each file contains one mouse and retained the dataset's animal identifiers.

## 1-c. How are the data split into sessions?

i. The first dimension of `trace` is treated as recording day/session. Every day becomes one output session, with the matching day from position, environment, and blocked geometry.

ii.
```python
n_days = trace.shape[0]
days = range(n_days) if max_sessions is None else range(min(n_days, max_sessions))
for day in days:
    blocked = blocked_indices(blocked_all[day])
    tr = trace[day]
    pos = pool_mean(position[day], FRAMES_PER_BIN)
```

iii. Repository inspection showed that the leading array dimension represents days, and the agent equated each recording day with a session.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping 60-second trials after 3-frame pooling. Thus each trial contains 600 100-ms bins. A trailing partial minute is discarded.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FPS / FRAMES_PER_BIN))
...
n_trials = n_bins // BINS_PER_TRIAL
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
```

iii. The instructions require artificial one-minute trials because the recordings have no trial structure. The agent stated that dropping the remainder ensures equal trial lengths.

## 1-e. How are trials filtered based on quality controls?

i. No completed 60-second trial is quality-filtered. Only the incomplete tail of a session is dropped; the agent deliberately does not remove low-running-speed frames or trials.

ii.
```python
n_trials = n_bins // BINS_PER_TRIAL
```

iii. The agent reasoned that the paper's speed threshold belongs to its Bayesian decoder, while applying it here would remove about 48% of samples and make one-minute trials ragged and non-contiguous.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the per-animal `trace` array, specifically `trace[day]`, described by the agent as binary rising phases of calcium transients.

ii.
```python
trace, position = dat["trace"], dat["position"]
...
tr = trace[day]
```

iii. The paper and repository describe this binary transient vector as the firing-rate surrogate used in subsequent analyses.

## 2-b. How is the `neural` data processed?

i. After neuron filtering, each session's traces are Gaussian-smoothed along time with sigma 3 native frames, mean-pooled over non-overlapping 3-frame windows, multiplied by 30, and cast to float32. The resulting neuron-by-time matrices are sliced into trials.

ii.
```python
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
...
neural_trials.append(np.ascontiguousarray(rates[:, sl]))
```

iii. The agent tried to reproduce `utils.fit_decoder`/`test_decoder`, which smooth and pool traces, and multiplied by frame rate to express pooled binary events as Hz like the paper's rate maps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is first deemed registered if its first sample is not NaN. Registered cells are then retained only if their session-wide transient sum is greater than 5. No place-cell or spatial-reliability selection is used.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered]
active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
tr = tr[active]
```

iii. The agent interpreted NaNs as unregistered cells and adopted the paper decoder's `cell_threshold = 5` sparsity rule. It reported that this removed 112 additional cell-sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Time zero is the beginning of each artificial one-minute segment, starting from recording onset.

ii.
```python
"temporal_alignment_event":
    "start of each 1-min trial; the session is a single continuous recording with "
    "no trial structure, cut into consecutive 1-min segments from recording onset",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The agent recognized that continuous free-foraging sessions contain no stimulus event and therefore used segment starts as the alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms. Native 30 Hz samples are mean-pooled in groups of three, after neural smoothing.

ii.
```python
FRAMES_PER_BIN = 3
TIME_BIN_MS = 1000.0 * FRAMES_PER_BIN / FPS
...
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
```

iii. The agent selected 3-frame bins because it found that temporal binning in the paper repository's decoder functions and viewed it as the applicable paper processing.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from each session's `blocked` entry, not from its `envs` name.

ii.
```python
blocked_all = dat["blocked"]
...
blocked = blocked_indices(blocked_all[day])
```

iii. The agent found that some named geometries were vertically mirrored for some animals, so it regarded `blocked` as the authoritative record of the actual configuration.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Values below zero (including the `-1` no-block sentinel) are removed. Remaining indices set entries in a 9-element float32 binary vector to one. A copy of this static vector is stored for every trial.

ii.
```python
vals = vals[vals >= 0]
...
geom = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
geom[blocked] = 1.0
...
input_trials.append(geom.copy())
```

iii. A nine-dimensional multi-hot representation directly indicates which arena partitions are unavailable and handles the open square as an all-zero vector.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the session's two-coordinate `position[day]` array.

ii.
```python
trace, position = dat["trace"], dat["position"]
...
pos = pool_mean(position[day], FRAMES_PER_BIN)
```

iii. The agent identified this as DeepLabCut head position in centimeters in the common arena frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Both coordinates are averaged over the same three-frame windows as neural data, converted to x/y bin indices, combined as `3*y + x`, and clipped to valid axis bins. Samples assigned to blocked partitions are moved to the nearest open partition by grid-center distance.

ii.
```python
pos = pool_mean(position[day], FRAMES_PER_BIN)
xb = np.clip((pos[0] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
part = N_SPATIAL_BINS * yb + xb
...
part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. The agent sought to follow the repository decoder's floor-based position rule and nearest-valid-bin cleaning. It empirically checked that `3*y+x` matched blocked indexing and interpreted occupancy inside walls as tracking noise.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. For each animal, a common bin width is `(nanmax(position across all days) + 1e-15) / 3`. Floored x and y quotients are clipped to 0–2 and combined into nine row-major categories.

ii.
```python
bin_size = (np.nanmax(position) + POSITION_BUFFER) / N_SPATIAL_BINS
...
xb = np.clip((pos[0] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
part = N_SPATIAL_BINS * yb + xb
```

iii. The agent generalized `decode_position_within` from 15 to 3 bins and retained that function's animal-wide maximum scaling, expecting a width close to the physical 25 cm partition size.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are assumed frame-aligned at 30 Hz. Each is independently pooled over identical non-overlapping three-frame windows, then truncated to their common number of bins and sliced with the same trial intervals.

ii.
```python
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
pos = pool_mean(position[day], FRAMES_PER_BIN)
n_bins = min(rates.shape[1], part.shape[0])
...
neural_trials.append(np.ascontiguousarray(rates[:, sl]))
output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. The agent stated that miniscope and behavioral acquisition shared a DAQ and frame rate, so matching pooling windows preserve alignment without interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered NaN-padded neurons are removed, the shorter of processed neural and position lengths is used, incomplete final minutes are dropped, category indices are clipped, and apparent position samples inside blocked regions are reassigned to the nearest open partition.

ii.
```python
registered = ~np.isnan(tr[:, 0])
...
n_bins = min(rates.shape[1], part.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
...
part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. The agent viewed NaNs as registration padding, partial segments as incompatible with uniform trials, and the very small blocked-region occupancy as wall-tracking noise analogous to the repository's nearest-valid-bin cleanup.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large per-animal joblib arrays, Gaussian-filtering every retained neural trace, pooling the large arrays, and serializing the 6.65 GB pickle dominate runtime. The trajectory's full conversion took several minutes.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent did not explicitly rank these in its final response, but its trajectory shows long full-conversion and validation runs; array I/O, convolution, and writing are the bulk operations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop could be replaced by reshape/split operations for neural and output arrays, although lists and per-trial copies are required by the target format. Subject and day loops cannot straightforwardly be vectorized because shapes vary.

ii.
```python
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. The trajectory contains no explicit efficiency justification. The loop is simple formatting work, while expensive smoothing and pooling are already vectorized over cells and time.

## 6-c. What processing does the code repeat multiple times?

i. Registration/activity filtering, smoothing, pooling, discretization, and trial-list construction repeat once per session. Mean pooling is separately applied to neural traces and position because both streams need the same temporal bins. There is no obvious repeated computation of the same result.

ii.
```python
for day in days:
    ...
    rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
    pos = pool_mean(position[day], FRAMES_PER_BIN)
```

iii. The agent organized processing session-wise to accommodate session-specific cells and geometry; it gave no separate rationale about repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No major computed array is immediately discarded: processed neural, geometry, categorical output, region indices, and session information are all saved. `n_registered` and `envs` are used only for metadata/reporting, not decoder fitting; diagnostic printing computes session summaries. The neural smoothing/rebinning and blocked-bin repair are consequential transformations, not discarded work.

ii.
```python
n_registered = int(registered.sum())
...
session_info.append({"environment": envs[day],
                     "n_neurons_registered": n_registered, ...})
```

iii. The trajectory shows that the extra metadata and diagnostics were retained to document and validate conversion choices. It does not identify any intentionally throwaway processing in the final converter.
