# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, opens each animal's MATLAB v7.3/HDF5 file with `h5py`, and dereferences the per-day `position`, `trace`, `envs`, and `blocked` objects. Every listed animal and every paired day are processed.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
with h5py.File(path, "r") as f:
    envs, blocked = load_session_list(f)
    for day, (env, blk) in enumerate(zip(envs, blocked)):
        position = f[f["position"][day, 0]][:]
        trace = f[f["trace"][day, 0]][:]
```

iii. The trajectory says MATLAB v7.3 requires HDF5-style loading and reports that processing all seven mice yielded 207 sessions and the paper's neuron counts, which the agent used as evidence that loading was complete.

## 1-b. How are the data split into subjects?

i. One named `.mat` file is treated as one mouse. The fixed `ANIMALS` list becomes `subjects`, and its index is stored for each resulting session.

ii.
```python
"subjects": ANIMALS, "subject_idx": [],
...
for a, animal in enumerate(ANIMALS):
    path = os.path.join(DATA_DIR, f"{animal}.mat")
    ...
    data["subject_idx"].append(a)
```

iii. The agent understood the source organization as one file per animal and used the filename/explicit animal ID as the subject identity.

## 1-c. How are the data split into sessions?

i. Each daily HDF5 reference entry is a session. `envs` and `blocked` are zipped and enumerated, then the same day index dereferences `position` and `trace`; each day produces one session dictionary.

ii.
```python
for day, (env, blk) in enumerate(zip(envs, blocked)):
    position = f[f["position"][day, 0]][:]
    trace = f[f["trace"][day, 0]][:]
...
sessions.append({"neural": neural, "input": inputs, "output": outputs, ...})
```

iii. The trajectory identifies these entries as daily recording sessions and notes that the resulting 207 sessions reproduce the paper's headline count.

## 1-d. How are the data split into trials?

i. After 15-frame temporal pooling, each session is cut into consecutive, non-overlapping 120-bin windows, corresponding to 60 seconds. Any trailing partial minute is discarded.

ii.
```python
TRIAL_SECONDS = 60.0
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / BIN_FRAMES))
...
n_trials = n_bins // TRIAL_BINS
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
```

iii. The instructions explicitly require one-minute trials. The agent chose fixed complete windows so every trial has equal duration and stated that sessions consequently contain 39 or 40 trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. All complete one-minute windows are retained; only the trailing incomplete window is dropped, and no velocity threshold is used.

ii.
```python
n_trials = n_bins // TRIAL_BINS
...
"exclusions":
    "None beyond dropping ... the trailing < 1 min fragment ... No velocity "
    "threshold is applied..."
```

iii. The agent reasoned that a locomotion-only filter would remove roughly 45% of frames and disrupt uniform one-minute trials, whereas the requested output is defined and scored at every time bin.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes solely from each session's raw `trace` matrix, described as binarized rising phases of calcium transients.

ii.
```python
trace = f[f["trace"][day, 0]][:]
```

iii. The agent says this is the signal treated as firing rate by the paper, rather than raw fluorescence or a separately inferred signal.

## 2-b. How is the `neural` data processed?

i. Unregistered columns are removed, the remaining trace is Gaussian-smoothed along time with sigma 15 frames, averaged in non-overlapping 15-frame windows, multiplied by 30 to express rates in Hz, cast to `float32`, sliced into trials, and transposed to neuron-by-time arrays.

ii.
```python
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA_FRAMES, axis=0)
chunks = smoothed[:n_bins * BIN_FRAMES].reshape(n_bins, BIN_FRAMES, trace.shape[1])
return (chunks.mean(axis=1) * FPS).astype(np.float32)
...
neural.append(np.ascontiguousarray(rates[sl].T))
```

iii. The agent says smoothing followed by average pooling follows the paper's decoder recipe. It deliberately widened the paper's cited 100 ms bin to 500 ms to reduce sparsity and dataset size while retaining spatial resolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Columns that are entirely NaN for a session are treated as unregistered cells and removed. The agent applies no further neuron curation and asserts that retained traces contain no NaNs.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.isnan(trace).any(), f"{animal} day {day}: partial NaN trace"
```

iii. The agent states that all-NaN columns represent cells not registered that day and that retaining all registered cells reproduces the paper's reported 5,413 neurons and 69,744 neuron-sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental-event alignment. The alignment event is defined as the start of each artificial consecutive one-minute trial, with offsets 0 to 60 seconds.

ii.
```python
"temporal_alignment_event":
    "start of the trial, i.e. of a consecutive non-overlapping 1-minute "
    "window of the continuous free-exploration session",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

iii. The agent explains that this is continuous free exploration with no stimulus event and that trace and position already share a frame clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 500 ms. Fifteen native 30 Hz frames are Gaussian-smoothed with sigma 15 frames and then average-pooled into each output bin.

ii.
```python
BIN_FRAMES = 15
BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FPS
SMOOTH_SIGMA_FRAMES = BIN_FRAMES
```

iii. This was a deliberate departure: the agent cited sparse events and an estimated 6.7 GB size at 100 ms, choosing 500 ms for denser rates and a 1.33 GB output while arguing that movement within 500 ms is small relative to a 25 cm spatial bin.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the session's `blocked` HDF5 reference; `envs` is loaded for session labels/metadata but does not determine the numeric input.

ii.
```python
for ref in f["blocked"][0]:
    idx = f[ref][:].ravel().astype(int)
    blocked.append(idx[idx >= 0])
```

iii. The agent interprets the entries as flat indices of walled-off partitions, with `-1` meaning the unblocked square environment.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Negative sentinel values are removed and the remaining flat indices set entries in a static nine-element binary vector. A copy of that vector is stored for every trial.

ii.
```python
geo = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
geo[blocked_flat] = 1.0
...
inputs.append(geo.copy())
```

iii. The agent derived the flat convention as `3*y + x` from repository functions and says an empirical occupancy check found less than 0.01% occupancy in marked blocked bins, ensuring input dimension `i` corresponds to output class `i`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the two coordinate columns of each session's `position` array.

ii.
```python
position = f[f["position"][day, 0]][:]
labels = bin_labels(process_position(position), n_bins)
```

iii. The agent describes these as DeepLabCut head coordinates in centimeters in the fixed 75 by 75 cm arena frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Each frame is mapped to a physical 3-by-3 grid, clipped to valid indices, and flattened as `3*y + x`. For every 15-frame/500 ms window, the most frequent categorical bin becomes the output label.

ii.
```python
bins = np.floor(position / (ARENA_SIZE / N_SPATIAL_BINS)).astype(int)
bins = np.clip(bins, 0, N_SPATIAL_BINS - 1)
return bins[:, 1] * N_SPATIAL_BINS + bins[:, 0]
...
counts[:, k] = (chunks == k).sum(axis=1)
return counts.argmax(axis=1).astype(np.int64)
```

iii. Fixed 25 cm physical boundaries preserve comparability across geometries. The agent chose the modal categorical label because averaging coordinates could create a label for a partition the mouse never entered.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Coordinate thresholds are 25 and 50 cm on both axes, yielding axis bins 0–2 and labels 0–8 under `3*y + x`; out-of-range tracking jitter is clipped to edge bins.

ii.
```python
bins = np.floor(position / (ARENA_SIZE / N_SPATIAL_BINS)).astype(int)
bins = np.clip(bins, 0, N_SPATIAL_BINS - 1)
return bins[:, 1] * N_SPATIAL_BINS + bins[:, 0]
```

iii. The agent rejected per-session min/max scaling because blocked edge columns would shift boundaries and instead followed the arena's physical partition boundaries and the inferred geometry index convention.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace are assumed frame-aligned. Their shared usable length is the shorter stream; both are reduced to the same number of complete 15-frame bins and sliced with the same trial indices.

ii.
```python
n_frames = min(trace.shape[0], position.shape[0])
n_bins = n_frames // BIN_FRAMES
rates = bin_traces(trace, n_bins)
labels = bin_labels(process_position(position), n_bins)
...
neural.append(np.ascontiguousarray(rates[sl].T))
outputs.append(labels[sl][np.newaxis, :])
```

iii. The agent says DAQ timestamps and the source pipeline place both streams on a common 30 Hz clock, so matching frame windows require no interpolation or lag correction.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neural columns are removed as unregistered cells. Any partial NaN in a retained neural column or any NaN position causes an assertion failure. Length discrepancies are handled by truncating to the shorter stream, and incomplete temporal bins/minute trials are dropped.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.isnan(trace).any()
assert not np.isnan(position).any()
n_frames = min(trace.shape[0], position.shape[0])
```

iii. The agent regarded all-NaN columns as intentional registration padding, not corrupt measurements. Assertions prevent silent propagation of unexpected missing values, while common-length truncation maintains alignment.

## 6-a. What are the most time-consuming steps of the code?

i. Reading the large HDF5 trace arrays and applying `gaussian_filter1d` to every full time-by-cell session are the dominant conversion operations; serializing the roughly 1.33 GB pickle is also substantial.

ii.
```python
trace = f[f["trace"][day, 0]][:]
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA_FRAMES, axis=0)
...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory shows the full conversion running for several minutes and emphasizes output size. Unlike the simpler reference, this implementation adds a whole-session Gaussian filtering pass.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The nine-class loop used to count labels per temporal bin could be replaced by a vectorized one-hot/count or mode operation. Session and trial loops mostly organize ragged nested outputs and HDF5 references, so full vectorization would provide less benefit.

ii.
```python
counts = np.zeros((n_bins, N_SPATIAL_BINS ** 2), dtype=np.int32)
for k in range(N_SPATIAL_BINS ** 2):
    counts[:, k] = (chunks == k).sum(axis=1)
```

iii. The trajectory does not explicitly justify this loop; it is a straightforward implementation over only nine classes, so its practical cost is likely modest.

## 6-c. What processing does the code repeat multiple times?

i. For each session it repeatedly filters/smooths full traces, discretizes positions, constructs geometry, and allocates a copied geometry vector for every trial. It also recomputes `len(set(envs))` for every session's metadata.

ii.
```python
inputs.append(geo.copy())
...
"sequence": day // len(set(envs)),
```

iii. No explicit trajectory justification addresses repetition. Copies avoid aliasing between trial inputs, while the repeated set computation and repeated static vectors are convenience choices rather than necessities.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It decodes environment names and builds extensive `session_info` metadata that the decoder does not use. It also Gaussian-smooths samples beyond the last complete minute before those trailing bins are discarded, and computes reporting totals after conversion.

ii.
```python
envs = [_read_string(f, ref) for ref in f["envs"][0]]
...
rates = bin_traces(trace, n_bins)
n_trials = n_bins // TRIAL_BINS
...
session_info.append(session["info"])
```

iii. The agent retained the metadata for provenance and diagnostics. The trajectory focuses on verification statistics, but gives no downstream-analysis need for environment strings or the processing of trailing data.
