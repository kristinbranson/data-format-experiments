# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script uses a fixed list of seven animals, opens each animal's MATLAB v7.3/HDF5 file lazily with `h5py`, reads session metadata, and then dereferences each day's `trace` and `position` arrays. Full mode traverses every day; sample mode selects two sessions.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
def open_animal(animal):
    return h5py.File(os.path.join(DATA_DIR, f'{animal}.mat'), 'r')
...
for ai, animal in enumerate(animals):
    f = open_animal(animal)
    envs, blocked = read_meta(f)
    days = SAMPLE_SESSIONS[animal] if sample else range(len(envs))
    for day in days:
        trace, position = read_session(f, day)
```

iii. The notes say lazy HDF5 reads are byte-identical to the joblib data, avoid loading an entire expanded animal (over 10 GB RAM), and reduce peak memory to about 1 GB. They report all 207 sessions were processed.

## 1-b. How are the data split into subjects?

i. Each named `.mat` file is one mouse. `subjects` is the fixed animal list, and every appended session receives that animal's index in the list.

ii.
```python
data = {'subjects': list(ANIMALS), 'subject_idx': [], ...}
subj_idx = ANIMALS.index(animal)
...
data['subject_idx'].append(subj_idx)
```

iii. The agent states the data contain seven per-animal files and confirms seven mice and the expected 31/31/31/21/31/31/31 sessions.

## 1-c. How are the data split into sessions?

i. Every HDF5 reference in `trace[day,0]` and `position[day,0]` is treated as one recording session/day. A session is appended only when at least two usable trials survive processing.

ii.
```python
def read_session(f, day):
    trace = f[f['trace'][day, 0]][:]
    position = f[f['position'][day, 0]][:]
    return trace, position
...
if len(ntr) < 2:
    continue
data['neural'].append(ntr)
```

iii. The notes identify one reference-array entry as one day/session and report that all 207 sessions pass the agent's checks.

## 1-d. How are the data split into trials?

i. After 3-frame pooling, sessions are cut into consecutive nominal 60-second blocks (600 pooled bins). A final block is retained if it contains at least 30 seconds before speed filtering; shorter remainders are dropped. Kept trials can have variable length because non-locomotion bins are removed.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN_FRAMES))
while start < n_bins:
    stop = min(start + BINS_PER_TRIAL, n_bins)
    if (stop - start) < MIN_TRIAL_FRAC * BINS_PER_TRIAL:
        break
    sel = np.where(moving_binned[start:stop])[0] + start
    ...
    start = stop
```

iii. The agent says keeping a remainder of at least 30 seconds avoids discarding about 2.5% of the data; the decoder task requires one-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. Within each 60-second block, bins at or below 5 cm/s are removed. A trial is retained only if at least 30 moving 100-ms bins (3 seconds) remain; sessions with fewer than two retained trials are skipped.

ii.
```python
moving_binned = speed_binned > V_THRESH
sel = np.where(moving_binned[start:stop])[0] + start
if sel.size >= MIN_BINS_PER_TRIAL:
    neural_trials.append(...)
...
if len(ntr) < 2:
    continue
```

iii. The speed cutoff is justified as the `decode_position_within` reference setting and as excluding immobility/replay periods. The three-second minimum is a pragmatic usability threshold; no paper-derived justification is given.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each session's raw `trace` HDF5 dataset, described as the authors' binarized rising phase of calcium transients.

ii.
```python
trace = f[f['trace'][day, 0]][:]
...
tr = trace[:, keep_cells]
```

iii. The paper says this distributed binary vector was treated as firing rate, so the agent concluded that dF/F need not be computed.

## 2-b. How is the `neural` data processed?

i. Selected traces are Gaussian-smoothed over time with sigma 3 frames, averaged in non-overlapping 3-frame blocks, cast to float32, multiplied by 30 to express events/s, and transposed to neuron-by-time for each trial.

ii.
```python
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SMOOTH_SIGMA, axis=0)
neural = pool_mean(tr_smooth).astype(np.float32) * FPS
...
neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
```

iii. The agent attributes smoothing and `AvgPool1d(3)` to `fit_decoder`/`test_decoder`; scaling changes units only and was intended to keep values numerically suitable for the supplied decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell must be registered on that day (`trace[0]` non-NaN) and have more than five summed events during raw-frame locomotion. No place-cell selection is performed.

ii.
```python
registered = ~np.isnan(trace[0, :])
events_moving = np.nansum(trace[moving, :], axis=0)
keep_cells = registered & (events_moving > CELL_THRESHOLD)
```

iii. This is justified as the `decode_position_within` cell threshold; the agent notes that place-cell selection would conflict with the paper's inclusion of all cells in later analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trial start is the alignment point. Neural and position streams are assumed frame-aligned at 30 Hz, pooled on the same three-frame grid, sliced into the same 60-second blocks, and selected with the same moving-bin indices.

ii.
```python
neural = pool_mean(tr_smooth)
pos_binned = pool_mean(position)
speed_binned = pool_mean(speed[:, None])[:, 0]
...
neural[sel, :]
out_class[sel]
```

iii. The notes say both streams were acquired simultaneously by the same DAQ and validate alignment by recomputing stored occupancy/rate maps and visually comparing raw and pooled position.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms (10 Hz), obtained by averaging non-overlapping groups of three native 30-Hz frames after neural smoothing.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    n = (x.shape[0] // k) * k
    return x[:n].reshape((n // k, k) + x.shape[1:]).mean(axis=1)
```

iii. The agent chose the temporal preprocessing used by the paper's decoder and reports equivalence to `AvgPool1d(kernel=3,stride=3)`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The saved input is computed from each session's `envs` geometry name through a hard-coded map of the ten geometries. The raw `blocked` indices are loaded only for an equality assertion.

ii.
```python
envs = [''.join(chr(c) for c in f[r][:].ravel()) for r in f['envs'][0]]
...
input_vec = env_blocked_vector(env_name)
expected = np.where(env_blocked_vector(envs[day]))[0]
got = np.sort(blocked[day][blocked[day] >= 0])
assert np.array_equal(expected, got)
```

iii. The agent re-derived orientation from `get_env_mat`/`clean_rate_maps` and verified exact agreement with raw `blocked` for all 207 sessions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The named 3x3 open-area matrix is flipped vertically, flattened using class order `3*y+x`, compared with zero, and cast to a nine-element float32 blocked/not-blocked vector. A copy is stored for each trial.

ii.
```python
def env_blocked_vector(env_name):
    m = np.array(ENV_MATS[env_name], dtype=float)
    return (np.flipud(m).ravel() == 0).astype(np.float32)
...
input_trials.append(input_vec.copy())
```

iii. This orientation was justified by exact agreement with the dataset's blocked field and by near-zero occupancy in blocked partitions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the per-session two-column `position` HDF5 dataset in centimeters.

ii.
```python
position = f[f['position'][day, 0]][:]
...
pos_binned = pool_mean(position)
out_class = position_to_class(pos_binned, blocked_vec=input_vec)
```

iii. The notes identify this as DeepLabCut head position, acquired at the same 30-Hz rate as calcium.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are averaged over each three-frame temporal bin, converted to a 3x3 class, and rare samples assigned to a physically blocked partition are reassigned to the nearest open partition center. Only moving bins are retained.

ii.
```python
pos_binned = pool_mean(position)
out_class = position_to_class(pos_binned, blocked_vec=input_vec)
moving_binned = speed_binned > V_THRESH
output_trials.append(out_class[sel][None, :].copy())
```

iii. Pooling is justified as matching the neural grid. Snapping is described as a 3x3 analogue of the reference decoder's environment cleaning and affects about 0.005% of samples.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is divided by 25 cm and truncated to bins 0, 1, or 2; values are clipped at arena bounds. The class is `3*ybin+xbin` (0-8), followed by the blocked-partition correction.

ii.
```python
part = ARENA_SIZE / N_PART
bins = np.clip((position / part).astype(int), 0, N_PART - 1)
cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
```

iii. The task requires 3x3 output bins, and the class ordering was verified against the raw blocked indices and occupancy maps.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are pooled over identical non-overlapping raw-frame groups, checked to have equal bin counts, then indexed by the identical per-trial locomotion selection.

ii.
```python
neural = pool_mean(tr_smooth)
pos_binned = pool_mean(position)
assert pos_binned.shape[0] == neural.shape[0] == speed_binned.shape[0]
...
neural[sel, :].T
out_class[sel][None, :]
```

iii. The agent cites simultaneous acquisition and reports independent all-close checks and plots showing no temporal shift.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered/NaN cells and low-event cells are removed; blocked-field metadata must agree with the geometry or conversion aborts; coordinates are clipped; positions in blocked cells are snapped open; incomplete pooling frames and remainders under 30 seconds are discarded; trials/sessions with too little usable data are dropped.

ii.
```python
keep_cells = registered & (events_moving > CELL_THRESHOLD)
assert np.array_equal(expected, got)
bins = np.clip(...)
n = (x.shape[0] // k) * k
if sel.size >= MIN_BINS_PER_TRIAL: ...
if len(ntr) < 2: continue
```

iii. The notes frame these as registration handling, consistency checks, boundary/noise cleanup, and minimum data requirements. They report no NaN/Inf in validation.

## 6-a. What are the most time-consuming steps of the code?

i. Per-session HDF5 reading and session processing (especially Gaussian filtering of large trace matrices) dominate; serializing the 3.4-GB pickle is another measured cost.

ii.
```python
t0 = time.time(); trace, position = read_session(f, day)
t1 = time.time(); ntr, inp, outp, keep = process_session(...)
...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes measure roughly 0.43 seconds reading and 0.46 seconds processing per session and estimate a several-minute full conversion. Lazy reads avoid the slower, memory-heavy whole-animal joblib route.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Core numeric work is already vectorized. The remaining animal/day and trial loops are structural and perform HDF5 reads or create required nested objects; `position_to_class`, pooling, speed, and cell filtering use NumPy/SciPy vector operations. Trial construction might be partially batched, but variable speed-filtered lengths limit benefit.

ii.
```python
return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)
bins = np.clip((position / part).astype(int), 0, N_PART - 1)
for day in days:
    ...
while start < n_bins:
```

iii. The notes explicitly say Python frame-binning loops were avoided and NumPy reshape-mean replaced a tensor round trip; no further vectorization was considered necessary.

## 6-c. What processing does the code repeat multiple times?

i. Each session independently recomputes its geometry vector, speed, cell mask, smoothing, pooling, and trial selections. Geometry conversion repeats across trials/sessions even though only ten named geometries exist, and the blocked vector is copied once per trial. Optional plotting also recomputes display-only bin labels.

ii.
```python
input_vec = env_blocked_vector(env_name)
...
input_trials.append(input_vec.copy())
...
xb = np.clip((diag['pos_binned'][wb, 0] / 25).astype(int), 0, 2)
```

iii. The agent does not identify these as problematic; the notes emphasize that runtime is small enough that caching/parallelism was unnecessary.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full conversion reads both environment names and raw blocked arrays but only stores geometry derived from names; raw blocked is used for assertions. It computes speed solely for filtering and does not save it. Diagnostic dictionaries and six-panel plots are created only with `--show-processing`. Timing and rich session metadata do not enter decoder training. Neural smoothing, rate scaling, and blocked-position snapping are also extra relative to the human conversion, though the agent intended them as substantive preprocessing rather than disposable work.

ii.
```python
envs, blocked = read_meta(f)
...
speed = compute_speed(position)
...
diag = {} if (args.show_processing and n_plotted < 2) else None
timings.append((t_read, t_proc))
```

iii. The agent justifies the assertion and diagnostics as validation and the timings as performance reporting. It does not claim that these values are decoder features.
