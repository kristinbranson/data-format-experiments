# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, then loads one joblib file per animal from `/app/data`. For each animal it reads the per-animal dictionary fields `trace`, `position`, `envs`, and `blocked`, and then iterates over every recording day. It does not enumerate files dynamically and does not load the `.mat` files directly.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

for ai, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    traces, positions = dat['trace'], dat['position']
    envs = [str(e[0]) for e in dat['envs']]
    blocked = dat['blocked']
    n_days = traces.shape[0]
```

iii. In trajectory steps 16-17, the agent says it inspected one joblib file and found a per-animal dict containing `envs`, `blocked`, `position`, and `trace`. In steps 30 and 48, it justifies using the authors' joblib files because it believed they were identical in content to the `.mat` files and consistent with the paper repository.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is treated as one subject. The subject list in the final output is exactly the `ANIMALS` constant, and the session-to-subject mapping is built from the loop index `ai`.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

for ai, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(ai)

data = {
    ...
    'subjects': ANIMALS,
    'subject_idx': np.array(subject_idx, dtype=np.int64),
}
```

iii. In steps 22, 30, and 48, the agent explicitly describes the dataset as "7 animals" and uses the per-animal joblib structure as the subject split.

## 1-c. How are the data split into sessions?

i. Within each subject, each recording day is treated as a separate session. The AI loops over `range(n_days)` and appends one output session per day that survives later filtering.

ii.
```python
n_days = traces.shape[0]

for day in range(n_days):
    env = envs[day]
    inp = blocked_vector(env, blocked[day])
    pos = np.asarray(positions[day], dtype=np.float64)
    tr = np.asarray(traces[day])
    ...
    neural_all.append(sess_neural)
    input_all.append(sess_input)
    output_all.append(sess_output)
```

iii. In steps 16, 22, 30, and 48, the agent says the joblib data are organized by day, that there are 207 total sessions/days, and that it will use "per animal/day session" processing.

## 1-d. How are the data split into trials?

i. Sessions are first rebinned to 100 ms bins, then divided into consecutive 1-minute windows of `600` bins each. Within each 1-minute window, only bins whose pooled running speed exceeds threshold are kept, so the saved trials have variable numbers of time bins. The final partial chunk is allowed because the slice end is `min(start + BINS_PER_TRIAL, nbins)`.

ii.
```python
TEMPORAL_BIN = 3
TRIAL_SEC = 60.0
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN))  # 600 bins

for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    if keep.sum() < MIN_BINS_PER_TRIAL:
        continue
    sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
    sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
    sess_input.append(inp.copy())
```

iii. In steps 23, 30, and 48, the agent states it is following the paper's decoding pipeline by smoothing and pooling to 100 ms, then "split into 1-min (600-bin) trials" and "keep running bins only."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by movement: a 1-minute chunk is dropped if it contains fewer than `30` running bins after speed thresholding. After that, an entire session is dropped if fewer than `2` usable trials remain.

ii.
```python
MIN_BINS_PER_TRIAL = 30

for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    if keep.sum() < MIN_BINS_PER_TRIAL:
        continue
    ...
if len(sess_neural) < 2:
    print(f'  day {day} ({env}): < 2 usable trials, skipped')
    continue
```

iii. In step 30 the agent says it will "speed-filter bins" and keep only usable 1-minute trials. In the saved metadata it also justifies this as `trial_curation`: trials with fewer than 30 running bins and sessions with fewer than 2 usable trials are dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data are derived from the raw `trace` field in each joblib animal file.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
traces, positions = dat['trace'], dat['position']
...
tr = np.asarray(traces[day])  # (n_cells, T)
```

iii. In steps 16, 17, 23, and 30, the agent identifies `trace` as the calcium-event matrix and notes that it is already binarized, with NaN rows for cells not registered on a given day.

## 2-b. How is the `neural` data processed?

i. The AI removes unregistered cells, computes running speed from position, filters out low-activity cells, Gaussian-smooths the traces, average-pools them over 3 frames to 100 ms bins, and finally keeps only the running bins inside each saved trial.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered].astype(np.float32)

speed = np.zeros(pos.shape[1])
speed[1:] = np.linalg.norm(np.diff(pos, axis=1), axis=0) * FPS
speed = gaussian_filter1d(speed, sigma=V_FILT_SIGMA)
running_frames = speed > V_THRESH

active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]

tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
```

iii. In steps 23, 30, and 48, the agent explicitly says it is following `decode_position_within` and `fit_decoder`: speed filter with Gaussian smoothing, activity threshold `> 5` events during running, then Gaussian smoothing of traces and 3-frame average pooling to 100 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two neuron filters are applied. First, cells not registered on that day are removed by checking for NaN in the first sample. Second, among the remaining cells, only those with more than `5` total events in running frames are kept. Sessions with zero surviving cells are skipped entirely.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered].astype(np.float32)

active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]
if tr.shape[0] == 0:
    print(f'  day {day} ({env}): no cells pass criteria, skipped')
    continue
```

iii. In steps 17, 23, 26, 30, and 48, the agent says unregistered cells appear as all-NaN rows and that it is intentionally adding the paper's `cell_threshold=5` activity criterion during running.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus-locked alignment. The AI treats the start of each recording day as the implicit alignment point, rebins the continuous session, and cuts it into consecutive 1-minute windows. Within each window, non-running bins are removed, so alignment is by session chronology plus a shared running-mask rather than by a discrete event.

ii.
```python
for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    ...
    sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))

'metadata': {
    'temporal_alignment_event': (
        'start of the recording session; each session is cut into consecutive '
        '1-minute trials'),
    'off_start': 0.0,
    'off_end': 60.0,
}
```

iii. In steps 30 and 48, the agent justifies this by saying the data are long continuous sessions and that the intended representation is consecutive 1-minute chunks, not event-triggered trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved data use 100 ms bins. The AI applies temporal rebinning by smoothing traces and averaging every 3 original frames, and it pools position and speed over the same 3-frame windows.

ii.
```python
FPS = 30.0
TEMPORAL_BIN = 3      # frames per time bin -> 100 ms

tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
pos_b = pool_mean(pos, TEMPORAL_BIN)
speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]

'metadata': {
    'time_bin_size': 100.0,
}
```

iii. In steps 23, 25, 30, and 48, the agent says it chose 100 ms bins because that matched `fit_decoder(... temporal_bin_size=3)` in the paper repository and reduced dataset size relative to keeping the original 30 Hz frames.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The final environment-geometry input is derived from the raw `blocked` field for each day. The `envs` field is also used, but only to cross-check that the stored blocked indices match a flipped version of the repository's `get_env_mat`.

ii.
```python
traces, positions = dat['trace'], dat['position']
envs = [str(e[0]) for e in dat['envs']]
blocked = dat['blocked']
...
inp = blocked_vector(env, blocked[day])

def blocked_vector(env_name, blocked_field):
    b = np.atleast_1d(np.asarray(blocked_field, dtype=float).ravel())
    vec = np.zeros(9, dtype=np.float32)
    if not (b.size == 1 and b[0] == -1):
        vec[b.astype(int)] = 1.0
    from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
    assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
    return vec
```

iii. In steps 19, 20, 23, and 32, the agent says it examined the relationship between occupancy, `blocked`, and `get_env_mat`, then decided to trust the dataset's own `blocked` indexing and use `envs` only for validation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the blocked partitions into a 9-element binary vector where `1` means blocked. If the raw field is `[-1]`, the vector stays all zeros. The same vector is copied into every trial from that session as a static input.

ii.
```python
vec = np.zeros(9, dtype=np.float32)
if not (b.size == 1 and b[0] == -1):
    vec[b.astype(int)] = 1.0

for start in range(0, nbins, BINS_PER_TRIAL):
    ...
    sess_input.append(inp.copy())
```

iii. In step 32, the agent says it will use the dataset's own `blocked` convention for both input and output and cross-check it against `flipud(get_env_mat)`. In steps 30 and 48, it describes the input as a static 9-dimensional blocked-geometry vector.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw per-day `position` field.

ii.
```python
traces, positions = dat['trace'], dat['position']
...
pos = np.asarray(positions[day], dtype=np.float64)
...
out_bins = position_to_bin(pos_b)[np.newaxis, :]
```

iii. In steps 16, 19, 20, and 23, the agent says `position` is a `(2, T)` trajectory in centimeters spanning approximately `0-75 cm`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first average-pools the continuous `x,y` position over 3-frame windows, then discretizes each pooled sample into one of 9 spatial bins. After that, it removes non-running bins by applying the same `keep` mask used for neural data within each 1-minute chunk.

ii.
```python
pos_b = pool_mean(pos, TEMPORAL_BIN)                        # (2, nbins)
speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]     # (nbins,)
run_b = speed_b > V_THRESH

out_bins = position_to_bin(pos_b)[np.newaxis, :]            # (1, nbins)

for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    ...
    sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
```

iii. In steps 23, 30, and 48, the agent says the paper's decoder averages position over the same 3-frame bins as neural data and uses only running periods for within-session position decoding.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 75 cm arena is divided into 3 equal bins per axis. The AI computes `x_bin = floor(x / 25)` and `y_bin = floor(y / 25)`, clips both to `0..2`, and encodes the class as `3 * y_bin + x_bin`, matching the dataset's blocked-partition indexing.

ii.
```python
edge = ARENA_SIZE / N_SPACE_BINS
xb = np.clip(np.floor(pos_xy[0] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
yb = np.clip(np.floor(pos_xy[1] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
return 3 * yb + xb
```

iii. In steps 20, 23, 32, and 33, the agent explains that it empirically checked occupancy against the blocked partitions and chose the `3 * y_bin + x_bin` convention to match the dataset's `blocked` field.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned by applying the same temporal pooling and the same per-trial running mask to both streams. Both are sliced with the same 1-minute windows and then indexed by the same boolean `keep` array.

ii.
```python
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
pos_b = pool_mean(pos, TEMPORAL_BIN)
speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]
run_b = speed_b > V_THRESH
out_bins = position_to_bin(pos_b)[np.newaxis, :]

for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
    sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
```

iii. In steps 30 and 48, the agent says both neural and position signals should undergo the same pooling and running-bin filtering so the decoder sees synchronized time points.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles one specific missing-data case: cells not registered on a day are identified by NaN values and removed. It also skips sessions where no cells survive, drops trials with too little running data, drops sessions with fewer than two usable trials, and uses an occupancy assertion to catch geometry/indexing mistakes. It does not impute or repair missing values.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered].astype(np.float32)

active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]
if tr.shape[0] == 0:
    ...

occ = np.bincount(out_bins[0], minlength=9) / nbins
assert occ[inp.astype(bool)].sum() < 0.02

if keep.sum() < MIN_BINS_PER_TRIAL:
    continue
...
if len(sess_neural) < 2:
    continue
```

iii. In steps 16, 17, and 22, the agent says unregistered cells appear as NaN rows. In steps 30, 32, and 48, it justifies the extra dropping rules as paper-inspired curation and uses the occupancy assertion as a guard against binning mistakes.

## 6-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are loading each large per-animal joblib file, Gaussian smoothing and average pooling all sessions, and serializing the very large output pickle. The code does no explicit profiling, but these are the heaviest operations visible in the implementation.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
...
with open(OUT_FILE, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The trajectory does not contain a formal profiling decision, but steps 25, 36, and 38 mention disk-space concerns, a long-running conversion, and a 3.3 GB output file, which is consistent with I/O plus whole-session smoothing/pooling dominating runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main avoidable Python loop is the per-session trial-building loop over `start` offsets. Because trial boundaries are regular after pooling, the code could have reshaped most full-length sessions into `(n_trials, ...)` blocks instead of slicing/appending one trial at a time. The outer animal/day loops are structurally necessary, but the inner trial construction is the clearest vectorization opportunity.

ii.
```python
for ai, animal in enumerate(ANIMALS):
    ...
    for day in range(n_days):
        ...
        for start in range(0, nbins, BINS_PER_TRIAL):
            sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
            keep = run_b[sl]
            ...
```

iii. No explicit efficiency justification appears in the trajectory. The agent focused on reproducing what it believed was the paper's preprocessing rather than reducing Python-loop overhead.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly applies the same per-trial slicing and boolean masking separately to neural and output arrays. It also copies the same static input vector into every trial with `inp.copy()`, even though the value is constant within a session.

ii.
```python
for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    ...
    sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
    sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
    sess_input.append(inp.copy())
```

iii. The trajectory does not give an explicit rationale for these repeated operations. They appear to be a straightforward implementation choice rather than a deliberate optimization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several intermediate computations are only used for filtering or sanity checks and are not saved for downstream decoding: full-resolution speed, the running masks, occupancy checks in blocked bins, and the flipped-geometry cross-check against `get_env_mat`. The decoder also does not use the verbose `session_info` metadata.

ii.
```python
speed = np.zeros(pos.shape[1])
speed[1:] = np.linalg.norm(np.diff(pos, axis=1), axis=0) * FPS
speed = gaussian_filter1d(speed, sigma=V_FILT_SIGMA)
running_frames = speed > V_THRESH
...
speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]
run_b = speed_b > V_THRESH

occ = np.bincount(out_bins[0], minlength=9) / nbins
assert occ[inp.astype(bool)].sum() < 0.02

from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
```

iii. The trajectory justifies these steps as safety checks and paper-inspired curation, especially in steps 30, 32, and 48. It does not claim they are needed by the downstream decoder itself.
