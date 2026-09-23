# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a fixed list of seven per-animal `joblib` files from `/app/data`, not the raw `.mat` files. For each animal file it reads the `position`, `trace`, `envs`, and `blocked` fields, then iterates over every day/session and later splits each session into trials.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for ai, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    pos_all, trace_all = dat['position'], dat['trace']
    envs = [str(e[0]) for e in dat['envs']]
    n_days = trace_all.shape[0]
    for day in range(n_days):
```

iii. In the trajectory, the agent said it was following the repo README’s “per animal joblib file” description and confirmed by inspection that those files contained `position`, `trace`, `envs`, and `blocked` for all seven animals. It treated those joblib files as the primary source for the conversion.

## 1-b. How are the data split into subjects?

i. Each animal file is treated as one subject. Subject IDs come from the hard-coded `ANIMALS` list, and each session appended for that animal gets the corresponding subject index.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
data = {
    ...
    'subjects': list(animals), 'subject_idx': [],
    ...
}
...
for ai, animal in enumerate(animals):
    ...
    data['subject_idx'].append(ai)
```

iii. In the trajectory, the agent concluded that there were seven mouse files and that each file corresponded to one mouse. It reported totals such as “207 sessions across 7 mice” and used those animals directly as subject labels.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day within an animal file as one session. It loops over `n_days = trace_all.shape[0]` and appends one output session per day unless the day is skipped for having no usable neurons or fewer than two full trials after processing.

ii.
```python
pos_all, trace_all = dat['position'], dat['trace']
envs = [str(e[0]) for e in dat['envs']]
n_days = trace_all.shape[0]
for day in range(n_days):
    env = envs[day]
    ...
    if tr.shape[0] == 0:
        print(f'  skipping {animal} day {day}: no cells pass criteria')
        continue
    ...
    if n_trials < 2:
        print(f'  skipping {animal} day {day}: < 2 full trials')
        continue
    ...
    data['neural'].append(neural_trials)
```

iii. In the trajectory, the agent explicitly said “one session = one recording day” and verified that the seven animals together produced 207 sessions, matching the paper’s session count.

## 1-d. How are the data split into trials?

i. Each session is split into consecutive, non-overlapping 1-minute trials after temporal pooling to 100 ms bins. With `TRIAL_SEC = 60` and `BIN_MS = 100`, each trial has `TRIAL_BINS = 600` time bins. Any trailing partial trial is dropped.

ii.
```python
BIN_MS = 1000.0 * TEMPORAL_BIN / FPS   # = 100 ms
TRIAL_SEC = 60.0
TRIAL_BINS = int(round(TRIAL_SEC * 1000.0 / BIN_MS))   # 600 bins per trial
...
n_bins = min(neural.shape[1], labels.shape[0])
n_trials = n_bins // TRIAL_BINS
...
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(labels[sl][None, :].copy())
```

iii. In the trajectory, the agent repeatedly stated that the instruction-defined trial was a 1-minute contiguous segment and that it would split each 40-minute recording into consecutive 1-minute chunks after the paper-style 100 ms temporal binning.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality filter beyond dropping any trailing incomplete chunk that does not make a full 1-minute trial. The code also skips an entire session if, after processing, it contains fewer than two full trials, because the downstream decoder requires at least two trials per session.

ii.
```python
n_trials = n_bins // TRIAL_BINS
if n_trials < 2:
    print(f'  skipping {animal} day {day}: < 2 full trials')
    continue
...
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
```

iii. In the trajectory, the agent focused on satisfying the decoder-format requirement of at least two trials per session. It did not describe any trial-level artifact rejection, only contiguous segmentation and dropping incomplete remainders.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from the raw `trace` field in each animal’s joblib file. The agent interprets this as the binarized rising phase of calcium transients for each cell over time.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
pos_all, trace_all = dat['position'], dat['trace']
...
trace = trace_all[day]                            # (n_cells, n_frames)
```

iii. In the trajectory, the agent noted from the paper and code that the analysis used “the binary vector of the rising phases of transients” as the firing-rate-like neural signal, and it described the loaded `trace` array as “binary events at 30 Hz.”

## 2-b. How is the `neural` data processed?

i. The agent keeps the per-day recorded cells, smooths each neural trace with a Gaussian kernel of sigma 3 frames, then average-pools over non-overlapping 3-frame windows. That converts the native 30 Hz traces into 100 ms binned neural activity.

ii.
```python
TEMPORAL_BIN = 3
...
def pool_mean(x, k):
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(x.shape[:-1] + (n // k, k)).mean(axis=-1)
...
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, n_bins)
```

iii. In trajectory steps 9, 20, 23, and 24, the agent said it would follow the paper’s decoder pipeline, specifically “gaussian smoothing sigma=3 frames” plus “3-frame average pooling (100 ms).” It justified this as matching `utils.fit_decoder / utils.test_decoder`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent removes cells that are all-NaN on a given day, treating them as unregistered in that session. Among the registered cells, it keeps only those with more than 5 events across the session after ignoring NaNs. If no cells survive, the entire session is skipped.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=1)
cell_ids = np.where(registered)[0]
tr = trace[cell_ids]
enough = np.nansum(tr, axis=1) > CELL_EVENT_THRESHOLD
cell_ids, tr = cell_ids[enough], tr[enough]
if tr.shape[0] == 0:
    print(f'  skipping {animal} day {day}: no cells pass criteria')
    continue
```

iii. In the trajectory, the agent first identified the paper’s decoder criterion as “cells with >5 events during movement,” then later changed the implementation rationale after deciding not to apply the velocity filter. It documented that the `>5` event threshold would instead be evaluated over the whole session so contiguous trials could be preserved.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no task-event alignment. The code defines alignment relative to the start of each artificial 1-minute trial, which is just an arbitrary segment of a continuous free-exploration session.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': (
        'start of each 1-min trial, i.e. arbitrary segmentation of the '
        'continuous 40-min free-exploration session (no task events)'),
    'off_start': 0.0,
    'off_end': TRIAL_SEC,
    ...
}
```

iii. In the trajectory, the agent described the recordings as continuous free exploration with “no task events,” so the only alignment it used was the start of each decoder-defined 1-minute segment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. The native 30 Hz streams are rebinned by smoothing and average-pooling every 3 frames, so yes, temporal rebinning is applied.

ii.
```python
FPS = 30.0
TEMPORAL_BIN = 3
BIN_MS = 1000.0 * TEMPORAL_BIN / FPS   # = 100 ms
...
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T
...
'time_bin_size': BIN_MS,
```

iii. In trajectory steps 9, 20, 23, and 24, the agent explicitly chose “3-frame average pooling (100 ms bins)” because it wanted to mirror the paper’s decoder implementation rather than keep the native frame rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment-geometry input is derived from the raw `blocked` field, which stores the blocked partition indices for each session/day. The `envs` field is loaded and stored in metadata, but the actual decoder input comes from `blocked`.

ii.
```python
env = envs[day]
blocked = np.array(dat['blocked'][day]).ravel().astype(int)
blocked = blocked[blocked >= 0]
...
geom = np.zeros(NGRID * NGRID, dtype=np.float32)
geom[blocked] = 1.0
```

iii. In the trajectory, the agent said it had confirmed the blocked-partition indexing convention and would use a 9-dimensional binary geometry vector as the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked partition indices are converted into a length-9 binary vector, with `1` for blocked partitions and `0` for open partitions. That vector is static within a session and copied once per trial.

ii.
```python
open_mask = np.ones(NGRID * NGRID, dtype=bool)
open_mask[blocked] = False
...
geom = np.zeros(NGRID * NGRID, dtype=np.float32)
geom[blocked] = 1.0
...
for t in range(n_trials):
    ...
    input_trials.append(geom.copy())
```

iii. In the trajectory, the agent described the input as “the static (per-trial) environment geometry: 9 binary values, 1 if that partition is blocked,” based on the paper’s 3x3 partition scheme.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The decoded output is derived from the raw `position` variable for each day/session.

ii.
```python
pos_all, trace_all = dat['position'], dat['trace']
...
pos = pos_all[day].T.astype(np.float64)          # (n_frames, 2), cm
```

iii. In the trajectory, the agent identified `position` as the DeepLabCut-tracked x-y coordinates recorded simultaneously with calcium imaging and used it as the source of the output labels.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The position stream is first average-pooled over 3-frame windows to match the 100 ms neural bins. The pooled 2D coordinates are then converted to 3x3 partition labels. If a pooled position falls into a blocked partition, the code reassigns it to the nearest open partition center.

ii.
```python
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T                    # (n_bins, 2)
labels = partition_labels(pos_b, open_mask).astype(np.int64)
...
def partition_labels(pos_binned_xy, open_mask):
    xb = np.clip(np.floor(pos_binned_xy[:, 0] / PART_SIZE).astype(int), 0, NGRID - 1)
    yb = np.clip(np.floor(pos_binned_xy[:, 1] / PART_SIZE).astype(int), 0, NGRID - 1)
    labels = NGRID * yb + xb
    bad = ~open_mask[labels]
    if np.any(bad):
        open_idx = np.where(open_mask)[0]
        centers = np.stack([(open_idx % NGRID + 0.5) * PART_SIZE,
                            (open_idx // NGRID + 0.5) * PART_SIZE], axis=1)
        d = np.linalg.norm(pos_binned_xy[bad][:, None, :] - centers[None], axis=2)
        labels[bad] = open_idx[np.argmin(d, axis=1)]
    return labels
```

iii. In the trajectory, the agent said it had empirically verified the partition-index convention from occupancy versus blocked partitions. It also justified snapping the rare blocked-bin samples as a way to handle tracking noise, citing the paper’s decoder behavior of snapping positions to visitable bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The x and y coordinates are each thresholded into three 25 cm bins across the 75 cm arena. The final category is `3 * y_bin + x_bin`, giving 9 spatial classes in the same index order as the blocked-partition representation.

ii.
```python
ARENA_SIZE = 75.0
NGRID = 3
PART_SIZE = ARENA_SIZE / NGRID   # 25 cm
...
xb = np.clip(np.floor(pos_binned_xy[:, 0] / PART_SIZE).astype(int), 0, NGRID - 1)
yb = np.clip(np.floor(pos_binned_xy[:, 1] / PART_SIZE).astype(int), 0, NGRID - 1)
labels = NGRID * yb + xb
```

iii. In the trajectory, the agent said it had confirmed that the correct indexing convention was `3 * y_bin + x_bin`, matching the `blocked` field’s `[[0,1,2],[3,4,5],[6,7,8]]` layout.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural activity and position are aligned by applying the same 3-frame temporal pooling to both streams, then slicing both pooled arrays with the same per-trial boundaries.

ii.
```python
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T
labels = partition_labels(pos_b, open_mask).astype(np.int64)
...
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    output_trials.append(labels[sl][None, :].copy())
```

iii. In the trajectory, the agent repeatedly framed the processing around a common 100 ms time base so neural, input, and output streams could be stored as contiguous aligned trials.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code removes cells that are all-NaN on a session, treats negative blocked indices as “no blocked partitions,” snaps rare position samples that fall into blocked bins to the nearest open partition, drops incomplete trailing trial fragments, and skips sessions with no surviving neurons or with fewer than two full trials.

ii.
```python
blocked = np.array(dat['blocked'][day]).ravel().astype(int)
blocked = blocked[blocked >= 0]
...
registered = ~np.all(np.isnan(trace), axis=1)
...
bad = ~open_mask[labels]
if np.any(bad):
    ...
    labels[bad] = open_idx[np.argmin(d, axis=1)]
...
n_trials = n_bins // TRIAL_BINS
if n_trials < 2:
    ...
```

iii. In the trajectory, the agent explicitly mentioned NaN-padded unregistered cells and an empirically observed small amount of occupancy leakage into blocked partitions. It justified the snapping as a tracking-noise fix and justified keeping all frames by arguing that dropping low-speed frames would break contiguous trial structure.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each large per-animal joblib file, running `gaussian_filter1d` over every retained neuron for every session, and writing the final very large pickle file. The per-trial appends are comparatively small.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
...
with open(out_file, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. In the trajectory, the agent monitored runtime around the full conversion, noted that the output pickle was about 6.65 GB, and waited several minutes for full conversion/training runs. That behavior is consistent with large I/O plus full-session smoothing and pooling dominating runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit loop over trials could be replaced with reshape-based chunking instead of repeated Python `append`s. The per-session/session-info accumulation also remains Python-loop driven. Most of the numerically heavy work is already vectorized.

ii.
```python
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(labels[sl][None, :].copy())
...
for ai, animal in enumerate(animals):
    ...
    for day in range(n_days):
```

iii. The trajectory does not show the agent optimizing for vectorization. Its focus was correctness and matching the paper’s decoder-style preprocessing, so it left the session/day/trial loops in straightforward Python form.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly copies the same static geometry vector once per trial and repeatedly builds trial lists by slicing each session one trial at a time. It also recomputes open-mask-related structures independently for every session, even when geometries repeat across days.

ii.
```python
open_mask = np.ones(NGRID * NGRID, dtype=bool)
open_mask[blocked] = False
...
geom = np.zeros(NGRID * NGRID, dtype=np.float32)
geom[blocked] = 1.0
...
for t in range(n_trials):
    ...
    input_trials.append(geom.copy())
```

iii. The trajectory shows the agent thinking mainly at the dataset and decoder level, not about removing repeated small computations. The repeated per-trial geometry copying follows directly from that straightforward implementation style.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores `envs`/`env` only for logging and metadata, and it stores large `session_info` entries including `cell_ids` that the downstream decoder does not use. That extra bookkeeping increases output size without affecting decoder inputs or targets.

ii.
```python
envs = [str(e[0]) for e in dat['envs']]
...
env = envs[day]
...
session_info.append({
    'subject': animal, 'day': int(day), 'environment': env,
    'blocked_partitions': blocked.tolist(),
    'n_neurons': int(len(cell_ids)), 'n_trials': int(n_trials),
    'cell_ids': [int(c) for c in cell_ids],
})
...
'session_info': session_info,
```

iii. In the trajectory, the agent used `env` and detailed session metadata to document choices and debug the conversion. It later had to patch the script because `cell_ids` in metadata initially broke JSON serialization during verification, which underscores that this metadata was ancillary to the actual decoder dataset.
