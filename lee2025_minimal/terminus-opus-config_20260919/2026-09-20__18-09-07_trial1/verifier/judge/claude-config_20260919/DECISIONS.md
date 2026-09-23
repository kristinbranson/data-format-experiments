# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the list of the 7 animal IDs and loads one **joblib** file per animal from `/app/data` (the `QLAK-CA1-*` files, i.e. the Python-native copies the paper's own `utils.mat2joblib`/`save_dat` produced from the `.mat` files). Each file is a nested dict keyed by the animal ID, from which it takes only `position`, `trace`, `envs` and `blocked`; `SFPs`, `centroids` and `maps` are never touched. Within an animal it iterates over the day (session) axis of `trace` (`n_days = trace_all.shape[0]`), and within a day it slices the continuous 40-min recording into 1-min trials. All 7 animals × 207 days are processed; the resulting file contains 207 sessions and 8,187 trials.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for ai, animal in enumerate(animals):
    print(f'loading {animal} ...', flush=True)
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    pos_all, trace_all = dat['position'], dat['trace']
    envs = [str(e[0]) for e in dat['envs']]
    n_days = trace_all.shape[0]
    for day in range(n_days):
        ...
    del dat, pos_all, trace_all
```

iii. From the trajectory: the AI read `/app/code/README.md` and `utils.load_dat`, noted that the repo's own loader defaults to `format="joblib"` (`dataset = joblib.load(os.path.join(p_data, f"{animal}"))`) and that the joblib files are the authors' converted copies of the `.mat` files, so it used them directly rather than going through `mat73`/`h5py`. It then verified by exploration that "Data per animal: 31 days, position (days,2,frames), trace (days,cells,frames), envs names, blocked partition indices per day", and confirmed "All 7 animals verified: 207 sessions total (matches paper), 30 Hz, positions in cm (0-75)", i.e. it cross-checked the load against the paper's reported 207 sessions.

## 1-b. How are the data split into subjects?

i. One subject = one animal = one data file. The 7 animal IDs are the entries of `subjects`, and `subject_idx` records, for every session appended, the index of the animal it came from.

ii.
```python
data = { ... 'subjects': list(animals), 'subject_idx': [], ... }

for ai, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    for day in range(n_days):
        ...
        data['subject_idx'].append(ai)
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The README states the data files "are given names of animal IDs from the original study", so the file name is the subject identifier. The AI's docstring records "all 207 sessions from all 7 animals are used". The resulting counts (31, 31, 31, 21, 31, 31, 31 sessions) match the per-animal day counts it verified during exploration.

## 1-c. How are the data split into sessions?

i. One session = one recording day = one 40-min free-exploration recording in one geometry. The AI iterates over the first (day) axis of `trace`/`position`/`envs`/`blocked` and emits one entry of `neural`/`input`/`output`/`subject_idx`/`brain_region_idx` per day. Each session also gets a `session_info` record (subject, day index, environment name, blocked partitions, n_neurons, n_trials, kept cell ids).

ii.
```python
n_days = trace_all.shape[0]
for day in range(n_days):
    env = envs[day]
    blocked = np.array(dat['blocked'][day]).ravel().astype(int)
    ...
    pos = pos_all[day].T.astype(np.float64)   # (n_frames, 2), cm
    trace = trace_all[day]                    # (n_cells, n_frames)
    ...
    session_info.append({
        'subject': animal, 'day': int(day), 'environment': env,
        'blocked_partitions': blocked.tolist(),
        'n_neurons': int(len(cell_ids)), 'n_trials': int(n_trials),
        'cell_ids': [int(c) for c in cell_ids],
    })
```

iii. Methods: "All sessions were 40 min, and one session was recorded per day", and one geometry is recorded per day. The AI's docstring states "one session = one recording day (40 min of free exploration)". Because cell registration (which cells are non-NaN) and the environment geometry both change day to day, a day is the natural session unit; the AI confirmed the day count summed to the paper's 207 sessions.

## 1-d. How are the data split into trials?

i. Trials are consecutive, non-overlapping 60-s segments of the continuous recording, cut **after** temporal rebinning: 60 s / 100 ms = 600 bins per trial. The trailing incomplete segment is dropped. Neural, input and output are cut with the identical slice, so all three streams share one time base. This yields 39–40 trials per session (8,187 trials total).

ii.
```python
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

iii. The instructions state "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session." There are no task events in a free-exploration paradigm, so the AI's docstring records "trials = consecutive, non-overlapping 1 min segments of each session (600 bins of 100 ms); the trailing incomplete segment is dropped."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied: every complete 1-min segment is kept. Two session-level guards exist — a session is skipped if no cell survives the neuron criteria, or if it yields fewer than 2 complete trials (the format requires ≥2 trials per session). Neither guard fires on this dataset (all 207 sessions are kept, min 39 trials). The only data discarded at the trial level is the trailing partial segment. The AI explicitly decided **not** to apply the paper's >5 cm/s velocity filter used in its Bayesian decoding analysis.

ii.
```python
if tr.shape[0] == 0:
    print(f'  skipping {animal} day {day}: no cells pass criteria')
    continue
...
n_trials = n_bins // TRIAL_BINS
if n_trials < 2:
    print(f'  skipping {animal} day {day}: < 2 full trials')
    continue
```

iii. From the module docstring: "the paper's within-session Bayesian decoding additionally discards frames in which the animal moves slower than 5 cm/s. That filter is NOT applied here: it would remove ~50% of the frames scattered through the recording and make it impossible to cut the session into contiguous 1-min trials with a common time base for neural, input and output streams (as required here). The animal still occupies a well defined partition while immobile, so every time bin carries a valid position label." The trajectory shows the AI had measured the "fraction of frames passing 5 cm/s filter" before making this call, i.e. the deviation was quantified rather than assumed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes solely from the `trace` field: the binarized rising phase of calcium transients (1 = significant event), shape `(n_days, n_cells, n_frames)` at 30 Hz, with all-NaN entries for cells not registered on a given day.

ii.
```python
pos_all, trace_all = dat['position'], dat['trace']
...
trace = trace_all[day]                            # (n_cells, n_frames)
```

iii. The AI's docstring quotes the repo README: "'trace': (n_days, n_cells, n_frames) binarized rising phase of calcium transients at 30 Hz ('1' = significant event). Cells that were not registered on a given day are NaN for that day." Methods: "This binary vector was treated as the firing rate in all further analyses." So `trace` is the paper's firing-rate variable and no other field carries neural activity (`maps` are derived rate maps, `SFPs`/`centroids` are anatomy).

## 2-b. How is the `neural` data processed?

i. After neuron selection (2-c), the binary trace is cast to float32, smoothed along time with a Gaussian kernel of σ = 3 frames, then average-pooled over non-overlapping 3-frame windows, giving continuous-valued "firing rates" in 100 ms bins, stored as float32 with shape `(n_neurons, 600)` per trial. This reproduces exactly the preprocessing inside the paper's own decoder (`utils.fit_decoder` / `utils.test_decoder`: `AvgPool1d(kernel_size=3, stride=3)` applied to `gaussian_filter1d(traces, sigma=3, axis=0)`). No z-scoring, normalization or deconvolution is applied on top.

ii.
```python
def pool_mean(x, k):
    """Average pool along the last axis with kernel = stride = k (drops remainder)."""
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(x.shape[:-1] + (n // k, k)).mean(axis=-1)
...
# temporal processing identical to the paper's decoder:
# gaussian smoothing (sigma = 3 frames) then 3-frame average pooling
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, n_bins)
```

iii. Docstring: "neural data = the binary transient-rising-phase vector used as 'firing rate' throughout the paper, smoothed and temporally binned exactly as in the paper's own position-decoding code (utils.fit_decoder / utils.test_decoder): gaussian_filter1d(sigma = 3 frames) followed by average pooling over 3 frames -> 100 ms time bins." Trajectory step 9: "decoding used 3-frame temporal binning (100 ms), gaussian smoothing sigma=3 frames on traces". The AI's stated principle was that since the downstream task is exactly the paper's position-decoding task, the paper's decoder-side preprocessing is the processing to match.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two criteria, applied per session: (1) cells not registered that day (all-NaN row) are dropped; (2) of the remaining cells, only those with more than 5 transients in the session are kept — the `cell_threshold=5` activity-sparsity criterion of `utils.decode_position_within`. The surviving indices are recorded in `session_info['cell_ids']` and `brain_region_idx` is sized to match. The paper evaluates that count over movement frames only; because the AI dropped the velocity filter, it evaluates it over the whole session instead. Across the dataset this criterion removes only ~0.4% of registered cells; 69,632 neuron-sessions remain.

ii.
```python
# keep cells registered on this day with > 5 transients in the session
registered = ~np.all(np.isnan(trace), axis=1)
cell_ids = np.where(registered)[0]
tr = trace[cell_ids]
enough = np.nansum(tr, axis=1) > CELL_EVENT_THRESHOLD
cell_ids, tr = cell_ids[enough], tr[enough]
if tr.shape[0] == 0:
    print(f'  skipping {animal} day {day}: no cells pass criteria')
    continue
```

iii. Docstring: "neurons: only cells registered on that day (non-NaN) and with more than 5 transients in the session are kept (activity-sparsity criterion of utils.decode_position_within, cell_threshold=5)" and "Correspondingly the cell activity criterion (> 5 transients) is evaluated over the whole session rather than over the moving frames only." The NaN criterion follows the README ("If cell is not registered on given day, will appear as nan"); the trajectory confirms the AI verified NaN is all-or-none per cell per day, so `np.all(...)` is a safe test.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no behavioural or stimulus event to align to — the paradigm is 40 min of continuous free exploration. Trials are therefore aligned to an arbitrary cut point: the start of each 1-min segment, taken from the start of the recording. This is recorded in the metadata as `temporal_alignment_event`, with `off_start = 0.0` s and `off_end = 60.0` s relative to that cut.

ii.
```python
'temporal_alignment_event': (
    'start of each 1-min trial, i.e. arbitrary segmentation of the '
    'continuous 40-min free-exploration session (no task events)'),
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. Docstring and metadata make the reasoning explicit: the session is a continuous free-exploration recording with "no task events", so the only meaningful alignment is the segmentation boundary itself, and the trial window is the full [0, 60) s of each segment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning is applied. The native acquisition rate of both the imaging and behaviour streams is 30 Hz (33.3 ms). The AI rebins by averaging groups of 3 frames, giving a uniform 100 ms bin for every trial and every session (600 bins per 1-min trial), and reports `time_bin_size = 100.0` ms in the metadata. The same pooling is applied to the position stream so the two remain bin-for-bin aligned.

ii.
```python
FPS = 30.0              # acquisition rate of both imaging and behavior streams
TEMPORAL_BIN = 3        # frames per time bin (paper's decoder: temporal_bin_size=3)
BIN_MS = 1000.0 * TEMPORAL_BIN / FPS   # = 100 ms
TRIAL_BINS = int(round(TRIAL_SEC * 1000.0 / BIN_MS))   # 600 bins per trial
...
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T
...
'time_bin_size': BIN_MS,
```

iii. The AI chose the paper's own decoder bin size: `fit_decoder(..., temporal_bin_size=3)` and `AvgPool1d(kernel_size=3, stride=3)` in `utils.py`. Its docstring says the smoothing and pooling are "exactly as in the paper's own position-decoding code". A side benefit noted in the trajectory was the 3× reduction in dataset size (6.65 GB rather than ~20 GB at native resolution).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From the `blocked` field: a per-day list of the indices of the occluded partitions in the 3×3 grid, indexed `[[0,1,2],[3,4,5],[6,7,8]]`, with the sentinel value `-1` meaning nothing is blocked (full square). The `envs` field (the geometry's string name, e.g. "square", "o", "l") is loaded too but only stored as descriptive `session_info` metadata, not used as a decoder input.

ii.
```python
envs = [str(e[0]) for e in dat['envs']]
...
blocked = np.array(dat['blocked'][day]).ravel().astype(int)
blocked = blocked[blocked >= 0]
```

iii. Docstring, quoting the README: "'blocked': list (n_days) of the indices of the blocked (occluded) partitions in the 3 x 3 grid, indexed [[0, 1, 2], [3, 4, 5], [6, 7, 8]]; -1 if nothing is blocked (full square)." This is the raw variable that literally encodes "which parts of the arena are blocked", which is what the instructions ask for as the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are turned into a 9-dimensional binary (float32) vector: 1 where the partition is occluded, 0 where accessible. The `-1` sentinel is filtered out first, so a full square yields the all-zero vector. The vector is constant within a session and is stored once per trial as a static `(9,)` input (no time axis), with `input_names = ['blocked_partition_0' ... 'blocked_partition_8']` in the same 3×3 indexing as the output labels.

ii.
```python
blocked = np.array(dat['blocked'][day]).ravel().astype(int)
blocked = blocked[blocked >= 0]
open_mask = np.ones(NGRID * NGRID, dtype=bool)
open_mask[blocked] = False
...
geom = np.zeros(NGRID * NGRID, dtype=np.float32)
geom[blocked] = 1.0
...
input_trials.append(geom.copy())
...
'input_names': [f'blocked_partition_{i}' for i in range(NGRID * NGRID)],
```

iii. Docstring: "input = the static (per-trial) environment geometry: 9 binary values, 1 if that partition is blocked in the current geometry", with metadata adding "Indexing [[0, 1, 2], [3, 4, 5], [6, 7, 8]] as in the dataset's 'blocked' field". The instructions specify the geometry input is "Static per-trial", and a 9-bit occupancy mask is a complete, lossless encoding of all 10 geometries that shares its index space with the output labels.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. From the `position` field: DeepLabCut head-tracking x–y coordinates in cm, `(n_days, 2, n_frames)`, acquired simultaneously with the imaging at 30 Hz. The AI verified the coordinates span 0–75 cm, matching the arena.

ii.
```python
pos_all, trace_all = dat['position'], dat['trace']
...
pos = pos_all[day].T.astype(np.float64)          # (n_frames, 2), cm
```

iii. Docstring: "'position': (n_days, 2, n_frames) x-y position (cm) from DeepLabCut tracking, simultaneously acquired with the imaging at 30 Hz", consistent with the Methods ("Position data were generated from tracking the head with DeepLabCut"). Trajectory: "positions in cm (0-75)" verified across all 7 animals.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The (x, y) trace is transposed to `(n_frames, 2)`, average-pooled over the same 3-frame windows as the neural data (so position is the mean position within each 100 ms bin), then discretized (4-c). The result is stored as an int64 array of shape `(1, 600)` per trial, i.e. a single time-varying categorical output named `position_bin` with 9 possible values.

ii.
```python
pos = pos_all[day].T.astype(np.float64)          # (n_frames, 2), cm
...
pos_b = pool_mean(pos.T, TEMPORAL_BIN).T                    # (n_bins, 2)
labels = partition_labels(pos_b, open_mask).astype(np.int64)
...
output_trials.append(labels[sl][None, :].copy())
...
'output_names': ['position_bin'],
'output_values': [[f'partition_{i}' for i in range(NGRID * NGRID)]],
```

iii. Pooling position with the identical kernel is what keeps the behavioural and neural streams on the same time base, mirroring `fit_decoder`, which pools `behav` and `traces` with the same `AvgPool1d`. The AI pools the *continuous* coordinates and then discretizes (as the paper does: it pools the binned-down continuous position and casts afterwards), rather than discretizing first and taking a mode.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 75 × 75 cm arena is cut into the 3 × 3 grid of 25 cm partitions that defines the experimental geometries. `x_bin = floor(x / 25)`, `y_bin = floor(y / 25)`, both clipped to [0, 2] to catch samples exactly at the 75 cm edge, and the class label is `3 * y_bin + x_bin` — the same indexing as the `blocked` field. One extra step: samples that land inside a *blocked* partition (tracking noise near the partition walls) are relabelled to the nearest accessible partition, by Euclidean distance from the sample to the accessible partition centres. Measured on the raw data this affects ~0.01% of samples overall (<1% in the worst single session).

ii.
```python
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

iii. The instructions require "Mouse position discretized into 3 x 3 = 9 spatial bins", and the 3×3 partition grid is also the paper's own experimental partition scheme, so the bins are the physically meaningful ones. The AI did not assume the label convention: the docstring says the indexing was "verified against the occupancy of the blocked partitions in every session", and the trajectory records "Mapping determined: 3x3 partition index = 3*y_bin + x_bin with bins = floor(pos/25), verified against `blocked` for several sessions (zero occupancy exactly in blocked partitions)" across all 207 sessions. The snapping is justified in the docstring as mirroring the paper's decoder, which snaps both actual and predicted positions to the visitable bins of the rate map (`true_bins` cleaning in `decode_position_within`).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame. The DAQ acquired the behavioural and imaging streams simultaneously at 30 Hz, and in the data file `position` and `trace` have identical frame counts for every day, so index *t* means the same instant in both. The AI applies the same 3-frame pooling to both, then cuts both with the same trial slice. A defensive `min()` guards against any length mismatch before the trial count is computed.

ii.
```python
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, n_bins)
pos_b  = pool_mean(pos.T, TEMPORAL_BIN).T                   # (n_bins, 2)
labels = partition_labels(pos_b, open_mask).astype(np.int64)

n_bins = min(neural.shape[1], labels.shape[0])
n_trials = n_bins // TRIAL_BINS
...
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    output_trials.append(labels[sl][None, :].copy())
```

iii. Methods: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz as uncompressed AVI files and all recorded frames were timestamped for post-hoc alignment", so the delivered arrays are already aligned and no resampling or lag correction is needed. The AI's exploration confirmed equal frame counts per day. No lag is introduced because `pool_mean` uses the same kernel, stride and remainder-dropping rule for both streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled:
- **Unregistered cells** (all-NaN rows on a given day) are dropped, so no NaNs reach the output and `brain_region_idx` length matches the kept neuron count.
- **Residual NaNs** in the transient count are tolerated by using `np.nansum` rather than `np.sum`.
- **Tracking noise placing the animal inside a blocked partition** is corrected by snapping to the nearest accessible partition (4-c).
- **Frames that do not fill a whole bin/trial** are dropped: `pool_mean` truncates the remainder before reshaping, and `n_bins // TRIAL_BINS` discards the trailing partial trial (≤60 s of a 40-min session).

Degenerate sessions (no surviving cells, or fewer than 2 complete trials) are skipped with a printed message; neither occurs in this dataset. Verification against the raw files confirms there are no NaNs in `position` and no partially-NaN cells, so these guards are conservative rather than load-bearing.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=1)
cell_ids = np.where(registered)[0]
tr = trace[cell_ids]
enough = np.nansum(tr, axis=1) > CELL_EVENT_THRESHOLD
...
def pool_mean(x, k):
    n = (x.shape[-1] // k) * k          # remainder frames dropped
    return x[..., :n].reshape(x.shape[:-1] + (n // k, k)).mean(axis=-1)
...
n_bins = min(neural.shape[1], labels.shape[0])   # guard against length mismatch
n_trials = n_bins // TRIAL_BINS                  # trailing partial trial dropped
```

iii. The README states cells unregistered on a day "will appear as nan the same shape", which the AI verified is all-or-none per cell per day, making `np.all(np.isnan(...), axis=1)` the right test. Dropping the remainder frames is the standard cost of fixed-length trials and is negligible (<1/40 of each session). The snapping rationale is given in `partition_labels`: "Rare samples that fall inside a blocked partition (tracking noise near the partition walls, <1% of samples in a single session) are snapped to the nearest accessible partition, as the paper's decoding code snaps actual and predicted positions to the visitable bins of the rate map."

## 6-a. What are the most time-consuming steps of the code?

i. Measured on this data, the ranking is:
1. **`joblib.load` of each animal file** — the dominant cost (~6.5 s for a 71 MB compressed file that expands to a 6.7 GB float64 `trace` array). Decompression plus the float64 materialization dominates; it is also wasteful because the load pulls in `SFPs`, `centroids` and `maps`, which the script never uses.
2. **Writing the 6.65 GB output pickle** in one `pickle.dump`.
3. **Building the per-trial copies** (`np.ascontiguousarray` / `.copy()` for 8,187 trials), which materialize a second full copy of the neural data and grow the in-memory `data` dict to the full 6.65 GB before the dump.
4. `gaussian_filter1d` + `pool_mean` — comparatively cheap (~0.06 s per session, ~13 s total).

The code contains no explicit optimization for any of these (no per-animal incremental writing, no float16/sparse storage, no selective field loading).

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]   # dominant cost
...
tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
...
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
...
with open(out_file, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The AI never profiled the code, but it did anticipate the I/O scale: trajectory step 26 records "Conversion running quickly, ~39 trials/session, hundreds of cells. Estimated total ~6.4 GB pickle", and it added `del dat, pos_all, trace_all` at the end of each animal's loop specifically to release the multi-GB per-animal arrays before loading the next one. Choosing 100 ms bins also cut both the compute and the output size by 3×. Overall runtime was a few minutes, so no further optimization was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two, both minor:
- The **per-trial loop** (`for t in range(n_trials)`) slices and copies each trial individually. Since all trials are equal-length contiguous blocks, this could be a single reshape/`np.split` of the truncated session array, avoiding 8,187 separate `ascontiguousarray`/`copy` calls and the transient double-memory.
- The **per-day loop** applies `gaussian_filter1d` one day at a time; since all days of an animal share a frame count, the smoothing could be done on the whole `(n_days, n_cells, n_frames)` block at once (though this would multiply peak memory and the per-day cell masks differ, so the gain is small).

The genuinely hot pieces are already vectorized: `pool_mean` is a reshape+mean, and the snapping in `partition_labels` uses a single broadcast distance computation rather than a Python loop over bad samples.

ii.
```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(labels[sl][None, :].copy())
```

iii. The AI gives no explicit justification; the loops it wrote are per-session/per-trial container construction, which the target format requires as a list of per-trial arrays anyway, so the loop bodies are copies rather than computation. The AI did deliberately vectorize the one place where a naive implementation would have looped (the out-of-geometry snapping).

## 6-c. What processing does the code repeat multiple times?

i. Small repeats only:
- `geom.copy()` is called once per trial, producing 8,187 identical 9-element copies of a per-session constant (one array per session would suffice, or `input_trials = [geom] * n_trials`).
- `pool_mean` is invoked twice per session (neural and position); this is necessary, not redundant, but it repeats the same truncate/reshape/mean logic on two streams.
- The blocked-partition information is derived twice per session, once as `open_mask` (for label snapping) and once as `geom` (for the decoder input), from the same `blocked` array.
- The whole conversion was run twice end-to-end during the session (trajectory steps 26 and 30–34) because the first pass stored `cell_ids` as an ndarray, which broke JSON serialization of the metadata; the script was patched to `[int(c) for c in cell_ids]` and re-run. That is wasted wall-clock, not repeated logic in the final script.

ii.
```python
open_mask = np.ones(NGRID * NGRID, dtype=bool)
open_mask[blocked] = False
...
geom = np.zeros(NGRID * NGRID, dtype=np.float32)
geom[blocked] = 1.0
...
for t in range(n_trials):
    input_trials.append(geom.copy())
```

iii. No justification is given in the code or trajectory. Copying `geom` per trial is defensive (it prevents a downstream consumer from mutating one trial's input and silently changing all of them) and costs 9 floats per trial, so the repetition is deliberate-looking and negligible.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, all small relative to the main pipeline:
- **`cell_ids` stored in `session_info`** — the full list of source cell indices for all 69,632 kept neuron-sessions is serialized into the metadata and never used by the decoder. (It was also the cause of the failed first run.)
- **Loading unused fields**: `joblib.load` reads the entire per-animal dict including `SFPs`, `centroids` and `maps`, of which only `trace`, `position`, `envs` and `blocked` are touched.
- **`envs`** is parsed and stored per session but is not a decoder input (the geometry is already fully encoded by the 9-bit `blocked` vector), so it is descriptive only.
- **Smoothing and pooling the trailing frames** that are later discarded when the last partial trial is dropped.
- **`n_bins = min(neural.shape[1], labels.shape[0])`** — a guard that can never bind, since both streams have identical frame counts and are pooled identically.
- **Redundant copies**: `np.ascontiguousarray` on slices of a freshly created C-contiguous array, plus the `geom.copy()` per trial.
- **Storing a near-binary, very sparse signal as dense float32** (mean value ≈0.005): the output is 6.65 GB where a sparse or float16 representation would be far smaller.

ii.
```python
session_info.append({
    'subject': animal, 'day': int(day), 'environment': env,
    'blocked_partitions': blocked.tolist(),
    'n_neurons': int(len(cell_ids)), 'n_trials': int(n_trials),
    'cell_ids': [int(c) for c in cell_ids],     # never used downstream
})
...
n_bins = min(neural.shape[1], labels.shape[0])  # guard that never binds
...
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. The AI's stated intent for `session_info` is provenance — the metadata block is documentation-heavy by design (`task_description`, `neural_data_description`, `input_description`, `output_description`, `velocity_filter`, `reference`), recording every processing choice alongside the data. `cell_ids` was explicitly converted to plain ints after the first run failed: "the stats JSON dump failed because I stored `cell_ids` as ndarray inside metadata... metadata must be JSON-serializable" (trajectory step 30). The remaining items are unexamined incidental costs rather than deliberate decisions.
