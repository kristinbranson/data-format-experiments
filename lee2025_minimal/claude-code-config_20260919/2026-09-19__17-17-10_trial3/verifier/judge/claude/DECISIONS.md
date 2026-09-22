# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the **joblib** version of the dataset (the extension-less per-animal files in `/app/data`, e.g. `/app/data/QLAK-CA1-51`), not the `.mat` files. It hard-codes the list of the 7 animal IDs and loads one file per animal with `joblib.load(...)`, then indexes the returned dict by the animal name to get the per-animal record. From that record it uses four fields: `trace` (n_days, n_cells, n_frames), `position` (n_days, 2, n_frames), `envs` (n_days,) and `blocked` (per-day list). Days (recording sessions) are iterated over inside each animal, and each day is then cut into 1-min trials. All 7 animals × all days = 207 sessions are loaded; the fields `SFPs`, `centroids` and `maps` are loaded into memory by `joblib.load` but never used.

ii.
```python
DATA_DIR = "/app/data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for subject, animal in enumerate(animals):
    print(f"Loading {animal} ...", flush=True)
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace, position = dat["trace"], dat["position"]
    envs = [str(e) for e in np.asarray(dat["envs"]).ravel()]
    blocked_all = dat["blocked"]
    n_days = trace.shape[0]
    ...
    days = range(n_days) if max_sessions is None else range(min(n_days, max_sessions))
    for day in days:
        ...
    del dat, trace, position
```

iii. From the script docstring and trajectory: *"Per-animal joblib files in /app/data (identical content to the .mat files, produced by the paper's own `mat2joblib`)."* The repository's own loader (`utils.load_dat`) defaults to `format="joblib"`, so the AI followed the paper's own access path rather than re-reading the HDF5 `.mat` files. It confirmed the field names/shapes and the `-1` "nothing blocked" code empirically (trajectory steps 21-23) against the repo README before writing the script. `del dat, trace, position` at the end of each animal is an explicit memory-management step, since one animal's `trace` is up to ~7 GB of float64.

## 1-b. How are the data split into subjects?

i. One subject per data file / per animal ID. The AI enumerates the hard-coded `ANIMALS` list; `subject` (the loop index) is appended to `subject_idx` once per session, and `subjects` in the output dict is exactly that list of 7 animal ID strings. The animal ID is also duplicated into each session's `session_info` entry.

ii.
```python
for subject, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    ...
    for day in days:
        ...
        subject_idx.append(subject)
        session_info.append({"subject": animal, "day": int(day), ...})
...
"subjects": list(animals),
"subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. The README states each file is named after an animal ID from the original study and holds all of that animal's registered cells and days, so file ↔ mouse is one-to-one. The AI verified the full set of 7 animals and their day counts (31/31/31/21/31/31/31 = 207) matches the "207 sessions" reported in the methods (trajectory step 34).

## 1-c. How are the data split into sessions?

i. One output session per **recording day** per animal (`trace.shape[0]` days per animal). No days are excluded, giving 207 sessions. Each session carries its own neuron set, its own geometry (`envs[day]`, `blocked[day]`), and its own list of trials.

ii.
```python
n_days = trace.shape[0]
days = range(n_days) if max_sessions is None else range(min(n_days, max_sessions))
for day in days:
    blocked = blocked_indices(blocked_all[day])
    ...
    neural_all.append(neural_trials)
    input_all.append(input_trials)
    output_all.append(output_trials)
    subject_idx.append(subject)
    brain_region_idx.append(np.zeros(rates.shape[0], dtype=np.int64))
    session_info.append({"subject": animal, "day": int(day),
                         "environment": envs[day], ...})
```

iii. "All sessions were 40 min, and one session was recorded per day"; the geometry changes between days, so a day is the natural session unit and the unit that a decoder input (geometry) is constant over. The AI's stated curation is *"All 7 mice, all 207 sessions and all 10 geometries are kept"*, matching the 207 sessions in the methods text.

## 1-d. How are the data split into trials?

i. Each continuous 40-min session is cut into consecutive, non-overlapping 1-minute trials, as the task instructions require. Because the AI temporally rebins to 100 ms (3 frames at 30 Hz), a trial is `BINS_PER_TRIAL = 60 s × 30 Hz / 3 = 600` bins. A trailing partial minute is dropped so every trial has exactly 600 bins. This yields 39-40 trials per session and 8,187 trials in total.

ii.
```python
FPS = 30
FRAMES_PER_BIN = 3
TRIAL_SECONDS = 60.0
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FPS / FRAMES_PER_BIN))   # 600
...
n_bins = min(rates.shape[1], part.shape[0])
n_trials = n_bins // BINS_PER_TRIAL     # drop a trailing partial minute

neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. *"Each session is a single continuous 40 min foraging session with no trial structure, so it is cut into consecutive 1 min trials (600 bins of 100 ms), as instructed. A trailing partial minute is dropped so that every trial has the same length."* The format requires equal-length time bins across trials and sessions, hence the uniform 600-bin trial and the dropped remainder (≤ 59 s of a 40-min session).

## 1-e. How are trials filtered based on quality controls?

i. **No trials are filtered.** All 39-40 trials of every one of the 207 sessions are kept. The one quality control the AI considered and explicitly rejected is the paper's decoder running-speed criterion (keep only frames > 5 cm/s); it kept all time points instead. It also deliberately applies no place-cell / spatial-reliability session or trial selection.

ii.
```python
# (no trial-level filter exists in the code; every t in range(n_trials) is emitted)
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. From the docstring: *"The paper's Bayesian decoder additionally restricts fitting and testing to frames faster than 5 cm/s. That criterion belongs to their decoding analysis rather than to the preparation of the dataset, and applying it here would delete ~48% of the recording and leave ragged, non-contiguous 1 min trials, so all time points are kept and the choice of whether to condition on running is left to the analysis."* The AI measured the moving fraction per animal (0.45-0.62, mean ≈ 0.52; trajectory step 40) and empirically A/B-tested a velocity-filtered variant on a 6-session subset (trajectory step 62: 0.402 balanced accuracy with velocity filtering, vs. 0.361 without on the same subset), i.e. it knew the filter would help accuracy but rejected it because it produces ragged, unequal-length trials that violate the target format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely the `trace` field: the binarised rising phase of calcium transients, shape (n_days, n_cells, n_frames), with NaN for a cell not registered on that day. No other neural field (`SFPs`, `centroids`, `maps`) contributes.

ii.
```python
trace, position = dat["trace"], dat["position"]
...
tr = trace[day]
```

iii. *"trace : (n_days, n_cells, n_frames) binarised rising phase of calcium transients, NaN for a cell that was not registered on that day. The paper treats this binary vector as the firing rate in every analysis."* The methods text states "This binary vector was treated as the firing rate in all further analyses", so `trace` is the paper's firing-rate surrogate. The AI verified `np.unique(trace)` is `{0, 1}` (trajectory step 23).

## 2-b. How is the `neural` data processed?

i. Three steps, applied per session after neuron selection:
1. `gaussian_filter1d` along the time axis with `sigma = 3` frames;
2. non-overlapping average pooling over 3 frames (equivalent to `torch.nn.AvgPool1d(kernel_size=3, stride=3)`), dropping the remainder;
3. multiplication by `FPS = 30` so the stored quantity is an event rate in Hz; cast to `float32`.

The result is an (n_neurons, n_100ms_bins) array which is then sliced into trials. Steps 1-2 replicate `utils.fit_decoder`/`utils.test_decoder`; step 3 replicates the `* fps` in `utils.get_rate_maps`.

ii.
```python
SMOOTH_SIGMA_FRAMES = FRAMES_PER_BIN   # gaussian smoothing of traces, as in fit_decoder

def pool_mean(x, factor):
    """Average-pool the last axis of x in non-overlapping windows, dropping the remainder.
    Equivalent to torch.nn.AvgPool1d(kernel_size=factor, stride=factor) ..."""
    n = (x.shape[-1] // factor) * factor
    return x[..., :n].reshape(*x.shape[:-1], n // factor, factor).mean(axis=-1)
...
# smooth along time, then average pool into 100 ms bins, and express in Hz
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
```

iii. *"It is smoothed and temporally binned exactly as in the paper's own position decoder (`utils.fit_decoder`/`test_decoder`): gaussian_filter1d along time with sigma = 3 frames, then average pooling over 3 frames... The pooled value is multiplied by the frame rate so that the stored quantity is an event rate in Hz, the same units the paper uses for its rate maps (`utils.get_rate_maps`)."* This is accurate: `fit_decoder` does `AvgPool1d(kernel_size=temporal_bin_size=3)` on `gaussian_filter1d(traces, sigma=3, axis=0)`, and `get_rate_maps` multiplies by `fps`. The AI also empirically checked that the overall scale factor does not matter to the provided decoder (trajectory step 60: balanced accuracy 0.358/0.361/0.367 for scales 0.0333/1/10), i.e. the `* FPS` is for interpretability, not performance.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two neuron-level filters, applied per session, in order:
1. **Registration**: cells not registered on that day are all-NaN and are dropped (`~np.isnan(tr[:, 0])`, valid because the AI verified NaN is all-or-none per cell-day). This keeps exactly 69,744 cell-sessions, matching the paper's reported count.
2. **Sparsity**: cells with ≤ 5 events in the session are dropped — the `cell_threshold = 5` of `utils.decode_position_within`. This removes 112 cell-sessions (0.16%), leaving 69,632 neuron-sessions (113-564 per session, mean 336).

No place-cell / split-half-reliability selection is applied. Note the threshold is evaluated over **all** frames, whereas the paper evaluates it over velocity-filtered (moving) frames only.

ii.
```python
CELL_EVENT_THRESHOLD = 5     # cell_threshold in decode_position_within
...
tr = trace[day]
registered = ~np.isnan(tr[:, 0])          # a cell is NaN for the whole day or not
tr = tr[registered]
n_registered = int(registered.sum())
# sparsity criterion of the paper's decoder: drop near-silent cells
active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
tr = tr[active]
```

iii. *"Within a session only cells registered on that day are kept (an unregistered cell is NaN, and NaNs are not allowed by the format); across the dataset that is the 69,744 cell-sessions the paper reports. Cells with 5 or fewer transients in the session are then dropped, which is the `cell_threshold = 5` sparsity criterion of the paper's decoder (~1% of cells). No place-cell/spatial-reliability selection is applied, matching the paper's decoding analysis, which uses all sufficiently active cells."* The AI verified empirically that the NaN pattern is all-or-none per cell-day (`nan_any == nan_all` is True for every animal) and that the registered totals sum to exactly 69,744 (trajectory step 40).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external alignment event. Neural and behavioural streams are frame-matched in the released arrays (same DAQ, same 30 Hz clock), so no resampling or shifting is applied; both streams are pooled over the *same* 3-frame windows and sliced by the same trial index. The only "alignment event" is the start of each 1-min trial, measured from recording onset; this is recorded in the metadata with `off_start = 0.0`, `off_end = 60.0`.

ii.
```python
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
...
pos = pool_mean(position[day], FRAMES_PER_BIN)          # (2, n_time_bins)
...
n_bins = min(rates.shape[1], part.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    output_trials.append(part[sl].astype(np.int64)[None, :])
...
"temporal_alignment_event":
    "start of each 1-min trial; the session is a single continuous recording with "
    "no trial structure, cut into consecutive 1-min segments from recording onset",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. *"The two streams are acquired by the same DAQ at 30 Hz and are already frame-aligned in the released arrays (trace and position have identical frame counts), so position is pooled over the same 3-frame windows as the neural data and no further alignment is needed."* This matches the methods text ("The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz... all recorded frames were timestamped for post-hoc alignment") and the repo, where `fit_decoder` feeds the two streams in as index-matched arrays.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes, rebinning is applied.** The native 30 Hz (33.33 ms) data is smoothed and average-pooled by a factor of 3, giving a uniform **100 ms** time bin (`time_bin_size: 100.0` in the metadata). Every trial is therefore exactly 600 bins (verified: T mean/min/max = 600). Position is pooled with the same factor over the same windows, so neural and behavioural bins coincide.

ii.
```python
FPS = 30                     # miniscope and behaviour camera acquisition rate (Hz)
FRAMES_PER_BIN = 3           # paper's temporal_bin_size in utils.fit_decoder -> 100 ms bins
TIME_BIN_MS = 1000.0 * FRAMES_PER_BIN / FPS
...
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
pos = pool_mean(position[day], FRAMES_PER_BIN)
...
"time_bin_size": float(TIME_BIN_MS),
```

iii. *"...exactly as in the paper's own position decoder (`utils.fit_decoder`/`test_decoder`)... At the 30 Hz acquisition rate of both the miniscope and the behavioural camera this gives 100 ms bins."* The paper's own within-session Bayesian position decoder uses `temporal_bin_size=3`, so 100 ms is the temporal resolution the source analysis decodes position at; the AI adopted it for the same task.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field of the per-animal record: a per-day list of the indices of the occluded partitions in the 3×3 layout `[[0,1,2],[3,4,5],[6,7,8]]`, or `-1` when nothing is blocked. `envs` (the geometry's string name) is loaded but used only for metadata/logging, **not** to build the input; `utils.get_env_mat(env_name)` is deliberately not used.

ii.
```python
def blocked_indices(blocked_entry):
    """Indices of the occluded partitions of one session, as an integer array (empty for none)."""
    vals = np.atleast_1d(np.asarray(blocked_entry[0], dtype=float)).ravel()
    vals = vals[vals >= 0]           # -1 codes "no partition blocked" (the full square)
    return vals.astype(int)
...
blocked = blocked_indices(blocked_all[day])
```

iii. *"`blocked` is used rather than `utils.get_env_mat(env_name)` because several geometries were run in a vertically mirrored version for some animals, and `blocked` records the configuration actually used."* This is supported by the repo: `utils.get_environment_label` has a `flipud` branch for `u`, `l`, `bit donut` and `glenn`, whereas `get_env_mat` returns a single canonical matrix per name. The `-1` sentinel is documented in the README and the AI confirmed it in the data (trajectory step 23).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted to a 9-dimensional binary (float32) vector, 1 = blocked, 0 = open. The vector is constant within a session and is replicated (as an independent copy) for every trial of that session, giving a static `(9,)` input per trial. Input names are `blocked_x0y0 ... blocked_x2y2`, i.e. index = 3·y + x.

ii.
```python
# ---- input: geometry of the environment ---------------------------------
geom = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
geom[blocked] = 1.0
...
    input_trials.append(geom.copy())
...
"input_names": [f"blocked_{n}" for n in partition_names()],
```
with
```python
def partition_names():
    """Human-readable name of each of the 9 spatial bins (x = column, y = row)."""
    return [f"x{i % N_SPATIAL_BINS}y{i // N_SPATIAL_BINS}"
            for i in range(N_SPATIAL_BINS ** 2)]
```

iii. The instructions state the decoder input is "Environment geometry, representing which parts of the arena are blocked. Static per-trial." A 9-d binary indicator is the direct encoding of the paper's 3×3 partition design and lets the decoder treat each partition independently rather than learning an arbitrary geometry-name code. The AI verified empirically that the `blocked` index convention is `3·y + x` by checking that occupancy is exactly zero in the partitions listed as blocked (e.g. for `rectangle`, blocked `[0,3,6]`, the entire x = 0 column has zero occupancy; trajectory steps 36 and 46).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field: (n_days, 2, n_frames) head position in cm from DeepLabCut, spanning 0-75 cm in both x and y and already expressed in a common frame across days.

ii.
```python
trace, position = dat["trace"], dat["position"]
...
pos = pool_mean(position[day], FRAMES_PER_BIN)          # (2, n_time_bins)
```

iii. *"position : (n_days, 2, n_frames) head position from DeepLabCut, in cm, 0-75 cm in both x and y, already in a common frame across days."* The AI verified the range (min 0.0, max 75.0) and that there are **no** NaNs anywhere in `position` for any animal (trajectory steps 23, 40).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2-D position is average-pooled over the same non-overlapping 3-frame windows as the neural data (so one position value per 100 ms bin), then discretised (see 4-c). The pooling is the numpy equivalent of the `AvgPool1d(kernel_size=3, stride=3)` the paper applies to `behav` in `fit_decoder`/`test_decoder`.

ii.
```python
pos = pool_mean(position[day], FRAMES_PER_BIN)          # (2, n_time_bins)
```

iii. Same justification as 2-b/2-e: the paper's position decoder temporally bins the behavioural stream with exactly this pooling, and using the identical windows for neural and behavioural data keeps the two streams aligned bin-for-bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Three steps:
1. **Bin size**: `bin_size = (np.nanmax(position) + 1e-15) / 3`, i.e. the maximum position over **all days of that animal**, divided by 3 — the rule of `utils.decode_position_within` / `utils.get_rate_maps` with `n_bins = 3` instead of 15. This evaluates to ≈ 25 cm, so one bin = one 25 cm partition.
2. **Discretise**: integer floor-division of x and y by `bin_size`, clipped to `[0, 2]`, combined as `part = 3·y_bin + x_bin` → 9 classes.
3. **Clean-up**: samples that land inside a *blocked* partition (tracking noise at partition walls; measured at 0.003% of frames on average, at most 0.59% in one session) are reassigned to the nearest **open** partition by Euclidean distance between partition centres.

Output is stored as an `(1, 600)` int64 array per trial, with `output_names = ['position_bin']` and `output_values` naming the 9 partitions `x0y0 ... x2y2`. The realised class distribution is 5.7%-20.0% per class.

ii.
```python
POSITION_BUFFER = 1e-15      # buffer used when binning position in decode_position_within
_PART_CENTRES = np.array([[i % N_SPATIAL_BINS, i // N_SPATIAL_BINS]
                          for i in range(N_SPATIAL_BINS ** 2)], dtype=float)
...
bin_size = (np.nanmax(position) + POSITION_BUFFER) / N_SPATIAL_BINS
...
open_parts = np.setdiff1d(np.arange(N_SPATIAL_BINS ** 2), blocked)
...
xb = np.clip((pos[0] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
part = N_SPATIAL_BINS * yb + xb

# tracking noise can put a few samples inside a wall; snap them to the nearest
# open partition, as decode_position_within snaps bins to the nearest valid one
stray = np.isin(part, blocked)
if stray.any():
    d = np.linalg.norm(_PART_CENTRES[part[stray]][:, None, :]
                       - _PART_CENTRES[open_parts][None, :, :], axis=2)
    part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. *"Position is binned with the paper's rule (`floor(position / ((max_position + buffer) / n_bins))`, `utils.decode_position_within`) with n_bins = 3 instead of 15, and with the scale taken from the maximum over all days of an animal, again as in `decode_position_within`. This makes one bin exactly one 25 cm partition of the 75 cm arena, so the 9 classes are the 9 partitions of the paper's design. The partition index is 3 * y_bin + x_bin, which is the indexing of the `blocked` field (verified empirically: with this convention the occupancy of blocked partitions is 0.003% of frames on average). The residual handful of samples that fall inside a blocked partition are tracking noise at partition walls; they are reassigned to the nearest open partition, mirroring the 'cleaning' of actual and predicted bins to the nearest valid bin in `decode_position_within`."* The 3×3 discretisation itself is dictated by the task instructions ("Mouse position discretized into 3 x 3 = 9 spatial bins").

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame in the raw data, then bin-for-bin after identical 3-frame average pooling, then trial-for-trial by the same slice object. A defensive `min()` guards against the two streams differing in length after pooling, and the shared `n_trials` count is derived from that minimum.

ii.
```python
n_bins = min(rates.shape[1], part.shape[0])
n_trials = n_bins // BINS_PER_TRIAL     # drop a trailing partial minute
...
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. Both streams come from the same 30 Hz DAQ and have identical frame counts in the released arrays, so no lag correction is needed; using one `slice` for both guarantees the neural and behavioural bins of a trial correspond. The `min()` is a safety net in case a day's `trace` and `position` differ in length.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases:
- **Unregistered cells (NaN rows)**: dropped per session (2-c). The AI first verified that NaN is all-or-none across time for a given cell-day, so testing frame 0 is sufficient and no partial-NaN traces exist.
- **Missing position samples**: none exist — the AI checked `np.isnan(position).sum() == 0` for all 7 animals — so no interpolation or masking is implemented.
- **`blocked = -1` sentinel**: filtered out by `vals >= 0`, yielding an empty index array and an all-zero geometry vector for the full square.
- **Position samples inside a blocked partition** (physically impossible, i.e. tracking noise): snapped to the nearest open partition (4-c).
- **Ragged tails**: a trailing partial minute, and any 1-2 frame mismatch between streams after pooling, are dropped.

ii.
```python
registered = ~np.isnan(tr[:, 0])          # a cell is NaN for the whole day or not
tr = tr[registered]
...
vals = vals[vals >= 0]           # -1 codes "no partition blocked" (the full square)
...
stray = np.isin(part, blocked)
if stray.any():
    ...
    part[stray] = open_parts[np.argmin(d, axis=1)]
...
n_bins = min(rates.shape[1], part.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
```

iii. NaNs are not allowed by the target format and an unregistered cell carries no information, so dropping is the only option; the AI confirmed empirically (trajectory step 40) that dropping is lossless with respect to the paper's reported 69,744 rate maps. The `-1` handling follows the README. The stray-position reassignment is justified as tracking noise at partition walls, quantified at 0.003% of frames. The converted file passes the provided validator with zero errors and zero warnings.

## 6-a. What are the most time-consuming steps of the code?

i. The code is dominated by I/O and bulk array arithmetic, in roughly this order:
1. `joblib.load` of each animal file — the decompressed `trace` array alone is float64 and up to ~7 GB per animal (e.g. QLAK-CA1-51 is (21, 554, 72219) = 6.7 GB); measured at ~6.6 s for that animal, and `joblib.load` also materialises the unused `SFPs`, `centroids` and `maps` fields.
2. `gaussian_filter1d` over each day's (n_cells, ~72,000) float64 trace — ~0.06 s for a 113-cell day, scaling to ~0.3 s for the largest 564-cell days, ≈ 1 min over all 207 sessions.
3. `pickle.dump` of the 6.65 GB output (and the `np.ascontiguousarray` copies that build it).

The whole conversion took on the order of 6 minutes in the trajectory. Memory is also a bottleneck: the entire output dict (6.65 GB) is accumulated in RAM alongside one animal's full float64 `trace`.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
...
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
...
with open(out_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI gives no explicit discussion of runtime. The one performance-relevant thing it does do deliberately is `del dat, trace, position` at the end of each animal, to release the multi-GB per-animal arrays before loading the next; and it casts to `float32` before storing, which halves the size of the output relative to the source float64. It ran the full conversion as a background job and polled it (trajectory steps 64-67), i.e. it treated runtime as acceptable rather than something to optimise.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops, all minor:
- The **per-trial slicing loop** `for t in range(n_trials)` builds 39-40 contiguous copies per session one at a time. It could be a single reshape/`np.split` of `rates[:, :n_trials*600]` into `(n_neurons, n_trials, 600)`, and likewise for `part`. (The reference solution uses an equivalent list comprehension, so this is not a difference in cost — but it is the one loop that is genuinely vectorisable.)
- `input_trials.append(geom.copy())` inside that loop makes `n_trials` identical 9-element copies; a single shared array (or `[geom] * n_trials`) would do, since nothing mutates them.
- `_PART_CENTRES` is built with a Python list comprehension at import time (negligible: 9 elements).

The per-day and per-animal loops cannot be vectorised — the arrays have different neuron counts and different geometries per day. The potentially expensive stray-snapping step is already fully vectorised as a single broadcast distance computation rather than a per-sample loop.

ii.
```python
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])
```
versus the already-vectorised clean-up:
```python
d = np.linalg.norm(_PART_CENTRES[part[stray]][:, None, :]
                   - _PART_CENTRES[open_parts][None, :, :], axis=2)
part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. No justification is offered; the AI does not discuss vectorisation. The choice to vectorise the stray-snapping (rather than loop over samples, as the paper's `decode_position_within` does with its per-sample `for i in range(actual.shape[0])` cleaning loop) is an implicit efficiency improvement over the reference implementation it is imitating.

## 6-c. What processing does the code repeat multiple times?

i. Very little is recomputed:
- `bin_size = (np.nanmax(position) + buffer) / 3` is computed **once per animal** and reused across that animal's days — matching `decode_position_within`, which also takes the maximum across all days, and avoiding 21-31 redundant `nanmax` passes over a multi-GB array.
- `geom` is built once per session and then `.copy()`-ed once per trial (39-40 identical 9-element copies) — trivial in cost but redundant.
- `part` is computed once per session and sliced; `rates` likewise.
- `n_registered`/`rates.shape[0]` are recomputed for logging and `session_info`, which is negligible.

There is no substantive repeated computation.

ii.
```python
# Spatial bin size, from the maximum position over all days of this animal, exactly
# as in utils.decode_position_within.  This is ~25 cm, i.e. one partition.
bin_size = (np.nanmax(position) + POSITION_BUFFER) / N_SPATIAL_BINS

days = range(n_days) if max_sessions is None else range(min(n_days, max_sessions))
for day in days:
    ...
```

iii. The hoisting of `bin_size` out of the day loop is explicitly justified in the code comment as replicating `decode_position_within`'s use of an animal-wide maximum ("needed for across-day training and decoding" in the original). That is a correctness requirement (the same physical location must map to the same class on every day) that happens also to avoid repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items, all small:
- **`* FPS` rescaling to Hz.** The AI itself measured that the provided decoder is invariant to a global scale of the neural data (balanced accuracy 0.358 / 0.361 / 0.367 for scale 0.0333 / 1 / 10). The multiplication is therefore cosmetic — kept only so the stored units match `get_rate_maps`.
- **Loading unused fields.** `joblib.load` materialises the whole per-animal dict, including `SFPs` (~0.11 GB), `centroids` and the precomputed `maps` (~0.16 GB), none of which are used. An h5py/lazy read of only `trace`, `position`, `blocked`, `envs` would avoid this.
- **`envs`** is parsed for every animal but used only for logging and the `session_info`/`environments` metadata, never for the decoder input (`blocked` is used instead).
- **Per-session `session_info`** (207 dicts, including `n_neurons_registered`) and the long descriptive metadata strings are written to the pickle but ignored by `train_decoder.py`. The same applies to the 39-40 duplicated `geom.copy()` arrays per session, and to the per-day progress printing.

None of these materially affect the 6.65 GB output size or the ~6 min runtime, which are dominated by the neural array itself.

ii.
```python
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)   # scale is decoder-irrelevant
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]          # also loads SFPs/centroids/maps
envs = [str(e) for e in np.asarray(dat["envs"]).ravel()]           # metadata/logging only
...
session_info.append({
    "subject": animal, "day": int(day), "environment": envs[day],
    "blocked_partitions": [int(b) for b in blocked],
    "n_neurons": int(rates.shape[0]),
    "n_neurons_registered": n_registered,
    "n_trials": int(n_trials),
})
```

iii. The AI justifies the `* FPS` as matching the paper's rate-map units (*"expressed as an event rate in Hz, the same units the paper uses for its rate maps"*) after explicitly testing that the decoder is scale-invariant, i.e. it knew the step was performance-neutral and kept it for interpretability. The rich metadata is justified by the target format, which asks for `task_description`, `temporal_alignment_event`, `off_start`/`off_end` and invites "other relevant fields, e.g. `session_info`". The unused fields loaded by `joblib.load` are an unremarked side effect of choosing the repo's own joblib access path over a lazy HDF5 read.
