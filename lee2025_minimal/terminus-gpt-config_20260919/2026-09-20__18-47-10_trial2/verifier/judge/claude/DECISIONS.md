# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the repository's native Python (joblib) copies of the dataset — the extension-less files `/app/data/QLAK-CA1-XX` — rather than the parallel `.mat` files. The seven animal IDs are hard-coded in an `ANIMALS` list. Each file is an outer dict keyed by the animal ID; the AI pulls `trace` (n_days, n_cells, n_frames), `position` (n_days, 2, n_frames) and `envs` (n_days,) from it, checks that the three have the same number of days, and iterates over days. One animal file is held in memory at a time and explicitly released (`del` + `gc.collect()`) before the next, because a single file expands to ~9 GB of float64 traces. All 7 animals × (31, 31, 31, 21, 31, 31, 31) days = 207 sessions are loaded; nothing is skipped.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for subj_i, animal in enumerate(ANIMALS):
    outer = joblib.load(DATA_DIR / animal)
    dat = outer[animal]
    traces = dat['trace']
    positions = dat['position']
    envs = np.asarray(dat['envs']).reshape(-1)
    if not (len(traces) == len(positions) == len(envs)):
        raise ValueError(f'inconsistent session count for {animal}')
    for day, env_raw in enumerate(envs):
        ...
    del outer, dat, traces, positions
    gc.collect()
```

iii. From the trajectory: "The README clarifies that the extensionless files are Python joblib datasets… The trace is already processed according to the paper, so it should be used rather than reprocessing raw calcium." The AI first inspected the README and `src/utils.py` (`load_dat`, whose default `format="joblib"`), confirmed the nested `dat[animal]` layout after an initial failed inspection, and then enumerated every animal to confirm "All seven animals total 207 sessions: six animals have 31 sessions and QLAK-CA1-51 has 21", matching the paper's reported 207 sessions / 69,744 rate maps.

## 1-b. How are the data split into subjects?

i. One subject per animal data file. The `ANIMALS` list is used directly as `data['subjects']`, and `subject_idx` records the index of the animal that produced each session, appended in loading order.

ii.
```python
for subj_i, animal in enumerate(ANIMALS):
    ...
        subject_idx.append(subj_i)
...
'subjects': ANIMALS,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. The repository README states that each file is named for the animal ID from the original study and contains all of that animal's sessions, so the file (animal ID) is the subject. The AI also records the animal in `metadata['session_info']` per session.

## 1-c. How are the data split into sessions?

i. One session per recording day, i.e. one entry of the `trace`/`position`/`envs` arrays. All 207 days across the 7 mice become 207 output sessions; the environment geometry label of that day and the day index are stored in `metadata['session_info']`.

ii.
```python
for day, env_raw in enumerate(envs):
    env = str(env_raw)
    counts, labels, n_cells = aggregate_session(traces[day], positions[day])
    ...
    session_info.append({
        'subject': animal, 'day_index': int(day), 'environment': env,
        'source_frames': int(traces.shape[-1]),
        'n_neurons': n_cells, 'n_trials': 40,
    })
```

iii. The methods state "All sessions were 40 min, and one session was recorded per day"; each day is a different geometry, and cells are registered across days but the recorded population differs per day, so each day must be modelled as its own session. The AI verified the resulting count (207) against the paper's reported 207 sessions.

## 1-d. How are the data split into trials?

i. Per the task instructions, trials are contiguous 1-minute segments. Instead of chopping the raw 30 Hz frame stream at fixed 1800-frame boundaries, the AI first re-bins each session onto exactly 2,400 one-second bins using proportional integer edges (`np.linspace(0, n_frames, 2401)`), then slices those 2,400 bins into 40 contiguous 60-bin (1-minute) trials. Because the aligned recordings are 71,866–72,219 frames (39.93–40.12 min) rather than exactly 72,000, this absorbs the small per-animal length differences instead of dropping a partial final trial: every session yields exactly 40 trials, and no frames are discarded (8,280 trials total).

ii.
```python
N_SECONDS = 40 * 60
TRIAL_SECONDS = 60
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
starts, widths = edges[:-1], np.diff(edges)
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. Docstring: "Recordings are nominally 40 min at 30 Hz but aligned arrays differ slightly from 72,000 frames. We use all aligned frames and proportional integer edges to form exactly 2,400 one-second bins… This avoids inventing/dropping a final minute and preserves the session duration specified in the paper." The trajectory adds that the slight length differences are "likely due to acquisition/frame alignment".

## 1-e. How are trials filtered based on quality controls?

i. No trial-, session-, or animal-level filtering is applied: all 207 sessions and all 40 trials per session are kept. The only curation is at the neuron level (see 2-c). Notably, the paper's own within-session Bayesian decoder (`decode_position_within` in `src/utils.py`) discards immobility frames (velocity < 5 cm/s, Gaussian-smoothed velocity) and low-activity cells (≤5 events); the AI applies neither, keeping the full continuous time series. The only integrity checks are hard failures: a session whose position array does not match the trace frame count, or a kept neuron with partially non-finite values, raises an exception rather than being dropped.

ii.
```python
if position.shape != (2, n_frames):
    raise ValueError(f'alignment mismatch: trace {trace.shape}, position {position.shape}')
...
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
...
if np.any(widths <= 0):
    raise ValueError('recording too short for one-second aggregation')
```

iii. The trajectory states "No place-cell filtering should be applied because place-cell detection was a separate analysis and the project describes all manually curated neurons", and the AI verified its totals match the paper's headline numbers (207 sessions, 69,744 session-neuron recordings). Velocity filtering is a decoding-analysis step in the paper rather than a data-curation step, and the target format requires continuous per-trial time series, so it was not applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely the `trace` field: the released, manually curated binarized rising phases of calcium transients (1 = significant event), shape (n_days, n_cells, n_frames). No re-derivation from fluorescence, no use of `maps`, `SFPs`, or `centroids`.

ii.
```python
traces = dat['trace']
...
counts, labels, n_cells = aggregate_session(traces[day], positions[day])
```

iii. Docstring: "Use the released `trace` arrays. These are the manually curated, binary rising phases of calcium transients (z > 2.5), which the paper treats as firing rate; we do not redo calcium extraction or select only place cells." This follows the methods text: "This binary vector was treated as the firing rate in all further analyses."

## 2-b. How is the `neural` data processed?

i. Three steps: (1) drop neurons not registered that day (all-NaN rows); (2) sum the binary events within each of the 2,400 one-second bins with `np.add.reduceat`, producing per-second event counts (integers 0–~30) stored as float32; (3) slice into 40 trials of 60 bins, each stored as a contiguous copy of shape (n_neurons, 60). No Gaussian smoothing, no z-scoring, no normalization, and no place-cell selection. (The repository's own decoder applies `gaussian_filter1d(sigma=3)` plus 3-frame average pooling; the AI keeps the summation but omits the smoothing.)

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
...
counts = np.add.reduceat(tr, starts, axis=1)
counts = counts[:, :N_SECONDS].astype(np.float32, copy=False)
```

iii. Docstring/metadata: "Counts per 1 s of released binary significant calcium-transient rising phases (z > 2.5)"; "Released events are 0/1. reduceat sums each variable-width (~30 frame) bin." The trajectory motivates aggregation as necessary for tractability: "Directly retaining 30 Hz for 69,744 neuron-sessions across 207 sessions would create an impractically huge converted dataset and decoder workload." float32 (rather than the AI's initial uint8) was chosen only to clear the validator's dtype warning: "the validator would convert them during training… these arrays should be saved as float32 despite the larger pickle."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons that were registered on that day are kept: a neuron is included iff its trace for that day contains at least one finite value. Cells absent on a day are all-NaN in the released data and are dropped per session; the AI additionally asserts that no *partially* NaN neuron survives the mask (it would raise, and did not). This yields 113–564 neurons per session and 69,744 session-neurons in total. No activity-rate threshold, no place-cell criterion, no SNR filter.

ii.
```python
# A neuron is either finite for the complete day or all NaN in released data.
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
...
region_idx.append(np.zeros(n_cells, dtype=np.int64))
```

iii. Docstring: "A registered cell is included in a recording session exactly when its trace on that day is not all NaN. NaN rows denote cells absent on that day," matching the README ("If cell is not registered on given day, will appear as nan the same shape"). The trajectory notes the decoder cannot accept NaNs and each session is modelled independently, and that the resulting total (69,744) reproduces the paper's reported number of rate maps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural alignment event — this is 40 min of continuous free exploration. The AI therefore defines the alignment event as the start of each contiguous 1-minute segment, with `off_start = 0.0` s and `off_end = 60.0` s, and documents this in `metadata`. Trial *t* of a session covers seconds [60t, 60t+60) of that recording, identically for neural, input and output streams.

ii.
```python
'temporal_alignment_event': 'start of each contiguous one-minute segment of a 40-minute free-exploration session',
'off_start': 0.0,
'off_end': 60.0,
'trial_definition': '40 contiguous one-minute trials per nominal 40-minute recording; all aligned source frames distributed across 2400 bins',
```

iii. The task instructions define trials as 1-minute splits of a long continuous session, so the only meaningful alignment reference is the segment boundary; the AI encodes that explicitly rather than leaving the metadata fields blank.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the data are rebinned from the native 30 Hz acquisition to **1-second bins** (`time_bin_size = 1000.0` ms), a 30× reduction; each trial is 60 timepoints. Bin edges are proportional integer edges over the whole session (`np.linspace(0, n_frames, 2401)`), so each bin covers ~29.94–30.09 frames depending on the animal (within 0.2% of exactly 1 s); the bins are exhaustive and non-overlapping. Neural events are **summed** within a bin (event counts); position is **averaged** within the same bins before discretization. The same edges are used for every stream, so bin sizes are identical across all trials and sessions.

ii.
```python
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
starts, widths = edges[:-1], np.diff(edges)
counts = np.add.reduceat(tr, starts, axis=1)
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
pos_mean = pos_sum / widths[None, :]
...
'time_bin_size': 1000.0,
'source_sampling_rate_hz': 30.0,
```

iii. Motivated in the trajectory purely by dataset size / decoder tractability: "To retain the full dataset at a practical size, each nominal 40-minute session will be aggregated into 2,400 one-second bins: binary events become per-second event counts and x/y positions become per-second means." The AI had earlier located the repository's decoder settings — "The reference within-session decoder smooths neural traces with a Gaussian and averages into non-overlapping 3-frame temporal bins (100 ms at 30 Hz). This is the appropriate temporal processing to preserve while making the dataset tractable" — but then chose a 10× coarser bin (1 s) and dropped the Gaussian smoothing.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From the per-day environment name string `envs`, mapped through a verbatim copy of the repository's `get_env_mat()` lookup table (`ENV_MATS`, all ten geometries: square, o, t, u, rectangle, +, i, l, bit donut, glenn), where 1 = accessible partition and 0 = blocked. The alternative raw field `blocked` (indices of occluded partitions, −1 if none) was inspected but not used.

ii.
```python
ENV_MATS = {
    'square':    [[1,1,1], [1,1,1], [1,1,1]],
    'o':         [[1,1,1], [1,0,1], [1,1,1]],
    ...
    'glenn':     [[1,1,0], [1,1,1], [0,1,1]],
}
...
envs = np.asarray(dat['envs']).reshape(-1)
for day, env_raw in enumerate(envs):
    env = str(env_raw)
    geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)
```

iii. Trajectory: "Geometry `blocked` exactly indicates inaccessible 3×3 cells, while `get_env_mat` supplies equivalent occupancy masks"; and, after checking the shared `behav_dict`: "The behavior dictionary only contains position, environment labels, and map shape—not blocked indices—so geometry should be derived from environment names via the repository's `get_env_mat` definitions." The matrices are copied exactly from `georepca1/src/utils.py`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3×3 matrix is flattened to a 9-vector of float32 accessibility indicators (1 = accessible, 0 = blocked) and attached unchanged to every trial of that session (static per trial, `(9,)`), with `input_names = ['accessible_x0_y0', 'accessible_x0_y1', …, 'accessible_x2_y2']`. The AI states the flattening uses the same x-major order as the position labels. **It does not**: the repository's matrices are in display orientation, and the repo converts them to map (x, y) indexing with `np.fliplr(get_env_mat(env).T)` (i.e. accessible(x, y) = E[2−y][x]), whereas the AI uses `E.reshape(9)`, i.e. entry 3x+y = E[x][y]. Empirically, with the AI's convention the animal spends 21.3% of its time in bins the input vector marks as inaccessible, versus 0.0% under the repository's convention. Because the mismatch is a single fixed permutation applied to every session, the geometry information is preserved up to relabeling (a trained decoder can absorb it), but the input dimension names do not correspond to the output position bins they claim to.

ii.
```python
geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)
...
    sess_input.append(geom.copy())
...
'input_names': [f'accessible_x{x}_y{y}' for x in range(3) for y in range(3)],
'input_description': 'Nine static binary indicators (1 accessible, 0 blocked), x-major flattening of repository get_env_mat geometry',
```

iii. Docstring: "Inputs are the nine accessible-region indicators for the day's geometry, in the same x-major flattened order [as the position labels]. They are static per trial as requested." The AI never verified the orientation empirically (e.g. against occupancy or against the repo's `np.fliplr(...T)` masking in `clean_rate_maps`), and the trajectory contains no check of geometry-vs-position correspondence.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, the DeepLabCut-tracked head x–y coordinates in cm, shape (2, n_frames) per day, already timestamp-aligned to the imaging frames at 30 Hz. Values are finite throughout (no NaNs) and span 0–75 cm.

ii.
```python
positions = dat['position']
...
counts, labels, n_cells = aggregate_session(traces[day], positions[day])
```

iii. The README documents `position` as "x-y position data for all days… x-y position in first dimension, and number of temporal bins / frames in second dimension"; the methods state position was tracked with DeepLabCut and streams were timestamped for post-hoc alignment at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is averaged within each 1-second bin (sum of frames / number of frames in that bin, using the same proportional edges as the neural data), then discretized (4-c) into a single categorical variable per timepoint, stored as int64 with shape (1, 60) per trial. `output_names = ['position_bin']`, `output_values = [['x0_y0', …, 'x2_y2']]`.

ii.
```python
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
pos_mean = pos_sum / widths[None, :]
...
labels = (3 * xy_bin[0] + xy_bin[1]).astype(np.int64)
...
sess_output.append(labels[None, start:stop].copy())
```

iii. Metadata: "Mean aligned x/y position per 1 s, discretized at 25 and 50 cm; label = 3*x_bin + y_bin." Averaging before discretizing follows from the decision to work in 1-second bins (2-e). (Cost of that choice, measured on QLAK-CA1-08: ~20% of 1-s bins contain frames from more than one spatial bin and ~6% of raw frames carry a label different from the bin label they are folded into; only 0.03% of bin-mean labels fall in a bin that is physically blocked for that geometry.)

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Fixed physical boundaries on the 75 × 75 cm arena: bin edges at 0/25/50/75 cm in each axis via `floor(pos / 25)`, clipped to 0–2 so that exact wall values (75.0 cm occurs in the data) fall in the last bin. The 2-D bin index is flattened x-major into 9 classes, `label = 3 * x_bin + y_bin`, giving integers 0–8. All 9 classes occur in the converted data (class fractions 0.057–0.201).

ii.
```python
# Physical grid boundaries are 0,25,50,75 cm. clip handles exact wall values.
xy_bin = np.floor(pos_mean / 25.0).astype(np.int64)
np.clip(xy_bin, 0, 2, out=xy_bin)
labels = (3 * xy_bin[0] + xy_bin[1]).astype(np.int64)
```

iii. Docstring: "Position is categorized on the physical 3 x 3, 75 cm arena grid. Labels use the repository's x-major convention: label = 3*x_bin + y_bin." The methods state the open square was 75 × 75 cm "partitioned into a 3 × 3 grid space", which is exactly the 3 × 3 = 9 discretization the task requests; the AI uses the true physical edges rather than the repository's per-session `max`-normalized binning used for rate maps.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace come from the same 30 Hz DAQ stream and are already frame-aligned in the released data; the AI asserts this (`position.shape == (2, n_frames)` of the trace) and raises on mismatch. Both are then aggregated with the *same* `starts`/`widths` bin edges and sliced with the same trial boundaries, so timepoint *k* of `neural` and of `output` cover exactly the same ~30 raw frames. Inputs are static, so alignment is trivial for them.

ii.
```python
n_frames = trace.shape[1]
if position.shape != (2, n_frames):
    raise ValueError(f'alignment mismatch: trace {trace.shape}, position {position.shape}')
...
counts = np.add.reduceat(tr, starts, axis=1)
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
```

iii. The methods state "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz… and all recorded frames were timestamped for post-hoc alignment", so no re-alignment is needed; the AI's contribution is an explicit assertion plus shared bin edges. No lag/offset between neural and behaviour is introduced.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) Cells not registered on a given day appear as all-NaN rows and are removed per session; a partially NaN kept neuron would raise rather than silently propagate NaNs. (b) Sessions that are not exactly 72,000 frames (all of them: 71,866–72,219) are handled by proportional integer bin edges, so no frames are dropped and no minute is fabricated; a recording too short to fill 2,400 bins would raise. (c) A position/trace frame-count mismatch raises. (d) Position contains no NaNs in this dataset, and none is imputed. (e) Copies (`np.ascontiguousarray`, `.copy()`) are taken so the pickle does not retain views into the multi-GB session buffers.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
if np.any(widths <= 0):
    raise ValueError('recording too short for one-second aggregation')
if position.shape != (2, n_frames):
    raise ValueError(f'alignment mismatch: trace {trace.shape}, position {position.shape}')
```

iii. The AI's stance is fail-loud rather than silently repair: every assumption it relies on (all-or-nothing NaN registration, stream alignment, session duration) is asserted, and the only data actually removed are unregistered cells. The docstring justifies the length handling as avoiding "inventing/dropping a final minute" while preserving "the session duration specified in the paper".

## 6-a. What are the most time-consuming steps of the code?

i. (1) `joblib.load` of each animal file — I/O plus decompression, and it materializes the entire outer dict, including `SFPs`, `centroids` and `maps`, which are never used; the `trace` array alone expands to ~9.2 GB of float64 for QLAK-CA1-08 (~9 s just to load that one file, more for the larger animals). (2) The two full passes of `np.isfinite` over that multi-GB array plus the fancy-index copy `trace[keep]`. (3) `np.add.reduceat` over the full trace (2,400 segments × up to 952 cells × 72k frames). (4) Pickling the 675 MB result. Per-trial slicing/copying and position work are negligible by comparison.

ii.
```python
outer = joblib.load(DATA_DIR / animal)   # whole file, incl. unused SFPs/centroids/maps
...
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
if not np.all(np.isfinite(tr)):
...
counts = np.add.reduceat(tr, starts, axis=1)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI was aware of the memory/time cost — the trajectory repeatedly emphasizes "memory-conscious converter", "inspect one processed animal … without dumping arrays", "load one animal at a time" — and mitigates it with `del` + `gc.collect()` between animals rather than by reading fields lazily. The joblib format forces whole-file loading; the `.mat` copies would have allowed per-field lazy reads via h5py.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code is already essentially vectorized: the per-second aggregation uses `np.add.reduceat` rather than a Python loop over 2,400 bins, and binning/discretization are pure NumPy. The only remaining Python loops are (1) the 40-iteration trial-slicing loop per session, which could be a single `counts.reshape(n_cells, 40, 60)` / `labels.reshape(40, 60)` view instead of 40 slices plus copies, and (2) the unavoidable loops over animals and days (each needs its own I/O and its own neuron mask). Neither is a meaningful cost — the trial loop copies ~40 × (n_neurons × 60) floats, a small fraction of the data already touched.

ii.
```python
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
    sess_input.append(geom.copy())
    sess_output.append(labels[None, start:stop].copy())
```

iii. The AI deliberately chose contiguous copies here ("Copies keep the pickle independent of the full-session buffers"), i.e. it traded a little copying for not pickling views onto multi-GB arrays — a correct trade given the target format is a list of per-trial arrays.

## 6-c. What processing does the code repeat multiple times?

i. The finiteness test is computed twice over the full (up to 9 GB) trace: once as `np.any(np.isfinite(trace), axis=1)` to build the keep mask, and again as `np.all(np.isfinite(tr))` as a sanity assertion — the second pass could have been derived from the first (e.g. comparing per-row finite counts to `n_frames`). `np.linspace` edges, `starts`/`widths` are recomputed for every session even though all sessions within an animal have identical frame counts. `geom.copy()` is made 40 times per session for an identical static 9-vector.

ii.
```python
keep = np.any(np.isfinite(trace), axis=1)
tr = trace[keep]
if not np.all(np.isfinite(tr)):
    raise ValueError('partially non-finite trace found in an included neuron')
...
edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)   # identical for all days of an animal
```

iii. These repeats are deliberate defensive checks rather than oversights (the docstring/comments flag the NaN convention as an assumption being verified); the AI's priority was validating its assumptions on every session rather than minimizing passes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `joblib.load` reads and decompresses `SFPs` (~156 MB), `centroids` and the precomputed `maps` for every animal although only `trace`, `position` and `envs` are used. (2) The redundant second full-array `np.isfinite` pass (6-c). (3) `counts[:, :N_SECONDS]` and `[:, :N_SECONDS]` on the `reduceat` results are no-ops — `reduceat` already returns exactly 2,400 bins. (4) Event counts are stored as float32 rather than the uint8 the AI originally used, quadrupling the pickle to 675 MB, purely to silence a validator dtype warning; the validator/decoder casts to float anyway. (5) 40 copies of the identical static geometry vector per session, and `metadata['session_info']` for all 207 sessions, are stored but unused by the decoder (though useful documentation).

ii.
```python
outer = joblib.load(DATA_DIR / animal)      # also loads SFPs / centroids / maps
counts = counts[:, :N_SECONDS].astype(np.float32, copy=False)   # slice is a no-op; uint8 would suffice
pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
```

iii. The dtype change was an explicit, documented choice: "The only validator warnings are that neural arrays are `uint8` rather than preferred `float32`… these arrays should be saved as float32 despite the larger pickle. This retains identical values/statistics while satisfying the validator's preferred dtype." The unused fields are a consequence of choosing the joblib format (whole-file load) over the `.mat` format; the defensive slices are harmless guards.
