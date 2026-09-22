# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the **joblib** version of the dataset (the extension-less per-animal files in `/app/data`, e.g. `QLAK-CA1-08`), not the `.mat` files. Subject names are hard-coded in a `SUBJECTS` list of the seven animals. For each subject it loads the whole nested dict once (`joblib.load(path)[subject]`) and pulls out four fields: `trace` (session × tracked-cell × frame), `position` (session × xy × frame), `envs` (session labels) and `blocked` (ragged per-session list of blocked compartments). It then iterates over the first (session) axis of `trace`. After each subject it deletes the arrays and calls `gc.collect()` to bound peak memory. This yields all 207 recordings from 7 subjects (31, 31, 31, 21, 31, 31, 31), and 69,744 session-specific neurons — exactly the counts stated in the paper's abstract.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
for subj_i, subject in enumerate(SUBJECTS):
    source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
    traces = source['trace']       # session x tracked cell x aligned frame
    positions = source['position'] # session x (x,y) x aligned frame
    envs = np.asarray(source['envs']).reshape(-1)
    blocked = source['blocked']
    for day in range(traces.shape[0]):
        ...
    del source, traces, positions
    gc.collect()
```

iii. From the trajectory: the AI first probed both file types (`file`, `h5py`, `pickle`, `joblib`), found the `.mat` files are MATLAB v7.3/HDF5 with per-session object references and the extension-less files are joblib dumps containing "the same structured data". It then read `utils.load_dat` in the paper's repo, whose **default** format is `joblib` (the repo ships `mat2joblib` to produce exactly these files), and concluded "the joblib files are confirmed as the intended reference format". It verified counts against the paper ("all seven animals … totalling 207 sessions exactly as reported in the paper") and chose one-subject-at-a-time loading explicitly "to control memory".

## 1-b. How are the data split into subjects?

i. One subject per data file / per animal ID. The `SUBJECTS` list of seven animal IDs is both the file-name list and the `subjects` field of the output; `subject_idx` records the index of the owning animal for each of the 207 sessions.

ii.
```python
'subjects': SUBJECTS,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
...
subject_idx.append(subj_i)
```

iii. The README states each file is named with the animal ID from the original study and contains that animal's fields, so file == subject. The AI confirmed seven animal files plus a shared `behav_dict`, and enumerated the IDs directly rather than globbing.

## 1-c. How are the data split into sessions?

i. One session in the output = one daily 40-minute recording in the source (the first axis of `trace`/`position`, one entry of `envs`/`blocked`). No recordings are excluded, giving 207 sessions. Each session carries its own neuron set, its own environment geometry, and a `session_info` metadata entry (subject, recording index, environment name, blocked bins, source frame count, neuron count, trial count).

ii.
```python
for day in range(traces.shape[0]):
    tr = traces[day]
    pos = positions[day]
    ...
    neural.append(ses_neural); inputs.append(ses_input); outputs.append(ses_output)
    subject_idx.append(subj_i)
    session_info.append({'subject': subject, 'recording_index': day,
                         'environment': str(envs[day]),
                         'blocked_spatial_bins': blocked_idx.tolist(),
                         'source_frames': int(nframes), ...})
```

iii. "All sessions were 40 min, and one session was recorded per day" (methods). The AI reasoned that the daily recording is "the correct session unit for the target format … with its own subset of valid recorded neurons", because cell identity/registration differs day to day and the decoder is fit per session.

## 1-d. How are the data split into trials?

i. Each session is first re-binned into exactly 2400 one-second bins (see 2-e) and then cut into 40 contiguous, non-overlapping 60-bin (= 1-minute) trials. Because the number of bins is fixed at 2400 regardless of the exact frame count (71,866–72,219 across animals), every session yields exactly 40 trials of 60 timepoints, and no frames are discarded.

ii.
```python
N_SECONDS = 40 * 60
TRIAL_SECONDS = 60
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    ses_neural.append(counts[:, start:stop].copy())
    ses_input.append(geometry_ts.copy())
    ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The instructions define the trial as a 1-minute segment of the continuous recording. The AI noted the arrays "differ slightly around 30 Hz after timestamp alignment" and argued that partitioning the 40-minute recording into 2400 equal bins "preserves all data and yields exactly forty one-minute trials, rather than incorrectly dropping a nearly complete minute for arrays slightly below 72,000 frames".

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied: all 40 trials of all 207 sessions are kept (8,280 trials). The only curation is at the neuron level (2-c). Two hard assertions guard data integrity — a session with zero valid neurons or with any NaN in position raises `ValueError` rather than being silently dropped; neither fired during the conversion. Notably, the paper's own Bayesian decoder applies a 5 cm/s velocity filter and a ≥5-event cell-sparsity filter (`decode_position_within`), and the AI did **not** port either.

ii.
```python
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
if np.isnan(pos).any():
    raise ValueError(f'NaN position: {subject}, day {day}')
```

iii. The AI treated the released dataset as already curated ("all 207 curated daily recordings", "69,744 session-neuron instances, exactly matching the paper's reported 69,744 rate maps") and used that exact match as its evidence that no further session/trial exclusion is warranted. It did not discuss the velocity or sparsity filters used in the paper's decoding routine.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Only the `trace` field: the paper's rise-extracted calcium traces, where 1 marks a significant transient rising phase and NaN marks a cell not registered on that day. No other neural field (`SFPs`, `centroids`, `maps`) is used.

ii.
```python
traces = source['trace']       # session x tracked cell x aligned frame
...
tr = traces[day]
```

iii. The README describes `trace` as "rise-extracted calcium traces, where '1' indicates a significant event", and the methods state "This binary vector was treated as the firing rate in all further analyses". The AI verified empirically that trace values are only `{0, 1, nan}`.

## 2-b. How is the `neural` data processed?

i. Three steps: (1) drop NaN (unregistered) cells; (2) **sum** the binary event vector within each of 2400 equal-width bins spanning the recording, i.e. convert the 30 Hz binary event train into an events-per-second count; (3) cast to `uint8` (max observed value 30, so no overflow) and slice into trials. No smoothing, normalisation, z-scoring, or re-derivation from raw fluorescence is applied — the released binary event vector is taken as-is as the firing-rate signal.

ii.
```python
def aggregate_equal_bins(x, edges, reducer='sum'):
    if reducer == 'sum':
        return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                         for i in range(len(edges)-1)], axis=-1)
    return np.stack([x[..., edges[i]:edges[i+1]].mean(axis=-1)
                     for i in range(len(edges)-1)], axis=-1)
...
nframes = tr.shape[1]
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
```

iii. Docstring: "The supplied `trace` is used directly: it is the paper's thresholded binary rising-phase calcium-event vector, aligned sample-for-sample with `position`." The summation is justified in the trajectory on practicality grounds: "Keeping raw 30 Hz data would produce a multi-gigabyte converted pickle and an impractically large decoder workload", and event counts per fixed bin are described as "a likely appropriate representation" of rate given the paper treats the binary vector as firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells whose trace contains any NaN in that recording are dropped, session by session. The AI first checked empirically that the NaN mask is constant across time within a recording (`partial_nan = 0` for every session it sampled), so "any NaN" and "all NaN" select the same cells. No firing-rate, sparsity, or place-cell criterion is applied. Retained populations range from 113 to 564 neurons per session; the total over sessions is 69,744.

ii.
```python
# In source files a cell is either present for every sample or NaN
# for every sample in a given daily recording.
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
```

iii. README: "If cell is not registered on given day, will appear as nan the same shape." The AI reasoned that NaN rows "are absent and should be removed" session-by-session, that decoder matrices cannot contain NaNs, and it validated the choice against the paper's reported 69,744 rate maps, which its output reproduces exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus/behavioural alignment event — the recording is continuous free exploration. The AI defines the alignment event as the start of each consecutive 1-minute segment and fills in the required metadata accordingly (`off_start = 0.0`, `off_end = 60.0`); trial *k* covers bins `[60k, 60k+60)` of the session. Neural, input and output streams are cut with identical indices.

ii.
```python
'temporal_alignment_event': 'start of each consecutive one-minute segment',
'off_start': 0.0,
'off_end': 60.0,
'trial_duration_seconds': 60.0,
```

iii. The AI treated the trial boundary itself as the alignment event so that the required `temporal_alignment_event`/`off_start`/`off_end` metadata fields are meaningful rather than null; the source streams are already timestamp-aligned by the original authors ("The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz … all recorded frames were timestamped for post-hoc alignment").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — aggressive rebinning.** The native 30 Hz data (33.3 ms) is rebinned to **1000 ms** bins: each recording is divided into `N_SECONDS = 2400` equal-width bins via `np.linspace(0, nframes, 2401)`, so each bin holds 29 or 30 frames depending on the animal's exact frame count (71,866 → 29.94 frames/bin; 72,219 → 30.09). `metadata['time_bin_size']` is reported as 1000.0 ms. Each trial is therefore 60 timepoints rather than 1800, a 30× reduction; the whole dataset is 496,800 timepoints. Bin edges are session-relative fractions, so nominal bin duration varies by ≤0.5 % across animals and by ±3 % (29 vs 30 frames) *within* a session.

ii.
```python
N_SECONDS = 40 * 60
...
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
'time_bin_size': 1000.0,
'source_sampling_rate_hz': 'approximately 30; streams timestamp-aligned by source authors',
```

iii. Two justifications given: (a) practicality — "Keeping raw 30 Hz data would produce a multi-gigabyte converted pickle and an impractically large decoder workload"; (b) robustness to the slightly variable recording lengths — equal proportional bins mean "all samples are partitioned into 2400 equal one-second bins (rather than assuming exactly 30 frames or dropping data)". The AI noted the paper's own decoder "uses spatially binned traces" but did not adopt the repo's temporal bin (`fit_decoder(..., temporal_bin_size=3)`, i.e. 100 ms with Gaussian smoothing).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field only (a ragged per-session list of blocked 3×3 compartment indices, with the sentinel `-1` for the open square). The `envs` string label (square, o, t, u, rectangle, …) is carried into `metadata['session_info']` for reference but is not used as a decoder input.

ii.
```python
blocked = source['blocked']
...
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
blocked_idx = blocked_idx[blocked_idx >= 0]  # square uses sentinel -1
```

iii. README: "**blocked**: location of blocked (occluded) partitions in 3x3 design of environment … organized in the following way – [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If no partitions are blocked, value is -1." The AI verified the row-major layout and the sentinel from the data, and noted geometry "can be represented exactly as nine blocked/open indicators from the source `blocked` field".

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A 9-dimensional binary vector (1 = compartment blocked, 0 = accessible) built by indexing with the non-negative blocked indices; `-1` is filtered out so the open square becomes all zeros. The vector is then tiled over the 60 time bins of each trial and repeated (as an independent copy) for all 40 trials of the session, giving `input` shape (9, 60) per trial. `input_names` are `'blocked spatial bin 0..8'`, in the same row-major indexing as the position output.

ii.
```python
geometry = np.zeros(9, dtype=np.float32)
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
blocked_idx = blocked_idx[blocked_idx >= 0]  # square uses sentinel -1
geometry[blocked_idx] = 1.0
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
...
ses_input.append(geometry_ts.copy())
```

iii. Docstring: "Source `blocked` indices are converted to nine blocked/not-blocked channels. They are repeated over trial time because decoder inputs are represented with a common time axis, although geometry is static within each recording." The nine-channel indicator coding preserves the exact geometry (and shares index conventions with the output bins) rather than collapsing 10 geometries into a categorical label.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field only: DeepLabCut head-tracking x–y coordinates in cm, one pair per imaging frame, already frame-aligned with `trace`.

ii.
```python
positions = source['position'] # session x (x,y) x aligned frame
...
pos = positions[day]
```

iii. The AI checked the ranges (0–75 cm in both dimensions) and the absence of NaNs, confirming this is the arena-scaled tracked position; methods: "Position data were generated from tracking the head with DeepLabCut".

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. x and y are **averaged within each 1-second bin** (the same bin edges used for the neural counts), and the averaged coordinates are then discretised. No velocity filtering, smoothing, interpolation or per-session rescaling is applied; the fixed 75 cm arena size from the methods is used rather than each session's observed max.

ii.
```python
xy = aggregate_equal_bins(pos, edges, 'mean')

col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
```

iii. The AI's stated plan was to "average aligned x/y position within each second" so the behavioural stream is reduced on exactly the same bin boundaries as the neural stream, keeping the two aligned sample-for-sample. (This mirrors the repo's `fit_decoder`, which average-pools the behavioural trace over each temporal bin before one-hot encoding, though there pooling is over 3 frames and applied to already-binned coordinates.)

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. A 3 × 3 grid over the 75 × 75 cm arena with fixed 25 cm edges: `col = floor(x/25)`, `row = floor(y/25)`, each clipped to [0, 2] so the occasional exact 75.0 cm coordinate falls in the last bin; the class label is `row*3 + col` ∈ {0..8}, matching the README's `[[0,1,2],[3,4,5],[6,7,8]]` convention used by `blocked`. One output variable (`output_names = ['mouse position']`) with nine values named `'row r, column c'`. Resulting class fractions are 0.057–0.201 (corner bin 8 most occupied, centre bin 4 least, as expected for an open field with thigmotaxis and blocked centres in several geometries).

ii.
```python
col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
...
'output_values': [[f'row {r}, column {c}' for r in range(3) for c in range(3)]],
```

iii. Docstring: "The 75 cm square was designed as a 3x3 grid. Position is discretized with 25-cm edges, clipping the occasional exact 75-cm coordinate into the last bin." The instructions require 3 × 3 = 9 spatial bins, and the paper explicitly "partitioned an open square (75 × 75 cm) into a 3 × 3 grid space", so the decoder's output bins coincide with the experimental compartments and with the `blocked` indexing.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Identically indexed at every stage: the source streams are already frame-for-frame aligned; the *same* `edges` array is used to aggregate both `trace` (sum) and `position` (mean); and the same `start:stop` slices cut both into trials. Output shape is (1, 60) per trial against neural (n_neurons, 60), i.e. one position label per neural time bin, with no lead/lag offset.

ii.
```python
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
ses_neural.append(counts[:, start:stop].copy())
ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. "the supplied `trace` … [is] aligned sample-for-sample with `position`" (docstring); the DAQ "simultaneously acquired behavioral and cellular imaging streams at 30 Hz" with post-hoc timestamp alignment, so no additional realignment is needed and using one shared set of bin edges preserves it.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases. (1) *Unregistered cells* (NaN rows) — removed per session (2-c), after the AI verified NaN masks are all-or-none in time. (2) *Missing position samples* — not expected; the code asserts `np.isnan(pos).any()` is false and raises otherwise. The assertion ran over all 207 sessions without firing, which establishes there are no NaN positions anywhere in the dataset. A session with zero valid neurons would likewise raise. (3) *Variable recording lengths* (71,866–72,219 frames) — absorbed by proportional binning instead of truncation, so no frames are dropped; the price is that a "1-second" bin is 29 or 30 frames depending on position in the session and animal. Boundary coordinates exactly at 75.0 cm are clipped into the last spatial bin. Per-session `source_frames` is recorded in `session_info` so the approximation is auditable.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
if np.isnan(pos).any():
    raise ValueError(f'NaN position: {subject}, day {day}')
...
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
...
col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
```

iii. The AI chose to fail loudly on unexpected missing data rather than impute or silently drop sessions ("decoder matrices cannot contain NaNs"), and chose proportional binning specifically so that short recordings do not lose "a nearly complete minute" of data.

## 6-a. What are the most time-consuming steps of the code?

i. (1) `joblib.load` of each subject file — these are `compress=3` dumps that expand to dense `session × cell × frame` float arrays (e.g. 31 × 952 × 72,071 ≈ 2.1 × 10⁹ elements for QLAK-CA1-75); decompression plus allocation dominates and is what the AI repeatedly waited on (the whole run took roughly 3–4 minutes of waiting in the trajectory). (2) The per-session Python binning loop in `aggregate_equal_bins`: 2400 slice-and-reduce calls per array × 2 arrays × 207 sessions ≈ 1 M small NumPy reductions. (3) `np.isnan(tr).any(axis=1)` over the full (cells × frames) array. (4) Pickling the 187 MB output. Memory, not CPU, is the binding constraint — hence the explicit per-subject `del` + `gc.collect()`.

ii.
```python
source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
...
return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                 for i in range(len(edges)-1)], axis=-1)
...
del source, traces, positions
gc.collect()
```

iii. The AI anticipated this: "Keeping raw 30 Hz data would produce a multi-gigabyte converted pickle", "we will avoid loading several animals simultaneously during conversion to control memory", and in the `aggregate_equal_bins` comment it accepts the loop because it is "memory-efficient and unambiguous".

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `aggregate_equal_bins` list comprehension is the main one: because bin widths are only ever 29 or 30 frames, it could be replaced by `np.add.reduceat(x, edges[:-1], axis=-1)` (one C-level pass), or — if the frame count were truncated to a multiple of 2400 — by a `reshape(..., 2400, w).sum(-1)`. The AI explicitly considered and rejected `reduceat` ("reduceat is unsafe for repeated final boundaries"), which is a real hazard only when a bin is empty; here widths are ≥29, so `reduceat` would have been safe. Secondarily, the 40-iteration trial-slicing loop could be a single `reshape(n_neurons, 40, 60)`, and `geometry_ts.copy()` need not be recreated per trial.

ii.
```python
# reduceat is unsafe for repeated final boundaries; lengths here are ~30,
# but explicit loop is memory-efficient and unambiguous.
if reducer == 'sum':
    return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                     for i in range(len(edges)-1)], axis=-1)
```

iii. The AI's stated reason is clarity and memory safety over speed; the runtime cost was acceptable (whole conversion completed in a few minutes), so it did not revisit it.

## 6-c. What processing does the code repeat multiple times?

i. The static 9-dimensional geometry vector is tiled to (9, 60) once per session and then `.copy()`-ed 40 times, producing 8,280 identical-per-session arrays (≈18 MB of redundant float32). Each neural/output trial slice is also `.copy()`-ed, duplicating the full session arrays before the originals go out of scope. `np.isnan(tr)` is computed over the whole array to build the valid mask (cheap but full-array), and `blocked_idx.tolist()` is recomputed for metadata. Nothing is recomputed across sessions or subjects — each file is loaded exactly once.

ii.
```python
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    ses_neural.append(counts[:, start:stop].copy())
    ses_input.append(geometry_ts.copy())
    ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The copies are deliberate: slices are views, and the AI wanted each trial to own contiguous memory so the parent session arrays (and the multi-GB source arrays) can be freed immediately — consistent with its per-subject `del`/`gc.collect()` memory strategy. Distinct per-trial input objects also avoid aliasing surprises if a consumer mutates one trial.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little is wasted, and the deliberate waste is small: (1) the per-trial tiling of the static geometry over 60 timepoints — the format permits a plain (9,) vector per trial, and the decoder derives no extra information from the 60 identical columns; (2) the 40 independent copies of that tiled array; (3) `envs` is decoded to strings for every session and stored in `session_info` but never used by the decoder (it is useful provenance, not computation); (4) storing `neural` as `uint8` saves disk but makes `verify_data_format` emit 8,280 "expected float32" warnings and forces a cast at train time. Conversely, the *information* thrown away — the 30× temporal downsampling — is not recoverable downstream. The AI also never ran the decoder itself (only `--verify-only`), so the downstream cost/benefit of the 1-second binning was never measured.

ii.
```python
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
...
    ses_input.append(geometry_ts.copy())
...
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
```

iii. Docstring: the geometry channels "are repeated over trial time because decoder inputs are represented with a common time axis, although geometry is static within each recording" — i.e. the redundancy is accepted for uniform (d_input, T) shapes, which the instructions prefer. `uint8` was chosen to keep the pickle compact (178 MB) given the AI's stated concern about a "multi-gigabyte converted pickle".
