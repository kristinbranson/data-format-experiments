# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the *joblib-serialized* per-animal files (the extensionless `data/QLAK-CA1-XX` files produced by the reference repo's `mat2joblib`/`load_dat`), **not** the original MATLAB v7.3 `.mat` files. The list of 7 animals is hard-coded in a module-level constant, and the data directory is hard-coded as the relative path `data`. For each animal, one `joblib.load` call returns a dict keyed by animal ID containing `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`; the AI uses only `envs`, `trace` and `position`. Sessions are iterated by enumerating `rec['envs']`, and trials are carved out of each session inside `session_to_trials`.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
for animal in animals:
    rec = joblib.load(Path('data') / animal)[animal]
    envs = rec['envs'].ravel().tolist()
    traces = rec['trace']
    positions = rec['position']
    for s, env_name in enumerate(envs):
        nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. From CONVERSION_NOTES Step 1/Step 2: the reference code base `georepca1` loads animal data through `load_dat(animal, p, to_convert=["envs","position","trace"], format="joblib")`, i.e. joblib is the format the paper's own analysis code consumes; `mat2joblib` is what produced these files from the `.mat` originals. The AI documented the per-animal joblib layout (`trace` `(n_sessions, n_neurons, n_frames)`, `position` `(n_sessions, 2, n_frames)`, `envs` `(n_sessions, 1)`) and used it directly. I verified that the joblib `trace`/`position` arrays are element-for-element identical (`np.allclose`) to the `.mat` contents read via `h5py`, so the two loading routes give the same data.

## 1-b. How are the data split into subjects (mice)?

i. One subject per animal file. The hard-coded `ANIMALS` list is used verbatim as `subjects`, and each session appends the animal's index into that list to `subject_idx`. Result: 7 subjects with 31/31/31/21/31/31/31 sessions.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
subjects = animals.copy()
subject_to_idx = {a: i for i, a in enumerate(subjects)}
...
        subject_idx.append(subject_to_idx[animal])
...
    'subjects': subjects,
    'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 records that `data/` contains 7 animals, each file being a top-level dict keyed by the animal ID, so the animal ID is the natural subject identifier. `--sample` mode is implemented by taking the first 2 animals.

## 1-c. How are the data split into sessions?

i. Each recording day stored inside an animal file becomes one output session. Sessions are enumerated over the `envs` array (length = number of recording sessions for that animal), and `traces[s]` / `positions[s]` are the leading-axis slices for that session. Sessions producing fewer than 2 trials are skipped. All 207 sessions survive (31+31+31+21+31+31+31).

ii.
```python
for s, env_name in enumerate(envs):
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
    if len(nt) < 2:
        continue
    neural.append(nt); inp.append(it); out.append(ot)
    subject_idx.append(subject_to_idx[animal])
    brain_region_idx.append(np.zeros(nt[0].shape[0], dtype=np.int64))
```

iii. CONVERSION_NOTES Steps 2/3: "All sessions were 40 min", one session per day, and the paper reports "5,413 unique neurons across 207 sessions in 10 geometries". The AI used the per-animal session axis directly and confirmed 207 sessions in the verification output. The `< 2 trials` guard exists because the target format requires "at least two trials within each session".

## 1-d. How are the data split into trials?

i. Trials are non-overlapping, consecutive 60-second chunks of the continuous session: 1800 frames at 30 Hz. The number of trials is `valid_frames // 1800`, where `valid_frames` is one past the last frame at which position is fully finite and at least one neuron is finite. The trailing remainder (< 1800 frames) is discarded. In practice this yields 39 trials/session for the three animals with 71,866 frames and 40 trials/session for the other four, for 8,187 trials total.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS          # 1800

def session_to_trials(trace_sxn, pos_sxn, env_name, min_trials=2):
    valid_frames = infer_valid_frames(pos_sxn, trace_sxn)
    n_trials = valid_frames // FRAMES_PER_TRIAL
    ...
    for t in range(n_trials):
        sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
        neural_trials.append(trace_sxn[:, sl])
        input_trials.append(env_vec.copy())
        output_trials.append(pos_bins[sl][None, :])
```

iii. There is no native trial structure in this dataset; the task instructions state "long recording sessions, which will be split into 1-minute trials within each session". CONVERSION_NOTES Step 5, Key Decision 1: "Split each native 40-minute session into 40 consecutive 1-minute trials at 30 Hz to satisfy decoder requirement of at least two trials per session." The AI noticed in Step 7 that it was getting 39 rather than 40 trials for the first animals and flagged it for later review; in Step 10 it checked `valid_frames` and found 71,866 valid frames out of 71,866 total, i.e. the 39 is simply `71866 // 1800` and not a data-validity artifact.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality metric. Three implicit filters exist: (1) frames after the last frame with finite position *and* at least one finite neuron are dropped before trialization (`infer_valid_frames`); (2) the trailing partial trial is dropped by integer division; (3) an entire session is dropped if it would produce fewer than 2 trials or if no neuron survives the finite-neuron filter. None of these actually removed anything in this dataset: position contains no NaNs and every session yields ≥ 39 trials.

ii.
```python
def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
    valid = pos_valid & neural_valid
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1] + 1)
...
    if n_trials < min_trials:
        return [], [], []
...
    if trace_sxn.shape[0] == 0:
        return [], [], []
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": "No native trial structure; analyses use full 40-min sessions and session halves (20 min / 20 min) for place-cell reliability" — i.e. the paper defines no trial-level exclusion criterion, so the AI applied none beyond guarding against missing/invalid frames and against sessions too short to evaluate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely `rec['trace'][session]`, the per-session calcium activity array of shape `(n_neurons, n_frames)`. No other neural variable (`SFPs`, `centroids`, `maps`) is used.

ii.
```python
traces = rec['trace']
...
    nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
```

iii. CONVERSION_NOTES Step 4: "Stored trace array dtype is float64; sampled values are exactly binary {0,1} with sparse events (~1.5% ones in sample)… Methods state analyses use binarized rising-phase vectors treated as firing rates → Resolved: stored traces are already binarized/preprocessed rising-phase vectors." The reference functions `get_rate_maps` and `decode_position_within` also consume `trace` directly.

## 2-b. How is the `neural` data processed?

i. Essentially no processing. The session array is already `(n_neurons, n_frames)`, so no transpose is needed; it is truncated to `n_trials * 1800` frames, cast to `float32`, neurons with any non-finite value are dropped, and the result is sliced into per-trial `(n_neurons, 1800)` blocks. No ΔF/F, no deconvolution, no smoothing, no z-scoring, no re-binarization, no normalization.

ii.
```python
n_keep = n_trials * FRAMES_PER_TRIAL
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
...
    neural_trials.append(trace_sxn[:, sl])
```
with metadata `'source_trace_representation': 'binary rising-phase vectors treated as firing rates'`.

iii. CONVERSION_NOTES Step 5, Key Decision 2: "Use stored binary `trace` arrays directly rather than recomputing calcium preprocessing, because methods + sampled values confirm they are already binarized rising-phase vectors." The methods describe the full pipeline (derivative → Gaussian smoothing SD = 5 frames → noise-normalized z-score → threshold z > 2.5), and the AI verified the released traces take only the values {0, 1}, confirming the pipeline had already been applied upstream. I independently confirmed the traces are exactly {0, 1} with ~0.7 % ones.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filter is the removal, per session, of neurons that contain **any** non-finite value in that session's kept frames. This removes the NaN-padded rows for neurons that were not registered/tracked in that particular session. No place-cell or reliability filtering is applied. Across the full dataset this leaves 69,744 session-neuron instances (mean 336.9 per session, range 113–564).

ii.
```python
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
```

iii. This filter was added reactively: the first sample verification reported "many trials in session 61 contain NaN or Inf values in neural data", and the AI reasoned (trajectory step 336) that "some neurons are not present/registered in some sessions, leaving NaNs in the per-session trace arrays… removal is more consistent with session-specific recorded neurons", noting that a per-session `brain_region_idx` permits variable neuron counts. CONVERSION_NOTES Step 5, Key Decision 6: "Initially include all stored neurons because artifact removal/preprocessing appears already reflected in the released data; place-cell filtering is analysis-specific and may not be appropriate for a general decoder."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no behavioural or stimulus alignment event — the recordings are continuous free foraging. Trials are aligned to session time: trial *t* spans `[t*60 s, (t+1)*60 s)` from the start of the session. Neural, input and output streams are sliced with the identical frame index, so they are aligned frame-for-frame at 30 Hz. This is recorded in the metadata.

ii.
```python
'temporal_alignment_event': 'session time; sessions split into consecutive 1-minute chunks',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
...
sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
neural_trials.append(trace_sxn[:, sl])
output_trials.append(pos_bins[sl][None, :])
```

iii. CONVERSION_NOTES Step 5, Key Decision 5: "Preserve native 30 Hz alignment between trace and position within each session when splitting into 1-minute trials", justified by the methods statement that "DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" with post-hoc timestamped alignment, so the two streams share a common frame index and need no further alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native acquisition resolution is kept: 30 Hz, i.e. `time_bin_size = 1000/30 ≈ 33.33 ms`, 1800 bins per 60-second trial, identical for every trial and session. No downsampling, upsampling, smoothing or re-binning is performed.

ii.
```python
FPS = 30
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
...
    'time_bin_size': 1000.0 / FPS,
    'fps': FPS,
```

iii. CONVERSION_NOTES Step 3 records "DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" as the source of the bin size. The AI noted that the reference decoder helper `test_decoder` uses a `temporal_bin_size=3` downsample internally, but treated that as an analysis choice inside the paper's own decoder rather than a property of the converted dataset, and kept the native resolution so the target-format requirement of uniform bins is satisfied at the highest available resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From `rec['envs'][session]` — the environment *name* string (`'square'`, `'o'`, `'t'`, `'u'`, `'rectangle'`, `'+'`, `'i'`, `'l'`, `'bit donut'`, `'glenn'`) — which is passed through a local re-implementation of the reference repo's `get_env_mat` to produce a 3×3 accessibility matrix (1 = open, 0 = blocked) flattened to a 9-vector. The `blocked` variable, which lists the blocked grid indices per session directly, is loaded in the file but **never read** by the conversion code.

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    ...
    elif env == 'glenn':
        return np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1]], dtype=np.float32)
    raise ValueError(f'Unknown environment: {env}')
...
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
...
'input_names': [f'geom_bin_{i}' for i in range(9)],
```

iii. Trajectory step 329: "We now have the explicit reference mapping from environment names to 3x3 occupancy matrices via `get_env_mat`… The matrices use 1 for accessible/open partitions and 0 for blocked partitions. This means we can derive the decoder input directly from `envs` via `get_env_mat` or equivalent logic, rather than relying on the more awkward `blocked` list; `blocked` can serve as a sanity check." CONVERSION_NOTES Step 5 states the input is "derived primarily from `envs` via the reference `get_env_mat` mapping and cross-checked against `blocked`".

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `get_env_mat(env_name)` returns a 3×3 float matrix, `.reshape(-1)` flattens it row-major to a 9-dim `float32` vector, and a `.copy()` of that vector is stored once per trial (static within a session, as the Decoder Task requires). No normalization or other transform. Two problems with the implementation:

- The **`'glenn'` matrix is wrong**. The AI wrote `convert_data.py` (trajectory step 328) *before* the terminal had displayed the reference `get_env_mat`'s `glenn` branch, and never went back to reconcile it. Its `glenn` = `[[1,1,1],[1,0,0],[1,0,1]]` marks 3 tiles blocked, whereas the data's `blocked` entry for every `glenn` session in every animal is `[0, 8]` — 2 tiles blocked. The vector is simply not the geometry the mouse experienced. `glenn` occurs once per 10-environment cycle, so ~20 of 207 sessions carry a fabricated geometry vector.
- The `get_env_mat` convention is **vertically flipped** relative to the tile indexing used for the output position bins. Empirically (occupancy histogram per session), `blocked` index *i* corresponds exactly to output bin `i = 3*y_bin + x_bin` — tiles listed in `blocked` have exactly 0.000 occupancy. `get_env_mat` matrices agree with `flipud(blocked_mask)`, so for the asymmetric geometries `t`, `l` and `bit donut` the AI's `geom_bin_i` does **not** refer to the same tile as `position_bin_i` (e.g. for `t`, AI marks tiles {0,2,3,5} blocked while the animal is actually excluded from {3,5,6,8}).
- The planned cross-check against `blocked` was never implemented or run; the only documented input check is that the `square` vector is all ones, which is invariant to both bugs. A visible symptom went unexamined: the verification log reports `geom_bin_1: [1.0, 1.0]`, i.e. a dimension that is constant across the entire dataset (with the correct encoding, tile 1 is blocked in the `l` geometry and would vary).

The encoding is still *injective* across the 10 geometries — all 10 vectors are distinct — so it still identifies the environment to the decoder; it just misdescribes which tiles are blocked in 4 of 10 geometries.

ii.
```python
env_vec = get_env_mat(env_name).reshape(-1).astype(np.float32)
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    ...
    input_trials.append(env_vec.copy())
```

iii. CONVERSION_NOTES Step 10, check 3: "Reference code comparison: environment input mapping now follows reference `get_env_mat`". Step 10, check 2 lists as the input sanity check only: "square environment input is `[1,1,1,1,1,1,1,1,1]`". The intended justification (Step 5) was that `get_env_mat` is the reference repo's own canonical name → geometry mapping, so reusing it guarantees consistency with the paper; the flip convention (the reference also exposes `get_environment_label(env_name, flipud=False)`) and the `glenn` entry were never verified against `blocked`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Solely `rec['position'][session]`, an array of shape `(2, n_frames)` giving DeepLabCut-tracked `(x, y)` in cm within the 75 × 75 cm arena.

ii.
```python
positions = rec['position']
...
nt, it, ot = session_to_trials(traces[s], positions[s], env_name)
...
pos_bins = discretize_position_3x3(pos_sxn)
```

iii. CONVERSION_NOTES Step 3: "Position was tracked with DeepLabCut"; Step 2 documents `position` as "numpy array of x/y trajectories, shape `(n_sessions, 2, n_frames_max)`". Step 5 maps `rec['position'][session]` → `output[time]`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are truncated to the kept frames, cast to `float32`, clipped into `[0, 75 − 1e-6)`, converted to integer x/y bin indices, combined into a single class label `3*y_bin + x_bin`, and shaped to `(1, n_timepoints)` per trial as a single categorical output variable with 9 values. No smoothing, no speed filtering, no interpolation, no removal of immobility periods.

ii.
```python
def discretize_position_3x3(position_xy, arena_size=75.0):
    x = np.clip(position_xy[0], 0, arena_size - 1e-6)
    y = np.clip(position_xy[1], 0, arena_size - 1e-6)
    xb = np.floor((x / arena_size) * 3).astype(int)
    yb = np.floor((y / arena_size) * 3).astype(int)
    xb = np.clip(xb, 0, 2)
    yb = np.clip(yb, 0, 2)
    return (yb * 3 + xb).astype(np.int64)
...
    output_trials.append(pos_bins[sl][None, :])
...
'output_names': ['position_bin'],
'output_values': [[f'bin_{i}' for i in range(9)]],
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "Use time-varying 3x3 discretized position labels from x/y coordinates, matching the task requirement and the paper's conceptual partition." The Step 3 notes quote the methods: "partitioned an open square (75 × 75 cm) into a 3 × 3 grid space", so the 75 cm arena extent and the 3 × 3 partition both come from the paper rather than from the empirical coordinate range. (The raw coordinates do in fact span exactly [0, 75] with no NaNs, so the clipping is a no-op guard except at exactly x = 75 or y = 75, which the `−1e-6` shift maps into the last bin.)

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is cut at the two internal edges 25 cm and 50 cm into three equal 25 cm bins by floor division of the normalized coordinate; the two axis bins are combined row-major into one 9-class label `3*y_bin + x_bin` (`bin_0` … `bin_8`). A value falling exactly on an edge goes to the upper bin; the arena's far edge (75) is forced into bin 2. The resulting class distribution over the full dataset is {0.100, 0.099, 0.135, 0.075, 0.057, 0.077, 0.116, 0.141, 0.200}, i.e. non-uniform but every class well populated — expected, since blocked tiles contribute zero occupancy in the non-square geometries.

ii.
```python
xb = np.floor((x / arena_size) * 3).astype(int)
yb = np.floor((y / arena_size) * 3).astype(int)
xb = np.clip(xb, 0, 2); yb = np.clip(yb, 0, 2)
return (yb * 3 + xb).astype(np.int64)
```

iii. The Decoder Task specifies "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying." The AI used equal-width bins over the paper's stated 75 cm arena extent. CONVERSION_NOTES Step 10 check 2 documents the verification: converted output "exactly matches direct 3x3 discretization of raw position (`np.allclose=True`)" when recomputed from the raw file outside the conversion code.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame. Position and trace share the same 30 Hz frame index in the source arrays (same `n_frames` per session), are truncated to the same `n_keep = n_trials * 1800` frames, and every trial uses the same `slice` object for both streams, so output bin *k* of trial *t* is the position at exactly the frame of neural column *k* of trial *t*. No lag, shift, or causal offset is introduced.

ii.
```python
n_keep = n_trials * FRAMES_PER_TRIAL
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_sxn   = pos_sxn[:, :n_keep].astype(np.float32, copy=False)
pos_bins  = discretize_position_3x3(pos_sxn)
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    output_trials.append(pos_bins[sl][None, :])
```

iii. The methods state the behavioural and imaging streams were acquired simultaneously at 30 Hz on one DAQ and timestamped for post-hoc alignment, so the released arrays are already co-registered and no re-alignment is needed. The AI's Step 10 sanity checks reload the raw file and confirm both the neural slice and the discretized position slice of session 0 / trial 0 match the converted arrays exactly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three mechanisms: (1) neurons not tracked in a given session appear as all-NaN rows and are dropped per session via `np.all(np.isfinite(...), axis=1)`, giving variable `n_neurons` per session (113–564) with a matching per-session `brain_region_idx`; (2) any trailing frames where position is not fully finite or where no neuron is finite are excluded by `infer_valid_frames` before trialization; (3) sessions left with 0 neurons or < 2 trials are skipped entirely. Unknown environment names raise a `ValueError` rather than being silently encoded. The partial trial at the end of each session is discarded (up to 59.9 s per session, ≤ 0.04 % of frames for the 71,866-frame animals). The `blocked == -1` sentinel for the unblocked square is never encountered because `blocked` is not used.

ii.
```python
def infer_valid_frames(position, trace):
    pos_valid = np.all(np.isfinite(position), axis=0)
    neural_valid = np.any(np.isfinite(trace), axis=0)
    valid = pos_valid & neural_valid
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return 0
    return int(idx[-1] + 1)
...
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)
trace_sxn = trace_sxn[finite_neurons]
if trace_sxn.shape[0] == 0:
    return [], [], []
...
    raise ValueError(f'Unknown environment: {env}')
```

iii. The NaN handling was driven by a concrete failure: the first `--verify-only` run on sample data reported NaN/Inf in the neural data of session 61, and the AI diagnosed it as cross-session cell registration padding (trajectory step 336), choosing removal over imputation because "removal is more consistent with session-specific recorded neurons". The `infer_valid_frames` guard was written defensively for possible tracking dropouts; Step 10 confirmed it is inert here (71,866 of 71,866 frames valid).

## 6-a. What are the most time-consuming steps of the code?

i. Full conversion took 291 s (`elapsed_sec 291.051`), which is well inside the 15-minute budget, so no optimization was pursued. The dominant costs are I/O-bound: (a) `joblib.load` of each per-animal file, which decompresses the *entire* animal dict (including `SFPs`, `centroids` and `maps`, which are never used — for QLAK-CA1-08 the `SFPs` array alone is 515 × 35 × 35 × 31 ≈ 155 M float64); and (b) the final `pickle.dump` of the ~20 GB `converted_data.pkl`. Inside the per-session work, the two full-array `np.isfinite` passes over the `(n_neurons, 71866)` trace dominate; the trial loop itself is trivial. The script instruments only total wall-clock time — no per-step timers — and the "Run Time Estimates" and "Code inefficiencies identified" tables in CONVERSION_NOTES were left as unfilled template placeholders.

ii.
```python
t0 = time.time()
data = convert_dataset(sample=sample)
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
...
print('elapsed_sec', round(time.time() - t0, 3))
```

iii. No explicit justification is given in CONVERSION_NOTES; the Step 6/Step 7 efficiency sections were never filled in. The implicit position is that a ~5-minute single-pass conversion needs no profiling.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop in `session_to_trials` is the only avoidable loop. Since trials are fixed-length contiguous chunks, `trace_sxn[:, :n_keep].reshape(n_neurons, n_trials, 1800)` (plus `np.array_split`/`transpose`) would produce all trials in one operation, and the output labels likewise via `pos_bins.reshape(n_trials, 1, 1800)`. `input_trials` is a list of 39–40 `.copy()` calls of the same 9-element vector, which could be a single shared array (or built once with a list multiplication). The outer loops over animals and sessions are inherently serial I/O and are not worth vectorizing; the animal loop is, however, embarrassingly parallel and could have been run with `joblib.Parallel`, which the instructions explicitly suggest.

ii.
```python
for t in range(n_trials):
    sl = slice(t * FRAMES_PER_TRIAL, (t + 1) * FRAMES_PER_TRIAL)
    neural_trials.append(trace_sxn[:, sl])
    input_trials.append(env_vec.copy())
    output_trials.append(pos_bins[sl][None, :])
```

iii. Not discussed in CONVERSION_NOTES — the Step 6 "Code speedups added" section is an unfilled placeholder. The loop is cheap because the appended arrays are views, so no measurable penalty results in practice.

## 6-c. What processing does the code repeat multiple times?

i. The finiteness of the trace array is computed twice over essentially the whole session array: once inside `infer_valid_frames` as `np.any(np.isfinite(trace), axis=0)` over all `(n_neurons, n_frames)` elements, and again immediately afterwards as `np.all(np.isfinite(trace_sxn), axis=1)` over the truncated copy. One pass (`np.isfinite(trace)` reduced along both axes) would give both answers. Similarly, `np.all(np.isfinite(position), axis=0)` is computed for a stream that turns out to contain no NaNs at all. The static 9-element environment vector is regenerated by `get_env_mat` once per session and then copied once per trial (~8,187 redundant copies overall). The float32 cast is applied to the whole session before slicing, which is correct, but the truncation `[:, :n_keep]` plus `astype` materializes a second full-size copy of each session's trace in memory.

ii.
```python
valid_frames = infer_valid_frames(pos_sxn, trace_sxn)   # np.isfinite over the whole trace
...
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)
finite_neurons = np.all(np.isfinite(trace_sxn), axis=1)  # np.isfinite over the whole trace again
```

iii. Not discussed in CONVERSION_NOTES. The duplicated scan is a side effect of `infer_valid_frames` having been written first (for frame validity) and the neuron filter having been bolted on later in response to the NaN verification error.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items:
- **Unused data loaded.** `joblib.load` deserializes the whole per-animal dict; `SFPs`, `centroids` and `maps` (hundreds of MB per animal) are decompressed and immediately discarded. Reading only `trace`/`position`/`envs` — as the `.mat`/`h5py` route does lazily — would avoid this entirely.
- **`infer_valid_frames`** performs a full-array finiteness scan whose result is always the full session length for this dataset, and whose position component is redundant with the fact that position has no NaNs.
- **Storage format.** The traces are binary {0, 1} with ~0.7 % ones but are stored as dense `float32`, producing a 20 GB pickle. `bool`/`int8` (5 GB) or a sparse representation (~150 MB) would carry identical information; the decoder casts to float anyway.
- **Per-trial duplication of the static input**, and `brain_region_idx` stored as `int64` for a single-region dataset.

Separately, the required `--show-processing` plotting mode was stubbed out and never implemented (`print('show-processing requested; plotting not yet implemented')`), so no processing plots exist; `README.md` was never written, and the full decoder training run (`train_decoder_full_out.txt`) stops at "Using device: cuda" — it never produced full-dataset accuracies, so CONVERSION_NOTES Steps 11–13 remain unfilled and only the 2-animal sample accuracies (train 0.5619 / validation 0.4852 balanced accuracy vs 0.1111 chance) are documented.

ii.
```python
rec = joblib.load(Path('data') / animal)[animal]   # loads SFPs, centroids, maps too
...
trace_sxn = trace_sxn[:, :n_keep].astype(np.float32, copy=False)   # binary data as float32
...
    if args.show_processing:
        print('show-processing requested; plotting not yet implemented')
```

iii. No justification is offered; the relevant CONVERSION_NOTES sections are unfilled template placeholders. The joblib route was chosen for fidelity to the reference repo's own `load_dat`, which implicitly accepts loading the unused fields; `float32` was chosen for "decoder compatibility" as noted in the target-format discussion.
