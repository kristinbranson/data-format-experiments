# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven per-animal **joblib** files (the extension-less files `QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`, `QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75` in `/app/data/`), not the equivalent `.mat` (HDF5) files. The animal list is hard-coded in `main()`, matching the animal list in the reference repo's `main.py`. Each file unpickles to `{animal_id: {SFPs, blocked, centroids, envs, maps, position, trace}}`; the whole dict is loaded into memory and only `trace`, `position` and `envs` are used. All 207 recording days across the 7 animals are processed; each day is split into 1-minute trials. Memory is released per animal with `del dat, d`.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
...
trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()  # (n_days,) string array
n_days, n_cells_total, n_frames_total = trace.shape
```
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for animal in animals_to_process:
    subject_id = animals.index(animal)
    result = process_animal(animal, data_dir, trial_duration_frames, show_processing=show)
```

iii. From CONVERSION_NOTES Step 1/Step 10-Check 3: the reference repo's own loader `load_dat(animal, p, format='joblib')` reads exactly these joblib files, so "same joblib loading" as the reference code. Step 2 documents the dict layout (`trace` (n_days, n_cells, n_frames) binary, `position` (n_days, 2, n_frames) in cm, `envs` (n_days,) shape names). Loading is timed and reported (~9–23 s/animal, 215 s total) and the resulting counts (7 animals, 207 sessions, 5,413 unique registered cells, 69,744 cell-sessions) were cross-checked against the paper.

## 1-b. How are the data split into subjects?

i. One subject per data file / animal ID. `subjects` is the hard-coded list of all 7 animal names; `subject_idx` for each session is the index of that animal in the list.

ii.
```python
subjects = animals  # All 7 animals are subjects
...
subject_id = animals.index(animal)
...
    subject_idx_list.append(subject_id)
...
'subjects': subjects,
'subject_idx': np.array(subject_idx_list, dtype=int),
```

iii. CONVERSION_NOTES Step 2/3: the data contain 7 animals (QLAK-CA1-{08,30,50,51,56,74,75}), matching the paper's "7 mice". Each file holds all of one animal's recording days, so file ⇒ subject. Verified against the paper: 31 sessions for six animals and 21 for QLAK-CA1-51.

## 1-c. How are the data split into sessions?

i. One session = one recording day = one index along the first axis of `trace`/`position`/`envs`. All days are kept (31, 31, 31, 21, 31, 31, 31 = 207 sessions). In `--sample` mode only the first 2 days of the first animal are kept (although all 31 days are processed first and then truncated).

ii.
```python
for day in range(n_days):
    trace_day = trace[day]       # (n_cells, n_frames)
    pos_day = position[day]      # (2, n_frames)
    env_name = str(envs[day])
    ...
    sessions_neural.append(trials_neural)
```
```python
n_sessions = len(result['neural'])
if max_sessions is not None:
    n_sessions = min(n_sessions, max_sessions)
for s in range(n_sessions):
    all_neural.append(result['neural'][s]); ...
```

iii. CONVERSION_NOTES Step 5 ("One session = one decoder session: Each recording day is a separate session in our output"). Each day is one ~40-min recording in one environment geometry, and the paper reports 207 sessions; the converted data reproduce 207 sessions exactly (verification_full_out.txt).

## 1-d. How are the data split into trials?

i. Each session is cut into consecutive, non-overlapping 1-minute trials of 1800 frames (30 Hz × 60 s), starting at frame 0. The trailing partial minute (~1,666–1,819 frames) is discarded. This yields 39 or 40 trials/session and 8,187 trials in total.

ii.
```python
fps = 30                      # recording frame rate
trial_duration_sec = 60       # 1 minute trials
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
n_trials = n_frames_total // trial_duration_frames
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input  = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The Decoder Task specifies "long recording sessions, which will be split into 1-minute trials within each session"; CONVERSION_NOTES Step 5 states 1 min = 1800 frames at 30 Hz, "~40 trials per session (last partial trial discarded if < 1800 frames)". Step 10 Check 5 verified trial boundaries (no gaps/overlap; last frame of trial 0 = frame 1799, first of trial 1 = frame 1800) and noted "~1666 frames discarded per session (last partial minute) — acceptable".

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. Every complete 1800-frame segment of every session of every animal is kept; only the incomplete final segment is dropped. The paper's decoding-specific curation (running-speed threshold v > 5 cm/s, cells with > 5 events) was explicitly considered and *not* applied.

ii. There is no filtering code; the only exclusion is the floor division that drops the remainder:
```python
n_trials = n_frames_total // trial_duration_frames
```

iii. CONVERSION_NOTES Step 3 ("Trial curation: None in original (1 session = 1 day). We split into 1-min trials") and Step 5 Key Decision 3: "No velocity filtering: The paper's velocity filter was for decoding within their pipeline; we let the decoder handle this". Keeping all timepoints is also required here because the decoder needs continuous, equal-length trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely from `trace`, the authors' binarized calcium-event matrix, `(n_days, n_cells, n_frames)`, with `NaN` rows for cells not registered on that day. Values are {0, 1} (1 = significant rising-phase calcium transient). No other field (`maps`, `SFPs`, `centroids`) feeds the neural stream.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]   # (n_cells, n_frames)
```

iii. CONVERSION_NOTES Step 1: "Neural data: Binary trace vector (1 = significant rising-phase event, 0 = no event). Already preprocessed"; "No delta F/F needed: The trace data is already binarized". Step 4 verified the data are indeed binary {0,1}. The rate maps in `maps` are derived products used by the paper's analyses, not raw activity, so they were not used.

## 2-b. How is the `neural` data processed?

i. Essentially no processing: rows of unregistered (all-NaN) cells are dropped, any residual NaN is replaced by 0 as a safety net, the array is kept in (neurons, time) orientation at the native 30 Hz, and each trial slice is cast to `float32`. No ΔF/F, deconvolution, smoothing, z-scoring, or rate conversion is applied. All registered cells are kept (no place-cell selection). `brain_region_idx` is all-zeros (single region, `CA1`).

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]  # (n_active, n_frames)
n_active = active_mask.sum()
# Replace any remaining NaN with 0 (shouldn't happen for active cells, but safety)
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Use raw binary trace (0/1 events) at native 30 Hz — NO additional processing needed. Trace is already binarized by the original authors... All registered cells included (no place cell filtering, matching paper's approach)" (the paper "motivated the inclusion of all cells in subsequent analyses"). The AI briefly stored the neural data as `int8` (5 GB file) but reverted to `float32` (20 GB) because `train_decoder.py` emitted dtype-conversion warnings (trajectory steps 103–109).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level quality control is removal of cells not registered on that day, identified as rows whose trace is entirely NaN (CellReg cross-day registration leaves NaN for unregistered cells). This is done per session, so neuron counts vary by session (113–564, mean 336.93; 69,744 cell-sessions in total). No event-count, SNR, or place-cell criterion is applied.

ii.
```python
# A cell is registered if its trace is not all NaN
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
n_active = active_mask.sum()
```

iii. CONVERSION_NOTES Step 3 "Neuron curation: Unregistered cells (NaN) excluded per day"; Step 4 notes the paper's place-cell threshold is irrelevant here because all cells are used. The resulting total of 69,744 active cell-sessions was checked against the paper's 69,744 rate maps, and per-animal registered-cell counts (515…952, mean 773) match the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the recordings are continuous 40-minute free-foraging sessions. The AI defines the alignment event as the start of each 1-minute segment: trials tile the session from frame 0, with `off_start = 0.0 s` and `off_end = 60.0 s`, and documents this in `metadata`. Neural, input and output streams are cut with identical frame indices, so they are aligned frame-for-frame by construction.

ii.
```python
'metadata': {
    'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric deformation task',
    'time_bin_size': 1000.0 / fps,  # ~33.33 ms
    'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
    'off_start': 0.0,
    'off_end': float(trial_duration_sec),
    ...
}
```
```python
start = trial_idx * trial_duration_frames
end = start + trial_duration_frames
trial_neural = active_trace[:, start:end]...
trial_output = bin_ids[start:end]...
```

iii. CONVERSION_NOTES Step 3/5: "All streams synchronous at 30 Hz" — position (DeepLabCut) and trace are sampled on the same imaging frames, so no resampling or lag correction is needed. Step 10 Check 5 verified trial-boundary alignment (trial 0 ends at frame 1799, trial 1 starts at frame 1800; last trial's last timepoint = frame 70199) and the `--show-processing` plots show raw position vs. discretized output for the same frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging resolution is preserved: 30 Hz, i.e. one bin per frame, `time_bin_size = 1000/30 ≈ 33.33 ms`. No rebinning, downsampling, smoothing or sliding-window averaging is applied, so every trial is exactly 1800 bins and all trials/sessions share the same bin size. (The reference paper's Gaussian Naive Bayes decoder used a temporal bin size of 3 frames; the AI did not carry that over.)

ii.
```python
fps = 30  # recording frame rate
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
'trial_duration_frames': trial_duration_frames,
```

iii. CONVERSION_NOTES Step 5: "Time bin size = 1/30 s ≈ 33.33 ms (raw frame rate)"; Step 10 Check 3(d): "Binning — No additional binning of neural data / Same — binary trace used directly" as the reference. Verification output confirms T = 1800 for every trial (mean/median/min/max = 1800).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From the per-day environment name in `envs` (one of `square, o, t, u, rectangle, +, i, l, bit donut, glenn`), converted to a 3×3 binary accessibility matrix with `get_env_mat()`, copied verbatim from the reference repo (`georepca1/src/utils.py`). The AI did **not** use the `blocked` field (the explicit list of blocked partition indices per day) that is also present in the data.

ii.
```python
def get_env_mat(env):
    """... From reference code: georepca1/src/utils.py"""
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
env_name = str(envs[day])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. CONVERSION_NOTES Step 1: "`get_env_mat()` returns 3x3 binary matrix: 1=accessible partition, 0=blocked. This will be our decoder input"; Step 5 Key Decision 5: "Environment geometry as input: Task says 'Environment geometry to represent which part of the arena is blocked'". Using the reference repo's own function is the AI's justification for matching reference processing (Step 10 Check 3(e): "Same function from reference code"). The `envs` names and `blocked` indices are redundant encodings of the same 10 geometries.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3×3 matrix is flattened row-major to a 9-element vector, cast to `float32`, and attached unchanged to every trial of the session (static per trial, shape `(9,)`, no time axis). Polarity is 1 = accessible, 0 = blocked (the inverse of a "blocked" one-hot). `input_names` are `partition_0 … partition_8`. The value depends only on the session's environment name, so it is constant within a session and identical across animals/repeats of the same geometry.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
trial_input = env_mat.astype(np.float32)
...
input_names = [f'partition_{i}' for i in range(9)]
```

iii. CONVERSION_NOTES Step 5: "Environment geometry as flattened 3x3 binary matrix (9 values); Static per trial (doesn't change within a trial); Shape: (9,) per trial — no time dimension since static". Sanity checks reported in Step 7/10: "Square = all 1s, O = center blocked, T = top/sides blocked — all MATCH `get_env_mat()`", and "the 'o' environment correctly has 0% in position bin 4 (center blocked)".

*Judge's note (verified independently against the raw data):* `get_env_mat`'s row/column convention is rotated relative to the AI's own position-bin indexing. E.g. for `t` the flattened env matrix is 0 at indices {0,2,3,5}, whereas the partitions the mouse never visits under the AI's `x_bin*3 + y_bin` labelling are {1,2,7,8}; the data's own `blocked` field for that day is {3,5,6,8}. I checked all 10 geometries: the mapping is a single fixed relabelling (env-matrix element (i,j) ↔ position bin `j*3 + (2-i)`) applied consistently to every session, so no information is lost and the encoding still identifies the geometry uniquely — but `input` element *i* does **not** refer to the same arena partition as `output` class *i*, despite the `partition_i` naming. The AI's sanity check only exercised `square` and `o`, which are invariant under that rotation, so it gave false assurance.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. From `position`, shape `(n_days, 2, n_frames)`: x-y coordinates (cm, range [0, 75]) tracked with DeepLabCut at 30 Hz, same frames as the neural data.

ii.
```python
position = d['position'] # (n_days, 2, n_frames)
...
pos_day = position[day]  # (2, n_frames)
bin_ids = discretize_position_3x3(pos_day)
```

iii. CONVERSION_NOTES Step 2/3: "position: shape (n_days, 2, n_frames), x-y coordinates in [0, 75] cm", "Position: DeepLabCut, 30 Hz, [0,75] cm". It is the only behavioural variable in the dataset and is exactly what the Decoder Task asks to be decoded.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous (x, y) trace is discretized to one of 9 spatial bins for the whole day at once, then sliced per trial to shape `(1, 1800)` and cast to `int64` (integer so it can index `output_values`). No smoothing, interpolation, velocity filtering or masking of blocked partitions is applied; the single output variable is named `position` with 9 value labels.

ii.
```python
bin_ids = discretize_position_3x3(pos_day)  # (n_frames,)
...
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
...
output_bin_names = []
for i in range(3):
    for j in range(3):
        output_bin_names.append(f"x[{i*25}-{(i+1)*25}]_y[{j*25}-{(j+1)*25}]")
'output_names': ['position'],
'output_values': [output_bin_names],
```

iii. CONVERSION_NOTES Step 5: "Position (x,y) in [0, 75] cm → 3x3 grid of 25 cm bins ... Output is time-varying (1, n_timepoints) per trial", following the Decoder Task requirement of "3 x 3 = 9 spatial bins. Time-varying." The `int64` cast was added after `train_decoder.py --verify-only` failed to index `output_values` with float32 labels (trajectory steps 82–86).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is split into 3 equal 25 cm bins over the 75 cm arena using `floor(coord / 25)`; coordinates are clipped to `[0, 75 − 1e-10]` (so the exact value 75.0 falls in bin 2) and bin indices are clamped to [0, 2]. The two axes are combined row-major as `bin_id = x_bin * 3 + y_bin`, giving classes 0–8.

ii.
```python
def discretize_position_3x3(position, env_size=75.0):
    bin_size = env_size / 3.0
    x = np.clip(position[0], 0, env_size - 1e-10)
    y = np.clip(position[1], 0, env_size - 1e-10)
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    x_bin = np.clip(x_bin, 0, 2)
    y_bin = np.clip(y_bin, 0, 2)
    bin_ids = x_bin * 3 + y_bin
    return bin_ids
```

iii. CONVERSION_NOTES Step 5: "Bin edges: [0, 25, 50, 75] for both x and y; Combine x_bin and y_bin into single label: bin_id = x_bin * 3 + y_bin (0-8); This gives 9 spatial bins matching the 3x3 partition structure." The 25 cm bins are chosen to coincide with the experiment's 3×3 partition layout. Step 10 Check 2 verified 4 individual frames by recomputing the bin by hand from raw x/y; the full-data distribution spans all 9 bins (0.057–0.200).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame: `position` and `trace` come from the same 30 Hz imaging frames of the same day and have identical frame counts, so the position bins are sliced with exactly the same `start:end` indices as the neural matrix. No lag, shift, or resampling is introduced; every trial has 1800 neural bins and 1800 output bins.

ii.
```python
bin_ids = discretize_position_3x3(pos_day)          # full session, (n_frames,)
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. CONVERSION_NOTES Step 3 ("All streams synchronous at 30 Hz") and Step 10 Checks 2/5 (boundary spot-checks at frames 1799/1800 and at the last trial's final frame 70199). The `--show-processing` figure plots raw x/y and the discretized output on a common time axis to demonstrate no misalignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled: (a) cells not registered on a given day are all-NaN and are dropped from that session; (b) any residual NaN inside a retained cell's trace is replaced by 0 via `np.nan_to_num` as a defensive measure; (c) position values are clipped into `[0, 75)` before binning, which absorbs the few samples recorded at exactly 75.0 (or marginally above, 75.00000000000001) and any out-of-arena tracking excursions. The incomplete final minute of each session is silently discarded. There is no explicit handling of NaN positions or of sessions with unequal stream lengths (neither occurs in this dataset — I verified position contains no NaNs and all days within an animal have identical frame counts).

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
# Replace any remaining NaN with 0 (shouldn't happen for active cells, but safety)
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
...
x_bin = np.clip(x_bin, 0, 2); y_bin = np.clip(y_bin, 0, 2)
```

iii. CONVERSION_NOTES Step 3/5: NaN in `trace` marks cells not registered on that day by CellReg, so they are excluded per day rather than zero-filled; the `nan_to_num` is documented in-code as a safety net. Step 10 Check 5 covers edge cases (trial boundaries, subject boundaries) and explicitly accepts the ~1,666-frame remainder loss per session.

## 6-a. What are the most time-consuming steps of the code?

i. Reading and decompressing the joblib files dominates: 8.9–22.6 s per animal (≈124 s of the 215 s total conversion), versus 0.15–0.69 s per session for all the actual processing (masking, discretizing, slicing). Writing the 20 GB pickle takes a further 14 s. The AI instruments and prints all of these timings. Because the whole joblib dict is loaded, the fields it never uses (`SFPs`, `maps`, `centroids`, `blocked`) are part of that I/O cost.

ii.
```python
t0 = time.time()
print(f"Loading {animal_name}...", flush=True)
dat = joblib.load(os.path.join(data_dir, animal_name))
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s", flush=True)
...
dt = time.time() - t_day
print(f"  Day {day}/{n_days}: {env_name}, {n_active} active cells, {n_trials} trials, {dt:.2f}s")
...
print(f"Saved in {time.time() - t_save:.1f}s")
```

iii. CONVERSION_NOTES Step 7 gives the estimate ("Load 1 animal ~13s; Process 31 sessions ~9s; Estimated full (7 animals) ~3-4 min. Well under 15-minute threshold, no optimization needed"), and Step 9 confirms the measured 215 s. Step 6 lists the efficiency measures taken: "Direct numpy array slicing for trial splitting (no loops over frames); Memory freed after each animal with `del`; Vectorized position discretization".

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python loops are the per-animal loop, the per-day loop and the per-trial loop. Only the per-trial loop is a candidate: the 39–40 slice-and-cast operations per session could be replaced by a single reshape of the truncated array (`active_trace[:, :n_trials*1800].reshape(n_active, n_trials, 1800)` plus one `.astype(np.float32)`), and `bin_ids` likewise. The gain would be small (the per-trial work is already a contiguous slice; the float32 copy has to happen regardless), which is consistent with the measured ~0.2–0.7 s per session. The genuinely hot inner operations — the all-NaN mask over (n_cells, n_frames) and the position discretization over the full session — are already vectorized numpy over the whole day.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)     # vectorized
bin_ids = discretize_position_3x3(pos_day)             # vectorized over whole session
```

iii. CONVERSION_NOTES Step 6 claims the loops that matter are already vectorized ("Direct numpy array slicing for trial splitting (no loops over frames)… Vectorized position discretization"), and Step 7 justifies not optimizing further because the projected runtime (~3–4 min) was far under the 15-minute threshold in the instructions.

## 6-c. What processing does the code repeat multiple times?

i. Little is recomputed: `get_env_mat` and `discretize_position_3x3` are each called once per session, not per trial. What *is* repeated is per-trial materialization of the same static input — `env_mat.astype(np.float32)` runs inside the trial loop and creates a separate 9-element array for each of the 8,187 trials instead of reusing one object (the reference does `[blocked] * n_trials`). Negligible in time, though it does make 8,187 copies in the pickle. In `--sample` mode the code also processes all 31 days of the first animal and then keeps only the first 2 — a repeat of work that is immediately thrown away. `plot_processing` re-plots for every animal when `--show-processing` is used in full mode rather than for "up to 2 sessions" as the instructions ask.

ii.
```python
for trial_idx in range(n_trials):
    ...
    trial_input = env_mat.astype(np.float32)   # re-created per trial
```
```python
result = process_animal(animal, data_dir, trial_duration_frames, show_processing=show)
n_sessions = len(result['neural'])
if max_sessions is not None:
    n_sessions = min(n_sessions, max_sessions)   # truncation happens after all days processed
```

iii. Not discussed as a problem in CONVERSION_NOTES; Step 6 asserts only that memory is freed per animal and that slicing/discretization are vectorized. The implicit justification is that these repeats cost negligible time relative to file I/O (sample mode still ran in ~22 s).

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `joblib.load` pulls the entire per-animal dict into memory — `SFPs` (35×35×n_cells×n_days), `maps` (15×15×n_cells×n_days), `centroids`, `blocked` — of which only `trace`, `position` and `envs` are used; the `.mat`/h5py route would have read only the needed arrays lazily. (2) The neural data are binary {0,1} but are stored as `float32`, making the pickle 20 GB where `int8` gave 5 GB (the AI implemented int8, measured the 4× saving, then reverted). (3) In `--sample` mode, 29 of 31 days are fully processed and discarded. (4) The static 9-element input is duplicated per trial (8,187 copies of the same vector). (5) Per-session statistics printed in the summary (`np.unique`, per-session neuron counts) are cosmetic. None of this changes the converted values.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal_name))   # loads SFPs/maps/centroids too
d = dat[animal_name]
trace = d['trace']; position = d['position']; envs = d['envs'].flatten()
```
```python
trial_neural = active_trace[:, start:end].astype(np.float32)   # binary data kept as 4-byte floats
```

iii. CONVERSION_NOTES Step 9 records the 20 GB file size without treating it as a problem; the trajectory (steps 101–109) shows the explicit trade-off: int8 reduced the file to 5 GB but `train_decoder.py` warned about dtype conversion, so the AI chose `float32` "to avoid these warnings and just accept the larger file size". The joblib route is justified in Step 10 Check 3 as matching the reference repo's own `load_dat(..., format='joblib')`, and `del dat, d` is used to bound peak memory.
