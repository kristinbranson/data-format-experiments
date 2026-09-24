# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven per-animal **joblib** files in `/app/data` (the extension-less files, e.g. `QLAK-CA1-08`), which is the format the reference code's `load_dat()` (utils.py:61) reads. The animal list is hard-coded. For each animal, `joblib.load` returns a dict keyed by the animal ID, from which it takes three fields: `trace` (n_days, n_cells, n_frames), `position` (n_days, 2, n_frames) and `envs` (n_days, 1). The whole animal dict (including the unused `maps`, `SFPs`, `centroids`, `blocked` fields) is loaded into memory at once; sessions (days) are then iterated in file order and each is split into trials. Parallel `.mat` (HDF5) copies of the same data exist in `/app/data` but are not used.

ii.
```python
DATA_DIR = 'data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    trace_all = dat[animal]['trace']        # (n_days, n_cells, n_frames)
    position_all = dat[animal]['position']  # (n_days, 2, n_frames)
    envs_all = dat[animal]['envs']          # (n_days, 1)
    n_days = trace_all.shape[0]
    for day in range(n_days):
        ...
        neural_trials, input_trials, output_trials, n_registered = process_session(
            trace_all[day], position_all[day], env_name)
```

iii. From CONVERSION_NOTES.md Step 10, Check 3(a): "Data loading: Using joblib.load() same as reference `load_dat()`." The AI identified `load_dat` as the reference loader in Step 1 and verified in Step 2/9 that the loaded data reproduce the paper's headline counts: 7 animals, 207 sessions (31/31/31/21/31/31/31), 5,413 unique neurons, 69,744 registered cell-days, ~72,000 frames/session at 30 Hz.

## 1-b. How are the data split into subjects (mice)?

i. One joblib file = one mouse. The seven animal IDs are hard-coded as `ANIMALS` and become `data['subjects']` verbatim; `subject_idx` records the animal index for each emitted session. Note that `subjects` is always the full 7-element list even in `--sample` mode, where only one animal is actually processed (the sample verification log therefore reports "Number of subjects: 7" with only one subject having sessions).

ii.
```python
subjects = list(animals)  # animal IDs as subject names
...
all_subject_idx.append(animal_idx)
...
'subjects': subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 2: each animal file "contains a dict with animal ID as key" holding all of that animal's days, so the file/ID is the natural subject identifier. Step 10 Check 5 verified the mapping ("Subject idx for first 5 sessions: [0 0 0 0 0] … session 31 → 1").

## 1-c. How are the data split into sessions?

i. Each recording **day** within an animal file is one session (`trace_all[day]`, `position_all[day]`, `envs_all[day]`), giving 6×31 + 1×21 = 207 sessions. A day is skipped only if it has zero registered cells (never triggered). Sessions are emitted in file order, animal by animal.

ii.
```python
n_days = trace_all.shape[0]
for day in range(n_days):
    env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name)
    if len(neural_trials) == 0:
        print(f"  Day {day}: skipped (no registered cells)")
        continue
    all_neural.append(neural_trials)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 6: "Sessions: Each day for each animal = 1 session. Total 207 sessions." This was cross-checked against the paper's "5,413 unique neurons across 207 sessions in 10 geometries" (Step 3/Step 9 consistency table).

## 1-d. How are the data split into trials?

i. Per the task instructions, each ~40-minute continuous session is cut into non-overlapping 1-minute trials. Because the AI temporally bins first (3 frames → 100 ms), a trial is 600 time bins = 1800 frames = 60 s. The number of trials is `n_timebins // 600`; the trailing partial trial is discarded. This yields 39 trials/session for animals with 71,866 frames and 40 for those with ≥72,000 frames (8,187 trials total).

ii.
```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS          # 1800
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
...
n_timebins_total = neural_binned.shape[1]
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    input_trial = env_flat.astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 2: "Split each 40-min session into 1-minute trials = 40 trials per session (discard remainder frames). Each trial = 1800 frames = 600 time bins after temporal binning." The AI explicitly noted the reference paper has no trial structure (continuous free exploration), so the 1-minute trial is a task-imposed segmentation, and checked the arithmetic (72,219 frames → 40 trials + 219 remainder frames) before writing the script.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is performed. All complete 60-s segments are kept. In particular, the AI deliberately did **not** apply the reference `decode_position_within` velocity filter (smoothed speed > 5 cm/s), because discarding immobility frames would destroy the fixed-length continuous trial structure required by the target format. Only the trailing incomplete segment of each session is dropped.

ii.
```python
# no filtering applied; every complete segment is appended
for t in range(n_trials):
    ...
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 7: "No velocity filtering: The task asks to decode position at all times, not just during movement." Step 10 Check 3(c): "Temporal alignment: No velocity filtering applied. Reference uses velocity>5cm/s for Bayesian decoder. We decode at all timepoints. Justified difference." The trajectory (step 17) shows the AI explicitly weighing "match the reference" against the decoder-format requirement before settling on this.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dat[animal]['trace']`, of shape (n_days, n_cells, n_frames). Values are the authors' binarized rising-phase calcium events (0/1), with `NaN` for cells not registered on that day. Only `trace` is used; `maps` (rate maps), `SFPs` and `centroids` are ignored.

ii.
```python
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
...
def process_session(trace_day, position_day, env_name):
    """trace_day: (n_cells, n_frames) binary trace, may contain NaN for unregistered cells"""
```

iii. CONVERSION_NOTES.md Step 1/Step 3: "Data is pre-processed: trace is already binarized rising-phase calcium events (0/1)"; "Binary vector treated as firing rate in all analyses". Because the authors already performed ΔF/F extraction and event detection, no further preprocessing of the fluorescence is needed (this also answers the instructions' "does ΔF/F need to be computed?" question — no).

## 2-b. How is the `neural` data processed?

i. Per session: (1) drop unregistered (NaN) cells; (2) `np.nan_to_num` any residual NaNs to 0; (3) Gaussian-smooth each cell's binary event train along time with σ = 3 frames; (4) temporally average-pool with kernel = stride = 3 frames, producing continuous rate estimates in 100 ms bins; (5) cast to float32 and slice into trials, leaving arrays of shape (n_neurons, 600). Smoothing and pooling are applied to the whole session before trial splitting. This reproduces the reference `fit_decoder` preprocessing (`gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` then `AvgPool1d(kernel_size=3, stride=3)`).

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)

smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                   sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()  # (n_registered, n_timebins)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: "Use reference code approach: Gaussian smooth trace with sigma=3, then AvgPool1d with kernel_size=3. This gives 100ms time bins at ~10 Hz." Step 10 Check 3(d): "Binning: Gaussian smooth sigma=3, AvgPool1d kernel=3. Matches reference `fit_decoder`. ✓". The AI verified (Step 10 sanity check 7) that the output is no longer binary (230 unique values in one sampled trial), and reproduced its own pipeline from the raw file to confirm an exact match (max diff 0.0).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level curation is removing cells that were not registered on that day (NaN trace). No place-cell (split-half reliability, p<0.01) filter and no "> 5 events while moving" activity threshold from `decode_position_within` are applied; all registered cells are kept. Implementation detail: registration is tested on the **first frame only** (`~np.isnan(trace_day[:, 0])`) rather than by an all-NaN test, although the docstring states the all-NaN rule. I verified on the raw files that the two criteria are identical for this dataset (e.g. QLAK-CA1-51 days 0–2: 441/411/406 NaN cells under both tests), and the AI's aggregate of 69,744 registered cell-days matches the paper's 69,744 rate maps.

ii.
```python
# A cell is registered if its trace is not all NaN
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]  # (n_registered, n_frames)
n_registered = registered_trace.shape[0]
if n_registered == 0:
    return [], [], [], 0
```

iii. CONVERSION_NOTES.md Step 5, Key Decisions 3 and 8: "Cells with NaN trace on a given day are excluded from that session… No place cell filtering: Use all registered cells to give the decoder maximum information." Step 10 Check 3(b) justifies dropping the reference's `cell_threshold > 5 events` rule as "specific to the Bayesian decoder", arguing an NN decoder can down-weight uninformative cells itself. The 69,744 total was used as the consistency check against the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the paradigm is 40 min of continuous free exploration. The AI therefore aligns trials to the start of each 1-minute segment (trial *k* = frames [k·1800, (k+1)·1800)) and documents this in metadata with `temporal_alignment_event = 'Start of 1-minute trial segment within 40-minute recording session'`, `off_start = 0.0`, `off_end = 60.0`. Neural, input and output are all cut with the same bin indices, so no relative shift is introduced.

ii.
```python
'metadata': {
    'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric environment exploration',
    'time_bin_size': TIME_BIN_MS,
    'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
    ...
}
```

iii. CONVERSION_NOTES.md Step 3 curation notes: "No explicit trial-level curation in the paper (sessions are continuous recordings)"; the 1-minute trial is imposed by the Decoder Task section of the instructions, so segment onset is the only meaningful alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the native 30 Hz (33.3 ms) data are rebinned to **100 ms** bins (3 frames per bin) by Gaussian smoothing (σ = 3 frames) followed by non-overlapping average pooling, giving 600 bins per 60-s trial. `metadata['time_bin_size'] = 100.0` ms. The same pooling is applied to the position stream so the streams stay bin-for-bin aligned.

ii.
```python
TEMPORAL_BIN_SIZE = 3  # frames per temporal bin (matches reference fit_decoder)
GAUSS_SIGMA = 3        # Gaussian smoothing sigma in frames (matches reference fit_decoder)
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600
```

iii. CONVERSION_NOTES.md Step 1 and Step 5: the reference decoder (`fit_decoder`, utils.py:1776) uses `temporal_bin_size=3`; the AI adopted the identical bin size "matches reference fit_decoder" so that its neural feature representation is the one the paper actually decodes from. It also noted the practical benefit: binary events at 30 Hz are extremely sparse, whereas the smoothed/pooled signal gives graded rate estimates (sanity check 7).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From `dat[animal]['envs']` — the per-day geometry **name** string ('square', 'o', 't', 'u', 'rectangle', '+', 'i', 'l', 'bit donut', 'glenn') — converted to a 3×3 binary matrix by a verbatim copy of the reference `get_env_mat()` (utils.py:215), in which 1 = partition present and 0 = partition omitted. The alternative `blocked` field (explicit indices of blocked partitions, present in the same files and used by the human reference) is not used.

ii.
```python
def get_env_mat(env):
    """Get binary 3x3 matrix for environment name. From reference code utils.py."""
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        ...
        'glenn':     np.array([[1,1,0],[1,1,1],[0,1,1]]),
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
env_mat = get_env_mat(env_name)
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "envs → get_env_mat() → input[0:9] … Reference Code Function: get_env_mat"; Step 10 Check 3(e): "Input construction: get_env_mat() copied from reference. ✓". The AI's rationale is that reusing the reference function guarantees the same geometry definitions as the paper.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3×3 matrix is flattened row-major into a 9-vector of floats (1 = open partition, 0 = blocked partition) and copied unchanged into every trial of the session, i.e. static per trial with shape (9,). Names are `env_00 … env_22`. Note the polarity is the inverse of the human reference (which one-hot-encodes *blocked* partitions as 1).

An orientation issue I verified against the raw data: `get_env_mat` returns the geometry in the paper's plotting orientation, which is vertically flipped relative to the coordinate frame of the `position` data (across all 10 geometries, `blocked` indices == zeros of `np.flipud(get_env_mat(env))`). Combined with the AI's `coord0*3 + coord1` output-bin ordering (4-c), input element *i* therefore does **not** correspond to output position bin *i* for 7 of the 10 geometries — e.g. for 'l' the input is 0 at {4,5,7,8} while the never-visited output bins are {3,4,6,7}; input dimension `env_01` is 1 in every session of the whole dataset, whereas the output bin that is never blocked is bin 7.

ii.
```python
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()  # (9,)
...
input_trial = env_flat.astype(np.float32)  # (9,) static per trial
...
input_names = ['env_00','env_01','env_02','env_10','env_11','env_12','env_20','env_21','env_22']
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 5: "Use get_env_mat() to convert environment name to 3x3 binary matrix. Flatten to 9 values. This is static per trial," matching the Decoder Task requirement that geometry be a static per-trial input. The AI planned a check "Verify environment geometry matches blocked field" but the executed check (Step 10, Check 2) only tested 'square' and 'o' — the two geometries that are invariant under the flip/transpose — and passed, so the misalignment was never caught.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `dat[animal]['position']`, shape (n_days, 2, n_frames), the DeepLabCut-tracked (x, y) coordinates of the mouse in centimetres in the 75 × 75 cm arena, sampled at the same 30 Hz as the imaging stream.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
...
def process_session(trace_day, position_day, env_name):
    """position_day: (2, n_frames) x,y position in cm"""
```

iii. CONVERSION_NOTES.md Step 2: "`position`: (n_days, 2, n_frames) - x,y position in cm (0-75)"; Step 3: "Position: DeepLabCut tracking". This is also the variable the reference `get_rate_maps`/`decode_position_within` use.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The (2, n_frames) coordinate stream is temporally average-pooled with the same `AvgPool1d(kernel_size=3, stride=3)` used for the neural data (so 100 ms bins), then discretized (see 4-c) and reshaped to (1, 600) per trial as int64. No smoothing, interpolation, velocity filtering or occupancy correction is applied. Position is never converted to a continuous regression target — the instructions require categorical outputs.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()  # (2, n_timebins)
...
output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)  # (1, 600)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: position is "Discretize[d] into 3x3 bins (0-8), temporally bin[ned] same as neural", so the two streams share an identical time base. Averaging the raw coordinates over 3 frames (100 ms) before binning follows the reference `fit_decoder`, which pools `behav` with the same kernel.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is divided by `(75 + 1e-5)/3 = 25.0000033` cm and truncated, giving three equal 25 cm bins ([0,25), [25,50), [50,75]), then clipped to [0,2]; the buffer ensures the boundary value exactly 75.0 (which occurs in the data) lands in bin 2 rather than 3. The two bin indices are combined as `pos_bins[0]*3 + pos_bins[1]` into a single 9-class label, and `output_values` labels these 'top-left' … 'bottom-right'.

The threshold locations are identical to the human reference's; the *flattening order* is the transpose of the reference's convention (reference rate maps and the `blocked` field use `coord1*3 + coord0`). This is inconsequential for decoding (a fixed relabelling of 9 classes) but means the human-readable 'top-left'…'bottom-right' names, and the correspondence with the geometry input discussed in 3-b, do not hold.

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS   # (75 + 1e-5)/3
pos_bins = (pos_binned_temporal / bin_size).astype(int)  # (2, n_timebins)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)      # ensure valid range
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]  # (n_timebins,)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 4 and "Position binning scheme": "Bin x,y position into 3x3 grid (each bin = 25x25 cm) … This matches the 3x3 partition grid of the environment," as required by the Decoder Task ("Mouse position discretized into 3 x 3 = 9 spatial bins"). The `buffer = 1e-5` device and the integer-division binning are lifted from the reference `get_rate_maps`, which uses `position // ((nanmax(position)+buffer)/n_bins)`; the AI substituted the known arena size 75 cm for the per-session maximum. The occupancy distribution was checked against the geometries (Step 7/9) and the verification log shows all 9 classes populated (0.057–0.200).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and calcium are acquired simultaneously at 30 Hz by the same DAQ and have identical frame counts, so they are aligned frame-for-frame in the source. The AI preserves this by applying the same `AvgPool1d(3,3)` to both streams and slicing both with the same bin indices `[t*600, (t+1)*600)`, giving (n_neurons, 600) and (1, 600) arrays that share a time base. No lead/lag is introduced (the Gaussian smoothing of the traces is symmetric/zero-phase).

ii.
```python
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
...
neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. CONVERSION_NOTES.md Step 3: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz … all recorded frames were timestamped," so no resampling or event-based re-alignment is needed. The `--show-processing` plots were produced to visually confirm no misalignment, and the high decoder accuracy (0.607 balanced vs 0.111 chance) is cited in Step 12 as evidence that the streams are synchronized.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases: (1) cells not registered on a day are all-NaN and are dropped (2-c); (2) any residual NaN in a registered cell is replaced by 0 via `np.nan_to_num` ("shouldn't happen but be safe"); (3) sessions whose registered-cell count is 0 are skipped entirely with a printed message (never triggered — the minimum is 113 neurons); (4) variable session lengths (71,866–72,219 frames instead of the nominal 72,000) are handled by integer division, discarding the trailing partial trial, so all trials are exactly 600 bins. Position needs no missing-data handling — I confirmed there are no NaNs in the position arrays and values lie in [0, 75]; the `+1e-5` buffer plus `np.clip` keeps the boundary value 75.0 inside bin 2.

ii.
```python
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
...
if n_registered == 0:
    return [], [], [], 0
...
if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue
...
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL   # trailing partial trial dropped
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
```

iii. CONVERSION_NOTES.md Step 10, Check 5 (edge cases): "Frames per session varies (71866-72219). Handled by discarding remainder after last complete trial. All cells registered on at least one day (no empty sessions). Position range [0, 75] handled with buffer for binning." One real bug was found and fixed during validation (Step 10, Issues): the output was emitted as float32, which broke indexing into `output_values` inside `decoder.py`; it was changed to int64.

## 6-a. What are the most time-consuming steps of the code?

i. Measured in the full run (427 s total, timing printed per animal and per session): **decompressing/loading the seven joblib files dominates at ~240 s** (9–80 s each, scaling with file size), per-session processing (NaN masking + float32 cast + `gaussian_filter1d` + `AvgPool1d` over ~550–950 cells × 72,000 frames) accounts for ~180 s (0.65–1.1 s × 207), and pickling the 6.35 GB result takes 11.6 s. The dominant memory cost is that `joblib.load` materializes every day of an animal at once — for QLAK-CA1-75 the `trace` array alone is 31 × 952 × 71,866 float64 ≈ 17 GB resident.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
print(f"  Loaded in {time.time()-t0:.1f}s")
...
print(f"  Day {day} ({env_name}): {n_registered} neurons, {n_trials_day} trials, "
      f"{time.time()-t_day_start:.2f}s")
```

iii. CONVERSION_NOTES.md Step 7 run-time table attributes "~20-80s per animal" to loading and "~0.7-1.0s per session" to processing, estimating ~380 s total; the actual 427 s was recorded as "within estimate" and comfortably below the 15-minute budget in the instructions, so no further optimization was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain, both cheap. The per-trial loop in `process_session` slices and copies each trial individually; it could be replaced by a single `reshape(n_neurons, n_trials, 600)` / `swapaxes` (or `np.array_split`) with no Python-level iteration, and the redundant `.astype(np.float32)` inside it copies data that is already float32 (~1.2 GB of pointless copying over the full dataset). The per-day loop cannot be meaningfully vectorized (each day has a different neuron count), though it is embarrassingly parallel across animals and `joblib.Parallel` — already imported via `joblib` — would have cut the ~240 s of load time substantially. Everything numerically heavy (`gaussian_filter1d`, `AvgPool1d`) is already vectorized.

ii.
```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)   # already float32
    input_trial = env_flat.astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. CONVERSION_NOTES.md Step 6 was left essentially empty ("Code inefficiencies identified" / "Code speedups added" were not filled in). The AI's stated reason for not optimizing further is in Step 7: the measured run time (~7 min) was already under the 15-minute threshold at which the instructions require optimization.

## 6-c. What processing does the code repeat multiple times?

i. Minor repetitions: `env_flat.astype(np.float32)` is re-executed for every trial (39–40 identical 9-element arrays per session, 8,187 in total, all stored separately in the pickle); `get_env_mat()` is called again inside `save_processing_plot` for a session it was already computed for; `registered_mask = ~np.isnan(trace_day[:, 0])` is likewise recomputed in `save_processing_plot`; and the NaN test itself is performed on data that is immediately re-scanned by `np.nan_to_num`. None of these is on the critical path.

ii.
```python
# in process_session
env_mat = get_env_mat(env_name); env_flat = env_mat.flatten()
for t in range(n_trials):
    input_trial = env_flat.astype(np.float32)   # repeated per trial
...
# in save_processing_plot
registered_mask = ~np.isnan(trace_day[:, 0])    # recomputed
env_mat = get_env_mat(env_name)                 # recomputed
```

iii. Not discussed in CONVERSION_NOTES.md. The repeated per-trial input array is effectively mandated by the target format, which requires one input entry per trial; the duplications in `save_processing_plot` only occur for the ≤2 sessions plotted in `--show-processing` mode.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `joblib.load` reads and decompresses the **entire** animal dict, including `maps`, `SFPs`, `centroids` and `blocked`, none of which are used — the parallel HDF5 `.mat` files would have allowed lazily reading only `trace`/`position`/`envs` for one day at a time; (2) unregistered (all-NaN) cells are decompressed and held in memory before being masked out — for most sessions these are the majority of rows (e.g. 441 of 554 on QLAK-CA1-51 day 0); (3) `np.nan_to_num` scans an array from which all NaNs have already been removed; (4) the per-trial `.astype(np.float32)` re-copies data that is already float32; (5) frames in the trailing partial trial are smoothed and pooled and then thrown away; (6) the `--full` flag is parsed but never read (full is the default path), and `--show-processing` renders an 8-panel figure whose contents are not consumed by any downstream step. The two genuinely wasteful items at scale are (1) and (2).

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))   # loads maps/SFPs/centroids too
...
registered_trace = np.nan_to_num(registered_trace, nan=0.0)  # already NaN-free
...
neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)  # already float32
...
parser.add_argument('--full', action='store_true', help='Process all sessions (default)')  # unused
```

iii. Not analyzed in CONVERSION_NOTES.md; Step 6's inefficiency sections were left blank. The implicit justification is again Step 7: total run time (427 s) was within budget, so the AI accepted the extra I/O and memory traffic rather than switching to the lazier HDF5 access path.
