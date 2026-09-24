# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the *preprocessed joblib* files in `/app/data/` (the extension-less files `QLAK-CA1-08`, `QLAK-CA1-30`, ...), one file per mouse, rather than the `.mat` (HDF5) files. Each file unpickles to `{animal_id: dict}` with keys `trace` (n_days, n_cells, n_frames), `position` (n_days, 2, n_frames), `envs` (n_days, 1), `blocked`, `maps`, `SFPs`, `centroids`. The list of the 7 animals is hard-coded. The whole animal dict is held in memory while its 31 (or 21) days are iterated; sessions and trials are produced by slicing along the day and frame axes.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_animal_data(animal):
    """Load preprocessed data for one animal from joblib file."""
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]

d = load_animal_data(animal)
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace = d['trace'][day]            # (n_cells, n_frames)
    position = d['position'][day]      # (2, n_frames)
```

iii. From CONVERSION_NOTES Step 1/Step 10 Check 3: the reference package's own loader `load_dat` (utils.py:61) loads exactly these joblib files, so `joblib.load(f'/app/data/{animal}')` is documented as matching the reference loading path ("Data loading | `joblib.load(...)` | `load_dat` via `joblib.load` | YES"). Verified independently here: the joblib `trace`/`position` arrays are bit-identical to the `.mat` contents (`np.allclose` True after transpose), so the two loading routes are equivalent.

## 1-b. How are the data split into subjects?

i. One subject per data file / per animal ID. The 7 IDs are hard-coded in `ANIMALS` and written to `data['subjects']`; `subject_idx` for each session is `ANIMALS.index(animal)`. Result: 7 subjects with 31, 31, 31, 21, 31, 31, 31 sessions.

ii.
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)
    subject_id = ANIMALS.index(animal)
    for s_idx in range(len(neural)):
        ...
        subject_idx_list.append(subject_id)
...
'subjects': ANIMALS,  # All animal names even in sample mode
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2/3: the paper reports 7 mice (4M/3F) and the data directory contains exactly 7 animal files; per-animal cell counts (515, 875, 942, 554, 862, 713, 952; total 5,413; mean 773) were checked against the paper's "5,413 unique neurons", "mean number of cells per animal = 773 ± 68 SE", "minimum cells per animal = 515".

## 1-c. How are the data split into sessions?

i. One session per recording day per animal: the `trace`/`position`/`envs` arrays are indexed along the day axis, giving 31 days for six animals and 21 for QLAK-CA1-51 = **207 sessions**, matching the paper. A day is dropped if it has zero registered cells or yields <2 trials (neither case occurs in this dataset).

ii.
```python
n_days = d['trace'].shape[0]
envs = d['envs'].squeeze()
for day in range(n_days):
    env_name = envs[day]
    trace = d['trace'][day]
    position = d['position'][day]
    ...
    if n_registered == 0:
        print(f"  Day {day} ({env_name}): No registered cells, skipping"); continue
    ...
    if n_trials < 2:
        print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)"); continue
```

iii. CONVERSION_NOTES Steps 2–4 and Step 9: "Each session (day) is one continuous 40-min recording"; session count (207), sessions/subject (31,31,31,21,31,31,31) and neuron-sessions (69,744 = the paper's "69,744 rate maps") were all cross-checked against the paper. The ≥2-trial guard is included because the target format requires at least two trials per session for decoder evaluation.

## 1-d. How are the data split into trials?

i. Each ~40-min session is cut into consecutive, non-overlapping 60 s segments = 1800 frames at 30 Hz, starting at frame 0. The trailing remainder (sessions are 71,866–72,219 frames, i.e. 39 full blocks + a 1,666–2,019 frame remainder) is **kept as a final shorter trial** as long as it is at least 30 s (900 frames); shorter remainders would be dropped. This yields 40 trials per session and 8,280 trials total, with per-trial T between 1,666 and 1,800.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S      # 1800 frames
MIN_TRIAL_FRAMES = FPS * 30                    # Minimum 30s for a partial trial at end
...
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    # Skip short partial trials
    if trial_len < MIN_TRIAL_FRAMES:
        continue
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    input_trial = env_mat.astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. From the instructions ("long recording sessions ... split into 1-minute trials") and CONVERSION_NOTES Step 5 Key Decision 1: "1-minute segments (1800 frames at 30Hz), ~40 trials per session". Step 10 Check 5 documents the edge case explicitly: "Last trial has fewer frames when session length isn't divisible by 1800 (min T=1666); partial trials <900 frames (30s) are dropped (correct behavior)". The time **bin** size is unchanged (33.33 ms) so ragged trial length is legal in the target format.

## 1-e. How are trials filtered based on quality controls?

i. No behavioural/quality-based trial filtering at all. The only trial-level rules are structural: (a) drop a trailing segment shorter than 30 s, (b) drop a session (not individual trials) if it produces <2 trials. Notably, the reference paper's *decoding* analysis applies a velocity filter (>5 cm/s) and a per-cell event threshold; the AI deliberately did **not** apply these.

ii.
```python
    if trial_len < MIN_TRIAL_FRAMES:
        continue
...
    if n_trials < 2:
        print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
        continue
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules: No trial filtering. Each session (day) is one continuous 40-min recording, split into 1-minute segments." and Step 5 Key Decision 6: "**No velocity filtering**: Not specified in decoder task." The paper itself does not curate trials (there are no behavioural trials in the original design — trials are an artefact of this conversion).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely from `d['trace']`, shape (n_days, n_cells, n_frames): the authors' preprocessed **binary calcium event raster** (1 = significant rising-phase transient, 0 otherwise, NaN for cells not registered on that day). Verified here: unique non-NaN values are {0, 1}, mean event rate ≈ 0.0042 per frame.

ii.
```python
trace = d['trace'][day]                     # (n_cells, n_frames)
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]   # (n_registered, n_frames)
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. CONVERSION_NOTES Step 1/3: "Data is already preprocessed: binary calcium trace (0/1 for significant events from rising-phase extraction)"; "No delta F/F computation needed — trace is already binarized" (paper: events detected by z-score > 2.5 on the fluorescence derivative). This is the same variable the reference code feeds to `get_rate_maps` and `decode_position_within`.

## 2-b. How is the `neural` data processed?

i. Essentially no processing: select the cells registered on that day, slice the 1800-frame trial window, cast `float64 → float32`. No ΔF/F, no smoothing, no z-scoring, no rate conversion, no velocity masking. Output shape per trial is (n_registered, T).

ii.
```python
trace_registered = trace[registered_mask]               # (n_registered, n_frames)
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. CONVERSION_NOTES Step 10 Check 3: "Binary trace | Used as-is (0/1) | 'treated as firing rate' | YES"; Step 1: the source traces are already the fully processed signal used for every analysis in the paper, so any further transformation would diverge from the reference. float32 is chosen for decoder/memory compatibility.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only unregistered cells are removed: a cell is kept for a day if its trace on that day is not NaN, tested on the **first frame** (`~np.isnan(trace[:, 0])`). No place-cell / spatial-information / split-half-reliability filtering is applied, and no minimum event-count threshold. This gives 69,744 neuron-sessions (113–564 per session, mean 337).

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue
trace_registered = trace[registered_mask]
```

iii. CONVERSION_NOTES Step 3 "Neuron curation rules: Use ALL registered cells on each day (non-NaN trace). No place-cell filtering." justified by the paper's statement that the reliability analysis "motivated the inclusion of all cells in subsequent analyses". The 69,744 total was checked against the paper's "69,744 rate maps". (Checked here: NaN cells are NaN for the *entire* day, so the first-frame test is exactly equivalent to an all-NaN test.)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external stimulus/behavioural alignment event — recordings are continuous free exploration. Trials are aligned to the start of each 1-minute segment: trial *k* spans frames [k·1800, (k+1)·1800) of the session for **all** streams. Neural and position share the same frame index in the source arrays (both 71,866–72,219 frames), so they are aligned by construction, and both are sliced with the identical `start:end`. Metadata records `temporal_alignment_event = 'Start of each 1-minute trial segment within a 40-minute recording session'`, `off_start = 0.0`, `off_end = 60.0`.

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
...
'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. CONVERSION_NOTES Step 5: trials are an artificial segmentation of a continuous 40-min session, so the segment start is the only meaningful alignment point. Step 10 Check 2 spot-checked that converted neural and position for session 62, trial 5 both reproduce frames 9000–10800 of the raw data (`np.allclose` True for both).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native acquisition resolution is kept: 30 Hz → `time_bin_size = 1000/30 = 33.33 ms`. No rebinning, downsampling, smoothing or rate conversion is performed; every frame of the recording is one time bin, identical for neural and behavioural streams.

ii.
```python
FPS = 30  # Recording frame rate (Hz)
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
'recording_fps': FPS,
```

iii. CONVERSION_NOTES Step 3/Step 5 Key Decision 3: "Time bin: 30Hz native sampling (33.33ms)", because the paper states imaging was "acquired at 30 Hz" and both `trace` and `position` are stored frame-by-frame at that rate; keeping the native rate avoids introducing any resampling artefacts or misalignment between streams.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From `d['envs']` — the per-day environment-shape label (one of `square, o, t, u, rectangle, +, i, l, bit donut, glenn`) — passed through the reference code's `get_env_mat()` lookup table, which returns the binary 3×3 accessibility matrix for that shape. The alternative raw field `d['blocked']` (indices of blocked partitions per day) is **not** used.

ii.
```python
def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry. From reference code."""
    if env == 'square':   return np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(float)
    elif env == 'o':      return np.array([[1,1,1],[1,0,1],[1,1,1]]).astype(float)
    elif env == 't':      return np.array([[0,1,0],[0,1,0],[1,1,1]]).astype(float)
    ...
envs = d['envs'].squeeze()
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. CONVERSION_NOTES Step 1 lists `get_env_mat` (utils.py:215) as the reference function for environment geometry, and Step 10 Check 3 records "Env geometry | `get_env_mat(env_name).flatten()` | `get_env_mat` identical | YES" — i.e. the decision is to reuse the authors' own function rather than re-derive geometry. (Verified here: the geometry implied by `envs`+`get_env_mat` is consistent with the `blocked` field on all 207 days, up to index ordering.)

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3×3 matrix is flattened row-major into a 9-dim vector, cast to float32, with **1 = accessible partition, 0 = blocked partition**, and attached unchanged to every trial of the session (static per trial, shape `(9,)`). `input_names` are `env_partition_00 … env_partition_22`. Note: the flattening order used for the input vector is *not* the same spatial ordering as the position-bin labels (see 4-c) — the code comment asserts "Using same convention as get_env_mat: row=x, col=y", but `get_env_mat`'s rows are in fact the y axis with the origin at the bottom row, so input index `r*3+c` corresponds to output bin `c*3+(2-r)`. Because the permutation is identical for every session, the geometry information supplied to the decoder is intact, but the input/output labels do not line up spatially.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
    # Input: environment geometry, static per trial -> (9,)
    input_trial = env_mat.astype(np.float32)
...
input_names = [f"env_partition_{r}{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]
```

iii. CONVERSION_NOTES Step 5 variable mapping: "Environment geometry 3x3 → input[0-8], `get_env_mat(env)` → flatten to 9 values, static per trial, 1=accessible, 0=blocked". Rationale: the decoder task specifies "Environment geometry, representing which parts of the arena are blocked. Static per-trial", and the geometry is constant within a day. The AI's sanity check of this input (Step 10 Check 2) only tested a `square` session (all ones) and a `+` session (a 4-fold-symmetric shape), which cannot reveal the axis-ordering issue above.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. From `d['position']`, shape (n_days, 2, n_frames): DeepLabCut-tracked (x, y) coordinates in cm within the 75 × 75 cm arena, sampled at 30 Hz, one sample per neural frame. Verified here: range 0–75 cm with no NaNs.

ii.
```python
position = d['position'][day]  # (2, n_frames)
pos_bins = bin_position_3x3(position)  # (n_frames,)
```

iii. CONVERSION_NOTES Step 2/3: `position` is the animal's tracked location in cm ("Position tracked with DeepLabCut", "75 x 75 cm" arena), and is the same variable the reference code passes to `get_rate_maps`/`decode_position_within`.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. No smoothing, interpolation, velocity filtering or unit conversion — the raw cm coordinates are discretized directly (see 4-c) into a single integer class per frame, stored as `(1, T)` int64 per trial, i.e. one time-varying categorical output named `position_bin` with 9 values.

ii.
```python
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
...
output_names = ['position_bin']
output_values = [[f"x{r}y{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]]
```

iii. CONVERSION_NOTES Step 5 Key Decisions 4 and 6: "Position discretization: 3x3 grid"; "No velocity filtering: Not specified in decoder task". The decoder task explicitly requires "Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying", and the instructions ask for time-varying outputs where possible.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is split into 3 equal bins over the fixed 0–75 cm arena extent (25 cm per bin) using `floor(coord / bin_size)` with a tiny buffer added to the arena size so that a coordinate of exactly 75.0 does not overflow, plus a `np.clip` to [0, 2] as a second guard. The two bin indices are combined as `bin_idx = x_bin*3 + y_bin` (labels `x{r}y{c}`), giving classes 0–8. Bins not physically accessible in a deformed environment simply receive ~0 samples. Resulting distribution over the full dataset: 9.9 / 7.5 / 11.6 / 9.8 / 5.7 / 14.1 / 13.5 / 7.7 / 20.1 % (corner/edge-biased, as expected for mice).

ii.
```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    # Row-major: bin_idx = x_bin * 3 + y_bin
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx
```

iii. CONVERSION_NOTES Step 5 Key Decision 4 ("3x3 grid, row-major ordering, 25 cm/bin"). The AI investigated the skewed class distribution (trajectory steps 63–67), checked the raw x/y histograms and the `+`-shaped session (0 % occupancy in bins 0, 2, 6, 8 exactly as the geometry predicts), and concluded the imbalance is genuine thigmotaxis rather than a binning bug. Fixed 0–75 cm edges (rather than per-session min/max) keep bins comparable across sessions and environments.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame: `position` and `trace` have the same number of samples in the source arrays, and for each trial the identical `start:end` frame window is applied to the binned position and to the trace, so output timepoint *t* corresponds to neural timepoint *t* with no lead/lag and no resampling. Every trial therefore has matching T in `neural` and `output` (verified by the format checker: no dimension errors/warnings).

ii.
```python
pos_bins = bin_position_3x3(position)      # (n_frames,) computed once for the whole day
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. CONVERSION_NOTES Step 10 Check 2: an independent spot check reloaded the raw joblib file and recomputed both streams for session 62 / trial 5 (frames 9000–10800); `np.allclose` returned True for the neural matrix and for the position-bin vector, confirming that the same frame window was used for both. The `--show-processing` plots additionally show the position trace with the 3×3 grid overlay and the resulting bin time series for the same trial.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) Cells not registered on a given day are stored as NaN in `trace` and are removed by the registration mask (no NaN padding survives into the output). (b) A day with zero registered cells is skipped. (c) Sessions are not all exactly 72,000 frames (71,866–72,219, i.e. slightly under/over 40 min) and are not divisible by 1800; the trailing partial segment is kept if ≥30 s, otherwise dropped, so no session-length assumption is hard-coded. (d) A session yielding <2 trials is dropped (target-format requirement). (e) Position has no missing samples in this dataset, so no gap-filling/interpolation is implemented. (f) QLAK-CA1-51 has only 21 days (2 instead of 3 repetitions) and is handled by reading `n_days` from the array.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0: ... continue
...
end = min(start + FRAMES_PER_TRIAL, n_frames)
if trial_len < MIN_TRIAL_FRAMES: continue
...
if n_trials < 2: ... continue
```

iii. CONVERSION_NOTES Step 10 Check 5 (Edge cases) lists exactly these cases: "Last trial has fewer frames when session length isn't divisible by 1800 (min T=1666)"; "Partial trials <900 frames (30s) are dropped"; "QLAK-CA1-51 has only 21 sessions (2 sequences) — correctly handled"; "All sessions produce >=2 trials". The NaN handling is justified in Step 2 ("NaN for unregistered cells") and Step 3 (all registered cells are used, per the paper).

## 6-a. What are the most time-consuming steps of the code?

i. Measured from `conversion_full_out.txt`: whole conversion 159.8 s + 20.1 s to pickle 19.3 GB. Per animal 11–29 s, of which the per-day processing loop is only ~0.1 s/day (≈3 s/animal). The dominant cost is therefore **`joblib.load()` of the whole animal dict** (measured directly here: 9.2 s for the smallest animal, QLAK-CA1-08) plus the final `pickle.dump` of the 19 GB dictionary. Both are I/O/decompression bound. The load is also the memory bottleneck: `d['trace']` alone is 9.2 GB in RAM for CA1-08 and ~17 GB for CA1-75 (float64), because the entire multi-day array (plus `SFPs`, `maps`, `centroids`) is materialized at once.

ii.
```python
def load_animal_data(animal):
    dat = joblib.load(filepath)     # decompresses the entire animal dict (all days, SFPs, maps, ...)
    return dat[animal]
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)   # 19,255 MB, 20.1 s
```

iii. The AI printed per-day and per-animal timings (`t_day`, `t0`) and reported in CONVERSION_NOTES Step 7 "~21s per animal average ... Full run estimated: ~160s (actual: 159.8s)". Since 160 s is far below the 15-minute budget in the instructions, no optimization was pursued and the load/memory cost was not called out.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: the per-day loop (inherently sequential, could be parallelized across animals/days with joblib but is cheap) and the inner per-trial loop. The per-trial loop does a Python-level slice + `astype(np.float32)` per trial, i.e. `float64→float32` conversion is done 40 times on overlapping views instead of once per day; it could be replaced by a single `trace_registered.astype(np.float32)` followed by `np.split`/`reshape` into 1800-frame blocks (the same applies to `pos_bins`). The per-trial `input_trial = env_mat.astype(np.float32)` is a redundant cast/copy executed once per trial. The actual savings are small (~0.1 s/session), so this is a code-cleanliness rather than a runtime issue. The underlying array operations (masking, binning) are already vectorized.

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    ...
    neural_trial = trace_registered[:, start:end].astype(np.float32)   # cast repeated per trial
    input_trial = env_mat.astype(np.float32)                           # identical every trial
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. CONVERSION_NOTES Step 6 claims "Efficient vectorized operations (no inner loops for trace/position)" — true at the array level (position binning and the registration mask are fully vectorized over all frames/cells at once); the remaining per-trial loop was judged acceptable because per-day time was 0.1 s and the total run was ~160 s.

## 6-c. What processing does the code repeat multiple times?

i. (a) `env_mat.astype(np.float32)` is recomputed for each of the 40 trials of a session (8,280 times overall) and stored as 8,280 separate 9-element arrays rather than one shared array per session. (b) The `float64→float32` cast of the trace is applied per trial rather than once per day (see 6-b). (c) `trace[registered_mask]` makes a full float64 copy of the day's trace (~0.3 GB) which is then copied again as float32 trial slices. (d) In `--show-processing` mode, `d['position'][day]` is re-read and re-plotted after it was already processed. None of these dominate runtime.

ii.
```python
trace_registered = trace[registered_mask]        # full float64 copy of the day
for start in trial_starts:
    neural_trial = trace_registered[:, start:end].astype(np.float32)  # second copy, per trial
    input_trial = env_mat.astype(np.float32)                          # recomputed per trial
```

iii. Not discussed in CONVERSION_NOTES; the AI's stated efficiency criterion was the total runtime (~160 s), which was comfortably inside the instructions' 15-minute budget, so these redundancies were never revisited.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (a) `joblib.load` pulls in `SFPs` (0.16 GB), `maps` (smoothed/unsmoothed/sampling rate maps), `centroids` and `blocked` for every animal, none of which are used — an h5py/lazy read (as in the reference) would touch only `trace`, `position`, `envs`. (b) All 31 days of every animal are decompressed into RAM even though days are consumed one at a time. (c) The binary (0/1) trace is stored as float32, inflating the pickle to 19.3 GB — 4× larger than an int8/uint8 representation would be — although the decoder does consume floats (the human reference makes the same float32 choice). (d) `--sample` mode processes 2 whole *animals* (62 sessions, 5.3 GB `sample_data.pkl`) rather than the 2 sessions the instructions ask for, which multiplies the cost of the sample-validation steps ~30×. (e) `session_info` (a 207-entry list of dicts) is carried in metadata and unused by the decoder, though it is useful provenance.

ii.
```python
def load_animal_data(animal):
    dat = joblib.load(filepath)   # loads SFPs / maps / centroids / all days as well
    return dat[animal]
...
if sample_mode:
    animals = animals[:2]         # 2 animals = 62 sessions, not 2 sessions
...
neural_trial = trace_registered[:, start:end].astype(np.float32)   # binary data kept as 4-byte floats
```

iii. CONVERSION_NOTES does not flag any of these; the AI's Step 7/9 justification is that the conversion is fast enough ("~160s") and that float32 is the decoder-compatible dtype. It did note the size consequence in passing during the sample run ("The file is quite large (5GB) because we're storing binary traces at 30Hz") but decided not to change the representation.
