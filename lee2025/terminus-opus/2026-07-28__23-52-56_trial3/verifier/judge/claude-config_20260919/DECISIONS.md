# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the data from the `joblib`-serialized per-animal files in `/app/data/` (the non-`.mat` files, e.g. `data/QLAK-CA1-08`), rather than from the MATLAB v7.3 `.mat` files. The list of 7 animals is hard-coded. Each file unpickles to a dict keyed by the animal ID, whose value is a dict with keys `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`. The AI uses only `trace` `(n_sessions, n_neurons, n_timepoints)`, `position` `(n_sessions, 2, n_timepoints)` and `envs` `(n_sessions, 1)`. Loading is one `joblib.load` per animal, processed animal-by-animal, with `del dat` afterwards to free memory. Timing is printed for the load and for each session.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = "data"
...
def process_animal(animal, data_dir=DATA_DIR, show_processing=False):
    t0 = time.time()
    print(f"  Loading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
    ...
    trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
    position = d['position'] # (n_sessions, 2, n_timepoints)
    envs = d['envs']         # (n_sessions, 1)
```
```python
for a_idx, animal in enumerate(animals_to_process):
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
    for sess in sessions:
        all_neural.append(sess['neural'])
        ...
```

iii. From CONVERSION_NOTES.md Step 1: the reference `utils.load_dat()` supports both a `"MATLAB"` and a `"joblib"` format, and the AI noted "Data loaded with `joblib.load()` from non-.mat files in data/". It therefore treats the joblib files as the reference-sanctioned, already-converted form of the `.mat` files (the reference repo also ships a `mat2joblib` helper). The joblib files are ~4–5× smaller than the `.mat` files. Step 10 Check 3 records "Data loading | joblib.load(animal) | joblib.load(animal) | ✓".

## 1-b. How are the data split into subjects?

i. One subject (mouse) per data file / per animal ID. The seven animal IDs are hard-coded in the `ANIMALS` constant and used directly as the `subjects` list; `subject_idx` is the index of the animal in the list being processed, repeated for each of that animal's sessions. All animals contribute all of their sessions (31, 31, 31, 21, 31, 31, 31 = 207 sessions). No subject is excluded.

ii.
```python
'subjects': [a for a in animals_to_process],
'subject_idx': np.array(all_subject_idx, dtype=int),
```
```python
for a_idx, animal in enumerate(animals_to_process):
    ...
    for sess in sessions:
        ...
        all_subject_idx.append(a_idx)
```

iii. CONVERSION_NOTES.md Step 2 documents "7 animals, each stored as joblib file in data/", matching the paper's 7 mice; Step 10 Check 2 item 4 states "Verified correct session counts per subject (31,31,31,21,31,31,31)". No justification is given for hard-coding the IDs rather than globbing the directory.

## 1-c. How are the data split into sessions?

i. One session per recording day, i.e. per index along axis 0 of `trace`/`position`/`envs`. Each animal's 21 or 31 recording days become 21/31 separate output sessions, giving 207 sessions total. Each session retains its environment name (`envs[day, 0]`) and is recorded in `metadata['session_info']` along with the animal, number of active neurons and number of trials.

ii.
```python
n_sessions = trace.shape[0]
...
for day in range(n_sessions):
    env_name = envs[day, 0]
    tr = trace[day]          # (n_neurons, n_timepoints)
    pos = position[day]      # (2, n_timepoints)
    ...
    sessions.append({'neural': trial_neural, 'input': trial_input,
                     'output': trial_output, 'env_name': env_name,
                     'n_active': n_active, 'animal': animal})
```
```python
session_info.append({'animal': sess['animal'], 'env_name': sess['env_name'],
                     'n_active': sess['n_active'], 'n_trials': len(sess['neural'])})
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 7: "Session definition: Each day/recording is one session. Each animal contributes multiple sessions." This is validated against the paper in Step 9/Step 10 Check 4 (207 sessions in paper = 207 in converted data). Each day is a separate ~40-min exposure to one of 10 geometries, so a day is the natural session unit.

## 1-d. How are the data split into trials?

i. There is no native trial structure (continuous ~40-min free exploration), so the AI cuts each session into non-overlapping consecutive 60-s segments of 1800 frames (30 fps × 60 s), starting at frame 0. The number of trials is `n_timepoints // 1800`; the remainder (1666–2219 frames, ~1–2 % of a session) is dropped. This yields 39 trials/session for animals with 71,866 frames and 40 for the longer ones, 8,187 trials in total.

ii.
```python
FPS = 30
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: "Trial definition: Split each 40-min session into 1-minute trials (1800 frames at 30fps). This gives ~40 trials per session." This follows directly from the instruction "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session." Step 5 sanity check list includes "Trial count per session is ~39-40 (71866/1800 ≈ 39.9)", and Step 10 Check 5 notes "Different session lengths handled correctly (39 vs 40 trials)".

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering of any kind is performed. All 8,187 complete 1-minute segments from all 207 sessions are kept. In particular, the velocity filter used by the reference decoding routine (`v_thresh=5`, `v_filt_size=5`, i.e. discarding low-speed frames) is deliberately **not** applied, and no trials are removed for low activity, poor tracking or unbalanced position occupancy. The only data loss is the incomplete final segment of each session (see 1-d).

ii. There is no filtering code; every segment produced by the trial loop is appended:
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    ...
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 4: "Velocity filtering: NOT applied at data conversion stage. The reference code applies it during decoding, but our decoder is different." Step 3 "Trial curation rules" notes "No explicit trial structure in original data (continuous 40-min sessions)". Step 10 Check 3 repeats that velocity/cell filtering are decoding-time choices in the reference rather than data-curation choices, so they are left to the downstream decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Only the `trace` field, `dat[animal]['trace']`, of shape `(n_sessions, n_neurons, n_timepoints)`. The AI verified that its values are binary 0/1 (plus NaN for untracked neurons) and identified it as the binarized rising-phase vector produced by the original authors' preprocessing. No other neural field (`SFPs`, `centroids`, `maps`) contributes to `neural`.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
...
tr = trace[day]          # (n_neurons, n_timepoints)
```

iii. CONVERSION_NOTES.md Step 1: "`trace` is binary (0/1) rising-phase vector treated as firing rate"; Step 3 quotes the methods: "binarized rising-phase vector", threshold "z > 2.5", smoothing "sigma=5 frames". The reference notebooks feed `trace` straight into `decode_position_within`, so it is the neural stream used by the paper.

## 2-b. How is the `neural` data processed?

i. Essentially no processing: the AI keeps the already-preprocessed binary trace as is. Per session it (a) selects the rows (neurons) that are not entirely NaN, (b) replaces any residual NaN with 0 as a defensive measure, (c) slices into 1800-frame trials and (d) casts each trial to `float32`. Orientation is already `(neurons, time)` in the joblib files, so no transpose is needed. No ΔF/F computation, no z-scoring, no smoothing, no normalisation, no temporal binning, and no place-cell/activity-based neuron selection are applied.

ii.
```python
tr = trace[day]  # (n_neurons, n_timepoints)
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]  # (n_active, n_timepoints)
n_active = active_neurons.shape[0]

# Replace any remaining NaN with 0 (shouldn't happen but safety)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. CONVERSION_NOTES.md Step 3: the traces are already "motion correction, cell segmentation, transient extraction" processed and binarised by the original authors, so no ΔF/F step is required (this directly answers the instruction's "does delta F over F need to be computed?"). Step 5, Key Decision 2: "Neural data: Use raw binary trace data (0/1) for active neurons. No additional temporal binning at this stage (decoder handles that)." Step 5, Key Decision 5: "Cell filtering: NOT applied at conversion. Include all active (non-NaN) neurons. The decoder can handle this." Step 10 Check 2 item 1 reports an `np.allclose` spot check of trial 5 of session 0 of QLAK-CA1-08 against the raw file.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The single curation rule is the removal, per session, of neurons whose trace is all-NaN — these are cells from the CellReg cross-session registration that were not detected on that day. Everything else is kept: no split-half-reliability / place-cell selection (the paper's `p < 0.01` criterion), and no application of the reference decoder's `cell_threshold = 5` activity criterion. This produces 69,744 active neuron-sessions out of 5,413 × 207 possible, 336.9 neurons/session on average (range 113–564).

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]  # (n_active, n_timepoints)
```
```python
all_brain_region_idx.append(np.zeros(sess['n_active'], dtype=int))
```

iii. CONVERSION_NOTES.md Step 2: "NaN neurons: neurons not tracked on a given day have all-NaN trace values"; Step 3 "Neuron curation rules": "Place cells identified by split-half reliability (p < 0.01), but decoding uses ALL active neurons, not just place cells; NaN neurons excluded per session". The 69,744 figure is used as the headline sanity check because the paper reports exactly "69,744 rate maps" (Step 4 and Step 10 Check 4). I independently re-derived this count from the raw files and confirm it is 69,744.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the recordings are continuous free exploration. The AI therefore uses the start of the recording session as the nominal alignment reference: trial *t* spans frames `[t*1800, (t+1)*1800)` measured from frame 0 of the session, and the metadata declares `temporal_alignment_event = 'Start of recording session'`, `off_start = 0.0`, `off_end = 60.0` (seconds). Neural, input and output streams for a trial are all cut with the same frame indices, so they are aligned frame-for-frame by construction.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```
```python
start = t * FRAMES_PER_TRIAL
end = (t + 1) * FRAMES_PER_TRIAL
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI does not argue the point at length; CONVERSION_NOTES.md Step 5 simply defines trials as fixed 1-minute segments, and the metadata fields are filled to satisfy the target-format specification. `--show-processing` plots neural activity and the position-bin trace on a common time axis for the first trial of the first three sessions to support the claim that there are no temporal misalignments (Step 7: "Plots generated for QLAK-CA1-08 and QLAK-CA1-30 ... position bins cover expected range").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native acquisition resolution is preserved: one time bin = one imaging frame = 1/30 s ≈ 33.33 ms, and every trial is exactly 1800 bins. No rebinning, downsampling, smoothing or averaging is applied. In particular the reference decoding routine's 3-frame `AvgPool1d` temporal binning (`temporal_bin_size=3` in `fit_decoder`) is deliberately not carried over.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
'fps': FPS,
'frames_per_trial': FRAMES_PER_TRIAL,
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 8: "Time bin size: 1 frame = 1/30 sec ≈ 33.33ms. Keep original 30Hz resolution", and Key Decision 2: no extra temporal binning "at this stage (decoder handles that)". Keeping the finest available resolution preserves information and satisfies the format requirement that bin size be identical across all trials and sessions; Step 4 lists the reference's 3-frame binning as a decoder-internal detail rather than a property of the dataset.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the per-session environment **name** string, `dat[animal]['envs'][day, 0]` (one of `square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`), which is mapped to a 3×3 binary occupancy matrix by a verbatim copy of the reference repo's `utils.get_env_mat()`. The AI explicitly examined the alternative source — the `blocked` field, which stores the flat 3×3 indices of the blocked partitions for each session — and decided **not** to use it.

ii.
```python
def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry.
    Copied from reference code utils.py."""
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        'u':         [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+':         [[0,1,0],[1,1,1],[0,1,0]],
        'i':         [[1,1,1],[0,1,0],[1,1,1]],
        'l':         [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn':     [[1,1,0],[1,1,1],[0,1,1]],
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. Trajectory steps 35–36: the AI compared `blocked` with `get_env_mat` for QLAK-CA1-08 and found they disagree (e.g. `t`: `blocked = [3,5,6,8]` vs `get_env_mat('t')` zeros at flat indices `[0,2,3,5]`). It concluded "the blocked field doesn't consistently map to get_env_mat in any single orientation. Some match original, some flipped_ud, some flipped_lr … For the decoder input, I should use get_env_mat(env_name) since that's the canonical representation used in the paper's analysis. The blocked field just stores the physical blocked positions." CONVERSION_NOTES.md Step 4 records this as "blocked vs env_mat | Different orientations | blocked uses flat indices | get_env_mat canonical | Use get_env_mat for decoder input". (Note: my own check of all 10 geometries shows the two *are* consistently related — `blocked` equals the zeros of `flipud(get_env_mat(env))` for every environment — so the stated premise for the choice is factually wrong.)

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3×3 matrix is flattened **row-major** into a 9-element vector, cast to `float32`, and attached unchanged to every trial of the session (static per trial, shape `(9,)`), with names `env_grid_0 … env_grid_8`. Polarity is 1 = accessible partition, 0 = omitted/blocked partition — the inverse of the "which parts are blocked" phrasing in the instructions. No normalisation or time expansion is performed.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
for t in range(n_trials):
    ...
    # Input: environment geometry, static per trial (9,)
    trial_input.append(env_mat.astype(np.float32))
```
```python
'input_names': [
    'env_grid_0', 'env_grid_1', 'env_grid_2',
    'env_grid_3', 'env_grid_4', 'env_grid_5',
    'env_grid_6', 'env_grid_7', 'env_grid_8',
],
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 6: "Environment input: Use get_env_mat(env_name).flatten() as 9-element binary vector. Static per trial (same for all timepoints in trial)", which follows the instruction "Environment geometry … Static per-trial". Step 10 Check 2 item 2 reports "Verified environment geometry for sessions 0-4 (square, o, t, u, rectangle) — all match get_env_mat output" — i.e. the check validates the input against `get_env_mat` itself, not against the animal's actual occupancy, so it could not detect an orientation problem. The `verification_full_out.txt` shows `env_grid_1: [1.0, 1.0]` (a constant, uninformative input dimension), which was not investigated.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, `dat[animal]['position']`, of shape `(n_sessions, 2, n_timepoints)`, holding the DeepLabCut-tracked (x, y) coordinates in cm within the 75 × 75 cm arena. Row 0 is x, row 1 is y. Nothing else contributes to the output.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
...
pos = position[day]  # (2, n_timepoints)
```

iii. CONVERSION_NOTES.md Step 2: "`position`: (n_sessions, 2, n_timepoints) - x,y position in cm" and "Position range: 0-75 cm in both x and y"; Step 3 "Position tracked with DeepLabCut". This is the same variable the reference `decode_position_within` spatially bins for its decoding analysis.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous (x, y) trace is used directly, at full 30 Hz, with no smoothing, interpolation, velocity filtering or unit conversion. The only transformation is the discretisation into the 3 × 3 grid (see 4-c), computed once per session for the whole time series and then sliced into trials. The result is stored as a `(1, 1800)` `int64` array per trial, one output variable named `position_bin` with nine values `bin_0 … bin_8`.

ii.
```python
pos = position[day]  # (2, n_timepoints)
# Compute position bins (3x3 = 9 categories)
pos_bins = position_to_bin(pos)  # (n_timepoints,)
...
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```
```python
'output_names': ['position_bin'],
'output_values': [
    ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
     'bin_5', 'bin_6', 'bin_7', 'bin_8'],
],
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 3 and the Decoder Task requirement that outputs be categorical and, where possible, time-varying. The AI kept the output time-varying at the native frame rate for maximum information (the reference paper's own decoding is also frame-wise, just into 15 × 15 bins).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is divided by a fixed 25 cm bin width (75 cm / 3), floored to an integer, and clipped to `[0, 2]`; the two indices are combined into a single class `0–8` as `x_bin * 3 + y_bin`. Clipping handles coordinates exactly at (or a floating-point hair above) the 75 cm boundary, which would otherwise floor to index 3. Fixed geometric edges are used rather than data-driven quantiles, so the class distribution is unbalanced (`[0.100, 0.075, 0.116, 0.099, 0.057, 0.141, 0.135, 0.077, 0.200]` over the whole dataset) and sessions in non-square geometries legitimately contain only a subset of the nine classes.

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    # Clip to valid range and compute bin indices
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    # Combine into single index: row * n_cols + col
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 3: "Position discretization: Bin x,y position into 3x3 grid (each cell = 25cm x 25cm). Position (0-75cm) / 25 = 0,1,2 for each axis. Combined bin = row*3 + col = 0-8." The 3 × 3 partition is exactly the partition the experiment itself uses to build the geometries, and it is mandated by the Decoder Task. Step 10 Check 5: "Position at boundaries (0 and 75 cm) correctly clipped to valid bin range [0,2]; All output values in [0,8] confirmed". Step 10 Check 2 item 3 verified bins for sessions 0–2, trials 0/10/20 against the raw position arrays.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, by construction. `trace[day]` and `position[day]` have identical time axes in the source file (verified: identical `n_timepoints` for every session), the binning is applied to the whole session before splitting, and both streams are then sliced with the same `start:end` indices. No lag, shift or offset is introduced in either direction.

ii.
```python
n_trials = n_timepoints // FRAMES_PER_TRIAL   # n_timepoints from trace.shape[2]
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI treats the two streams as already synchronised by the original acquisition (CONVERSION_NOTES.md Step 2 lists both at the same `n_timepoints`). The `--show-processing` plots put neural rasters and the position-bin trace on a common time axis for the same trial, and Step 10 Check 2 spot-checks neural values and position bins for matching (session, trial, frame) triplets with `np.allclose`, which would break if the two were shifted relative to one another.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled: (1) neurons not registered on a given day appear as all-NaN rows and are dropped per session; (2) any residual NaN inside a kept neuron is replaced by 0 via `np.nan_to_num` as a defensive measure (the AI notes it "shouldn't happen"; I confirmed there are in fact zero partially-NaN neurons in the dataset, so this is a no-op); (3) position samples at or marginally beyond the arena edge (`max = 75.00000000000001`) are clipped into the valid bin range; (4) sessions whose length is not a multiple of 1800 frames simply lose their incomplete final segment, and animals with fewer recording days (QLAK-CA1-51, 21 sessions) are handled by driving every loop off the array shapes. Missing position samples are not handled explicitly — there are none in this dataset.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
# Replace any remaining NaN with 0 (shouldn't happen but safety)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```
```python
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
```
```python
n_trials = n_timepoints // FRAMES_PER_TRIAL   # remainder frames dropped
```

iii. CONVERSION_NOTES.md Step 2 and trajectory step 39: "The NaN pattern in trace data is per-neuron, not per-timepoint. Neurons are either entirely NaN (not tracked on that day) or entirely valid … NaN indicates the neuron wasn't detected on that day", so dropping them (rather than zero-filling them) is the correct semantics and is what makes the 69,744 active-neuron-session count reproduce the paper. Step 10 Check 5 covers the boundary-clipping and the variable trial counts.

## 6-a. What are the most time-consuming steps of the code?

i. The full conversion takes 253.7 s end to end. The breakdown printed by the script shows two dominant costs: (1) `joblib.load` of each animal file, 12.7–22.8 s each, ≈ 127 s total — half the runtime, and it deserialises the *entire* animal dict including the large unused `SFPs`, `maps` and `centroids` arrays; and (2) the final `pickle.dump` of the 19.98 GB result, 40.7 s. The actual per-session work (NaN masking, `nan_to_num`, position binning, trial slicing and `float32` casting) is only 0.23–1.16 s per session, ≈ 80 s total. Decoder-side costs dwarf all of this (the 20 GB pickle must be re-read and SVD-initialised per session in `train_decoder.py`).

ii.
```python
t0 = time.time()
print(f"  Loading {animal}...")
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s")
```
```python
t_save = time.time()
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
print(f"Saved in {time.time() - t_save:.1f}s")
```

iii. The instructions require timing instrumentation and a < 15 min budget. CONVERSION_NOTES.md Step 7 gives the estimate ("Load data 12-23s … ~35s/animal, ~4 min total"), which matched the realised 4.2 min, so the AI concluded no optimisation was needed. It did register the size problem — trajectory step 46: "The file size is concerning - 5.21 GB for 2 animals means ~18 GB for all 7 … The neural data is binary (0/1) stored as float32. I should consider using a more compact representation" — and step 48: "I could use uint8 or even bit-pack it, but the decoder likely expects float32. Let me just proceed."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner `for t in range(n_trials)` loop is the only hot loop and is a pure reshape in disguise: 39–40 iterations per session × 207 sessions = 8,187 iterations, each doing a slice plus three `astype` calls. It could be replaced by a single `active_neurons[:, :n_trials*1800].reshape(n_active, n_trials, 1800)` (plus `np.split`/`transpose`) for the neural data and `pos_bins[:n_trials*1800].reshape(n_trials, 1, 1800)` for the output, with the `float32`/`int64` casts hoisted to one whole-session call. The outer per-animal loop is inherently serial as written but is embarrassingly parallel across the 7 files (`joblib.Parallel`), which would have cut the ~127 s of I/O substantially; the AI mentions parallelism nowhere. In practice these loops are cheap relative to I/O, so the achievable saving is modest (tens of seconds).

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI offers no justification: CONVERSION_NOTES.md Step 6 ("Script Development") is left essentially empty, with no "Code inefficiencies identified" or "Code speedups added" entries despite those template headings, and the Step 7 run-time table concludes the ~4 min total is acceptable.

## 6-c. What processing does the code repeat multiple times?

i. Three repeats, all inside the trial loop and all avoidable: (1) `env_mat.astype(np.float32)` is recomputed and re-allocated for each of the 8,187 trials even though it is a session constant — the human reference instead binds one array and repeats the reference (`[blocked] * len(neural_trials)`); (2) the `float32` cast of the neural data and the `int64` cast of the position bins are done 8,187 times on slices instead of once per session on the full array; (3) `active_neurons = tr[active_mask]` followed by `np.nan_to_num(...)` makes two full copies of each session's trace before any slicing, and the trial slices then make a third. At session scale this is ~3× the minimum memory traffic on the largest array in the pipeline. `get_env_mat` is also rebuilt (dict literal + `np.array`) once per session rather than cached across the ten geometries, but that cost is negligible.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)  -- computed once per session
for t in range(n_trials):
    ...
    trial_input.append(env_mat.astype(np.float32))   # re-cast every trial
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```
```python
active_neurons = tr[active_mask]                      # copy 1
active_neurons = np.nan_to_num(active_neurons, nan=0.0)  # copy 2
```

iii. No justification is given; the AI never identifies these as redundant. The `nan_to_num` copy is explicitly labelled a safety net ("shouldn't happen but safety"), which is the only documented rationale for any of the repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five items. (1) `joblib.load` deserialises the whole animal dict, so `SFPs` (35×35×n_neurons×n_sessions), `maps` (15×15×n_neurons×n_sessions ×2 plus sampling) and `centroids` are fully materialised and then thrown away — these dominate the ~127 s of load time, and the reference's h5py approach reads only the three needed datasets. (2) `np.nan_to_num` scans and copies every session's trace to fix NaNs that do not exist. (3) The output labels are stored as `int64` for a 9-class variable (`int8` would do), ~118 MB of pure padding. (4) The neural data, which is strictly 0/1, is stored as `float32`, making the pickle 19.98 GB when `uint8` would be 5 GB and `bool`/bit-packing less still — the AI noticed this and chose not to act. (5) `--sample` mode processes two whole *animals* (62 sessions, 2,418 trials, a 5.21 GB `sample_data.pkl`) rather than the two sessions the instructions specify for a quick test, so the "small sample" validation step costs nearly as much as the full run and leaves a 5 GB artefact behind. Separately, one input dimension (`env_grid_1`) is constant at 1.0 across the entire dataset and carries no information.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
trace = d['trace']; position = d['position']; envs = d['envs']   # SFPs/maps/centroids unused
```
```python
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```
```python
if args.sample:
    animals_to_process = ANIMALS[:2]
    print(f"SAMPLE mode: processing {len(animals_to_process)} animals")
```

iii. For the dtype choice, trajectory step 48: "I could use uint8 or even bit-pack it, but the decoder likely expects float32. Let me just proceed with training" — i.e. a deliberate trade of disk for decoder compatibility. For the rest there is no documented rationale; CONVERSION_NOTES.md Step 6 contains no inefficiency analysis, and Step 7's timing table concludes the pipeline is fast enough to leave alone.
