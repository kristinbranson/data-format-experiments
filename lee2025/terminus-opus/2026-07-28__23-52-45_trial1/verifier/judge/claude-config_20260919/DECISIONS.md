# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the per-animal **joblib** files in `/app/data/` (the extension-less files, e.g. `QLAK-CA1-08`), not the `.mat` files. This mirrors the reference repo's `load_dat(animal, p, format="joblib")`, which `main.py` uses by default. The list of 7 animals is hard-coded from the reference `main.py`. Each loaded object is `{animal: {'trace', 'position', 'envs', 'blocked', 'maps', 'SFPs', 'centroids'}}`, where `trace` is `(n_days, n_cells, n_timepoints)`, `position` is `(n_days, 2, n_timepoints)` and `envs` is `(n_days, 1)` strings. The AI iterates animals → days → trials, and frees each animal (`del dat`) before loading the next.

ii.
```python
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal in animals:
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
    n_days = d['trace'].shape[0]
    n_cells_total = d['trace'].shape[1]
    n_timepoints = d['trace'].shape[2]
    ...
    for day in range(n_days):
        ...
    del dat
```

iii. From CONVERSION_NOTES Step 1: "Data is loaded via `load_dat(animal, p, format="joblib")` which returns `{animal: dataset_dict}`". The AI documented that the joblib files are the pre-converted form of the `.mat` files used by the paper's own pipeline, so loading them reproduces exactly what the reference analyses consume. It verified the load against the paper: 7 animals, 207 sessions, 5,413 unique cells, 69,744 valid cell-days — all matching the paper's reported numbers.

## 1-b. How are the data split into subjects?

i. One subject per data file / animal ID. The animal ID string is used directly as the subject name, and `subject_idx` is the index of that ID in the sorted subject list. All sessions from one file get that subject index.

ii.
```python
all_subjects = sorted(set(animals))
...
subject_id = animal
if subject_id not in all_subjects:
    all_subjects.append(subject_id)
subj_idx = all_subjects.index(subject_id)
...
subject_idx_list.append(subj_idx)
...
'subjects': all_subjects,
'subject_idx': np.array(subject_idx_list),
```

iii. Each joblib file contains all recording days for a single mouse (confirmed by the reference code's `load_dat(animal, ...)` signature and by the paper's 7-animal cohort). The verification output confirms 31/31/31/21/31/31/31 sessions per subject.

## 1-c. How are the data split into sessions?

i. One session = one recording **day** = one index along the first axis of `trace`/`position`/`envs`. There are 31 days for six animals and 21 for `QLAK-CA1-51`, giving 207 sessions, which is exactly the number reported in the paper.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    env_name = str(d['envs'][day, 0])
    trace_day = d['trace'][day]      # (n_cells, n_timepoints)
    pos_day = d['position'][day]     # (2, n_timepoints)
    ...
    neural_sessions.append(session_neural)
```

iii. CONVERSION_NOTES Step 4: "Session = Day … Confirmed: 1 session per day", cross-checked against the paper's "5,413 unique neurons across 207 sessions in 10 geometries". Each day is a single ~40-min recording in one environment geometry, so a day is the natural session unit.

## 1-d. How are the data split into trials?

i. Each session is cut into non-overlapping 60-second trials. Because the AI first rebins time into 1-second bins, a trial is 60 time bins (`TRIAL_DURATION_BINS = 60`), equivalent to 1800 raw frames. The trailing partial trial is dropped. This yields 39 trials/session for the 71,866-frame animals and 40 for the longer ones, 8,187 trials in total.

ii.
```python
FPS = 30
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)          # 30
TRIAL_DURATION_SEC = 60
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)   # 60
...
neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
...
def split_into_trials(data, trial_length):
    n_timebins = data.shape[-1]
    n_trials = n_timebins // trial_length
    trials = []
    for t in range(n_trials):
        start = t * trial_length
        end = start + trial_length
        trials.append(data[..., start:end])
    return trials
```

iii. The recordings are continuous free exploration with no trial structure; the instructions state "long recording sessions, which will be split into 1-minute trials within each session." CONVERSION_NOTES Step 5 decision 2: "Split each ~40 min session into 1-minute trials (60 seconds = 60 time bins at 1s resolution). This gives ~39-40 trials per session."

## 1-e. How are trials filtered based on quality controls?

i. No trial-quality filtering. The only curation at this level is (a) dropping the trailing partial trial (55 s at most) and (b) a guard that skips an entire session if it would yield fewer than 2 trials — a requirement of the target format. The guard never fires on this dataset (every session gives 39–40 trials).

ii.
```python
n_trials = len(neural_trials)

if n_trials < 2:
    print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
    continue
```

iii. CONVERSION_NOTES Step 3: "No explicit trial curation in reference code - all days/sessions are used." The AI deliberately did **not** port the reference decoder's velocity filter (`v_thresh=5` cm/s), reasoning (Step 5, decision 7) that "for our format we include all timepoints. The decoder can learn to handle stationary periods" — dropping low-speed frames would break the fixed-length, uniformly-binned trial structure the target format requires.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely `d['trace']`, sliced per day: `(n_cells, n_timepoints)`. The AI verified the values are **binary 0/1** — the binarized rising phase of calcium transients that the paper's preprocessing produces, not raw fluorescence.

ii.
```python
trace_day = d['trace'][day]      # (n_cells, n_timepoints)
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]   # (n_valid_cells, n_timepoints)
```

iii. CONVERSION_NOTES Step 1: "The trace data is BINARY (0/1) - rising phase of calcium transients. No additional dF/F computation needed - data is already preprocessed." Step 3 records the paper's pipeline: "derivative smoothed with Gaussian (sigma=5 frames), z-scored, binarized at z=2.5. Binary vector treated as firing rate." This explicitly answers the instruction's "does delta F over F need to be computed?" with no.

## 2-b. How is the `neural` data processed?

i. Two operations: (1) drop unregistered (NaN) cells for that day, (2) average the binary event train within each 30-frame (1 s) window, producing an event rate in [0, 1] per cell per bin, cast to float32. No dF/F, no z-scoring, no Gaussian smoothing, no place-cell selection, no activity threshold.

ii.
```python
def bin_trace_temporal(trace, time_bin_frames):
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)
...
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
```

iii. CONVERSION_NOTES Step 5, decision 3: "Use the binary trace data directly. Average within each 1-second time bin to get firing rates. Only include cells that are registered (non-NaN) on that day." The AI noted that the reference decoder itself temporally averages traces (`AvgPool1d(kernel_size=3)` in `fit_decoder`), so averaging binary events into a rate is the reference's own treatment — only the window length differs (see 2-e). Decision 8 records the choice not to apply the reference's `cell_threshold=5` activity filter: "we include all cells and let the decoder handle it."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only registration-based filtering: cells whose trace is NaN on a given day were not registered in that session and are removed, per day. The test is done on the **first timepoint only**. No other neuron QC (no place-cell / split-half-reliability selection, no minimum event count).

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
n_valid = valid_mask.sum()
...
brain_region_idx_sessions.append(np.zeros(n_valid, dtype=int))  # All CA1
```

iii. CONVERSION_NOTES Step 2: "Cells not registered on a given day have NaN traces." Step 3 explicitly notes: "The reference code does NOT filter to only place cells for decoding - it uses ALL registered cells (with only the activity threshold filter in decode_position_within)." The AI validated the filter against the paper: summing `n_valid` over all 207 sessions gives 69,744, matching the paper's "69,744 rate maps" exactly, and the union of registered cells is 5,413, matching "5,413 unique neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no behavioural or stimulus alignment event — this is continuous free exploration, and trials are arbitrary consecutive 60-s windows. The AI anchors trial 0 at frame 0 of the recording and records this in metadata as `temporal_alignment_event = 'Start of recording session'`, `off_start = 0.0`, `off_end = 60`.

ii.
```python
'metadata': {
    'task_description': 'Decode mouse position (3x3 grid) from CA1 neural activity during geometric environment exploration',
    'time_bin_size': TIME_BIN_SEC * 1000,  # in ms
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
    ...
}
```

iii. CONVERSION_NOTES Step 10 Check 3 lists "Temporal alignment | Reference: Start of recording | Ours: Start of recording | ✓". The reference code likewise treats each session as one continuous stream indexed from frame 0 (`decode_position_within` operates on the whole session with a k-fold split). Note the `off_start`/`off_end` pair literally describes only the first trial; for trial *k* the offsets from the recording start are 60·k to 60·(k+1).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — aggressive rebinning.** The native 30 Hz (33.3 ms) data is rebinned to **1-second bins** (30 frames averaged), a 30× reduction. Every trial is therefore 60 time bins, and `metadata['time_bin_size'] = 1000.0` ms. Neural data is averaged within the bin; position takes the mode within the bin.

ii.
```python
FPS = 30
TIME_BIN_SEC = 1.0
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
...
trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
...
'time_bin_size': TIME_BIN_SEC * 1000,  # in ms
```

iii. CONVERSION_NOTES Step 5, decision 1: "Use 1 second (30 frames) bins. This provides reasonable temporal resolution while reducing data size. The reference code uses temporal_bin_size=3 (100ms) for decoding, but for our decoder format, 1-second bins are more practical and still capture spatial behavior well." Note that the AI correctly identified the reference bin size (100 ms) and then chose a bin 10× coarser; the justification offered is convenience/data size rather than anything in the paper or the decoder specification.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From `d['envs'][day, 0]` — the environment **name** string for that day (one of `square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`) — which is mapped to a 3×3 binary matrix by a verbatim copy of the reference repo's `get_env_mat` (`utils.py:215`). The AI deliberately did **not** use the `blocked` field, which is the other available encoding of the same information (per-day indices of blocked 3×3 partitions).

ii.
```python
def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry.
    Copied from reference code utils.py:215."""
    env_mats = {
        'square': [[1,1,1],[1,1,1],[1,1,1]],
        'o':      [[1,1,1],[1,0,1],[1,1,1]],
        't':      [[0,1,0],[0,1,0],[1,1,1]],
        'u':      [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+':      [[0,1,0],[1,1,1],[0,1,0]],
        'i':      [[1,1,1],[0,1,0],[1,1,1]],
        'l':      [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn':  [[1,1,0],[1,1,1],[0,1,1]],
    }
    ...
...
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. Trajectory step 31: the AI compared `blocked` with the flattened `env_mat` and found they disagree — "Interesting - the blocked indices don't directly correspond to the flattened env_mat for some environments… The env_mat is the correct representation of the environment shape. The blocked field might use a different convention or orientation. For my task, I should use the env_mat (from the environment name) as the decoder input, since it correctly represents the environment geometry." CONVERSION_NOTES Step 4 records the same resolution: "Blocked vs env_mat … Use `get_env_mat(env_name)` for input, not blocked field directly."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3×3 matrix is flattened row-major to a length-9 float32 vector with **1 = open, 0 = blocked** (the opposite polarity to the raw `blocked` field), and is replicated identically for every trial of the session — a static per-trial input of shape `(9,)`. Input names are `env_grid_0 … env_grid_8`.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
for trial_idx in range(n_trials):
    ...
    session_input.append(env_mat.astype(np.float32))
...
input_names = [f'env_grid_{i}' for i in range(N_OUTPUT_CLASSES)]
```

iii. CONVERSION_NOTES Step 5, decision 6: "Use `get_env_mat(env_name).flatten()` → 9 binary values. Static per trial", matching the instruction that environment geometry is "Static per-trial." The AI's Step 10 sanity checks report "Input data check (square env = all 1s): PASS", "Input data check (o env = center blocked): PASS", and "Blocked environment occupancy (o env, bin 4 = 0.000): PASS", and README.md states position bins are "indexed 0-8 in row-major order [0,1,2]/[3,4,5]/[6,7,8]" — i.e. the AI asserts that `env_grid_i` refers to the same arena partition as output bin `i`. Both sanity checks used only *rotationally symmetric* geometries (`square`, `o`), so they cannot detect an orientation error; the asymmetric geometries were never checked.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. From `d['position'][day]`, shape `(2, n_timepoints)` — the DeepLabCut-tracked x (row 0) and y (row 1) coordinates in cm, sampled at 30 Hz, same length as `trace`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
...
pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
```

iii. CONVERSION_NOTES Step 2 documents `position`: "(n_days, 2, max_timepoints) - x,y position at 30 Hz", and Step 3: "Position tracked with DeepLabCut". The AI verified the range is 0–75 cm, consistent with the paper's "square open field (75×75 cm²)".

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Two stages. (1) Per-frame spatial discretization: the bin width is derived from the data as `(np.nanmax(position) + 1e-5) / 3`, then `floor(position / bin_size)`, clipped to [0, 2] per axis, combined as `x_bin * 3 + y_bin`. (2) Temporal aggregation: within each 1-second (30-frame) window the **mode** of the per-frame bin index is taken. The result is reshaped to `(1, 60)` per trial and stored as int64.

ii.
```python
def bin_position_to_grid(position, n_spatial_bins=3):
    buffer = 1e-5
    pos_max = np.nanmax(position) + buffer
    bin_size = pos_max / n_spatial_bins
    pos_binned = np.floor(position / bin_size).astype(int)
    pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
    bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
    return bin_idx

def bin_position_temporal(position, time_bin_frames, n_spatial_bins=3):
    bin_idx = bin_position_to_grid(position, n_spatial_bins)
    n_bins = len(bin_idx) // time_bin_frames
    bin_idx_reshaped = bin_idx[:n_bins * time_bin_frames].reshape(n_bins, time_bin_frames)
    result = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result
...
session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. CONVERSION_NOTES Step 5, decision 4: "Use the same binning approach as the reference code: `pos_binned = floor(pos / bin_size)` where `bin_size = (max_pos + buffer) / 3`." This is indeed the reference `get_rate_maps` recipe (`(np.nanmax(position, axis=0) + buffer) // n_bins`), applied with 3 bins instead of 15 per the Decoder Task; the reference's `rate_maps[:, x, y]` indexing also matches the AI's `x*3 + y` ordering. Decision 5: "Use the mode (most frequent) position bin within each 1-second window" — a mode is the correct aggregator for a categorical variable.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Into exactly 9 categories (3×3 grid) as the Decoder Task requires. Thresholds are the two internal edges of three equal-width bins spanning `[0, nanmax(position)]` per session (≈ 25 cm and ≈ 50 cm, since the arena is 75 cm and mice reach both walls). The bin scale is taken from a **single global max over both x and y combined** rather than a fixed 75 cm arena or a per-axis max. `np.clip` guards the top edge. Category labels are `bin_0 … bin_8` with index `x_bin*3 + y_bin`.

ii.
```python
N_SPATIAL_BINS = 3
N_OUTPUT_CLASSES = N_SPATIAL_BINS ** 2  # 9
...
pos_max = np.nanmax(position) + buffer
bin_size = pos_max / n_spatial_bins
pos_binned = np.clip(np.floor(position / bin_size).astype(int), 0, n_spatial_bins - 1)
bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
...
output_names = ['position']
output_values = [[f'bin_{i}' for i in range(N_OUTPUT_CLASSES)]]
```

iii. The Decoder Task specifies "Mouse position discretized into 3 x 3 = 9 spatial bins", and the 3×3 grid is also the paper's own physical partitioning of the arena ("75×75 cm² arena … partitioned into a 3×3 grid … blocked with 25 cm walls"), so the output classes coincide with the physical partitions. The `+1e-5` buffer and floor-division are copied from the reference `get_rate_maps` so that a coordinate exactly at the wall does not fall into a 4th bin. The resulting marginal distribution (bin_4 = 0.056 lowest, bin_8 = 0.200 highest) was checked against the expectation that the centre partition is blocked in several geometries.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, with no shift. `trace[day]` and `position[day]` share the same time axis and length in the source data; both are truncated to `n_timepoints // 30` bins using the same rule, then split with the same `split_into_trials` boundaries, so bin *t* of the neural array and bin *t* of the output array cover the identical 30-frame window.

ii.
```python
neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)          # (n_cells, n_bins)
pos_binned   = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)  # (n_bins,)

neural_trials     = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
output_trials_raw = split_into_trials(pos_binned,   TRIAL_DURATION_BINS)
```

iii. CONVERSION_NOTES Step 10 Check 2 reports spot-checks that reload the raw joblib independently and confirm "Neural data spot-check (session 0, trial 0, bin 0): PASS - exact match", "(session 0, trial 5, bin 10): PASS", and "Output data check (position bin mode, 2 spot checks): PASS". The `--show-processing` plots additionally show the position trajectory, per-frame grid index and binned neural traces on a shared time axis. The number of bins is computed independently inside each helper from its own array length; that is safe here only because `trace` and `position` have identical `n_timepoints` in every session (verified: no length mismatch and no NaN in `position` in any of the 7 files).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled: (1) **unregistered cells** — NaN-trace cells are dropped per day (2-c); (2) **variable session length** — sessions run 71,866–72,219 frames, so the trailing incomplete trial (60–1,666 frames, i.e. up to ~55 s) is silently discarded by integer division, both at the 1-s binning step and at the trial-splitting step; (3) **NaN positions** — `np.nanmax` is used so the bin width is robust to NaN, though a NaN coordinate would still propagate to a meaningless bin index (in fact there are no NaNs in `position` in any file). There is no handling of dropped/jumpy tracking frames and no interpolation.

ii.
```python
valid_mask = ~np.isnan(trace_day[:, 0])
valid_trace = trace_day[valid_mask]
...
n_bins = n_timepoints // time_bin_frames
trace_truncated = trace[:, :n_bins * time_bin_frames]
...
n_trials = n_timebins // trial_length      # remainder implicitly dropped
...
pos_max = np.nanmax(position) + buffer
```

iii. CONVERSION_NOTES Step 10 Check 5: "End of session: ~55s remainder discarded (39-40 full 1-min trials from ~40 min). NaN cells: Properly excluded per session. No off-by-one errors in trial splitting or temporal binning." The AI also verified "Last trial boundary check: PASS - all values finite, correct shape", i.e. no NaN or padding leaks into the final trial of a session.

## 6-a. What are the most time-consuming steps of the code?

i. **Deserializing the per-animal joblib files.** The timing printed by the conversion shows 15.5–27.7 s per animal (153 s total for the full run) while per-day processing is only 0.1–0.2 s — so >95% of wall-clock time is the 7 `joblib.load` calls, which are I/O- plus decompression-bound and pull in ~100–150 MB per animal. Within the per-day work, the largest cost is the Python mode loop in `bin_position_temporal` (~2,395 `np.unique` calls per session).

ii.
```python
t_animal_start = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
...
t_animal_end = time.time()
print(f"  {animal} done in {t_animal_end - t_animal_start:.1f}s")
```

iii. CONVERSION_NOTES Step 6: "Data loading is sequential (could parallelize but not needed given short runtime)." Step 7 estimated ~23 s/animal → ~3 min total, well under the instruction's 15-minute threshold, so the AI explicitly decided further optimization was unnecessary; the realized 153 s matched the estimate.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest one is the per-time-bin mode loop in `bin_position_temporal`, which calls `np.unique(..., return_counts=True)` once per 1-second bin (~2,395 per session × 207 sessions ≈ 500k calls). It is fully vectorizable, e.g. with a one-hot `bincount` over the reshaped `(n_bins, 30)` array and a single `argmax(axis=1)`, or `scipy.stats.mode(..., axis=1)`. Secondary candidates: the per-trial `for trial_idx in range(n_trials)` loop that just appends already-computed slices and re-appends the same `env_mat`, and the `for t in range(n_trials)` slicing loop inside `split_into_trials` (both could be a single reshape/`np.split`).

ii.
```python
    result = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
...
    for trial_idx in range(n_trials):
        session_neural.append(neural_trials[trial_idx])
        session_input.append(env_mat.astype(np.float32))
        session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
```

iii. The AI identified this itself in CONVERSION_NOTES Step 6: "Code inefficiencies identified: Mode computation in `bin_position_temporal` uses a loop (could vectorize with `scipy.stats.mode`)." It chose not to fix it because "Code speedups added: Vectorized temporal binning using reshape+mean; Efficient position binning using floor division" already brought the estimated total runtime to ~3 min, below the 15-min threshold in the instructions. (The docstring comment `# Mode for each time bin (vectorized)` sitting directly above the Python loop is inaccurate.)

## 6-c. What processing does the code repeat multiple times?

i. In the main conversion path there is essentially no redundant recomputation — each day's trace binning and position binning happens once. Redundancy appears in two places: (1) `plot_processing`, run under `--show-processing`, re-derives `valid_mask`, re-runs `bin_trace_temporal` and `bin_position_temporal` for days 0–2 of each animal even though the main loop already computed exactly those arrays; (2) `env_mat.astype(np.float32)` is re-executed once per trial (39–40× per session) even though the value is constant within a session, and `get_env_mat` rebuilds its whole 10-entry dictionary on every call. Both are negligible in cost.

ii.
```python
def plot_processing(d, animal, data_dir):
    ...
        trace_day = d['trace'][day]
        valid_mask = ~np.isnan(trace_day[:, 0])
        valid_trace = trace_day[valid_mask]
        neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)   # recomputed
        ...
        pos_binned = bin_position_temporal(d['position'][day], TIME_BIN_FRAMES, N_SPATIAL_BINS)  # recomputed
...
    for trial_idx in range(n_trials):
        session_input.append(env_mat.astype(np.float32))   # same value re-cast per trial
```

iii. The AI did not flag repeated computation in CONVERSION_NOTES; its efficiency notes cover only the mode loop and sequential loading. The duplication in `plot_processing` is a deliberate separation of the plotting path from the conversion path (it takes a raw `d` dict rather than the processed arrays), which keeps the plotting function self-contained at the cost of recomputing three days of binning.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mainly at load time: `joblib.load` deserializes the entire per-animal dict, including `SFPs` (35×35×n_cells×n_days spatial footprints), `centroids`, `maps` (15×15 smoothed/unsmoothed/sampling rate maps) and `blocked` — none of which is used; only `trace`, `position` and `envs` are read. That is the dominant cost of the whole script (see 6-a). Smaller items: `n_cells_total` and `n_timepoints` are computed only to be printed; `bin_position_to_grid` discretizes all ~72,000 frames per session at 30 Hz and then throws away 29 of every 30 values via the mode; the outputs are stored as int64 when the values are 0–8 (an int8 would be 8× smaller); and `--sample` mode converts two whole *animals* (62 sessions, 2,418 trials, 174 MB) rather than the 2 sessions the instructions call for, so most of the sample run's work is redundant with the full run.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))   # loads SFPs, centroids, maps, blocked too
d = dat[animal]
n_cells_total = d['trace'].shape[1]      # only printed
n_timepoints = d['trace'].shape[2]       # only printed
...
session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))  # int64
...
if args.sample:
    animals = ALL_ANIMALS[:2]            # 2 animals = 62 sessions, not 2 sessions
```

iii. The AI's justification for loading via joblib is fidelity to the reference pipeline (`load_dat(..., format="joblib")`), which returns the full dict; it did not consider a partial/lazy read. Since the measured total runtime (153 s) was far below the 15-minute budget set by the instructions, CONVERSION_NOTES Step 6/7 treat further optimization as unnecessary: "could parallelize but not needed given short runtime."
