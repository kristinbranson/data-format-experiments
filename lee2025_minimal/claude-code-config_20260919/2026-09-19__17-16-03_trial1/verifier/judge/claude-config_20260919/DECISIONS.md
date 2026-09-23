# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the **joblib** version of the Zenodo dataset (`/app/data/<ANIMAL>`, no extension), which is the default format used by the paper's own loader (`load_dat(..., format="joblib")` in `georepca1/src/utils.py`, called from `main.py`). One file per mouse; the seven animal IDs are hard-coded to exactly the list in the paper's `main.py`. From each file it takes only the fields it needs: `trace`, `position`, `envs`, `blocked`. All days (sessions) in each file are processed, and every session is cut into trials; nothing is subsampled. The result is 207 sessions / 8,187 trials / 69,632 neuron-sessions, i.e. all 7 mice × 21–31 days, matching the paper's stated "207 sessions" and "69,744 rate maps" (69,744 registered cell-sessions before the >5-event cell criterion).

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal_idx, animal in enumerate(ANIMALS):
    print(f'loading {animal}', flush=True)
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace, position = dat['trace'], dat['position']
    envs, blocked = dat['envs'], dat['blocked']
    n_days = trace.shape[0]
```

iii. From the trajectory: the AI read `code/README.md` and `main.py`, saw that the analysis code loads the joblib files with `load_dat(animal, p, format="joblib")` and that `animals` is exactly this list of 7 IDs, then inspected one joblib file to confirm the field shapes (`trace (31, 515, 71866)`, `position (31, 2, 71866)`, `envs (31,1)`, `blocked` list of 31). It verified across all 7 animals that the total number of registered cell-sessions is 69,744, matching the paper's reported rate-map count, and concluded that "no session or mouse was excluded."

## 1-b. How are the data split into subjects?

i. One subject per data file / animal ID. `subjects` is the hard-coded list of the 7 animal IDs; every session produced from a file gets that animal's index in `subject_idx`.

ii.
```python
data = {..., 'subjects': list(ANIMALS), 'subject_idx': [], ...}
...
for animal_idx, animal in enumerate(ANIMALS):
    ...
    for day in range(n_days):
        ...
        data['subject_idx'].append(animal_idx)
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. Each Zenodo file is named after one mouse and contains all of that mouse's recording days (README: "datasets ... are given names of animal IDs"). The AI took the paper's `animals` list from `main.py` verbatim as the canonical subject list.

## 1-c. How are the data split into sessions?

i. One output session per mouse-**day**. The first axis of `trace` / `position` / `envs` / `blocked` indexes recording days (one 40-min session per day, one geometry per day), so the AI iterates `for day in range(n_days)` and emits one session per day. 31 days for six mice and 21 for `QLAK-CA1-51` → 207 sessions. Each session additionally gets a `session_info` entry recording subject, day, geometry name, which sequence repetition it belongs to, blocked partitions, neuron count and trial count.

ii.
```python
n_days = trace.shape[0]
for day in range(n_days):
    neural, keep = session_neural(trace[day])
    bins = session_position_bins(position[day], bin_width)
    geometry = blocked_vector(blocked[day])
    ...
    session_info.append({
        'subject': animal, 'day': int(day),
        'sequence': int(day // 10) + 1,
        'environment': str(envs[day][0]),
        'blocked_partitions': np.where(geometry > 0)[0].tolist(),
        'n_neurons': int(keep.sum()), 'n_trials': int(n_trials)})
```

iii. Methods: "All sessions were 40 min, and one session was recorded per day"; each day has its own geometry, its own cell registration and its own position stream, so a day is the natural session unit. The AI printed the environment sequence for each animal and confirmed the 10-geometry sequence repeated up to three times, matching the paper.

## 1-d. How are the data split into trials?

i. The recordings are continuous free exploration with no trial structure, so, as the instructions require, each session is cut into consecutive, non-overlapping **1-minute** trials. Because the AI rebins time into 100 ms bins (3 frames at 30 Hz), one trial is 600 bins. The incomplete remainder at the end of the session is dropped, giving 39 trials for the 71,866-frame sessions and 40 for slightly longer ones (8,187 trials total).

ii.
```python
TRIAL_BINS = TRIAL_SECONDS * FPS // TEMPORAL_BIN_FRAMES   # 600 bins per trial
...
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geometry.copy())
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. Directly from the task instructions ("long recording sessions, which will be split into 1-minute trials within each session"). `min(neural.shape[1], bins.shape[0])` guards against the neural and position streams ending on different bins, and taking the floor guarantees every trial has exactly the same length (required by the target format). Metadata records this explicitly: `temporal_alignment_event = 'Start of each 1-minute trial ... sessions are continuous 40-minute free exploration recordings with no discrete trial events'`, `off_start = 0.0`, `off_end = 60.0`.

## 1-e. How are trials filtered based on quality controls?

i. **No trial is discarded.** The only data dropped at the trial level is the sub-1-minute remainder at the end of each session. Critically, the AI deliberately did **not** apply the paper's decoding speed filter (keep only frames with running speed > 5 cm/s, computed with a Gaussian velocity filter of 5 frames), which would remove ~40–50% of frames and destroy trial contiguity. This deviation is documented in the module docstring and in `metadata['speed_filter']`.

ii.
```python
# module docstring
# Deviation from the paper, required by the decoder format: the paper's decoder analysis
# keeps only frames in which the animal ran faster than 5 cm/s. Here every frame of the
# session is kept, because the task asks for contiguous 1-minute trials covering the
# session and because the animal's spatial bin is equally well defined while it is
# immobile. ...
'speed_filter': (
    'None. Unlike the paper\'s decoding analysis, frames with running speed below '
    '5 cm/s are retained so that trials are contiguous 1-minute epochs; the '
    'occupied partition is equally well defined during immobility.'),
```

iii. The AI explicitly quantified the cost of the choice before making it: it reimplemented the paper's velocity filter and measured that 34–59% of frames per session fall below 5 cm/s. Its stated reasoning: the speed filter in the paper "serves rate-map/place-coding estimation, which is biased by long stationary epochs; it is not needed to define the decoding target", and the requirement of contiguous, equal-length 1-minute trials is incompatible with removing scattered frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `trace` field only: the authors' binarized rising phase of calcium transients, shape `(n_days, n_cells, n_frames)`, values in {0, 1} with all-NaN rows for cells not registered on that day. No other neural field (`SFPs`, `centroids`, `maps`) is used for `neural`.

ii.
```python
trace, position = dat['trace'], dat['position']
...
neural, keep = session_neural(trace[day])
```
```python
def session_neural(trace_day):
    """(n_cells, n_frames) binary trace -> (n_kept_cells, n_bins) binned rate + cell mask."""
```

iii. Methods: "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5 ... This binary vector was treated as the firing rate in all further analyses", and "All analyses were conducted using the binary vector of the rising phases of transients". The AI verified the values are exactly {0, 1} plus NaN.

## 2-b. How is the `neural` data processed?

i. Exactly the preprocessing the paper's decoder applies (`fit_decoder`/`test_decoder` in `utils.py`): (1) select cells (see 2-c), (2) cast to float32, (3) Gaussian-smooth along time with σ = 3 frames, (4) average-pool into non-overlapping 3-frame bins (`AvgPool1d(kernel_size=3, stride=3)`), yielding a continuous binned "firing rate" at 100 ms resolution. Smoothing and pooling are done once per whole session (before trial cutting), so there are no edge artifacts at trial boundaries. Final per-trial array is `(n_kept_cells, 600)` float32.

ii.
```python
def pool_time(x):
    n = x.shape[-1] // TEMPORAL_BIN_FRAMES
    return x[..., :n * TEMPORAL_BIN_FRAMES].reshape(
        x.shape[:-1] + (n, TEMPORAL_BIN_FRAMES)).mean(-1)

def session_neural(trace_day):
    registered = ~np.isnan(trace_day[:, 0])
    events = np.nansum(trace_day, axis=1)
    keep = registered & (events > MIN_EVENTS)
    trace = trace_day[keep].astype(np.float32)
    # Gaussian smoothing along time followed by average pooling, as in `fit_decoder`.
    smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
    return pool_time(smoothed).astype(np.float32), keep
```

iii. The AI read `fit_decoder` and copied its two preprocessing steps verbatim in order and parameters:
`pooling(torch.tensor(gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0).T)).numpy().T` with `temporal_bin_size=3`. Its docstring states: "Processing follows the within-session Bayesian position decoding of the paper (`decode_position_within` / `fit_decoder`)". Metadata documents it as "Binarised rising phase of calcium transients (treated as firing rate in the paper), Gaussian-smoothed along time (sigma = 3 frames) and averaged within non-overlapping 100 ms bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two per-session criteria, both applied to cells (never to sessions): (1) the cell must be registered on that day (its trace row is not NaN) and (2) it must have **more than 5 transients** in the session — the paper's `cell_threshold=5` used in `decode_position_within`. Cells are tracked across days, so the same physical neuron can appear in several sessions of the same mouse; `brain_region_idx` for the session is sized to the kept cells. Effect measured by the AI: 69,744 registered cell-sessions → 69,632 kept (112 dropped, 0.16%).

ii.
```python
MIN_EVENTS = 5                # paper's `cell_threshold` for the decoding analysis
...
registered = ~np.isnan(trace_day[:, 0])
events = np.nansum(trace_day, axis=1)
keep = registered & (events > MIN_EVENTS)
trace = trace_day[keep].astype(np.float32)
...
data['brain_region_idx'].append(np.zeros(int(keep.sum()), dtype=np.int64))
```

iii. From `decode_position_within`: `cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold` with `cell_threshold=5`. The AI reproduced this ("the paper's `cell_threshold` for the decoding analysis"), with the one difference that the event count is taken over all frames rather than only the speed-filtered frames, consistent with its decision not to apply the speed filter. It also verified empirically that NaN is all-or-none per cell-day (`nan per cell-day all-or-none: True`), which is what makes the cheap `~np.isnan(trace_day[:, 0])` registration test valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event: the sessions are continuous 40-min free-exploration recordings. The AI defines the alignment event as the **start of each 1-minute trial**, with `off_start = 0.0 s` and `off_end = 60.0 s`, and states this in the metadata. Alignment across streams is by frame index: imaging and behaviour were acquired simultaneously at 30 Hz by the same DAQ and are stored with identical frame counts, so neural, position and geometry are cut with the same bin indices.

ii.
```python
'temporal_alignment_event': (
    'Start of each 1-minute trial. Sessions are continuous 40-minute free '
    'exploration recordings with no discrete trial events, so each session is cut '
    'into consecutive non-overlapping 1-minute trials.'),
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```
```python
sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
output_trials.append(bins[sl][np.newaxis, :])
```

iii. Methods: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz ... and all recorded frames were timestamped for post-hoc alignment", so the two streams are already aligned sample-for-sample; the paper's own decoder indexes `behav` and `traces` with the same row indices. The AI confirmed the two streams have identical frame counts per day (`trace (31, 515, 71866)`, `position (31, 2, 71866)`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms** (`time_bin_size = 100.0`). Yes — the native 30 Hz (33.3 ms) data is rebinned by a factor of 3 by averaging non-overlapping 3-frame windows, after Gaussian smoothing, for both the neural and the position streams. This is the paper's `temporal_bin_size=3` for decoding. 1-minute trials therefore contain 600 bins; the leftover < 3 frames at the end of a session are dropped by the pooling.

ii.
```python
FPS = 30                      # acquisition rate of both imaging and behaviour streams
TEMPORAL_BIN_FRAMES = 3       # paper's `temporal_bin_size` for decoding -> 100 ms bins
TRIAL_BINS = TRIAL_SECONDS * FPS // TEMPORAL_BIN_FRAMES   # 600 bins per trial
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS          # 100 ms
...
'time_bin_size': TIME_BIN_MS,
```

iii. `fit_decoder`/`test_decoder` use `AvgPool1d(kernel_size=temporal_bin_size, stride=temporal_bin_size)` with `temporal_bin_size=3` on both behaviour and traces; the AI mirrors that exactly ("Mirrors torch.nn.AvgPool1d(kernel_size=3, stride=3) used in the paper's decoder"). Binarized single-frame calcium events are extremely sparse (~0.007 events/frame/cell, measured by the AI), so binning also gives the decoder a usable rate estimate per bin and cuts the dataset size 3×.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field, one entry per day, holding the indices of the occluded partitions in the 3 × 3 grid indexed `[[0,1,2],[3,4,5],[6,7,8]]`, or `-1` when nothing is blocked. `envs` (the geometry's string name) is read as well but only stored in `session_info`, not used to build the input.

ii.
```python
envs, blocked = dat['envs'], dat['blocked']
...
geometry = blocked_vector(blocked[day])
```

iii. The README documents `blocked` as "location of blocked (occluded) partitions in 3x3 design of environment ... organized in the following way – [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If no partitions are blocked, value is -1." The AI cross-checked `blocked` against the paper code's geometry masks for all 207 sessions (`blocked == flipud(get_env_mat(env))`: True), so `blocked` fully determines the geometry and the string name adds nothing.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A 9-dimensional binary (float32) vector per session: 1 for each blocked partition, all-zero for the full square (`-1`). It is static, so the same vector is attached to every trial of the session as a 1-D array of shape `(9,)` (the format explicitly permits `(n_input)`); `input_names` are `partition_0_blocked` … `partition_8_blocked`. The AI kept all 9 dimensions even though partition 7 is never blocked in any of the 10 geometries (a constant-zero input), for a complete geometry description.

ii.
```python
def blocked_vector(blocked_day):
    """Dataset `blocked` entry -> (9,) float32, 1 where the partition is walled off."""
    idx = np.array(blocked_day[0]).ravel().astype(int)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if idx.size and idx[0] != -1:
        vec[idx] = 1.0
    return vec
...
input_trials.append(geometry.copy())
...
'input_names': [f'partition_{i}_blocked' for i in range(N_SPATIAL_BINS ** 2)],
'input_description': ('Binary vector of length 9, 1 for each partition of the 3 x 3 '
                      'grid that is walled off in that session (all zeros for the full square).'),
```

iii. The instructions call for "Environment geometry, representing which parts of the arena are blocked. Static per-trial", and the format spec asks for one-hot/binary encodings of discrete inputs. A multi-hot 9-vector is the lossless representation of the blocking pattern and shares structure across geometries (unlike a 10-way one-hot over geometry names). The AI verified the indexing convention two ways — against the code's environment matrices and against occupancy (blocked partitions are exactly the never-visited bins).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, `(n_days, 2, n_frames)`, x-y head position in cm from DeepLabCut tracking, at the same 30 Hz as the imaging.

ii.
```python
trace, position = dat['trace'], dat['position']
...
bins = session_position_bins(position[day], bin_width)
```

iii. README: "position: x-y position data for all days ... x-y position in first dimension, and number of temporal bins / frames in second dimension"; Methods: "Position data were generated from tracking the head with DeepLabCut". The AI checked the ranges (0–75 cm, no NaNs).

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is average-pooled into the same 100 ms bins as the neural data (so it is the mean x and mean y within each bin), then converted to a partition index (see 4-c). The spatial bin width is derived per animal as `(max position over all of that animal's sessions + 1e-15) / 3`, i.e. the paper's `bin_down` with `n_bins = 3` instead of 15; in practice this evaluates to 25.0 cm for every animal (verified: each animal's position maximum is 75.0 cm).

ii.
```python
BUFFER = 1e-15                # paper's rounding buffer for spatial binning
...
# One spatial grid for the whole animal, from the largest coordinate reached in
# any session (as `decode_position_within` does), so that the 3 x 3 grid is the
# physical partition grid in every geometry, including those the animal cannot
# fully cover.
bin_width = (np.nanmax(position) + BUFFER) / N_SPATIAL_BINS
...
def session_position_bins(position_day, bin_width):
    pooled = pool_time(position_day.astype(np.float64))          # (2, n_bins), cm
    xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. `decode_position_within` computes `bin_down = (behav_max.max() + buffer) / n_bins` from the maxima across *all* of an animal's days, precisely so that the grid is identical in every geometry — important here because in a geometry like "rectangle" the animal can only reach x ≥ 25 cm, and a per-session rescaling would misalign the grid with the physical partitions. The AI states this reason in the code comment. The `+buffer` reproduces the paper's rounding guard so that a position exactly at the maximum does not fall into a 4th bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Into the 9 physical partitions of the 3 × 3 grid: column = `floor(x / 25 cm)`, row = `floor(y / 25 cm)`, class = `3 * row + column`, clipped to 0…2 per axis. This is one categorical output (`output_names = ['position_bin']`) of shape `(1, 600)`, dtype int64, with `output_values` naming each partition together with its x/y indices. The class ordering was chosen to coincide with the dataset's own `blocked` indexing so that input and output live in the same 9-partition index space.

ii.
```python
xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
...
'output_names': ['position_bin'],
'output_values': [[f'partition_{i}_x{i % 3}_y{i // 3}'
                   for i in range(N_SPATIAL_BINS ** 2)]],
'output_description': ('Index of the occupied partition of the 3 x 3 grid, 3 * (y // 25 cm) + '
                       '(x // 25 cm), computed from the head position averaged within each 100 ms '
                       'bin. Partitions blocked in a given geometry are never occupied in that session.'),
```

iii. Required by the task ("Mouse position discretized into 3 x 3 = 9 spatial bins. Time-varying") and matching the experimental design (the arena is physically partitioned into a 3 × 3 grid with 25 cm inserts). The AI justified the specific index convention empirically, in the trajectory and in the docstring: it computed occupancy per partition for several sessions and confirmed that with `3*row + column` the partitions listed in `blocked` are exactly the partitions with (near-)zero occupancy. It then checked the final pickle: only 152 of 4,912,200 output timepoints (0.003%) fall in a blocked partition, i.e. tracking jitter at the walls. The resulting class distribution is mildly non-uniform (0.057–0.200).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, then bin-for-bin. Both streams have the same number of 30 Hz frames per session, both are pooled with the identical `pool_time` (same kernel, same dropped remainder), and both are sliced with the same trial slice, so bin *t* of `output` and bin *t* of `neural` cover the same 100 ms of the session. The trial count is taken as the minimum of the two stream lengths as a safety check.

ii.
```python
neural, keep = session_neural(trace[day])        # pool_time inside
bins = session_position_bins(position[day], bin_width)   # same pool_time
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. The two streams were acquired simultaneously at 30 Hz by the same DAQ and are stored with equal frame counts, and the paper's decoder pools behaviour and traces with the same `AvgPool1d`. One deliberate difference from the paper: the paper casts the *pooled* position to int (mean position within the bin), which is what the AI does too — no shift or offset is introduced anywhere. Only one asymmetry exists: the neural stream is Gaussian-smoothed before pooling while position is not, exactly as in `fit_decoder`; smoothing is symmetric (zero-phase) so it introduces no lag.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases: (1) **Unregistered cells** appear as all-NaN rows and are dropped per session; `np.nansum` is used for the event count so NaNs cannot poison it. (2) **Near-silent cells** (≤ 5 transients) are dropped (2-c). (3) **Position outside the grid** (rounding at the arena edge, e.g. x exactly 75.0) is handled by the `+1e-15` buffer in the bin width plus `np.clip(..., 0, 2)`, so no out-of-range class can be produced. (4) **Leftover frames** that do not fill a whole 3-frame bin, or a whole 1-minute trial, are dropped, so all trials are exactly 600 bins. Position contains no NaNs (checked by the AI), so no interpolation is needed. Residual tracking jitter that places 0.003% of bins inside a walled-off partition is left as-is rather than corrected.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
events = np.nansum(trace_day, axis=1)
keep = registered & (events > MIN_EVENTS)
...
xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
...
n = x.shape[-1] // TEMPORAL_BIN_FRAMES          # trailing frames dropped
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS   # trailing trial dropped
```

iii. The AI verified the assumptions behind each guard before relying on them: NaN is all-or-none per cell-day (so the single-frame registration test is safe), `np.isnan(pos).sum() == 0` (so no position imputation is needed), and position maxima are exactly 75.0 (so the buffer/clip matter only at the boundary). Dropping < 1 minute of a 40-minute session is negligible and is the price of equal-length trials.

## 6-a. What are the most time-consuming steps of the code?

i. Two steps dominate, both per animal/session:
  - `joblib.load` of each animal file — the file is decompressed and *every* field is materialised, including `SFPs` (35×35×n_cells×n_days) and `maps` (three arrays of up to 15×15×n_cells×n_days), which are never used. The `trace` array alone is float64, e.g. 31 × 875 × 71,866 ≈ 15 GB for `QLAK-CA1-50`.
  - `gaussian_filter1d(trace, sigma=3, axis=1)` on a (n_cells × 71,866) float32 array for each of the 207 sessions, plus the float64→float32 cast and the `np.nansum`/`np.isnan` full passes over the raw day array.
Pickling the 6.65 GB result at the end is also a significant I/O cost. The whole conversion took roughly 7–8 minutes wall-clock.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]   # loads SFPs, centroids, maps too
...
events = np.nansum(trace_day, axis=1)
trace = trace_day[keep].astype(np.float32)
smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
...
pickle.dump(data, f, protocol=4)
```

iii. Not discussed explicitly by the AI, but it did structure the code to limit the cost: it slices `trace[day]` rather than processing the full 3-D array, casts to float32 *after* cell selection so smoothing runs on the smallest possible array, and `del dat, trace, position` at the end of each animal so only one animal's raw data is resident at a time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two:
  - The **trial loop** (`for t in range(n_trials)`) is pure slicing and could be replaced by a single reshape/`np.split` of the session arrays; as written it also calls `np.ascontiguousarray`, which copies the neural data a second time.
  - The **day loop** could be partly vectorized: `gaussian_filter1d` accepts the full `(n_days, n_cells, n_frames)` array with `axis=-1`, and `pool_time` is already n-dimensional, so smoothing and pooling could be done once per animal instead of 31 times (though at a large memory cost, so the per-day loop is the reasonable trade-off).
Everything inside `session_neural`, `session_position_bins` and `blocked_vector` is already vectorized.

ii.
```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geometry.copy())
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. Not discussed by the AI. The loops are shallow (207 sessions × ~40 trials) and the per-iteration work is a memory copy, so the potential saving is small relative to load and smoothing time; the target format itself requires a Python list of per-trial arrays, so the trial loop cannot be removed entirely.

## 6-c. What processing does the code repeat multiple times?

i. Minor repetitions only:
  - The static 9-dim geometry vector is **copied once per trial** (`geometry.copy()`), so ~40 identical arrays per session (8,187 copies overall) are built and pickled instead of one shared object. Negligible in bytes, but it is redundant work and redundant storage.
  - `np.ascontiguousarray(neural[:, sl])` materialises a second copy of the whole session's neural data one trial at a time (the slice is already a valid view; the copy is needed only because the session array is discarded afterwards).
  - `np.isnan` / `np.nansum` each make a full pass over the raw day array; the registration mask could have been derived from the same pass.
  - `pool_time` is invoked separately for the neural and position streams, which is unavoidable since they are different arrays.
  - `int(keep.sum())` is computed twice per session (once for `brain_region_idx`, once for `session_info`).

ii.
```python
input_trials.append(geometry.copy())
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
...
data['brain_region_idx'].append(np.zeros(int(keep.sum()), dtype=np.int64))
session_info.append({..., 'n_neurons': int(keep.sum()), ...})
```

iii. Not discussed by the AI. `geometry.copy()` per trial is defensive (it guarantees that no two trials alias the same mutable array, which matters if downstream code modifies inputs in place) and is required anyway by the "one array per trial" format.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
  - **Loading unused fields.** `joblib.load` reads and decompresses the entire per-animal dictionary, including `SFPs`, `centroids` and the three `maps` arrays, none of which is ever touched. Reading the `.mat` file with `h5py` (or the paper's `load_dat(to_convert=[...])`) would have pulled only `trace`, `position`, `blocked`, saving substantial I/O and memory.
  - **Smoothing frames that are then thrown away**: the trailing frames that do not fill a complete trial are smoothed and pooled before being dropped (< 1 min out of 40).
  - **`np.nansum` over unregistered cells**: the event count is computed for all cells including the all-NaN ones that are excluded anyway (~50% of rows in some animals).
  - **A constant input dimension**: `partition_7_blocked` is 0 in every one of the 207 sessions (no geometry blocks that partition), so one of the nine input features carries no information for the decoder.
  - **Metadata that the decoder ignores**: `envs`, `session_info` (including the `sequence` field, whose `day // 10 + 1` formula labels the 31st day as "sequence 4"). This is descriptive information the task asks for, not waste, but it is not consumed by the decoder.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position = dat['trace'], dat['position']
envs, blocked = dat['envs'], dat['blocked']       # SFPs / centroids / maps loaded, never used
...
events = np.nansum(trace_day, axis=1)             # includes all-NaN rows
...
'sequence': int(day // 10) + 1,
```

iii. Not discussed by the AI. The joblib route was chosen because it is the format the paper's own `main.py` uses, which trades some redundant I/O for exact consistency with the reference pipeline; the machine had ~1 TB of RAM, so the extra fields were not a practical constraint. Retaining all nine geometry dimensions, including the never-blocked one, was a deliberate choice for a complete and uniform description of the environment.
