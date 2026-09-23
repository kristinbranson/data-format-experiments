# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI chose to load the paper's Python `joblib` subject files rather than the `.mat` files. It hard-coded the seven animal IDs in `ANIMALS`, then for each animal loaded `/app/data/<animal>` with `joblib.load(...)`, extracted `trace`, `position`, `envs`, and `blocked`, and iterated over all days in the loaded arrays.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for ai, animal in enumerate(ANIMALS):
    print(f'Loading {animal} ...', flush=True)
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace = dat['trace']
    position = dat['position']
    envs = [e[0] for e in dat['envs']]
    blocked = dat['blocked']
    n_days = trace.shape[0]
```

iii. In the trajectory, the AI said it was following `georepca1/src/utils.py` and that the data layout in the paper repository used Python joblib files with fields such as `trace`, `position`, `envs`, and `blocked`. It explicitly decided to load "the joblib files distributed with the paper" rather than the `.mat` files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the hard-coded `ANIMALS` list. Each list entry is treated as one mouse, and the loop index `ai` becomes that mouse's subject index for every session from that animal.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for ai, animal in enumerate(ANIMALS):
    ...
    subject_idx.append(ai)
...
'subjects': ANIMALS,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. In the trajectory, the AI summarized the dataset as "7 mice, 207 sessions" and chose to enumerate those seven mice explicitly. Its justification was that the dataset structure was already known from exploring the joblib files and paper code.

## 1-c. How are the data split into sessions?

i. Each day within a subject's joblib file is treated as a separate session. The AI loops over `range(n_days)`, where `n_days = trace.shape[0]`, and appends one session entry per day.

ii.
```python
n_days = trace.shape[0]

for day in range(n_days):
    tr = trace[day]
    ...
    neural_all.append(neural_trials)
    input_all.append(input_trials)
    output_all.append(output_trials)
    subject_idx.append(ai)
    brain_region_idx.append(np.zeros(neural.shape[0], dtype=np.int64))
    session_info.append({'animal': animal, 'day': int(day),
                         'environment': str(envs[day]),
                         'n_neurons': int(neural.shape[0]),
                         'n_trials': n_trials})
```

iii. The trajectory states that the AI understood the data as "7 mice, 207 sessions (days)" and therefore used the day axis of the loaded arrays as the session axis.

## 1-d. How are the data split into trials?

i. Sessions are split into consecutive non-overlapping 1-minute trials after temporal rebinning to 100 ms bins. The AI defined `BINS_PER_TRIAL` as 600 bins per minute and sliced each session into `n_trials = n_bins // BINS_PER_TRIAL`, dropping any remainder.

ii.
```python
TRIAL_SECONDS = 60.0
TEMPORAL_BIN_FRAMES = 3
BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES))
...
n_bins = min(neural.shape[1], pos_bin.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
...
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(pos_bin[sl][None, :].copy())
```

iii. In the trajectory, the AI explicitly decided to "split each session into consecutive 1-minute trials (600 bins)" after applying the paper's temporal binning.

## 1-e. How are trials filtered based on quality controls?

i. There is no separate per-trial quality-control filter. The AI only enforces that a session must yield at least two full 1-minute trials; otherwise that session is skipped. Partial remainder bins at the end of a session are also discarded implicitly.

ii.
```python
n_bins = min(neural.shape[1], pos_bin.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
if n_trials < 2:
    print(f'  skipping {animal} day {day}: too short')
    continue
```

iii. The trajectory shows the AI was aware of the decoder requirement that each session needs at least two trials, so it added a guard for too-short sessions. It did not describe any further trial-specific QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` variable in the joblib files. The AI interpreted this field as the paper's binarized transient rising-phase activity.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace = dat['trace']        # (n_days, n_cells, n_frames), binary, NaN if unregistered
...
tr = trace[day]
```

iii. In the trajectory, the AI cited the README and methods text and concluded that `trace` contains "binary rise-extracted traces" and that this binarized vector was treated as the firing rate in the paper.

## 2-b. How is the `neural` data processed?

i. The AI kept the binarized traces, removed some cells, then Gaussian-smoothed them with `sigma=3` frames and average-pooled them into non-overlapping 3-frame bins. It treated the resulting 100 ms binned signal as the final neural data.

ii.
```python
SMOOTH_SIGMA = 3
TEMPORAL_BIN_FRAMES = 3
...
def bin_time(x, n_frames_bin):
    n = x.shape[-1] // n_frames_bin
    x = x[..., :n * n_frames_bin]
    return x.reshape(*x.shape[:-1], n, n_frames_bin).mean(axis=-1)
...
tr = tr[keep].astype(np.float32)
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA, axis=1)
neural = bin_time(tr, TEMPORAL_BIN_FRAMES).astype(np.float32)
```

iii. The trajectory repeatedly says the AI was trying to match `utils.fit_decoder` from the paper code. It explicitly justified the smoothing and 3-frame average pooling as the paper's decoding pipeline: "gaussian smooth traces (sigma=3 frames), average-pool 3 frames (100 ms bins)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applied two neural filters per session: it removed cells that were not registered on that day, identified by NaNs, and it removed cells with `<= 5` total transients in that session.

ii.
```python
tr = trace[day]
registered = ~np.isnan(tr).any(axis=1)
events = np.nansum(tr, axis=1)
keep = registered & (events > EVENT_THRESHOLD)
...
tr = tr[keep].astype(np.float32)
```

iii. In the trajectory, the AI justified these filters by reference to the paper code: unregistered cells are NaN, and "as in `utils.decode_position_within` (`cell_threshold=5`) cells with <= 5 transients in a session are also dropped."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI did not use a stimulus or behavior event. It treated the start of each recording session as the effective alignment point and then cut the continuous session into consecutive non-overlapping 1-minute windows.

ii.
```python
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
...
'temporal_alignment_event': (
    'start of the recording session; each session is cut into consecutive '
    'non-overlapping 1-minute trials'),
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The trajectory states that there was no task event to align to and that the correct organization was continuous recording segmented into one-minute trials. The AI therefore documented session start as the alignment reference in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins rather than the original 30 Hz frames. The AI rebinned both neural and position data by averaging non-overlapping groups of 3 frames.

ii.
```python
FPS = 30.0
TEMPORAL_BIN_FRAMES = 3
BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0
...
neural = bin_time(tr, TEMPORAL_BIN_FRAMES).astype(np.float32)
pos = bin_time(position[day].astype(np.float64), TEMPORAL_BIN_FRAMES)
...
'time_bin_size': BIN_MS,
```

iii. In the trajectory, the AI explicitly chose "100 ms bins" because it wanted to mirror `utils.fit_decoder/test_decoder` from the paper repository.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the `blocked` field for each day, not from the `envs` string labels. The AI used `blocked[day]` to determine which of the 9 partitions were inaccessible.

ii.
```python
blocked = dat['blocked']
...
geom = geometry_vector(blocked[day])
```

iii. In the trajectory, the AI said it verified that the `blocked` field uses the same flat 3x3 index convention as its position bins, so it used `blocked` as the decoder input for geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converted the blocked-partition list for each day into a 9-dimensional binary vector, with `1` meaning blocked and `0` meaning accessible. The resulting vector is static within a session and copied once per trial.

ii.
```python
def geometry_vector(blocked_day):
    vec = np.zeros(N_SPATIAL_BINS * N_SPATIAL_BINS, dtype=np.float32)
    b = np.atleast_1d(np.array(blocked_day[0]).ravel())
    for i in b:
        if i >= 0:
            vec[int(i)] = 1.0
    return vec
...
geom = geometry_vector(blocked[day])
...
input_trials.append(geom.copy())
```

iii. The trajectory states that the AI wanted a static "9-dim binary geometry vector (1 = blocked)" and chose that representation because it matches the experiment's 3x3 partition design and the dataset's indexing.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the raw `position` variable in the joblib file for each day.

ii.
```python
position = dat['position']  # (n_days, 2, n_frames), cm
...
pos = bin_time(position[day].astype(np.float64), TEMPORAL_BIN_FRAMES)
```

iii. In the trajectory, the AI summarized `position` as the x-y location in centimeters recorded at 30 Hz and used it as the source for the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first averaged position over 3-frame temporal bins, then discretized each averaged x-y sample into one of nine 25 cm by 25 cm spatial bins covering the 75 cm square arena. The category label is `ybin * 3 + xbin`.

ii.
```python
pos = bin_time(position[day].astype(np.float64), TEMPORAL_BIN_FRAMES)
bin_size_cm = ARENA_SIZE / N_SPATIAL_BINS
xb = np.clip((pos[0] // bin_size_cm).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size_cm).astype(int), 0, N_SPATIAL_BINS - 1)
pos_bin = (yb * N_SPATIAL_BINS + xb).astype(np.int64)
```

iii. The trajectory says the AI selected 25 cm bins because the arena is 75 x 75 cm and the experiment uses a 3x3 partition design. It also said it verified that `ybin*3 + xbin` matches the indexing used in `blocked`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The continuous x and y coordinates are thresholded into 3 equal bins along each axis using 25 cm boundaries. The resulting 3 x 3 grid is flattened to category labels `0` through `8` using `ybin * 3 + xbin`.

ii.
```python
bin_size_cm = ARENA_SIZE / N_SPATIAL_BINS
xb = np.clip((pos[0] // bin_size_cm).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size_cm).astype(int), 0, N_SPATIAL_BINS - 1)
pos_bin = (yb * N_SPATIAL_BINS + xb).astype(np.int64)
```

iii. In the trajectory, the AI described this explicitly as "position discretized into the experiment's 3x3 partition grid with 25 cm bins, bin index = ybin*3+xbin".

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligned output and neural data by applying the same temporal binning to both streams, truncating them to the same binned length with `min(...)`, and then slicing both into trials with the same trial boundaries.

ii.
```python
neural = bin_time(tr, TEMPORAL_BIN_FRAMES).astype(np.float32)
pos = bin_time(position[day].astype(np.float64), TEMPORAL_BIN_FRAMES)
...
n_bins = min(neural.shape[1], pos_bin.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
...
sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
output_trials.append(pos_bin[sl][None, :].copy())
```

iii. The trajectory states that both behavior and neural activity were acquired at 30 Hz and that the AI intentionally binned them together into the same 100 ms bins before trializing so they would remain aligned.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled NaN-marked unregistered cells by dropping them, removed low-event cells, truncated neural and position streams to the shorter binned length with `min(...)`, discarded leftover partial bins and partial trials, and skipped sessions that would have fewer than two full trials.

ii.
```python
registered = ~np.isnan(tr).any(axis=1)
events = np.nansum(tr, axis=1)
keep = registered & (events > EVENT_THRESHOLD)
...
n = x.shape[-1] // n_frames_bin
x = x[..., :n * n_frames_bin]
...
n_bins = min(neural.shape[1], pos_bin.shape[0])
n_trials = n_bins // BINS_PER_TRIAL
if n_trials < 2:
    print(f'  skipping {animal} day {day}: too short')
    continue
```

iii. The trajectory justifies the NaN handling by the paper's representation of unregistered cells and the low-event filter by `cell_threshold=5` from the paper decoder. The length truncation and short-session skip were practical guards to satisfy the decoder format.

## 6-a. What are the most time-consuming steps of the code?

i. The AI's code spends most of its time loading large per-animal joblib files and processing full-session neural arrays with Gaussian smoothing and temporal binning. Trialization itself is relatively cheap compared with the session-wide I/O and filtering work.

ii.
```python
for ai, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    ...
    tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA, axis=1)
    neural = bin_time(tr, TEMPORAL_BIN_FRAMES).astype(np.float32)
```

iii. In the trajectory, the AI noted that the decoder-sized dataset was large, that the output pickle reached 6.2 GB, and that the full conversion run took about 3 minutes 39 seconds while repeatedly loading large joblib files.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization opportunities are the Python loop over trials when appending trial slices and the loop over blocked indices in `geometry_vector`. Both could be replaced with more array-based operations or precomputed reshapes.

ii.
```python
for i in b:
    if i >= 0:
        vec[int(i)] = 1.0
...
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(pos_bin[sl][None, :].copy())
```

iii. The trajectory does not contain an explicit optimization discussion here. This conclusion comes from the implemented code structure: the AI favored straightforward Python loops over more vectorized reshaping or broadcasting.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same day-wise workflow for every session: compute registration and event filters, smooth and bin neural data, bin position, and then repeatedly copy the same static geometry vector once per trial. It also re-runs the same trial-slicing logic session by session.

ii.
```python
for day in range(n_days):
    tr = trace[day]
    registered = ~np.isnan(tr).any(axis=1)
    events = np.nansum(tr, axis=1)
    keep = registered & (events > EVENT_THRESHOLD)
    ...
    geom = geometry_vector(blocked[day])
    ...
    for t in range(n_trials):
        ...
        input_trials.append(geom.copy())
```

iii. The trajectory shows the AI intentionally used a uniform per-session pipeline derived from the paper code, so this repeated work is a consequence of that design rather than a separate justified optimization choice.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores some information that the downstream decoder does not need, including per-session metadata from `envs`, counters such as `n_cells_total` and `n_dropped_silent` used only for logging, and repeated per-trial copies of the same static geometry vector. It also computes `n_bins = min(...)` even though the two streams are expected to be co-timed.

ii.
```python
envs = [e[0] for e in dat['envs']]
...
n_dropped_silent = 0
n_cells_total = 0
...
n_bins = min(neural.shape[1], pos_bin.shape[0])
...
input_trials.append(geom.copy())
...
session_info.append({'animal': animal, 'day': int(day),
                     'environment': str(envs[day]),
                     'n_neurons': int(neural.shape[0]),
                     'n_trials': n_trials})
```

iii. The trajectory indicates these additions were mainly for bookkeeping, validation, and documentation. They do not drive the decoder itself, which only consumes the final `neural`, `input`, and `output` arrays.
