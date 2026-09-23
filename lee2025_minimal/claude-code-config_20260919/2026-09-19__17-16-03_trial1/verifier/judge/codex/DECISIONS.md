# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from seven hard-coded per-animal `joblib` files in `/app/data`, not from the `.mat` files. For each animal, it loads the animal dictionary and reads session-wise arrays for `trace`, `position`, `envs`, and `blocked`, then iterates over all recording days.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for animal_idx, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace, position = dat['trace'], dat['position']
    envs, blocked = dat['envs'], dat['blocked']
    n_days = trace.shape[0]
```

iii. In trajectory step 15, the agent inspected `/app/data/QLAK-CA1-08` and found a per-animal dictionary with `trace`, `position`, `envs`, and `blocked`. In step 50 it justified the loader by saying the source data were “one joblib file per mouse” and that it wanted to mirror the paper’s decoding pipeline from the provided code.

## 1-b. How are the data split into subjects?

i. Subjects are the seven animal IDs listed in `ANIMALS`. Each loaded joblib file corresponds to one mouse, and the subject list is pre-populated from that constant list.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

data = {
    ...
    'subjects': list(ANIMALS), 'subject_idx': [],
    ...
}

for animal_idx, animal in enumerate(ANIMALS):
    ...
    data['subject_idx'].append(animal_idx)
```

iii. In steps 14 and 50, the agent adopted the same seven-animal list used in the paper code and treated each animal/day block as belonging to that mouse. Step 70 explicitly described the output as “all 7 mice.”

## 1-c. How are the data split into sessions?

i. Each day within an animal’s arrays is treated as one session. The script iterates over `range(n_days)` and appends one session entry per day to `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
n_days = trace.shape[0]

for day in range(n_days):
    neural, keep = session_neural(trace[day])
    bins = session_position_bins(position[day], bin_width)
    geometry = blocked_vector(blocked[day])
    ...
    data['neural'].append(neural_trials)
    data['input'].append(input_trials)
    data['output'].append(output_trials)
```

iii. In step 15, the agent confirmed `trace` had shape `(31, 515, 71866)` for one animal, which it interpreted as days × cells × frames. In step 70 it summarized this decision as “one per mouse-day.”

## 1-d. How are the data split into trials?

i. Within each session/day, the AI cuts the session into consecutive, non-overlapping 1-minute trials after temporal pooling. Because it rebins to 100 ms first, each trial has `600` bins (`60 s * 30 Hz / 3`), and any incomplete remainder is dropped.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TRIAL_SECONDS = 60
TRIAL_BINS = TRIAL_SECONDS * FPS // TEMPORAL_BIN_FRAMES

n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geometry.copy())
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. In step 50, the agent wrote that each 40-minute session is cut into “consecutive, non-overlapping 1-min trials (600 bins of 100 ms)” and that the incomplete remainder is dropped. Step 70 repeated the same justification.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a separate trial-level quality-control filter. It keeps every contiguous 1-minute segment that survives the session-length truncation; trial count is limited only by the available binned timepoints.

ii.
```python
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geometry.copy())
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. In step 50 and again in step 70, the agent explicitly justified *not* removing slow or stationary periods at the trial level, saying it kept all frames so trials would remain contiguous 1-minute epochs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays are derived from the raw `trace` variable in the per-animal joblib data. The script treats `trace[day]` as the per-session cell-by-frame activity matrix.

ii.
```python
Source data ... contains, for each of the ~31 recording days of each mouse:
    trace     (n_cells, n_frames)  binarised rising phase of calcium transients (30 Hz).

...
trace, position = dat['trace'], dat['position']
...
neural, keep = session_neural(trace[day])
```

iii. In steps 15 and 17, the agent inspected the data and verified that `trace` exists, has binary values `0/1`, and contains NaN rows for cells not registered that day. In step 50 it described `trace` as the “binarised transient-rising-phase vector.”

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps to `trace`: Gaussian smoothing along time with `sigma=3` frames, then non-overlapping average pooling over groups of 3 frames. The result is stored as `float32` and has shape `(kept_cells, n_time_bins)`.

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
    smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
    return pool_time(smoothed).astype(np.float32), keep
```

iii. In step 12, the agent examined `fit_decoder` and saw that the paper’s code smooths `traces` with `gaussian_filter1d(..., sigma=temporal_bin_size)` and average-pools them. In step 50 it said this was chosen to “mirror” `decode_position_within` / `fit_decoder`, and step 70 repeated that the conversion used the paper’s 100 ms decoder preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells that are registered on that day and have more than five events in the session. Registration is checked from NaNs in the first frame; event count is computed with `np.nansum(trace_day, axis=1)`.

ii.
```python
MIN_EVENTS = 5

def session_neural(trace_day):
    registered = ~np.isnan(trace_day[:, 0])
    events = np.nansum(trace_day, axis=1)
    keep = registered & (events > MIN_EVENTS)
    trace = trace_day[keep].astype(np.float32)
```

iii. In step 34, the agent verified that NaNs are “all-or-none” for each cell/day. In steps 50 and 70 it justified the extra `>5` event threshold by citing the paper decoder’s `cell_threshold=5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. The AI defines the alignment event as the start of each artificial 1-minute trial cut from a continuous recording, then slices the already binned neural data into those contiguous trial windows.

ii.
```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))

...
'temporal_alignment_event': (
    'Start of each 1-minute trial. Sessions are continuous 40-minute free '
    'exploration recordings with no discrete trial events, so each session is cut '
    'into consecutive non-overlapping 1-minute trials.'),
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. In step 50, the agent explicitly stated that sessions are continuous free-exploration recordings with “no discrete trial events,” so trial starts were used as the alignment marker. Step 70 repeated that rationale.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. The AI rebins the original 30 Hz data by average-pooling every 3 frames into one time bin after Gaussian smoothing.

ii.
```python
FPS = 30
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS

def pool_time(x):
    n = x.shape[-1] // TEMPORAL_BIN_FRAMES
    return x[..., :n * TEMPORAL_BIN_FRAMES].reshape(
        x.shape[:-1] + (n, TEMPORAL_BIN_FRAMES)).mean(-1)
```

iii. In step 12, the agent extracted `temporal_bin_size=3` from the paper code. In steps 50 and 70 it justified the 100 ms representation as reproducing the paper’s decoder preprocessing.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the raw `blocked` variable for each day/session. The AI does not derive geometry from `envs`; it only reads `envs` for metadata.

ii.
```python
trace, position = dat['trace'], dat['position']
envs, blocked = dat['envs'], dat['blocked']
...
geometry = blocked_vector(blocked[day])

def blocked_vector(blocked_day):
    idx = np.array(blocked_day[0]).ravel().astype(int)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if idx.size and idx[0] != -1:
        vec[idx] = 1.0
    return vec
```

iii. In steps 15 and 17, the agent inspected the `blocked` field and verified its structure, including `-1` for unblocked sessions. Step 70 says the input is “taken from the `blocked` field.”

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the blocked-partition indices into a length-9 binary vector. The vector is static for the whole session and is copied once per trial.

ii.
```python
def blocked_vector(blocked_day):
    idx = np.array(blocked_day[0]).ravel().astype(int)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if idx.size and idx[0] != -1:
        vec[idx] = 1.0
    return vec

...
geometry = blocked_vector(blocked[day])
...
input_trials.append(geometry.copy())
```

iii. In step 50, the agent described the input as “which of the 9 partitions are blocked (static/trial).” Step 70 further justified this as a 9-dimensional binary vector and noted it kept all nine entries even if some partitions are never blocked.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw `position` variable for each session/day. The script uses `position[day]`, which is the 2D `(x, y)` trajectory in centimeters.

ii.
```python
Source data ... contains, for each of the ~31 recording days of each mouse:
    position  (2, n_frames)        head position in cm (x, y) in the 75 x 75 cm arena.

...
trace, position = dat['trace'], dat['position']
...
bins = session_position_bins(position[day], bin_width)
```

iii. In steps 15 and 17, the agent verified the shape and numeric range of `position`, including that the coordinates span roughly `0` to `75` cm. Step 50 described the output as coming from head position in the `75 x 75 cm` arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first average-pools position over 3-frame windows to match the 100 ms neural bins. It then divides pooled `x` and `y` by a per-animal `bin_width`, floors by integer cast, clips to `0..2`, and converts the 2D location into a single 0..8 partition index.

ii.
```python
def session_position_bins(position_day, bin_width):
    pooled = pool_time(position_day.astype(np.float64))
    xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)

...
bin_width = (np.nanmax(position) + BUFFER) / N_SPATIAL_BINS
...
bins = session_position_bins(position[day], bin_width)
```

iii. In step 12, the agent inspected `decode_position_within`, where behavior is temporally pooled before decoding. In steps 50 and 70 it justified the per-animal `bin_width` as mirroring the paper code so that the spatial grid is consistent across an animal’s sessions and lines up with the physical partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The pooled continuous position is thresholded into a 3 x 3 grid. The final category is `3 * y_bin + x_bin`, where each axis index is clipped to the set `{0, 1, 2}`.

ii.
```python
xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. In step 50, the agent explained that blocked partitions are indexed as `[[0,1,2],[3,4,5],[6,7,8]]` and used that same numbering for position classes. In step 70 it summarized the output as partition index `3*(y//25cm) + (x//25cm)`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns position and neural data by applying the same temporal pooling factor to both streams and then slicing both with the same trial boundaries. After pooling, both arrays are indexed by the same 100 ms bins.

ii.
```python
def session_neural(trace_day):
    ...
    smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
    return pool_time(smoothed).astype(np.float32), keep

def session_position_bins(position_day, bin_width):
    pooled = pool_time(position_day.astype(np.float64))
    ...

for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    output_trials.append(bins[sl][np.newaxis, :])
```

iii. In step 50, the agent said both streams were converted into the paper’s 100 ms decoder bins. The rationale in steps 50 and 70 is that the output should be temporally matched to the rebinned neural activity.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes cells with NaN traces for a day, removes cells with at most five events, clips pooled positions back into valid spatial bins, and drops incomplete trailing time at two points: leftover frames inside `pool_time` and any leftover sub-minute remainder when forming trials. It does not impute missing values.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
events = np.nansum(trace_day, axis=1)
keep = registered & (events > MIN_EVENTS)

def pool_time(x):
    n = x.shape[-1] // TEMPORAL_BIN_FRAMES
    return x[..., :n * TEMPORAL_BIN_FRAMES].reshape(
        x.shape[:-1] + (n, TEMPORAL_BIN_FRAMES)).mean(-1)

xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
```

iii. In step 34, the agent checked that NaNs in `trace` occur as whole missing cell/day rows, which motivated dropping unregistered cells rather than imputing them. In steps 50 and 70 it justified the extra cell-event threshold from the paper decoder and justified retaining slow frames instead of filtering them out.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming work in the AI code is likely loading the large per-animal joblib files and then running Gaussian smoothing over every kept neuron for every session. Trial construction and metadata assembly are smaller by comparison.

ii.
```python
for animal_idx, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    ...
    for day in range(n_days):
        neural, keep = session_neural(trace[day])

def session_neural(trace_day):
    ...
    smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
    return pool_time(smoothed).astype(np.float32), keep
```

iii. The trajectory does not contain an explicit performance analysis, but step 15 shows the agent inspected very large arrays and step 50 added `gaussian_filter1d` across all sessions as part of the chosen preprocessing. That makes file loading and smoothing the obvious dominant costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop that slices `neural`, duplicates `geometry`, and slices `bins` could have been replaced with a reshape/split-based approach. The day/session loop is still needed structurally, but the inner trial-building loop is the clearest vectorization target.

ii.
```python
for day in range(n_days):
    ...
    n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
    neural_trials, input_trials, output_trials = [], [], []
    for t in range(n_trials):
        sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
        neural_trials.append(np.ascontiguousarray(neural[:, sl]))
        input_trials.append(geometry.copy())
        output_trials.append(bins[sl][np.newaxis, :])
```

iii. The trajectory does not show the agent optimizing for vectorization; it focused on reproducing paper logic and obtaining a decoder-valid dataset. The repeated trial loop is therefore an implementation convenience rather than an explicitly justified design choice.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly copies the same static geometry vector once per trial, repeatedly slices sessions trial-by-trial in Python, and repeatedly computes pooled outputs per day. It also repeats string/metadata work for `session_info` on every session.

ii.
```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(geometry.copy())
    output_trials.append(bins[sl][np.newaxis, :])

session_info.append({
    'subject': animal,
    'day': int(day),
    'sequence': int(day // 10) + 1,
    'environment': str(envs[day][0]),
    'blocked_partitions': np.where(geometry > 0)[0].tolist(),
    'n_neurons': int(keep.sum()),
    'n_trials': int(n_trials),
})
```

iii. There is no explicit trajectory justification for repeated processing. The agent’s discussion in steps 50 and 70 is about fidelity to the paper, not computational efficiency.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `envs` even though it only uses it for metadata and logging, builds a detailed `session_info` structure that the downstream decoder does not consume, and copies a per-session geometry vector into every trial even though it is static. The decoder also does not use many of the verbose metadata strings.

ii.
```python
envs, blocked = dat['envs'], dat['blocked']
...
input_trials.append(geometry.copy())
...
session_info.append({
    'subject': animal,
    'day': int(day),
    'sequence': int(day // 10) + 1,
    'environment': str(envs[day][0]),
    'blocked_partitions': np.where(geometry > 0)[0].tolist(),
    'n_neurons': int(keep.sum()),
    'n_trials': int(n_trials),
})
...
'session_info': session_info,
```

iii. The trajectory does not explicitly defend these extra pieces as necessary for decoding. They appear to have been added for interpretability and record-keeping rather than because the downstream analysis requires them.
