# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from extensionless joblib-serialized files (one per subject) in `/app/data/`. Each file is a dictionary keyed by the subject name, containing arrays for `trace`, `position`, `blocked`, and `envs`. The AI loads each subject file sequentially using `joblib.load()`.

ii.
```python
source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
traces = source['trace']       # session x tracked cell x aligned frame
positions = source['position'] # session x (x,y) x aligned frame
envs = np.asarray(source['envs']).reshape(-1)
blocked = source['blocked']
```

iii. The agent discovered that both `.mat` (HDF5) and extensionless (joblib) files exist for each subject and contain the same data. It chose joblib because it provides direct numpy array access without HDF5 reference dereferencing. The agent verified that `joblib.load()` successfully returns dictionaries with the expected keys (`trace`, `position`, `blocked`, `envs`).

## 1-b. How are the data split into subjects?

i. The AI hardcodes a list of seven subject names matching the seven data files. Each subject corresponds to one joblib file. The subject index is tracked via enumeration.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
for subj_i, subject in enumerate(SUBJECTS):
    ...
    subject_idx.append(subj_i)
```

iii. The agent identified seven subjects from the data directory listing and confirmed this matches the paper's description.

## 1-c. How are the data split into sessions?

i. Each subject's data contains multiple daily recordings (31 for most subjects, 21 for QLAK-CA1-51), totaling 207 sessions. Each daily recording becomes one session in the output. The agent iterates over `traces.shape[0]` (number of recordings per subject).

ii.
```python
for day in range(traces.shape[0]):
    tr = traces[day]
    pos = positions[day]
    ...
```

iii. The agent confirmed that the paper reports 207 sessions across 7 mice. Each daily recording is a natural session boundary since neuron populations differ by day.

## 1-d. How are the data split into trials?

i. Each 40-minute recording session is split into 40 non-overlapping one-minute trials. Since the AI rebins data to 2400 one-second bins per session, each trial contains 60 consecutive one-second bins.

ii.
```python
N_SECONDS = 40 * 60  # 2400
TRIAL_SECONDS = 60
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    ses_neural.append(counts[:, start:stop].copy())
    ses_input.append(geometry_ts.copy())
    ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The agent reasoned that since the paper specifies 40-minute recordings and the instructions specify 1-minute trials, each session yields exactly 40 trials. By first rebinning to exactly 2400 one-second bins, the trial splitting is clean with no remainder.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All 40 trials from each session are included.

ii. N/A

iii. No trial quality filtering is mentioned in the paper or instructions, and the agent did not apply any.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains the paper's thresholded binary rising-phase calcium-event vectors. The trace array has shape `(n_sessions, n_tracked_neurons, n_frames)`.

ii.
```python
traces = source['trace']       # session x tracked cell x aligned frame
...
tr = traces[day]
```

iii. The agent inspected the trace data and confirmed it contains only values 0, 1, and NaN, consistent with the paper's description of binarized rising-phase calcium transient vectors.

## 2-b. How is the `neural` data processed?

i. The AI applies temporal rebinning: the raw ~30 Hz binary trace is rebinned into 2400 equal one-second bins by summing the binary events within each bin. The result is stored as uint8 (event counts per second). NaN neurons are first removed (see 2-c).

ii.
```python
nframes = tr.shape[1]
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
```

```python
def aggregate_equal_bins(x, edges, reducer='sum'):
    if reducer == 'sum':
        return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                         for i in range(len(edges)-1)], axis=-1)
```

iii. The agent reasoned that keeping the raw 30 Hz data would produce a multi-gigabyte file and noted that the paper's own decoder uses spatially binned data. It chose 1-second bins to balance temporal resolution with data size, using `np.linspace` to create equal-sized bins that accommodate the slight variation in frame counts across subjects (~71,866–72,219 frames).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all-NaN across time (not recorded in that session) are removed. The agent verified that NaN masks are constant across time within each recording (no partial NaN neurons).

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
```

iii. The agent confirmed that each neuron is either fully present or fully NaN within a session, with no partial NaN patterns. This yields 69,744 total session-neuron instances, matching the paper's reported number of rate maps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is performed. The recording is continuous, and trials are consecutive one-minute segments starting from the beginning of the session.

ii. N/A (implicit from the sequential trial splitting)

iii. There is no stimulus onset or discrete event to align to; the experiment involves continuous free exploration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins from the native ~30 Hz sampling rate to 1-second bins (1000 ms). Binary events are summed within each bin. This produces 2400 time bins per 40-minute session.

ii.
```python
N_SECONDS = 40 * 60  # 2400
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
```

Metadata:
```python
'time_bin_size': 1000.0,
```

iii. The agent reasoned that temporal rebinning was necessary to keep file size manageable and noted that arrays differ slightly around 30 Hz after timestamp alignment, so equal-sized bins via `np.linspace` would preserve all data without dropping frames.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable in the source data, which contains indices of blocked spatial compartments for each recording session. A value of `-1` indicates no positions are blocked (the square environment).

ii.
```python
blocked = source['blocked']
...
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
blocked_idx = blocked_idx[blocked_idx >= 0]  # square uses sentinel -1
```

iii. The agent inspected the blocked data for all subjects and confirmed it encodes which of the 9 spatial compartments are blocked, with -1 as sentinel for the open square.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary vector (one-hot encoding), where 1 means blocked and 0 means accessible. The vector is then repeated over time to create a (9, 60) array per trial, matching the trial's temporal dimension.

ii.
```python
geometry = np.zeros(9, dtype=np.float32)
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
blocked_idx = blocked_idx[blocked_idx >= 0]
geometry[blocked_idx] = 1.0
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
```

iii. The agent made geometry time-varying (repeated over time) because the decoder input format expects `(d_input, n_timepoints)`. This ensures compatibility with the time-varying decoder framework.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable in the source data, which contains 2D (x, y) coordinates of the mouse in the arena at each frame, with shape `(n_sessions, 2, n_frames)`.

ii.
```python
positions = source['position']  # session x (x,y) x aligned frame
...
pos = positions[day]
```

iii. The agent confirmed that position data has no NaN values and coordinates range from 0 to 75 cm, matching the paper's 75×75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The continuous 2D position is first temporally averaged within each 1-second bin (matching the neural data rebinning), then discretized into a 3×3 grid using 25 cm bin edges. The grid label is computed as `row * 3 + col`, yielding 9 classes (0–8).

ii.
```python
xy = aggregate_equal_bins(pos, edges, 'mean')

col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
```

iii. The agent used `np.floor(coord / 25.0)` to discretize each axis into 3 bins of 25 cm each, clipping to [0, 2] to handle boundary values at exactly 75 cm.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by dividing each coordinate by 25 cm and flooring, then clipping to valid bin indices [0, 2]. The combined label is `row * 3 + col`, producing 9 spatial categories.

ii.
```python
col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
```

iii. The 25 cm bins divide the 75 cm arena into a 3×3 grid as specified in the instructions. Clipping handles edge cases where position equals exactly 75 cm.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by using the same temporal rebinning boundaries (`edges`). Both are aggregated from the same raw frames: neural events are summed and positions are averaged within each 1-second bin. The same bin indices are then used for trial splitting.

ii.
```python
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
```

iii. Since trace and position are already sample-aligned in the source data (both at ~30 Hz), using the same bin edges for both ensures continued alignment after rebinning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neurons with all-NaN values are removed per session (see 2-c). The code raises an error if any session has zero valid neurons or if position data contains NaN. No other missing data handling is applied.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
if np.isnan(pos).any():
    raise ValueError(f'NaN position: {subject}, day {day}')
```

iii. The agent verified that NaN patterns are all-or-nothing per neuron per session, and that position data has no NaN values.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the large joblib files (each subject's file is 70–780 MB), followed by the temporal rebinning (`aggregate_equal_bins`) which loops over 2400 one-second bins for each neuron array.

ii.
```python
source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
...
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
```

iii. The agent observed that loading and processing all seven subjects takes several minutes due to I/O and the large array sizes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `aggregate_equal_bins` function uses a Python loop over 2400 bins with `np.stack`, creating 2400 intermediate sliced arrays. This could be replaced with `np.add.reduceat` or reshaped operations for better vectorization.

ii.
```python
def aggregate_equal_bins(x, edges, reducer='sum'):
    if reducer == 'sum':
        return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                         for i in range(len(edges)-1)], axis=-1)
```

iii. The agent noted in comments that `reduceat` has edge cases but chose the explicit loop for clarity. This is the primary candidate for vectorization.

## 6-c. What processing does the code repeat multiple times?

i. The geometry time-series (`geometry_ts`) is created once per session but `.copy()` is called 40 times (once per trial), creating 40 identical copies per session. This is redundant since the array could be shared.

ii.
```python
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
...
ses_input.append(geometry_ts.copy())
```

iii. The agent did not discuss this explicitly; the copies ensure independent trial arrays but are wasteful for a static quantity.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code makes the geometry input time-varying by repeating it over the 60 time bins per trial, producing (9, 60) arrays. However, the geometry is static per session and the decoder could accept a static (9,) vector. The temporal repetition adds data volume without information content.

ii.
```python
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
```

iii. The agent's docstring states this was done "because decoder inputs are represented with a common time axis," but the target format specification allows `(n_input)` (static) as well as `(n_input, n_timepoints)`.
