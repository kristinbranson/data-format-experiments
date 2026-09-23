# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from seven hard-coded extensionless subject files in `/app/data` using `joblib.load`, not from the `.mat` files. For each subject, it reads a dictionary containing `trace`, `position`, `envs`, and `blocked`, then iterates over every recording day and later splits each day into trials.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
for subj_i, subject in enumerate(SUBJECTS):
    source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
    traces = source['trace']
    positions = source['position']
    envs = np.asarray(source['envs']).reshape(-1)
    blocked = source['blocked']
```

iii. In the trajectory, the agent inspected both `.mat` and extensionless files, then concluded that the extensionless `joblib` files were "the intended reference format" and that they contained the same structured data while being easier to load and inspect.

## 1-b. How are the data split into subjects?

i. Subjects are split by a hard-coded list of seven subject IDs. Each loaded file corresponds to one mouse, and the output `subject_idx` records which subject produced each session.

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

iii. In the trajectory, the agent identified seven animals in the dataset and treated each per-animal file as one subject.

## 1-c. How are the data split into sessions?

i. The AI treats each daily recording within a subject file as one session. It iterates over the first dimension of `trace`, so every day becomes one output session.

ii.
```python
traces = source['trace']       # session x tracked cell x aligned frame
...
for day in range(traces.shape[0]):
    tr = traces[day]
    pos = positions[day]
    ...
    neural.append(ses_neural)
    inputs.append(ses_input)
    outputs.append(ses_output)
```

iii. In trajectory steps 10 and 11, the agent justified this by noting that the paper reports 207 daily recordings and that neuron availability changes across days, so each daily recording should be its own decoder session.

## 1-d. How are the data split into trials?

i. The AI first rebins each session into 2,400 one-second bins across the full 40-minute recording, then groups those bins into forty 60-second trials. Each output trial therefore has 60 time bins, not 1,800 native frames.

ii.
```python
N_SECONDS = 40 * 60
TRIAL_SECONDS = 60
...
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    ses_neural.append(counts[:, start:stop].copy())
    ses_input.append(geometry_ts.copy())
    ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. In trajectory steps 10 and 11, the agent argued that sessions were nominally 40 minutes but slightly variable in frame count around 30 Hz, so equal one-second bins preserved all samples and guaranteed exactly forty one-minute trials per session.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filtering. Every session is turned into forty consecutive one-minute trials.

ii.
```python
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    ses_neural.append(counts[:, start:stop].copy())
    ses_input.append(geometry_ts.copy())
    ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The trajectory does not describe any trial rejection rule. The agent instead focused on preserving all 40 minutes of each recording after rebinning.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` array in each subject file.

ii.
```python
source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
traces = source['trace']       # session x tracked cell x aligned frame
...
tr = traces[day]
```

iii. In the trajectory, the agent concluded from inspection that `trace` already contains binary rising-phase calcium-event vectors and should therefore be used directly.

## 2-b. How is the `neural` data processed?

i. The AI removes invalid neurons, then temporally rebins the binary trace into one-second event counts by summing within 2,400 equal bins across the session. The result is stored as `uint8`.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
...
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
```

iii. In trajectory steps 10 and 11, the agent justified this by saying the supplied traces were already thresholded binary events and that summing them into one-second bins would preserve all data despite session lengths not being exactly 72,000 frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with any NaNs across time are removed from that session. In practice, the agent had earlier verified that these NaN masks are all-or-none across time for a neuron, so this removes neurons absent on that recording day.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
```

iii. In trajectory steps 8, 10, and 11, the agent observed that NaN rows mark neurons not detected in a given session and explicitly checked that NaNs were constant across time within a neuron.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to any behavioral or stimulus event. Instead, it declares the temporal alignment event to be the start of each consecutive one-minute segment and stores trials as fixed windows from that artificial boundary.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'start of each consecutive one-minute segment',
    'off_start': 0.0,
    'off_end': 60.0,
    'trial_duration_seconds': 60.0,
    ...
}
```

iii. In the trajectory, the agent reasoned that the recordings are continuous 40-minute sessions and therefore used consecutive one-minute segments as the organizing event for trialization.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 1,000 ms bins. Yes, the AI rebins the original sample stream into one-second bins before splitting into trials.

ii.
```python
N_SECONDS = 40 * 60
...
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
...
'metadata': {
    ...
    'time_bin_size': 1000.0,
    ...
}
```

iii. In trajectory steps 10 and 11, the agent justified the rebinning as a way to avoid dropping data when session lengths vary slightly around the nominal 30 Hz sampling rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the raw `blocked` variable for each session.

ii.
```python
blocked = source['blocked']
...
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
```

iii. In the trajectory, the agent identified `blocked` as a list of inaccessible 3x3 compartments, with `-1` used as a sentinel for the open square.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the blocked indices into a nine-element binary geometry vector and then repeats that vector across 60 time bins so each trial gets a `(9, 60)` time series, even though the geometry is static within a recording.

ii.
```python
geometry = np.zeros(9, dtype=np.float32)
blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
blocked_idx = blocked_idx[blocked_idx >= 0]
geometry[blocked_idx] = 1.0
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
...
ses_input.append(geometry_ts.copy())
```

iii. In trajectory steps 10 and 11 and in the script docstring, the agent justified the time repetition by saying decoder inputs were easier to represent on a common time axis, even though geometry does not vary within the recording.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the raw `position` array in each subject file.

ii.
```python
positions = source['position'] # session x (x,y) x aligned frame
...
pos = positions[day]
```

iii. In the trajectory, the agent described `position` as already aligned sample-for-sample with `trace` and spanning the 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI averages x and y positions within each one-second bin, then converts the averaged location into a 3x3 spatial class for each second.

ii.
```python
xy = aggregate_equal_bins(pos, edges, 'mean')
...
col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
```

iii. In trajectory steps 10 and 11, the agent justified this by keeping position aligned to the same one-second binning applied to the neural data and by using the paper’s 75 cm square divided into a 3x3 grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI uses a row-major 3x3 discretization. It divides both x and y by 25 cm, floors the result, clips to `[0, 2]`, and computes the final class as `row * 3 + col`.

ii.
```python
col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
spatial_bin = (row * 3 + col).astype(np.int8)[None, :]
```

iii. In the trajectory and script docstring, the agent justified this using the paper’s explicit 3x3 partition of the 75 cm square and noted that clipping handles exact boundary values such as 75 cm.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output with neural data by using the same equal-bin edges for both `trace` and `position`, then slicing both rebinned arrays into the same one-minute trial boundaries.

ii.
```python
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
ses_neural.append(counts[:, start:stop].copy())
ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. In trajectory steps 10 and 11, the agent said `trace` and `position` were already sample-aligned and therefore should be rebinned with identical temporal boundaries.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes neuron rows containing NaNs, raises an error if position contains NaNs, and avoids dropping leftover frames by repartitioning each session into 2,400 equal one-second bins regardless of exact frame count.

ii.
```python
valid = ~np.isnan(tr).any(axis=1)
tr = tr[valid]
if not len(tr):
    raise ValueError(f'No valid neurons: {subject}, day {day}')
if np.isnan(pos).any():
    raise ValueError(f'NaN position: {subject}, day {day}')
...
edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
```

iii. In trajectory steps 8 through 11, the agent justified these choices by observing that NaNs mark absent neurons, that sampled positions had no NaNs, and that varying frame counts around 30 Hz should not cause a nearly complete minute of data to be discarded.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading each large `joblib` subject file and repeatedly aggregating each session into 2,400 bins with explicit Python-level loops inside `aggregate_equal_bins`.

ii.
```python
source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
...
return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                 for i in range(len(edges)-1)], axis=-1)
...
return np.stack([x[..., edges[i]:edges[i+1]].mean(axis=-1)
                 for i in range(len(edges)-1)], axis=-1)
```

iii. In the trajectory, the agent repeatedly discussed memory control and I/O, and explicitly described the explicit loop in `aggregate_equal_bins` as "memory-efficient and unambiguous."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit list-comprehension loops in `aggregate_equal_bins` could have been vectorized or reshaped-based, and the per-trial loop that repeatedly copies fixed-size slices and duplicated geometry could also be reduced.

ii.
```python
return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                 for i in range(len(edges)-1)], axis=-1)
...
for start in range(0, N_SECONDS, TRIAL_SECONDS):
    stop = start + TRIAL_SECONDS
    ses_neural.append(counts[:, start:stop].copy())
    ses_input.append(geometry_ts.copy())
    ses_output.append(spatial_bin[:, start:stop].copy())
```

iii. The trajectory shows that the agent knowingly kept the explicit aggregation loop for clarity and memory behavior rather than for raw speed.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same geometry time series for every trial within a session, recomputes trial slicing with repeated `.copy()` operations, and runs separate aggregation passes over the same session boundaries for neural and position data.

ii.
```python
counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
xy = aggregate_equal_bins(pos, edges, 'mean')
...
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
...
ses_input.append(geometry_ts.copy())
```

iii. The trajectory emphasizes preserving a common time axis and exact forty-trial structure, which led the agent to accept this repeated processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code expands static session geometry into a full `(9, 60)` time series for every trial even though the environment geometry is constant within a trial and session. It also stores extensive `session_info` metadata that is not needed for decoder training.

ii.
```python
geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)
...
ses_input.append(geometry_ts.copy())
...
'session_info': session_info,
```

iii. In the trajectory and script docstring, the agent justified the geometry expansion as a convenience for keeping decoder inputs on a shared time axis, despite also acknowledging that geometry is static within a recording.
