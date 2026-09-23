# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, opens one MATLAB v7.3 / HDF5 `.mat` file per animal with `h5py`, reads per-animal session lists from `envs` and `blocked`, and then loads `position` and `trace` for each day/session from HDF5 object references. Trials are not loaded separately from disk; they are created later from each continuous session.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a, animal in enumerate(ANIMALS):
    path = os.path.join(DATA_DIR, f"{animal}.mat")
    for session in process_animal(path, animal):
        data["neural"].append(session["neural"])
```

```python
with h5py.File(path, "r") as f:
    envs, blocked = load_session_list(f)
    for day, (env, blk) in enumerate(zip(envs, blocked)):
        position = f[f["position"][day, 0]][:]
        trace = f[f["trace"][day, 0]][:]
```

iii. In the trajectory summary, the AI said it would "Read the MATLAB v7.3 files directly with h5py" and process "All 7 mice, all sessions," using the paper's headline counts as a sanity check that it had loaded everything.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. Each listed `.mat` file is treated as one mouse, and `subject_idx` is the index of that mouse in `ANIMALS`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

data = {
    "subjects": ANIMALS, "subject_idx": [],
}

for a, animal in enumerate(ANIMALS):
    ...
    data["subject_idx"].append(a)
```

iii. The AI's trajectory justification was implicit rather than separate: in its final summary it described processing "All 7 mice" and used those seven animal IDs as the subject list.

## 1-c. How are the data split into sessions?

i. Each day/session inside one animal's `.mat` file becomes one output session. The code iterates over the paired `envs` and `blocked` entries and, for each `day`, loads one `position` array and one `trace` array.

ii.
```python
def process_animal(path, animal):
    sessions = []
    with h5py.File(path, "r") as f:
        envs, blocked = load_session_list(f)
        for day, (env, blk) in enumerate(zip(envs, blocked)):
            position = f[f["position"][day, 0]][:]
            trace = f[f["trace"][day, 0]][:]
            ...
            sessions.append({
                "neural": neural,
                "input": inputs,
                "output": outputs,
            })
```

iii. The AI justified this by matching the paper/data organization: one file per mouse, one continuous recording per day. It also cited reproducing the 207-session headline count as confirmation that session splitting matched the source data.

## 1-d. How are the data split into trials?

i. Each continuous session is first temporally rebinned into 500 ms bins, then split into consecutive, non-overlapping 60 s windows. Because 60 s / 0.5 s = 120, each trial is 120 time bins. Any trailing fragment shorter than one full 60 s trial is dropped.

ii.
```python
BIN_FRAMES = 15                                  # 15 frames @ 30 Hz = 500 ms
TRIAL_SECONDS = 60.0
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / BIN_FRAMES))   # 120 bins per trial
```

```python
n_bins = n_frames // BIN_FRAMES
...
n_trials = n_bins // TRIAL_BINS
...
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural.append(np.ascontiguousarray(rates[sl].T))
    inputs.append(geo.copy())
    outputs.append(labels[sl][np.newaxis, :])
```

iii. The AI explicitly justified trials as "Consecutive non-overlapping 60 s windows (120 bins); the trailing < 1 min fragment is dropped so all trials span equal time."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. The only trial-level exclusion is implicit: incomplete trailing data shorter than one full 60 s trial is discarded.

ii.
```python
n_bins = n_frames // BIN_FRAMES
...
n_trials = n_bins // TRIAL_BINS
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
```

iii. The trajectory summary states that the "trailing < 1 min fragment is dropped so all trials span equal time." No separate per-trial quality-control rule was described.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays are derived from the raw `trace` variable in each `.mat` file. The AI interprets this as the paper's binarized rising-phase calcium-transient signal.

ii.
```python
trace = f[f["trace"][day, 0]][:]
```

iii. In its final summary, the AI said "`trace` is the paper's binarised transient rising-phase vector, treated as firing rate."

## 2-b. How is the `neural` data processed?

i. The AI removes unregistered neurons, Gaussian-smooths the traces along time with `sigma = 15` frames, average-pools in non-overlapping 15-frame windows, rescales the pooled values to Hz by multiplying by `FPS`, and finally transposes each trial to `(n_cells, n_timepoints)`.

ii.
```python
BIN_FRAMES = 15
SMOOTH_SIGMA_FRAMES = BIN_FRAMES
```

```python
def bin_traces(trace, n_bins):
    smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA_FRAMES, axis=0)
    chunks = smoothed[:n_bins * BIN_FRAMES].reshape(n_bins, BIN_FRAMES, trace.shape[1])
    return (chunks.mean(axis=1) * FPS).astype(np.float32)   # (n_bins, n_cells)
```

```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
...
rates = bin_traces(trace, n_bins)
...
neural.append(np.ascontiguousarray(rates[sl].T))    # (n_cells, T)
```

iii. The AI explicitly justified this as following the paper's decoder recipe: "Gaussian-smooth along time with sigma = bin width, then average-pool, expressed in Hz." It also justified widening the bin from the paper's 100 ms decoder bins to 500 ms because the events were sparse and 100 ms output would be too large.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters out neurons that are all `NaN` within a session, treating them as cells not registered on that day. It does not apply any further neural curation, but it asserts that remaining traces contain no partial `NaN` values.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.isnan(trace).any(), f"{animal} day {day}: partial NaN trace"
```

iii. The trajectory summary says: "Per session I keep only the cells registered that day (unregistered cells are all-NaN columns)." It also says this was chosen to match the paper's headline neuron/session counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not perform a stimulus- or behavior-centered event alignment. It assumes `trace` and `position` already share a common 30 Hz frame clock, rebins both on that same clock, and then defines the start of each 60 s trial window as the alignment event in metadata.

ii.
```python
# Position and trace share the frame index ...
# ... so no further temporal alignment is required here.
```

```python
rates = bin_traces(trace, n_bins)
labels = bin_labels(process_position(position), n_bins)
...
sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
neural.append(np.ascontiguousarray(rates[sl].T))
outputs.append(labels[sl][np.newaxis, :])
```

```python
"temporal_alignment_event":
    "start of the trial, i.e. of a consecutive non-overlapping 1-minute "
    "window of the continuous free-exploration session",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

iii. The AI justified this twice in the trajectory: first by noting that `position` and `trace` "share a frame index at 30 Hz," and later by describing the alignment event as simply the start of each consecutive 1-minute trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 500 ms bins. Yes, temporal rebinning is applied: 15 original 30 Hz frames are smoothed and pooled into one output time bin.

ii.
```python
BIN_FRAMES = 15                                  # 15 frames @ 30 Hz = 500 ms
BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FPS
SMOOTH_SIGMA_FRAMES = BIN_FRAMES
```

```python
"time_bin_size": BIN_SIZE_MS,
```

iii. The AI explicitly called this a "deliberate departure" from the paper's 100 ms decoder bins. It justified 500 ms by arguing that the traces are sparse and that a 100 ms dataset would be much larger.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment-geometry input is derived from the raw `blocked` variable, which stores flat indices of blocked 3 x 3 partitions. The `envs` variable is read alongside it but is only used for metadata, not for the actual decoder input.

ii.
```python
def load_session_list(f):
    envs = [_read_string(f, ref) for ref in f["envs"][0]]
    blocked = []
    for ref in f["blocked"][0]:
        idx = f[ref][:].ravel().astype(int)
        blocked.append(idx[idx >= 0])
    return envs, blocked
```

iii. The AI's justification in the trajectory focused on decoding the `blocked` convention correctly and verifying that its flat indices matched the physically inaccessible bins.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI removes negative `blocked` entries (`-1` means no blocked bin in the square), converts the remaining flat indices into a 9-dimensional binary vector, and then copies that same static vector into every trial from that session.

ii.
```python
def load_session_list(f):
    ...
    for ref in f["blocked"][0]:
        idx = f[ref][:].ravel().astype(int)
        blocked.append(idx[idx >= 0])   # 'square' is stored as -1 (nothing blocked)
```

```python
def geometry_vector(blocked_flat):
    geo = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    geo[blocked_flat] = 1.0
    return geo
```

```python
geo = geometry_vector(blk)
...
inputs.append(geo.copy())                           # (9,) static
```

iii. The AI justified this with reference-code conventions plus empirical checks. In the trajectory it said it decoded the flat-index convention as `3*y + x` and verified across all animals that blocked bins had essentially zero occupancy.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw `position` variable in each session.

ii.
```python
position = f[f["position"][day, 0]][:]
```

iii. The trajectory summary states that `position` contains the mouse's head position in centimeters in the fixed 75 x 75 cm arena frame.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first converts the continuous `(x, y)` positions into one of nine 3 x 3 arena partitions using fixed 25 cm boundaries over the full 75 cm arena. It then temporally aggregates these framewise labels into 500 ms bins by taking the majority label within each 15-frame window.

ii.
```python
def process_position(position):
    bins = np.floor(position / (ARENA_SIZE / N_SPATIAL_BINS)).astype(int)
    bins = np.clip(bins, 0, N_SPATIAL_BINS - 1)
    return bins[:, 1] * N_SPATIAL_BINS + bins[:, 0]
```

```python
def bin_labels(frame_labels, n_bins):
    chunks = frame_labels[:n_bins * BIN_FRAMES].reshape(n_bins, BIN_FRAMES)
    counts = np.zeros((n_bins, N_SPATIAL_BINS ** 2), dtype=np.int32)
    for k in range(N_SPATIAL_BINS ** 2):
        counts[:, k] = (chunks == k).sum(axis=1)
    return counts.argmax(axis=1).astype(np.int64)
```

iii. The AI justified the fixed 25 cm boundaries by saying positions are already in the full arena coordinate frame, and justified majority-vote labels by saying coordinate averaging could place the mouse in a partition it never occupied, including a blocked one.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI thresholds position into 9 categories by clipping each coordinate into 3 bins and flattening the resulting `(x_bin, y_bin)` pair into one class index using `flat = 3 * y_bin + x_bin`.

ii.
```python
bins = np.floor(position / (ARENA_SIZE / N_SPATIAL_BINS)).astype(int)
bins = np.clip(bins, 0, N_SPATIAL_BINS - 1)
return bins[:, 1] * N_SPATIAL_BINS + bins[:, 0]
```

iii. The AI justified this as matching the blocked-bin convention and the arena's 3 x 3 partition structure from the paper.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position data are aligned by sharing the same session clock. The code uses a common `n_frames`, rebins both streams using the same `BIN_FRAMES`, and slices them with the same trial boundaries.

ii.
```python
n_frames = min(trace.shape[0], position.shape[0])
n_bins = n_frames // BIN_FRAMES

rates = bin_traces(trace, n_bins)
labels = bin_labels(process_position(position), n_bins)
```

```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural.append(np.ascontiguousarray(rates[sl].T))
    outputs.append(labels[sl][np.newaxis, :])
```

iii. The AI explicitly justified alignment by saying `position` and `trace` "share a frame index at 30 Hz," so both can be rebinned and trial-sliced identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes all-`NaN` neuron columns, asserts that no partial `NaN` traces remain, asserts that positions contain no `NaN`s, crops both streams to the shorter common length via `n_frames = min(...)`, ignores negative blocked indices (`-1`), and drops any trailing data that does not fill a complete 500 ms bin or a complete 60 s trial. It does not impute missing values.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.isnan(trace).any(), f"{animal} day {day}: partial NaN trace"
assert not np.isnan(position).any(), f"{animal} day {day}: NaN position"

n_frames = min(trace.shape[0], position.shape[0])
n_bins = n_frames // BIN_FRAMES
```

```python
for ref in f["blocked"][0]:
    idx = f[ref][:].ravel().astype(int)
    blocked.append(idx[idx >= 0])
```

iii. In the trajectory, the AI justified NaN-column removal as keeping only registered cells, and described small blocked-bin occupancy as tracking jitter rather than something to filter away. The rest of the handling is implicit in the code rather than separately justified.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading large HDF5 `.mat` files for all sessions, smoothing and pooling the large `trace` arrays (`gaussian_filter1d` plus reshape/mean), and writing the 1.33 GB pickle.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    trace = f[f["trace"][day, 0]][:]
```

```python
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA_FRAMES, axis=0)
chunks = smoothed[:n_bins * BIN_FRAMES].reshape(n_bins, BIN_FRAMES, trace.shape[1])
```

```python
with open(OUT_FILE, "wb") as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory shows the full conversion took about 5 minutes 20 seconds and produced a 1.33 GB file, which supports HDF5 I/O and full-trace smoothing/writing as the expensive steps. The AI also explicitly motivated the 500 ms binning partly as a way to reduce data size.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loop is the `for k in range(9)` loop inside `bin_labels`, which builds per-bin counts one category at a time. The per-trial `for t in range(n_trials)` loop is also a repeated Python-level append pattern that could be replaced with reshaping/splitting plus list conversion.

ii.
```python
def bin_labels(frame_labels, n_bins):
    chunks = frame_labels[:n_bins * BIN_FRAMES].reshape(n_bins, BIN_FRAMES)
    counts = np.zeros((n_bins, N_SPATIAL_BINS ** 2), dtype=np.int32)
    for k in range(N_SPATIAL_BINS ** 2):
        counts[:, k] = (chunks == k).sum(axis=1)
    return counts.argmax(axis=1).astype(np.int64)
```

```python
for t in range(n_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    neural.append(np.ascontiguousarray(rates[sl].T))
    inputs.append(geo.copy())
    outputs.append(labels[sl][np.newaxis, :])
```

iii. The AI did not explicitly discuss vectorization in the trajectory. This assessment is based on the code structure it wrote.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats static geometry copying once per trial even though geometry is session-constant, repeats the per-trial slicing/appending logic for neural/input/output in every session, and repeats human-readable metadata construction (`blocked_bins`, `sequence`) for every session although those fields are not needed to build the main arrays.

ii.
```python
geo = geometry_vector(blk)
...
for t in range(n_trials):
    ...
    inputs.append(geo.copy())                           # (9,) static
```

```python
"info": {
    "animal": animal,
    "day": day,
    "environment": env,
    "blocked_bins": [BIN_NAMES[i] for i in blk],
    "sequence": day // len(set(envs)),
    "n_neurons": int(registered.sum()),
    "n_trials": n_trials,
},
```

iii. The AI did not explicitly call these repetitions out in the trajectory. They are evident from the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `envs` and constructs extensive `session_info` metadata (`environment`, `blocked_bins`, `sequence`) that the downstream decoder does not use. It also duplicates the same static geometry vector once per trial with `geo.copy()`. Those steps increase work and storage without affecting the decoder inputs/outputs used later.

ii.
```python
envs = [_read_string(f, ref) for ref in f["envs"][0]]
```

```python
session_info = []
...
session_info.append(session["info"])
...
"session_info": session_info,
```

```python
inputs.append(geo.copy())                           # (9,) static
```

iii. The trajectory does not explicitly justify these steps as necessary for decoding. They appear to be for interpretability/reporting rather than for the downstream analysis itself.
