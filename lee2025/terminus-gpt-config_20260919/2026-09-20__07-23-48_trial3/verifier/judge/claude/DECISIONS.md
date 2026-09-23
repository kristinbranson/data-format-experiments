# Decisions

**Note on code provenance**: The file currently at `/app/convert_data.py` on disk is a simplified h5py-based version that does NOT match the agent's trajectory, CONVERSION_NOTES, or output files. The trajectory (step 39, with fixes at steps 44 and 57) shows the agent wrote a sophisticated version using joblib, Gaussian smoothing, and 3-frame pooling -- nearly identical to the reference. All code snippets below are from the agent's actual code as recorded in the trajectory, which produced the output files.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each subject's data is loaded from a compressed joblib file in `/app/data/`. The joblib file contains a dict keyed by animal ID, with fields `trace`, `position`, `blocked`, and `envs` (among others). Each field is a list indexed by recording day. The agent iterates over a hard-coded list of 7 animal IDs and loads each animal's data one at a time.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
for subj,animal in enumerate(selected):
    root=joblib.load('/app/data/'+animal); dat=root[animal]
    for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
        ...
```

iii. The agent identified in Step 1 of CONVERSION_NOTES that the reference code's `load_dat` function uses joblib by default. The agent adopted the same approach, loading one animal at a time to manage memory.

## 1-b. How are the data split into subjects?

i. Each animal corresponds to one joblib file. The agent iterates over a hard-coded list of 7 animal IDs (matching the reference repository). The animal ID becomes the subject name.

ii.
```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
'subjects': selected,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. The agent noted in Step 2 that `/app/data` contains seven animal files matching the seven IDs listed in the reference repository README and paper.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject becomes a separate session. The agent iterates over the `trace`, `position`, `blocked`, and `envs` lists for each animal, where each list element corresponds to one day.

ii.
```python
for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
    nt,it,ot,ncells,nfixed,tail = process_session(tr,pos,blk,animal,day)
    neural.append(nt); inputs.append(it); outputs.append(ot); subject_idx.append(subj)
```

iii. The agent documented in Step 4 that "each animal-day is a decoder session, matching source and paper terminology." This yields 207 total sessions (31 per animal for 6 animals, 21 for QLAK-CA1-51).

## 1-d. How are the data split into trials?

i. Each continuous ~40-minute recording session is split into non-overlapping 60-second segments (1800 source frames at 30 Hz). After 3-frame pooling, each trial has 600 time bins. Incomplete final segments are discarded.

ii.
```python
TRIAL_FRAMES = 60 * FPS  # 1800
TRIAL_BINS = TRIAL_FRAMES // POOL  # 600
...
ntrials = trace.shape[1] // TRIAL_FRAMES
nframes = ntrials * TRIAL_FRAMES
...
neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. Per CONVERSION_NOTES Step 5: "Split each session into consecutive non-overlapping 60 s windows, as required. At 30 Hz this is 1,800 source frames or 600 target bins." The agent noted this yields 39-40 trials per session and 8,187 total trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied beyond requiring at least 2 complete trials per session (to allow decoder evaluation). The agent explicitly chose not to apply speed filtering.

ii.
```python
if ntrials < 2:
    raise ValueError(f'{animal} day {day}: fewer than two complete trials')
```

iii. CONVERSION_NOTES Step 4: "Target requires continuous 1-minute trials and time-varying position. Retain all valid frames; speed filtering would destroy regular temporal trials and is not global data curation."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium event traces (0/1 values indicating significant rising-phase events). Shape is `(registered_cells, n_timepoints)` per session.

ii.
```python
trace = np.asarray(trace)
...
raw = trace[finite_all]
vals = np.unique(raw)
if not np.all(np.isin(vals, [0, 1])):
    raise ValueError(f'{animal} day {day}: nonbinary finite trace values {vals[:10]}')
```

iii. CONVERSION_NOTES Step 1: "trace is already a rise-extracted binary calcium event series (1 = significant event), not raw fluorescence. Therefore delta-F/F must not be recomputed."

## 2-b. How is the `neural` data processed?

i. The binary trace is Gaussian-smoothed with sigma=3 source frames, then non-overlapping 3-frame mean pooling is applied, resulting in 100 ms time bins. Cast to float32.

ii.
```python
POOL = 3
...
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
pooled = pooled.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Neural events are Gaussian-smoothed with sigma 3 source frames before average pooling, exactly matching `fit_decoder`/`test_decoder`." The agent identified this processing from the reference code's `fit_decoder` function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons with wholly finite trace rows are kept. All-NaN rows (unregistered cells for that session) are removed. The agent also validates that no partially-missing rows exist and that all finite values are binary (0 or 1). No place-cell or event-count filtering is applied.

ii.
```python
finite_all = np.all(np.isfinite(trace), axis=1)
finite_any = np.any(np.isfinite(trace), axis=1)
if np.any(finite_any != finite_all):
    raise ValueError(f'{animal} day {day}: partially missing neural row')
raw = trace[finite_all]
```

iii. CONVERSION_NOTES Step 4: "Paper Results explicitly state that high reliability 'motivated the inclusion of all cells in subsequent analyses.' Place-cell filtering was used only where specified, not as global data curation."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment. The recording is continuous, and trials are consecutive non-overlapping 60-second segments starting from the beginning of the session. Neural and position data share the same frame indices from simultaneous acquisition.

ii.
```python
'temporal_alignment_event': 'start of each consecutive non-overlapping 60-second segment within a recording session',
'off_start': 0.0, 'off_end': 60.0,
```

iii. CONVERSION_NOTES Step 5: "Trials align to their own start (consecutive 60 s segments), with off_start=0, off_end=60, time_bin_size=100 ms."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz data is rebinned to 100 ms time bins via non-overlapping 3-frame mean pooling (matching the reference decoder). This produces 600 time bins per 60-second trial.

ii.
```python
POOL = 3
BIN_MS = 100.0
TRIAL_BINS = TRIAL_FRAMES // POOL  # 600
...
'time_bin_size': BIN_MS,
```

iii. CONVERSION_NOTES Step 3: "Reference within-session decoder... smooths neural traces with Gaussian sigma 3 frames, then average-pools behavior and neural data in non-overlapping 3-frame (100 ms) bins." Step 5 confirms: "use 3-frame/100 ms bins, matching reference position decoder."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` variable in each animal's data, which contains a list of blocked 3x3 partition indices per day. A value of `[-1]` indicates no positions are blocked.

ii.
```python
def blocked_vector(blocked):
    """Return row-major 3x3 mask (1 blocked, 0 accessible)."""
    out = np.zeros(9, dtype=np.float32)
    idx = np.asarray(blocked).reshape(-1).astype(int)
    idx = idx[idx >= 0]
    if len(idx):
        if np.any(idx > 8):
            raise ValueError(f'Invalid blocked indices: {idx}')
        out[idx] = 1
    return out
```

iii. CONVERSION_NOTES Step 1: "`blocked` stores omitted partition indices using row-major 3x3 labels [[0,1,2],[3,4,5],[6,7,8]]; -1 means no omitted partition."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a 9-element binary vector (1=blocked, 0=accessible) in row-major order. The vector is static per session (constant across all trials). Negative indices (-1) are filtered out, leaving an all-zeros vector for fully open geometries.

ii.
```python
bmask = blocked_vector(blocked)
...
input_trials = [bmask.copy() for _ in range(ntrials)]
...
'input_names': [f'blocked_row{r}_col{c}' for r in range(3) for c in range(3)],
```

iii. CONVERSION_NOTES Step 5: "Encode the supplied blocked indices directly as a 9-element binary vector (1=blocked). All repetitions/animals show one canonical mask per geometry."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Position is derived from the `position` variable, which contains 2D (x, y) coordinates of the animal's head location tracked by DeepLabCut, at 30 Hz, in a 75x75 cm arena.

ii.
```python
position = np.asarray(position)
...
xy = pool_position(position, nframes)
labels, nfixed = position_labels(xy, bmask)
```

iii. CONVERSION_NOTES Step 3: "Position is head location from DeepLabCut." Step 2: "Position is finite, in the range 0-75 cm on each axis."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position coordinates are mean-pooled over the same 3-frame windows used for neural data, then discretized into a 3x3 grid of 25 cm bins using floor division, with clipping at boundaries.

ii.
```python
def pool_position(position, nframes):
    p = np.asarray(position, dtype=np.float64)[:, :nframes]
    return p.reshape(2, -1, POOL).mean(axis=2)

def position_labels(xy, blocked_mask):
    col = np.clip(np.floor(xy[0] / 25.0), 0, 2).astype(np.int64)
    row = np.clip(np.floor(xy[1] / 25.0), 0, 2).astype(np.int64)
    labels = row * 3 + col
    ...
```

iii. CONVERSION_NOTES Step 5: "Position uses matching average pooling and integer spatial discretization." The agent chose half-open 25 cm bins via floor, consistent with the arena's natural 3x3 partition structure.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis (x, y) is divided into 3 bins of 25 cm each using `floor(coord / 25.0)`, clipped to [0, 2]. The 2D bin label is computed as `row * 3 + col` (row-major), giving 9 categories (0-8). Rare labels falling in blocked cells are snapped to the nearest accessible cell by Euclidean distance.

ii.
```python
def position_labels(xy, blocked_mask):
    col = np.clip(np.floor(xy[0] / 25.0), 0, 2).astype(np.int64)
    row = np.clip(np.floor(xy[1] / 25.0), 0, 2).astype(np.int64)
    labels = row * 3 + col
    invalid = blocked_mask[labels].astype(bool)
    nfixed = int(invalid.sum())
    if nfixed:
        open_labels = np.flatnonzero(blocked_mask == 0)
        centers = np.column_stack(((open_labels % 3 + 0.5) * 25.0,
                                   (open_labels // 3 + 0.5) * 25.0))
        pts = xy[:, invalid].T
        nearest = np.argmin(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        labels[invalid] = open_labels[nearest]
    return labels.astype(np.uint8), nfixed
```

iii. CONVERSION_NOTES Step 5: "Source coordinate 0 is x (column), coordinate 1 is y (row)... class=y*3+x." The agent validated the orientation by checking that the alternative (x*3+y) placed 17.1% of frames in blocked cells vs only 0.003% with the correct orientation.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same source frame rate (30 Hz, simultaneously acquired). Both are pooled using the same 3-frame windows and split into trials using the same indices, ensuring perfect temporal alignment.

ii.
```python
nframes = ntrials * TRIAL_FRAMES
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
xy = pool_position(position, nframes)
...
neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
output_trials = [np.ascontiguousarray(labels[None, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. CONVERSION_NOTES Step 10: "Raw and converted checks use identical 1,800-frame windows and matching three-frame pooling for neural and position."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of data issues are handled: (1) All-NaN neural rows (unregistered cells) are removed per session. (2) Incomplete final segments that don't fill a 60-second trial are discarded. (3) Rare pooled position bins falling in blocked cells (152 out of 4,912,200 total bins, ~0.003%) are snapped to the nearest accessible cell.

ii.
```python
# NaN removal
raw = trace[finite_all]
# Tail discard
ntrials = trace.shape[1] // TRIAL_FRAMES
nframes = ntrials * TRIAL_FRAMES
# Blocked label correction
invalid = blocked_mask[labels].astype(bool)
if nfixed:
    ...
    labels[invalid] = open_labels[nearest]
```

iii. CONVERSION_NOTES Step 10: "152/4,912,200 samples to nearest accessible bin, matching reference decoder cleanup logic." The agent tracked discarded tail frames per session (ranging from 60 to 1666 frames) in the conversion output.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the joblib files is the most time-consuming step (8-16 seconds per animal). Processing each session takes 0.2-0.9 seconds. Total conversion time for all 207 sessions was ~204 seconds.

ii. N/A (timing from output logs)

iii. CONVERSION_NOTES Step 7: "Sample conversion including two large animal loads, plots, write: 54.9 s / 62 = 0.89 s average." The conversion output shows per-animal load times: 8.9s (QLAK-CA1-08) to 16.1s (QLAK-CA1-75).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent's code is already well-vectorized. Gaussian smoothing and pooling are applied to whole sessions at once. The only remaining loop is over trials for slicing, which is unavoidable given the target list-of-arrays format.

ii.
```python
# Whole-session vectorized operations
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
# Trial slicing loop (unavoidable)
neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. CONVERSION_NOTES Step 6: "Vectorized Gaussian filtering, pooling, coordinate discretization, and blocked-bin correction."

## 6-c. What processing does the code repeat multiple times?

i. The blocked vector is computed once per session and copied for each trial (`bmask.copy()`). No other processing is repeated. Each session is processed only once.

ii.
```python
bmask = blocked_vector(blocked)
...
input_trials = [bmask.copy() for _ in range(ntrials)]
```

iii. CONVERSION_NOTES Step 6: "Avoids recomputing source processing for each trial by processing a whole session at once."

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent processes frames beyond the last complete trial boundary before discarding them (the Gaussian smoothing operates on the full session length before truncation). The tail frames (60-1666 per session) are smoothed but then discarded. However, this is a minor inefficiency.

ii.
```python
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)  # full session
pooled = smooth[:, :nframes].reshape(...)  # truncate to complete trials only
```

iii. No explicit justification given. This is a standard trade-off: smoothing the full session before truncation avoids edge effects at the last trial boundary.
