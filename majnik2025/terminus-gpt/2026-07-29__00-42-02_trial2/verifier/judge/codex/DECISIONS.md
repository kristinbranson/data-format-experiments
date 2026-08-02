# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `data/` for subject directories whose names start with `jm`, then scans each subject for session directories whose first four characters are digits. It flattens all selected `(subject, session)` pairs into one list. For each session it loads Suite2p calcium files (`F.npy`, `Fneu.npy`, `ops.npy`, `stat.npy`) and behavior files (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`). Trials are not loaded natively; they are created later by splitting each processed session into consecutive blocks.

ii.
```python
def discover_subjects(data_root: Path):
    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
    sessions_by_subject = {}
    for subj in subjects:
        sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        sessions_by_subject[subj.name] = sessions
    return subjects, sessions_by_subject

def load_session(session_dir: Path):
    pl0 = session_dir / 'suite2p' / 'plane0'
    move = session_dir / 'move_deve'
    F = np.load(pl0 / 'F.npy', allow_pickle=True)
    Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
    ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
    stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
    motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
    tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
    interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
    return F, Fneu, ops, stat, motion, tstamps, interframe
```

iii. In `CONVERSION_NOTES.md`, the agent states that the dataset consists of 6 `jm*` subject folders, each containing dated session folders with `suite2p/plane0` and `move_deve`, and that one session should correspond to one recording day.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories under `data/` whose names start with `jm`, sorted lexicographically. The final `subjects` field is the list of those directory names.

ii.
```python
subjects, sessions_by_subject = discover_subjects(data_root)
subject_names = [p.name for p in subjects]
```

iii. The notes say each `jm*` directory is one mouse and that the subject mapping should follow the native organization directly.

## 1-c. How are the data split into sessions?

i. Sessions are split as subdirectories within each subject folder, restricted to names whose first four characters are digits and then sorted. Each session remains one top-level session entry in the converted dataset.

ii.
```python
for subj in subjects:
    sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    sessions_by_subject[subj.name] = sessions

selected = []
for subj in subject_names:
    for sess in sessions_by_subject[subj]:
        selected.append((subj, sess))
```

iii. In Step 5 of the notes, the agent explicitly records the decision “Use one session per recording day and one subject per mouse.”

## 1-d. How are the data split into trials?

i. The agent assumes there are no native trials and creates artificial trials as consecutive 2-minute blocks after first averaging both neural and motion data in 10-frame bins. Each block therefore contains `BLOCK_BINS = 360` time bins.

ii.
```python
BIN_FRAMES = 10
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES

def session_to_blocks(neural_binned: np.ndarray, motion_binned: np.ndarray):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    neural_binned = neural_binned[:, :n_bins]
    motion_binned = motion_binned[:n_bins]
    n_blocks = n_bins // BLOCK_BINS
    neural_blocks = []
    motion_blocks = []
    time_blocks = []
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
    return neural_blocks, motion_blocks, time_blocks
```

iii. The notes and trajectory say the recordings are continuous and that the paper’s decoder used consecutive 2-minute blocks, so the agent treated those blocks as the trial unit.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is based on output-label completeness, not on native trial metadata. A block is dropped if fewer than `max(10, 0.8 * len(y))` output bins are valid after motion alignment. Entire sessions are dropped unless at least two blocks survive.

ii.
```python
for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
    y = np.digitize(mb, edges, right=False).astype(np.int64)
    y[np.isnan(mb)] = -1
    valid = y >= 0
    if valid.sum() < max(10, int(0.8 * len(y))):
        continue
    ...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. The trajectory shows this filtering was introduced because motion alignment produced missing values, and the notes repeatedly mention the decoder requirement that each session contain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from `suite2p/plane0/F.npy` and `Fneu.npy`, with `ops.npy` used to fetch processing parameters such as `neucoeff`, `win_baseline`, `fs`, and `prctile_baseline`. `stat.npy` is loaded but not used in the computation.

ii.
```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()

def compute_dff(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
```

iii. The notes say the notebook indicated `F.npy` contains raw fluorescence and that “dF/F should be computed,” while `ops.npy` contains Suite2p baseline parameters.

## 2-b. How is the `neural` data processed?

i. The agent computes a custom dF/F-like signal: neuropil subtraction, then for each neuron a reflected moving average over a 60-second window, then a single per-neuron percentile baseline, then `(Fcorr - baseline) / baseline`. After that, the neural traces are averaged into non-overlapping 10-frame bins.

ii.
```python
def compute_dff(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    win_seconds = float(ops.get('win_baseline', 60.0))
    win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
    prct = float(ops.get('prctile_baseline', 8.0))
    baseline = np.empty_like(Fcorr, dtype=np.float32)
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fcorr - baseline) / baseline
    return dff.astype(np.float32)

def bin_neural(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)
```

iii. The notes say the intended rationale was to create a “Suite2p-consistent neuropil-corrected / baseline-corrected fluorescence (dF/F-like signal)” and to follow the methods text stating both neural and behavior traces were averaged in bins of 10 timestamps. The same notes also admit this baseline computation is a simplified approximation rather than Suite2p internals.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not apply any explicit neuron-level filtering. It loads all rows from `F.npy` and `Fneu.npy`, and does not use `iscell.npy` or `stat.npy` to exclude neurons.

ii.
```python
F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
dff = compute_dff(F, Fneu, ops)
...
'n_neurons': dff.shape[0],
```

iii. The notes justify this by saying the released `suite2p` folders already contain only tracked neurons present across all days and that observed `iscell` values already satisfy the default cell threshold, so no extra filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns each neural trial to the start of each artificial 2-minute block, not to the start of the full session. Within each block, time starts at zero again.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'start of each 2-minute continuous recording block',
    'off_start': 0.0,
    'off_end': BLOCK_SECONDS,
}

time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The notes and trajectory say the recordings are continuous and that 2-minute blocks were chosen as the natural “trial” unit because the paper’s decoding used consecutive 2-minute blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 333.33 ms time bins, produced by averaging over non-overlapping 10-frame windows at 30 Hz. Yes, temporal rebinning is applied.

ii.
```python
FS = 30.0
BIN_FRAMES = 10

'metadata': {
    'time_bin_size': 1000.0 * BIN_FRAMES / FS,
}

motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
neural_binned = bin_neural(dff, BIN_FRAMES)
```

iii. The notes cite the methods text statement that both dF/F and behavior traces were “averaging in bins of 10 consecutive timestamps.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a raw file variable. It is synthesized from `FS`, `BIN_FRAMES`, and the bin index within each 2-minute block.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. In the notes, the agent says this input was “constructed to satisfy decoder input specification” and operationalized as elapsed time within each 2-minute block.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent creates a `(1, T)` vector for each block using equally spaced times `0, 10/30, 20/30, ...` seconds. The clock resets to zero at every 2-minute block rather than continuing from the experiment start.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The notes explicitly describe this as “elapsed time within each 2-minute block,” and the trajectory says this was chosen to align with the block-based trial definition.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is generated inside `session_to_blocks` using the same block boundaries used for neural data, and the resulting `time_blocks` are zipped with `neural_blocks` and `motion_blocks`. Thus each time vector has exactly the same number of bins as the corresponding neural trial.

ii.
```python
neural_blocks, motion_blocks, time_blocks = session_to_blocks(neural_binned, motion_binned)
...
for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
    sess_neural.append(nb.astype(np.float32))
    sess_input.append(tb.astype(np.float32))
```

iii. The metadata says the alignment event is the start of each 2-minute block, and the notes say this input was meant to represent time after that block-alignment point.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`. `tstamps.npy` and `interframe_int.npy` are loaded, but the final alignment function ignores their values and uses only the motion array length plus the neural frame count.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)

def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
```

iii. The justification changed over time. The notes initially say missing camera frames should be handled using `tstamps.npy` and `interframe_int.npy`, but the trajectory later records a bug fix: `tstamps.npy` stores seconds, so the agent switched to sample-order alignment and simple pad/truncate behavior.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is padded or truncated to the neural frame count, averaged in 10-frame bins while ignoring `NaN`s, pooled across all sessions to compute global quantile thresholds, and then discretized into 5 bins. The code does not standardize motion energy by session standard deviation.

ii.
```python
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
...
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. The notes and trajectory justify 10-frame averaging from the methods text and equal-percentile binning from the decoder specification. After a failed attempt to use `tstamps` as frame indices, the trajectory says the agent chose direct sample-order alignment because the motion array is already framewise with occasional missing frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent pools all finite binned motion values across all sessions, computes the 20th/40th/60th/80th percentiles, and uses `np.digitize` to assign 5 categories. Missing bins are temporarily labeled `-1`, then filled by nearest previous/next valid labels, with any remaining negatives set to `0`.

ii.
```python
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
y = np.digitize(mb, edges, right=False).astype(np.int64)
y[np.isnan(mb)] = -1
...
if not np.all(valid):
    yy = y.copy()
    last = None
    for i in range(len(yy)):
        if yy[i] >= 0:
            last = yy[i]
        elif last is not None:
            yy[i] = last
    nxt = None
    for i in range(len(yy)-1, -1, -1):
        if yy[i] >= 0:
            nxt = yy[i]
        elif nxt is not None:
            yy[i] = nxt
    yy[yy < 0] = 0
    y = yy
```

iii. The notes explicitly list “Discretize motion energy into 5 equal-percentile bins using valid binned samples across the full converted dataset” as a key decision.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The output is aligned to neural data by placing the motion series in sample order into an array of neural-frame length, padding the end with `NaN` if motion is shorter, then applying the same 10-frame binning and the same 2-minute block boundaries used for neural data. The code trims both streams to `min(neural_binned.shape[1], motion_binned.shape[0])`.

ii.
```python
def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

def session_to_blocks(neural_binned: np.ndarray, motion_binned: np.ndarray):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    neural_binned = neural_binned[:, :n_bins]
    motion_binned = motion_binned[:n_bins]
```

iii. The trajectory documents that an earlier timestamp-based alignment dropped 9 sessions, and that the final justification was to preserve sample order because `motion_energy_glob.npy` is already framewise and `tstamps.npy` stores seconds rather than frame indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion frames are represented as trailing `NaN`s after pad/truncate alignment. Binning ignores `NaN`s. Blocks with too many missing labels are dropped, and remaining missing labels are imputed by forward fill then backward fill, then set to `0` if still missing. Partial trailing data that do not fill a whole 2-minute block are implicitly discarded because `n_blocks = n_bins // BLOCK_BINS`.

ii.
```python
aligned = np.full(nframes, np.nan, dtype=np.float32)
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]

valid = np.isfinite(x)
sums = np.where(valid, x, 0.0).sum(axis=1)
counts = valid.sum(axis=1)

if valid.sum() < max(10, int(0.8 * len(y))):
    continue

yy[yy < 0] = 0
...
n_blocks = n_bins // BLOCK_BINS
```

iii. The trajectory explicitly records one resolved mistake: the agent first misread `tstamps.npy` as frame indices, which caused many sessions to be dropped, and then changed to sample-order alignment. The notes also say missing camera frames should be treated as missing values and that sessions must keep at least two trials for decoder evaluation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the custom neural baseline / dF/F computation, especially the per-neuron loop inside `compute_dff`. Secondarily, the first pass over all sessions stores all prepared data in memory before global motion discretization.

ii.
```python
baseline = np.empty_like(Fcorr, dtype=np.float32)
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base

prepared = []
for idx, (subj, sessdir) in enumerate(selected):
    ...
    prepared.append({
        'subject': subj,
        'session': sessdir.name,
        'neural_blocks': neural_blocks,
        'motion_blocks': motion_blocks,
        'time_blocks': time_blocks,
        'n_neurons': dff.shape[0],
    })
```

iii. Step 6 of the notes says “baseline computation loops over neurons” and “all sessions are prepared in memory before global motion discretization,” and identifies the baseline computation as the main inefficiency.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent’s own notes identify the per-neuron baseline loop in `compute_dff` as the main vectorization opportunity. Additional scalar loops remain in block construction and in forward/backward filling missing labels.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base

for b in range(n_blocks):
    s = b * BLOCK_BINS
    e = (b + 1) * BLOCK_BINS
    neural_blocks.append(neural_binned[:, s:e].astype(np.float32))

for i in range(len(yy)):
    ...
for i in range(len(yy)-1, -1, -1):
    ...
```

iii. The explicit justification in the notes is that the baseline computation “loops over neurons and uses a simplified approximation instead of Suite2p internals.”

## 6-c. What processing does the code repeat multiple times?

i. The code makes two passes over the prepared session/block data: one pass to collect all motion values for global quantile estimation, and a second pass to discretize motion, filter blocks, and assemble the final output structure. It also repeatedly casts arrays to `float32`.

ii.
```python
for idx, (subj, sessdir) in enumerate(selected):
    ...
    prepared.append({...})
    for mb in motion_blocks:
        all_motion_values.append(mb[np.isfinite(mb)])

...
for item in prepared:
    ...
    for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
        ...
        sess_neural.append(nb.astype(np.float32))
        sess_input.append(tb.astype(np.float32))
```

iii. The notes do not call this out directly, but they do note that “all sessions are prepared in memory before global motion discretization,” which implies this two-pass structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `stat.npy` and `interframe_int.npy` but never uses them. It also loads `tstamps.npy`, passes it into `align_motion_to_frames`, and then ignores its contents. In addition, `sessdir.name` is stored in `prepared` but never written to the output pickle.

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
...
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
return F, Fneu, ops, stat, motion, tstamps, interframe

def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

prepared.append({
    'subject': subj,
    'session': sessdir.name,
    ...
})
```

iii. The trajectory shows that `tstamps.npy` and `interframe_int.npy` were originally meant to support frame-drop handling, but after the timestamp-alignment bug the final code fell back to sample-order alignment and no longer used those arrays.
