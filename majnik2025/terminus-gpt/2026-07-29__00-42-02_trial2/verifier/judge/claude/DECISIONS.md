# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subject directories matching `jm*` in the data root, then iterates over sorted session subdirectories within each subject. For each session, it loads suite2p fluorescence files (`F.npy`, `Fneu.npy`, `ops.npy`, `stat.npy`) and motion energy files (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`) from the respective subdirectories.

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

iii. The AI noted that the directory structure follows a standard convention with subject folders containing session subfolders, each with suite2p output and motion energy files. It also noted from the data README that the suite2p folders contain pre-tracked neurons matched across days.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data root, sorted alphabetically. Six subjects are identified: jm031, jm032, jm038, jm039, jm040, jm046.

ii.
```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
subject_names = [p.name for p in subjects]
```

iii. Each `jm*` directory represents one mouse, consistent with the paper's description of 6 mice.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a dated subdirectory within a subject's folder, filtered to directories whose name starts with 4 digits, sorted alphabetically. 41 total sessions are found across 6 subjects.

ii.
```python
sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The AI noted that the data contains both 20-minute (jm031, jm032) and 30-minute (jm038-jm046) sessions, which it preserved as-is despite the methods text stating 20 minutes.

## 1-d. How are the data split into trials?

i. The AI splits each continuous recording into consecutive **2-minute (120-second) blocks**, yielding `BLOCK_BINS = 360` time bins per trial (120s * 30Hz / 10 frames). This produces 10 blocks for 20-min sessions and 15 blocks for 30-min sessions.

ii.
```python
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES

def session_to_blocks(neural_binned, motion_binned):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    n_blocks = n_bins // BLOCK_BINS
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The AI chose 2-minute blocks because the paper states "splits were done on consecutive 2 minute blocks of the recording" for its decoding analysis. However, the task instructions explicitly state "Split sessions into 60-second trials."

## 1-e. How are trials filtered based on quality controls?

i. Blocks where fewer than 80% of bins have valid (non-NaN) motion energy labels are discarded. Sessions with fewer than 2 valid blocks are excluded entirely.

ii.
```python
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. The AI implemented this to ensure decoder training has sufficient valid data per block and that cross-validation requires at least 2 trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (parameters) from `suite2p/plane0/`.

ii.
```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence. The data README notes that dF/F should be computed for proper analysis.

## 2-b. How is the `neural` data processed?

i. The AI implements its own dF/F computation: (1) neuropil subtraction (`Fcorr = F - 0.7 * Fneu`), (2) moving average smoothing with reflected padding, (3) a single scalar percentile baseline per neuron (`np.percentile(smooth, 8.0)`), (4) dF/F = `(Fcorr - baseline) / baseline`. This differs from the reference which uses suite2p's `dcnv.preprocess` with `maximin` baseline method.

ii.
```python
def compute_dff(F, Fneu, ops):
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
```

iii. The AI acknowledged this is "only an approximation and may not exactly match Suite2p's baseline-corrected fluorescence." It noted the suite2p ops parameters (`baseline=maximin`, `win_baseline=60`, etc.) but did not use suite2p's actual `dcnv.preprocess` function, instead implementing a simplified version. The key difference is that the AI computes a single global percentile baseline per neuron rather than the time-varying `maximin` baseline that suite2p uses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons from `F.npy` are included. The AI noted that all cells in the dataset already have `iscell[:,0] == 1` with probability > 0.5, meaning the released data is pre-filtered.

ii. No filtering code exists; all rows of `F.npy` are used directly.

iii. The AI verified that all cells satisfy the iscell threshold and that the data README states the suite2p folders already contain only tracked, valid cells. This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 2-minute block. Since the recording is continuous and blocks are contiguous segments, no event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'start of each 2-minute continuous recording block',
'off_start': 0.0,
'off_end': BLOCK_SECONDS,
```

iii. There is no stimulus event in this spontaneous activity paradigm, so alignment is to the start of each artificially defined trial segment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy data are averaged into non-overlapping bins of 10 consecutive frames, converting 30 Hz to 3 Hz (333.33 ms bins). This matches the paper's description.

ii.
```python
BIN_FRAMES = 10

def bin_neural(x, bin_frames):
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)

def nanbin_mean_1d(x, bin_frames):
    n = (len(x) // bin_frames) * bin_frames
    x = x[:n].reshape(-1, bin_frames)
    ...
```

iii. The methods text states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is constructed from the bin index within each 2-minute block and the bin duration.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The AI operationalized "time from start of experiment" as time elapsed within each 2-minute block, ranging from 0 to ~120 seconds per block, identical for every block.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (10 / 30)` seconds, giving time from the start of each block in increments of 1/3 second. The time vector resets to 0 at the start of each block.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. No justification was provided for using within-block time rather than cumulative session time.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time bins are computed to have the same number of bins as the neural data within each block, so alignment is automatic by construction.

ii. Same as above -- `BLOCK_BINS` bins are created for both time and neural arrays.

iii. Both the time input and neural data use the same block segmentation and bin count, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. The AI also loads `tstamps.npy` and `interframe_int.npy` but does not use them in the final version.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) align motion to imaging frames by simple order-based placement (prefix copy, NaN padding for missing trailing frames), (2) average into 10-frame bins using NaN-aware binning, (3) discretize into 5 bins using **global** percentile edges computed across all sessions.

ii.
```python
def align_motion_to_frames(motion, tstamps, nframes):
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. The AI initially tried using `tstamps.npy` as frame indices but found they are timestamps in seconds, so it switched to simple order-based alignment. For discretization, it chose global percentile bins to ensure balanced class frequencies across the entire dataset.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories using global percentile edges at the 20th, 40th, 60th, and 80th percentiles, computed across all valid motion energy values from all sessions. `np.digitize` maps values to bins 0-4.

ii.
```python
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. The AI chose global bins without explicitly discussing the alternative of per-session bins. The task instructions state "five equal-percentile bins, selected per session."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to imaging frames by simple positional correspondence: the first `min(nframes, len(motion))` motion values are placed into the first positions, with any remaining positions filled with NaN. No interpolation of dropped frames is performed. After 10-frame binning (NaN-aware), both streams share the same time axis.

ii.
```python
def align_motion_to_frames(motion, tstamps, nframes):
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
```

iii. The AI initially tried to use `tstamps.npy` for frame-level alignment but discovered they are timestamps in seconds, not indices. It then switched to simple order-based alignment, noting that "Data README states motion_energy_glob is framewise with occasional missing camera frames."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames result in NaN values at the end of the aligned motion energy array. After binning, blocks with >20% NaN motion bins are discarded. For remaining blocks with some NaN values, missing output labels are forward-filled then backward-filled from nearest valid values. Neural data tail frames that don't fill a complete bin or block are discarded.

ii.
```python
y[np.isnan(mb)] = -1
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
if not np.all(valid):
    yy = y.copy()
    last = None
    for i in range(len(yy)):
        if yy[i] >= 0: last = yy[i]
        elif last is not None: yy[i] = last
    nxt = None
    for i in range(len(yy)-1, -1, -1):
        if yy[i] >= 0: nxt = yy[i]
        elif nxt is not None: yy[i] = nxt
    yy[yy < 0] = 0
    y = yy
```

iii. The AI's approach handles missing motion data through NaN propagation and filling, which is a reasonable but different strategy from the reference's approach of detecting and interpolating dropped frames at the raw frame level.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the custom `compute_dff` function, which loops over every neuron individually to compute a moving average with reflected padding and a percentile baseline. Loading `.npy` files is also I/O bound.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
```

iii. The per-neuron loop for baseline computation is inherently sequential and cannot benefit from GPU acceleration, unlike the reference's use of suite2p's `dcnv.preprocess` which supports batched GPU processing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `compute_dff` function loops over neurons individually for moving average and percentile computation. The forward-fill/backward-fill loop for NaN labels could also be vectorized using pandas or numpy methods.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
```

iii. The per-neuron loop processes each neuron's full trace sequentially. While the moving average could potentially be vectorized across neurons, the percentile computation remains per-neuron.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing is apparent. The code processes each session once.

ii. N/A

iii. The code follows a single-pass architecture: load, process, bin, then discretize all sessions.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `stat.npy`, `tstamps.npy`, and `interframe_int.npy` but does not use them in the final processing pipeline. `stat.npy` is loaded but never referenced. `tstamps.npy` and `interframe_int.npy` are loaded but ignored after the initial alignment bug was fixed.

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
```

iii. These files are loaded as part of the general session loading function but are not used in downstream processing, wasting I/O time.
