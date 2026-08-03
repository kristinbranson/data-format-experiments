# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` in the `data/` folder, then discovers sessions as subdirectories within each subject. For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, plus `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy` from `move_deve/`. It also loads `stat.npy`. All sessions are iterated over in a loop, processed, and stored in a list before final assembly.

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

iii. The AI identified the directory structure convention from the data README and the reference code. It loads `ops.npy` and `stat.npy` in addition to the core data files, though these are not heavily used. Session directories are filtered to those whose name starts with 4 digits (a date pattern).

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data root, sorted alphabetically. Each subject directory represents one mouse.

ii.
```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
subject_names = [p.name for p in subjects]
```

iii. The naming convention is consistent across the dataset. The AI preserves the full list of subject names for subject indexing.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, filtered to those whose name starts with 4 digits (date format), and sorted alphabetically. Each subdirectory represents one recording day.

ii.
```python
sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The AI applies a date-pattern filter (`name[:4].isdigit()`) to avoid picking up non-session directories. Sorting ensures chronological order.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive 2-minute (120-second) non-overlapping blocks of the continuous recording. After 10-frame binning, each block has BLOCK_BINS = 360 time bins. Any remainder that doesn't fill a complete block is discarded.

ii.
```python
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)  # 3600
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES  # 360

def session_to_blocks(neural_binned, motion_binned):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    n_blocks = n_bins // BLOCK_BINS
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
    return neural_blocks, motion_blocks, time_blocks
```

iii. The AI chose 2-minute blocks based on the paper's statement that "splits were done on consecutive 2 minute blocks of the recording" for cross-validation in decoding analyses. This is documented in CONVERSION_NOTES.md Step 3 and Step 5.

## 1-e. How are trials filtered based on quality controls?

i. Blocks (trials) are filtered based on the amount of valid (non-NaN) motion energy data. A block is excluded if valid motion energy samples are fewer than max(10, 80% of block length). Additionally, sessions with fewer than 2 valid blocks are excluded entirely.

ii.
```python
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. This filtering ensures trials with too much missing motion data are excluded and that every session has at least 2 trials for decoder cross-validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The `ops.npy` dictionary is also loaded to extract preprocessing parameters.

ii.
```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI implements a custom dF/F computation: (1) neuropil subtraction with coefficient 0.7 (`Fcorr = F - 0.7 * Fneu`), (2) smoothing each neuron's trace with a moving average (reflect-padded, window = `win_baseline * fs` = 1800 frames), (3) computing a scalar percentile baseline per neuron (8th percentile of the smoothed trace), (4) computing dF/F as `(Fcorr - baseline) / baseline`, and (5) binning in 10-frame windows by averaging.

ii.
```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fcorr - baseline) / baseline
    return dff.astype(np.float32)

def bin_neural(x, bin_frames):
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES.md states it follows suite2p-consistent neuropil correction and baseline normalization. However, the implementation differs from the reference solution, which uses suite2p's `dcnv.preprocess` with the `maximin` baseline method (a sliding-window min-of-max filter, not a global percentile of a moving average).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons present in `F.npy` are included.

ii. N/A (no filtering code)

iii. The data README states that the `suite2p` folders already contain only cells tracked across all days. The AI noted in CONVERSION_NOTES.md that the released data has already been curated by Track2p.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the recording session. Since data is continuous and trials are contiguous blocks, no event-based alignment is performed. The temporal alignment event is described as "start of each 2-minute continuous recording block."

ii.
```python
'temporal_alignment_event': 'start of each 2-minute continuous recording block',
'off_start': 0.0,
'off_end': BLOCK_SECONDS,
```

iii. There is no stimulus or task event to align to in this spontaneous behavior dataset. Trials are artificial segments of the continuous recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal binning, resulting in a time bin size of 333.33 ms (10 frames / 30 Hz). Both neural and motion energy data are averaged in 10-frame bins.

ii.
```python
BIN_FRAMES = 10
...
'time_bin_size': 1000.0 * BIN_FRAMES / FS,  # 333.33 ms
...
neural_binned = bin_neural(dff, BIN_FRAMES)
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
```

iii. The AI chose 10-frame binning based on the paper's methods: "averaging in bins of 10 consecutive timestamps" for decoding analyses.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed as the bin index within each 2-minute block multiplied by the bin duration.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. Since the frame rate is constant at 30 Hz, time can be computed from indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (10 / 30)` seconds within each block, giving values from 0 to ~119.7 seconds per block. This resets to 0 at the start of each 2-minute block.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
# produces values: [0.0, 0.333, 0.667, ..., 119.667]
```

iii. The AI interpreted "time from the beginning of the experiment" as time within each block rather than cumulative time from the session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is constructed directly from neural data bin indices, so alignment is inherent. Each time point corresponds exactly to one neural data time bin.

ii. Same as 3-b above.

iii. No separate alignment step is needed since time is derived from the same indexing scheme as neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. `tstamps.npy` and `interframe_int.npy` are also loaded but used only indirectly.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
```

iii. The motion energy file contains pre-computed global motion energy from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) motion energy is aligned to imaging frames by copying in order and padding/truncating to match the neural frame count, with missing frames as NaN, (2) 10-frame bin averaging (using NaN-aware binning), (3) discretization into 5 bins using global quantile edges computed across all valid binned motion energy values.

ii.
```python
def align_motion_to_frames(motion, tstamps, nframes):
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

def nanbin_mean_1d(x, bin_frames):
    ...
    out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return out

edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. The AI chose to handle missing camera frames by treating them as NaN rather than interpolating. The 10-frame bin averaging and 5-bin discretization follow the paper's methods.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories using 4 quantile edges at the 20th, 40th, 60th, and 80th percentiles, computed across all valid binned motion energy values from all sessions. `np.digitize` maps values into bins 0-4.

ii.
```python
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. Global percentile-based discretization ensures balanced class counts across the dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to imaging frames by simple sample-order copying: the first `min(nframes, len(motion))` motion values are placed into an array of length `nframes`, with any remaining positions filled with NaN. After 10-frame binning, NaN bins are handled by nearest-neighbor fill in the discretized output.

ii.
```python
def align_motion_to_frames(motion, tstamps, nframes):
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
```

iii. The AI initially tried to use `tstamps.npy` as frame indices but this caused 9 sessions to be dropped. The final approach treats motion energy as ordered samples and pads/truncates, per the data README's description that motion energy is "framewise."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames result in the motion energy array being shorter than the neural data. The AI handles this by padding with NaN and using NaN-aware binning. After discretization, NaN labels are filled using nearest-neighbor interpolation (forward then backward). Blocks with >20% missing data are excluded entirely. Remainder frames at the end of a session that don't fill a complete block are discarded.

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
        if yy[i] >= 0:
            last = yy[i]
        elif last is not None:
            yy[i] = last
    ...
```

iii. The NaN-padding approach is simpler than interpolation but introduces missing values that must be handled downstream.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the custom `compute_dff` function, which loops over each neuron to compute a moving average with reflect padding and a percentile baseline. This is done on CPU with Python loops. The total conversion for 41 sessions takes ~130 seconds.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
```

iii. The per-neuron loop with moving average and percentile computation is the bottleneck. The reference solution uses suite2p's GPU-accelerated `dcnv.preprocess`.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The baseline computation loop in `compute_dff` iterates over neurons one at a time. This could potentially be vectorized using 2D convolution operations. The NaN-fill loop in discretization output also iterates element by element.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
```

iii. The number of neurons per session ranges from 221 to 746, so vectorization would provide meaningful speedup.

## 6-c. What processing does the code repeat multiple times?

i. The code loads and processes all session data in a first pass, then iterates over the prepared data again to assemble the final dictionary. This is a two-pass approach but doesn't duplicate computation since the first pass is preprocessing and the second is assembly/discretization.

ii.
```python
# First pass: preprocess
for idx, (subj, sessdir) in enumerate(selected):
    ...
    prepared.append({...})

# Second pass: discretize and assemble
for item in prepared:
    ...
```

iii. The two-pass design is necessary because global percentile bin edges must be computed from all sessions before discretization can occur.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `stat.npy`, `tstamps.npy`, and `interframe_int.npy` for every session, but `stat.npy` is never used and `tstamps.npy`/`interframe_int.npy` are loaded but not meaningfully used in the final alignment approach (which simply copies in order and pads with NaN). The code also computes `plot_processing` visualizations when `--show-processing` is enabled, which are for debugging only.

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)  # never used
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)  # loaded but not used in alignment
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)  # loaded but not used
```

iii. These files are loaded as part of a general `load_session` function but are vestigial from an earlier alignment approach that was abandoned.
