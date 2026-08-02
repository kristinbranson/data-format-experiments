# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` in the `data/` directory, then discovers sessions as subdirectories with names starting with 4 digits. For each session, it loads `F.npy`, `Fneu.npy`, `ops.npy`, `stat.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` from `move_deve/`. All sessions are loaded in a loop over subjects and sessions.

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

iii. The AI documented this in CONVERSION_NOTES.md under Steps 1-2, noting the directory structure follows standard suite2p conventions. The AI loads `ops.npy` and `stat.npy` as well, though these are only used to read baseline parameters for dF/F computation.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data root, sorted alphabetically. 6 subjects are identified: jm031, jm032, jm038, jm039, jm040, jm046.

ii.
```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
subject_names = [p.name for p in subjects]
```

iii. Each `jm*` directory represents one mouse. The AI documented 6 subjects matching the paper's description.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a dated subdirectory within a subject folder (filtered by names starting with 4 digits), sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The AI identified 41 total sessions across 6 subjects (7,7,7,7,6,7 per subject).

## 1-d. How are the data split into trials?

i. The AI defines trials as consecutive 2-minute (120-second) non-overlapping blocks of the continuous recording. After 10-frame binning, each block has `BLOCK_BINS = 360` time bins. Remainder frames that don't fill a complete block are discarded.

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
        neural_blocks.append(neural_binned[:, s:e])
        motion_blocks.append(motion_binned[s:e])
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The AI justified this by referencing the paper's description: "splits were done on consecutive 2 minute blocks of the recording" for the decoding analysis.

## 1-e. How are trials filtered based on quality controls?

i. Trials (blocks) are filtered based on the fraction of valid (non-NaN) motion energy values. A block is excluded if fewer than 80% of its time bins have valid motion energy data, or if fewer than 10 bins are valid. Sessions with fewer than 2 valid blocks are also excluded entirely.

ii.
```python
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. The AI implemented this to handle missing motion energy data from dropped camera frames, ensuring blocks with too much missing data don't corrupt the decoder training.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The `ops.npy` file is also loaded for baseline correction parameters.

ii.
```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
```

iii. These are standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI implements a custom dF/F computation: (1) neuropil subtraction with coefficient 0.7, (2) moving average smoothing with reflect-padded convolution using a window of `win_baseline * fs` frames, (3) scalar percentile baseline estimation (8th percentile of the smoothed trace), (4) dF/F = (Fcorr - baseline) / baseline, with baseline floored at 1e-3.

ii.
```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
    prct = float(ops.get('prctile_baseline', 8.0))
    baseline = np.empty_like(Fcorr, dtype=np.float32)
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fcorr - baseline) / baseline
    return dff
```

iii. The AI noted this follows the paper's description of using "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." However, the AI implemented a custom approximation rather than using suite2p's actual `dcnv.preprocess` function. The baseline is a single scalar per neuron (the 8th percentile of the smoothed trace), not a time-varying baseline as suite2p's `maximin` method produces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons in the `F.npy` output are included without checking `iscell.npy`.

ii. N/A (no filtering code)

iii. The AI noted in CONVERSION_NOTES.md Step 3 that the paper says "We considered all ROIs above the default threshold of 0.5 as true cells" and documented this as a curation rule, but did not implement `iscell` filtering in the code. The CONVERSION_NOTES also state "All observed `iscell[:,0] == 1`", suggesting the AI believed all ROIs already passed the threshold in the released data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 2-minute block. Since blocks are contiguous segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'start of each 2-minute continuous recording block',
'off_start': 0.0,
'off_end': BLOCK_SECONDS,
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal rebinning. The native 30 Hz data is averaged in bins of 10 frames, resulting in a temporal resolution of ~333.33 ms (3 Hz effective rate). Each 2-minute block has 360 time bins.

ii.
```python
BIN_FRAMES = 10

def bin_neural(x, bin_frames):
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)

'time_bin_size': 1000.0 * BIN_FRAMES / FS,  # 333.33 ms
```

iii. The AI justified this based on the paper's decoding methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index within each 2-minute block, using the known frame rate and bin size.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. Since the frame rate is constant at 30 Hz and the bin size is 10 frames, time can be computed directly from indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (10 / 30)` seconds, giving time elapsed from the start of each 2-minute block. The time resets to 0 at the beginning of each block/trial, ranging from 0 to ~119.7 seconds.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
# Results in [0.0, 0.333, 0.667, ..., 119.667]
```

iii. The AI documented this as "time elapsed from beginning of experiment; operationalize as time within session/block after alignment."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is constructed to have exactly the same number of time bins as the binned neural data within each block (360 bins), so alignment is trivial by construction.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
# BLOCK_BINS = 360, same as neural_binned[:, s:e] shape
```

iii. No special alignment is needed since both are derived from the same frame indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory. `tstamps.npy` is also loaded but ultimately not used for alignment (the AI switched to simple pad/truncate).

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
```

iii. The motion energy file contains pre-computed global motion energy from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Motion energy is aligned to neural frames by simple copy of the first `min(nframes, motion.shape[0])` values, with any excess frames filled with NaN; (2) 10-frame binning with NaN-aware mean; (3) Discretization into 5 bins using quantile edges [0.2, 0.4, 0.6, 0.8] computed across all valid binned motion values; (4) NaN bins are forward/backward filled in the output labels.

ii.
```python
def align_motion_to_frames(motion, tstamps, nframes):
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned

motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)

edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. The AI initially tried using timestamps for alignment but found it caused 9 sessions to fail, so switched to simple pad/truncate per the data README guidance.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using 4 quantile edges at the 20th, 40th, 60th, and 80th percentiles of all valid binned motion values across all sessions. `np.digitize` maps values to bins 0-4.

ii.
```python
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. This follows the instruction to discretize into "five equal-percentile bins."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural frames by copying the first `min(nframes, len(motion))` values, padding remaining frames with NaN. Both are then binned by 10 frames. NaN bins in the discretized output are forward/backward filled.

ii.
```python
def align_motion_to_frames(motion, tstamps, nframes):
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
```

iii. The AI noted: "Data README states motion_energy_glob is framewise with occasional missing camera frames. `tstamps.npy` stores timestamps (seconds), not frame indices, so alignment should preserve sample order and pad/truncate to imaging length."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames result in the motion energy array being shorter than the neural array. The AI handles this by padding the motion energy with NaN to match the neural length. After binning, NaN bins are forward/backward filled in the output labels. Blocks with too many NaN bins (>20%) are excluded. Sessions with fewer than 2 valid blocks are excluded.

ii.
```python
aligned = np.full(nframes, np.nan, dtype=np.float32)
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]

# NaN handling in output labels:
if not np.all(valid):
    yy = y.copy()
    last = None
    for i in range(len(yy)):
        if yy[i] >= 0:
            last = yy[i]
        elif last is not None:
            yy[i] = last
```

iii. The AI documented this approach after initially encountering issues with timestamp-based alignment. The forward/backward fill ensures all time bins have valid output labels.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the custom `compute_dff` baseline computation, which loops over every neuron and applies a moving average followed by percentile computation. This is done on CPU without GPU acceleration (unlike the reference solution which uses suite2p's GPU-accelerated `dcnv.preprocess`). The full conversion took ~130 seconds.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
```

iii. The AI acknowledged this as a potential inefficiency in CONVERSION_NOTES.md.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `compute_dff` function loops over neurons individually to compute moving averages and percentile baselines. The forward/backward NaN fill for output labels also uses a Python loop. The moving average convolution could be vectorized across neurons using 2D convolution or scipy functions.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)

# Forward/backward fill loop:
for i in range(len(yy)):
    if yy[i] >= 0:
        last = yy[i]
    elif last is not None:
        yy[i] = last
```

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. The AI processes all sessions in a single pass, then does a second pass to assemble the final data structure (discretize and segment into blocks). The motion energy values are collected once for global quantile computation, then digitized per block in the second pass. No significant repeated computation.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `stat.npy` and `tstamps.npy` for each session but does not meaningfully use them. `stat.npy` is loaded but never referenced. `tstamps.npy` is passed to `align_motion_to_frames` but is not used inside the function (the function uses simple copy/pad instead).

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)

def align_motion_to_frames(motion, tstamps, nframes):
    # tstamps is not actually used
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
```

iii. These appear to be remnants of an earlier approach that was abandoned.
