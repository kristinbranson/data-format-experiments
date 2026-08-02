# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the `data/` directory for subject folders, iterates through each subject's session subdirectories, and loads suite2p neural data (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) and behavioral data (`motion_energy_glob.npy`, `tstamps.npy`) for each session. All sessions are collected into flat lists before assembling the final output.

ii.
```python
def load_session(sess_path):
    suite = sess_path / 'suite2p' / 'plane0'
    move = sess_path / 'move_deve'
    F = np.load(suite / 'F.npy', mmap_mode='r')
    Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(suite / 'iscell.npy')
    ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)
```

iii. The AI noted that the directory structure follows a subject/session hierarchy with suite2p and motion energy subdirectories. It loads all available directories as subjects/sessions without filtering by name prefix (unlike the reference which filters for `jm*` prefixes). In practice all directories in the data folder are subject directories so this produces the same result.

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted top-level directories under `data/`. Unlike the reference which filters for directories starting with `jm`, the AI includes all directories.

ii.
```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
```

iii. The AI noted from data exploration that all subject folders follow the `jm*` naming convention, so including all directories produces the same result as filtering by prefix.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject folder. Each subdirectory corresponds to one daily recording session.

ii.
```python
for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
    session_paths.append(sess)
    session_subjects.append(subj)
```

iii. The AI documented that each subdirectory contains suite2p output and motion energy files for one recording session, matching the reference approach.

## 1-d. How are the data split into trials?

i. There is no natural trial structure in this dataset. The AI splits each session into consecutive non-overlapping 2-minute blocks (after 10-frame temporal averaging). With 30 Hz raw data and 10-frame bins, each 2-minute block is 360 bins. Any remainder data not filling a complete block is discarded.

ii.
```python
bin_size_frames = 10
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning = 360 bins

def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    T2 = n_blocks * block_bins
    ...
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
```

iii. The AI justified this choice by citing the paper's methods: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" and that decoding used "consecutive 2-minute blocks." The reference solution uses 60-second trials at native 30 Hz instead.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 blocks are skipped. No other trial-level filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The instructions require at least two trials per session for decoder evaluation. The AI added this guard to ensure compliance.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `iscell.npy` (cell classification), and `ops.npy` (suite2p parameters including `neucoeff` and `prctile_baseline`).

ii.
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
```

iii. The AI correctly identified these as the standard suite2p output files needed for computing fluorescence-based neural traces.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - neucoeff * Fneu` with neucoeff from ops, default 0.7). Then a simple baseline correction is performed: the 8th percentile of each neuron's trace is computed as F0, and dF/F is calculated as `(Fc - F0) / F0`. This is followed by 10-frame temporal averaging.

ii.
```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile_baseline = float(ops.get('prctile_baseline', 8.0))
    F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
    F0 = np.maximum(F0, 1e-3)
    dff = (Fc - F0) / F0
    return dff.astype(np.float32)
```

```python
def bin_time_series(x, bin_size):
    T = x.shape[-1]
    T2 = (T // bin_size) * bin_size
    x = x[..., :T2]
    new_shape = x.shape[:-1] + (T2 // bin_size, bin_size)
    return x.reshape(new_shape).mean(axis=-1)
```

iii. The AI documented this as a "fast approximation to suite2p baseline-corrected fluorescence." It initially attempted a sliding-window percentile baseline but found it too slow and switched to a global per-neuron percentile baseline. The reference solution uses suite2p's `dcnv.preprocess` with `baseline='maximin'`, `win_baseline=60.0`, which is a more faithful reproduction of the paper's described processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the suite2p cell classification: only ROIs with `iscell[:,1] > 0.5` (cell probability > 0.5) are retained.

ii.
```python
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32), np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The AI cited both the paper methods ("We considered all ROIs above the default threshold of 0.5 as true cells") and the Track2p reference code default (`iscell_thr = 0.50`). The reference solution does NOT filter by iscell.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since the recording is continuous and trials are contiguous blocks from the beginning of the session, no event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 120.0,
```

iii. The AI documented that there is no stimulus event to align to, as the recording captures spontaneous activity. This matches the reference approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal averaging, resulting in a time bin size of ~333.3 ms (10 frames / 30 Hz * 1000). This is applied to both neural and behavioral data.

ii.
```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs  # 333.33 ms

dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
```

iii. The AI cited the paper's methods: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The reference solution keeps the native 30 Hz resolution (33.33 ms bins) without rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the frame index divided by the raw frame rate (30 Hz), giving elapsed seconds from the start of the session. It is not derived from any stored timestamp variable.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The AI initially attempted to use `tstamps.npy` but discovered the values were not in seconds (range ~0-1.21 with tiny increments). It switched to computing time from frame indices, which is consistent with the reference approach.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices are divided by 30 Hz to get seconds, then averaged in 10-frame bins to match the neural data's temporal resolution. The result is cumulative time from session start.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The 10-frame binning of the time vector produces bin-center times that correspond to the same temporal bins as the neural and output data.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is constructed from the same frame indices used for neural data, so alignment is inherent. Both are binned with the same 10-frame averaging procedure.

ii.
```python
# In make_blocks:
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. Since time is derived from frame indices and processed with the same binning as neural data, they are perfectly aligned by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session. The AI also loads `tstamps.npy` but ultimately does not use it for output computation.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video. Unlike the reference solution, the AI does NOT load `interframe_int.npy` for dropped frame detection.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is (1) truncated to the minimum length shared with neural data, (2) averaged in 10-frame bins, then (3) discretized into 5 equal-percentile bins computed across all sessions. No normalization by standard deviation is applied.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
...
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. The AI applies percentile-based discretization globally across all sessions, which produces approximately balanced bins. The reference solution additionally normalizes motion energy by its standard deviation per session before discretizing, though this does not affect percentile-based bin assignments.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Continuous motion energy values are discretized into 5 bins using percentile edges at the 20th, 40th, 60th, and 80th percentiles, computed across all sessions' binned motion energy values. `np.digitize` assigns bin labels 0-4.

ii.
```python
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. This produces 5 equal-percentile bins as required by the instructions. The approach is functionally equivalent to the reference solution's method of computing `np.linspace(0, 100, 6)` percentiles and using `bin_edges[1:-1]`.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy array is truncated to the minimum length shared with the neural data (`T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])`). Then both are binned with the same 10-frame averaging and split into the same 2-minute blocks.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
```

iii. The AI relied on the paper's statement that video acquisition was triggered by microscope acquisition, so the streams are inherently synchronized frame-by-frame. However, the AI does NOT handle dropped video frames using `interframe_int.npy`, which the reference solution does. Instead, length mismatches are resolved by truncation to the shortest stream.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Length mismatches between neural and motion energy arrays are handled by truncating both to the minimum length. Sessions with fewer than 2 trial blocks are skipped. Remainder frames not filling a complete block are discarded. The AI initially discovered that `tstamps.npy` was not in seconds and switched to frame-index-based time computation.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The reference solution handles dropped video frames by detecting them via `interframe_int.npy` and interpolating missing values. The AI's truncation approach is simpler but may result in slight misalignment between neural and motion energy data when dropped frames cause the motion energy array to be shorter than the neural array.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified its initial sliding-window percentile baseline estimation as the major bottleneck, which it replaced with a fast per-neuron global percentile computation. The final code's bottleneck is the dF/F computation and file I/O.

ii. N/A (timing analysis documented in CONVERSION_NOTES.md)

iii. The AI documented that sample conversion completed in ~1 second per session after optimization. The reference solution's most time-consuming step is suite2p's `dcnv.preprocess` GPU-accelerated baseline correction.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The block-splitting loop in `make_blocks` iterates over blocks and slices arrays. This could potentially be vectorized using reshape operations, though the loop is simple and fast.

ii.
```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
    input_trials.append(time_binned[sl][None, :].astype(np.float32))
    output_cont.append(motion_binned[sl][None, :].astype(np.float32))
```

iii. With only ~10 blocks per session, this loop has negligible performance impact.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. The code makes one pass through sessions, with a second pass only for discretization (applying bin edges to continuous motion energy values).

ii.
```python
# First pass: compute continuous motion energy
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
# Second pass: discretize
for sess_trials in all_output_cont:
    for arr in sess_trials:
        labels = np.digitize(x, edges, right=False)
```

iii. The two-pass approach is necessary because global percentile edges must be computed before discretization can be applied.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `tstamps.npy` for every session but does not use it in the final output (it constructs time from frame indices instead). This is unnecessary file I/O.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

iii. The timestamps were originally intended to be used for the time input, but the AI discovered they were not in the expected units and switched to frame-index-based computation. The loading of `tstamps.npy` was not removed from the final code.
