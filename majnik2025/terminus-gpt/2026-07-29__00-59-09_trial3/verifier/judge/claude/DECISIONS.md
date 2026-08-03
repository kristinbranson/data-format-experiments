# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories in the `data/` root folder (sorted alphabetically). For each subject, sessions are sorted subdirectories. Each session's neural data is loaded from `suite2p/plane0/F.npy` and `Fneu.npy`, and motion energy from `move_deve/motion_energy_glob.npy`. The AI also loads `iscell.npy` for neuron filtering, `ops.npy` for preprocessing parameters, and `tstamps.npy` for timestamps. All sessions are collected into flat lists before trial segmentation.

ii.
```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)

# In load_session:
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
```

iii. The AI noted that the directory structure follows standard conventions from the dataset, with subject folders containing session subfolders. The AI used memory-mapped loading (`mmap_mode='r'`) for efficiency. The AI also loaded `tstamps.npy` and `iscell.npy` which the reference solution does not use.

## 1-b. How are the data split into subjects?

i. Subjects are all sorted directories under the `data/` root. Each subject has a unique folder name (e.g., `jm031`).

ii.
```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
```

iii. The AI identified 6 subjects matching the paper's description. Unlike the reference which filters for `jm*` prefix, the AI includes all directories, but since the data directory only contains `jm*` folders, this produces the same result.

## 1-c. How are the data split into sessions?

i. Sessions are all sorted subdirectories within each subject folder. Each session represents one daily recording.

ii.
```python
for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
    session_paths.append(sess)
```

iii. The AI correctly identified each subdirectory as a session, matching the reference approach.

## 1-d. How are the data split into trials?

i. The AI splits continuous recordings into consecutive 2-minute (120-second) non-overlapping blocks after 10-frame temporal averaging. After binning, each block is 360 bins long. Any remainder frames that don't fill a complete block are discarded. Sessions with fewer than 2 blocks are skipped.

ii.
```python
bin_size_frames = 10
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning = 360 bins

def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    T2 = n_blocks * block_bins
    dff_binned = dff_binned[:, :T2]
    # ...
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
        neural_trials.append(dff_binned[:, sl].astype(np.float32))
```

iii. The AI chose 2-minute blocks based on the paper's decoding methodology which used "consecutive 2-minute blocks" for cross-validation splits. The reference solution uses 60-second (1-minute) trials with 1800 frames each at native 30 Hz. The AI's choice of 2 minutes is based on the paper's decoding block size, but these blocks were used for cross-validation grouping, not necessarily as individual trial units.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 blocks are skipped. No other trial-level filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The instructions require at least two trials per session for decoder evaluation. Both the AI and reference discard remainder frames. The reference does not filter sessions by minimum trial count (all sessions produce sufficient trials at 60s duration).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `iscell.npy` (cell classification), and `ops.npy` (suite2p parameters) from `suite2p/plane0/`.

ii.
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
```

iii. The AI correctly identified F.npy and Fneu.npy as the core source variables, matching the reference. The AI additionally loads iscell.npy and ops.npy for filtering and parameter extraction.

## 2-b. How is the `neural` data processed?

i. The AI applies: (1) neuron filtering via `iscell[:,1] > 0.5`, (2) neuropil subtraction `Fc = F - 0.7 * Fneu`, (3) a simplified dF/F computation using a per-neuron low-percentile (8th percentile) baseline: `F0 = np.percentile(Fc, 8, axis=1)`, then `dff = (Fc - F0) / F0`, and (4) 10-frame temporal averaging.

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

# Then temporal binning:
dff_binned = bin_time_series(dff, bin_size_frames)  # 10-frame averaging
```

iii. The AI noted this was a "fast approximation to suite2p baseline-corrected fluorescence." The reference solution uses suite2p's `dcnv.preprocess` with the `maximin` baseline method and a 60s sliding window, which is the actual Suite2p preprocessing pipeline. The AI's approach is a much simpler static percentile baseline rather than the time-varying baseline correction used by suite2p.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `iscell[:,1] > 0.5`, keeping only ROIs classified as cells by suite2p's classifier with probability above 0.5.

ii.
```python
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32), np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The AI justified this based on both the paper's methods ("We considered all ROIs above the default threshold of 0.5 as true cells") and the Track2p reference code default `iscell_thr = 0.50`. The reference solution does NOT apply iscell filtering, keeping all neurons from F.npy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since recordings are continuous with no stimulus events, the neural data is simply segmented into consecutive blocks from the beginning.

ii.
```python
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 120.0,
```

iii. Both the AI and reference align to session start, as there are no stimulus-driven events in this spontaneous activity dataset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal averaging, resulting in a time bin size of ~333.33 ms (10 frames / 30 Hz * 1000). Each trial has 360 time bins (2 minutes at this resolution).

ii.
```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs  # 333.33 ms

def bin_time_series(x, bin_size):
    T = x.shape[-1]
    T2 = (T // bin_size) * bin_size
    x = x[..., :T2]
    new_shape = x.shape[:-1] + (T2 // bin_size, bin_size)
    return x.reshape(new_shape).mean(axis=-1)
```

iii. The AI chose 10-frame binning based on the paper's methods: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The reference solution keeps data at native 30 Hz (no rebinning), with time_bin_size = 33.33 ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the frame index divided by the frame rate (30 Hz), giving elapsed seconds from session start. The AI also loads `tstamps.npy` but ultimately constructs time from frame indices.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. The AI initially tried using `tstamps.npy` but found it produced a compressed range, so switched to computing time from frame indices. The reference similarly computes time from frame indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as frame_index / 30.0 to get seconds, then averaged over 10-frame bins (same as neural and output data), then segmented into 2-minute blocks.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
# Then in make_blocks:
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The reference computes time similarly but without temporal binning: `t = ((s + np.arange(trial_frames)) / FS).astype(np.float32)`. The AI's binned time values represent the center of each 10-frame bin.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same frame indices as the neural data, then binned and segmented identically, ensuring perfect alignment.

ii.
```python
# Same bin_time_series and make_blocks applied to time as to neural and motion
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. Both AI and reference ensure alignment by deriving time from the same frame indices used for neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
```

iii. Both AI and reference use the same source file. The reference additionally loads `interframe_int.npy` for dropped frame detection, which the AI does not.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI: (1) truncates motion energy to match the minimum length across neural and motion data, (2) applies 10-frame temporal averaging, (3) segments into 2-minute blocks, then (4) discretizes into 5 bins using global percentile edges at [20, 40, 60, 80].

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
# ...
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. The reference solution: (1) detects dropped video frames via `interframe_int.npy` and interpolates them, (2) normalizes motion energy by its standard deviation, (3) computes 5 equal-percentile bins using `np.linspace(0, 100, 6)` percentiles, (4) uses `np.digitize(me, bin_edges[1:-1])`. The AI skips dropped frame handling and normalization, and uses a different percentile computation method.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global percentile edges at [20, 40, 60, 80] across all binned motion energy values, then uses `np.digitize` with clipping to produce 5 categories (0-4).

ii.
```python
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. The reference uses `np.linspace(0, 100, 6)` = [0, 20, 40, 60, 80, 100] and `np.digitize(me, bin_edges[1:-1])` which uses edges at indices [1:-1] = [20, 40, 60, 80] percentiles. Both approaches produce equivalent 5-bin discretization with equal-percentile boundaries.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI truncates the motion energy array to the minimum length across neural frames, motion frames, and timestamps, then applies the same 10-frame binning and block segmentation as neural data.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
# Same binning applied
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
```

iii. The reference handles alignment by detecting dropped video frames via `interframe_int.npy` and interpolating to match the neural data length exactly. The AI instead truncates to the shortest stream. Since some sessions have dropped frames (motion energy shorter than neural data), the AI's approach may silently lose some neural data at the end rather than properly accounting for the mismatch.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles length mismatches between neural and motion data by truncating to the minimum length. Remainder frames that don't fill a complete block are discarded. Sessions with fewer than 2 blocks are skipped.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
# ...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The reference explicitly detects dropped video frames via interframe intervals and interpolates them, ensuring the motion energy signal exactly matches the neural data length. The AI's truncation approach is simpler but less precise -- it drops the last few frames of neural data rather than properly interpolating the missing motion energy frames.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified that the initial sliding-window percentile baseline estimation was too slow and replaced it with a fast per-neuron global percentile baseline. The final code's most time-consuming step is the dF/F computation and file I/O, taking ~0.2-1.0 seconds per session. Total conversion time was ~25 seconds.

ii. N/A

iii. The reference code's most time-consuming step is `dcnv.preprocess` (suite2p baseline correction with GPU acceleration). The AI avoided this bottleneck entirely by using a simplified baseline approach.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The block creation loop in `make_blocks` iterates over blocks and slices arrays. This could potentially be reshaped into a single vectorized operation. However, the loop is simple and fast.

ii.
```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
```

iii. This loop is simple slicing and is not a performance bottleneck. The reference has a similar loop structure.

## 6-c. What processing does the code repeat multiple times?

i. The discretization is done in two passes: first, all continuous motion energy values are collected, then percentile edges are computed, then each session's trials are discretized. This requires iterating through all session data twice.

ii.
```python
# First pass: collect continuous values
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))
# Compute edges
edges = np.percentile(all_motion_values, [20, 40, 60, 80])
# Second pass: discretize
for sess_trials in all_output_cont:
    for arr in sess_trials:
        labels = np.digitize(x, edges, right=False)
```

iii. The two-pass approach is necessary to compute global percentiles before discretization. The reference uses a similar two-pass pattern.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `tstamps.npy` in every session but doesn't use it for the final time computation (time is derived from frame indices instead). The timestamps array is only used for truncation length calculation.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
```

iii. Loading timestamps is minimal overhead but strictly unnecessary since time is reconstructed from frame indices.
