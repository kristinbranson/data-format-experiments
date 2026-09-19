# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over all subdirectories in the `data/` root directory as subjects, then iterating over subdirectories within each subject as sessions. For each session, it loads suite2p files (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) from `suite2p/plane0/`, and behavioral files (`motion_energy_glob.npy`, `tstamps.npy`) from `move_deve/`. Notably, it does NOT filter subject directories by a `jm` prefix — all directories are treated as subjects.

ii.
```python
root = Path('data')
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
# ...
for subj in subjects:
    subj_path = root / subj
    for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
        session_paths.append(sess)
        session_subjects.append(subj)
```

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

iii. The AI documented in CONVERSION_NOTES.md that data is organized as subject folders containing session subfolders, each with `suite2p/plane0/` and `move_deve/` subdirectories. The AI also loads `iscell.npy` and `ops.npy` for neuron filtering and processing parameters, which the reference does not use.

## 1-b. How are the data split into subjects?

i. Subjects are identified as all subdirectories within the `data/` root. Unlike the reference which filters for directories starting with `jm`, the AI includes all directories.

ii.
```python
subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
```

iii. The AI's CONVERSION_NOTES.md identifies subject folders as directories in `data/` (e.g. `jm031`). In practice, since all directories in the data folder happen to be `jm*` prefixed, this produces the same result.

## 1-c. How are the data split into sessions?

i. Sessions are all subdirectories within each subject folder, sorted alphabetically. Each subdirectory contains one recording session's data.

ii.
```python
for sess in sorted([p for p in subj_path.iterdir() if p.is_dir()]):
    session_paths.append(sess)
    session_subjects.append(subj)
```

iii. The AI documented that each subject folder contains multiple session/day folders. This matches the reference approach.

## 1-d. How are the data split into trials?

i. The AI splits continuous sessions into consecutive **2-minute** (120-second) non-overlapping blocks, yielding 360 bins per trial (120s * 30Hz / 10 frames). This differs from the reference which uses 60-second trials (180 bins per trial) as specified in the instructions. Sessions with fewer than 2 blocks are skipped.

ii.
```python
block_bins = int((2 * 60 * raw_fs) / bin_size_frames)  # 2 minutes after binning
# ...
def make_blocks(dff_binned, motion_binned, time_binned, block_bins):
    T = dff_binned.shape[1]
    n_blocks = T // block_bins
    # ...
    for i in range(n_blocks):
        sl = slice(i * block_bins, (i + 1) * block_bins)
```

iii. The AI justified this decision based on the paper's decoding analyses which split recordings into consecutive 2-minute blocks. CONVERSION_NOTES.md Step 4 states: "Decoding uses consecutive 2-minute blocks and 10-frame averaging" and Step 5 states: "Continuous sessions will be segmented into consecutive 2-minute blocks: This follows the paper's decoding split unit."

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 blocks (trials) are skipped. No other trial-level filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. The instructions require at least two trials per session for decoder evaluation. The AI implemented this minimum check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `iscell.npy` (cell classification), and `ops.npy` (suite2p parameters) from `suite2p/plane0/`.

ii.
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
Fneu = np.load(suite / 'Fneu.npy', mmap_mode='r')
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
```

iii. The AI identified these as standard suite2p output files, consistent with the reference code's use of `F.npy` and `Fneu.npy`. The AI additionally loads `iscell.npy` for neuron filtering and `ops.npy` for processing parameters.

## 2-b. How is the `neural` data processed?

i. The AI applies: (1) neuron filtering with `iscell[:,1] > 0.5`, (2) neuropil subtraction (`Fc = F - 0.7 * Fneu`), (3) a simplified baseline normalization using a single per-neuron low-percentile value (`F0 = np.percentile(Fc, 8.0, axis=1)`, then `dff = (Fc - F0) / F0`), and (4) 10-frame bin averaging.

This differs significantly from the reference which uses suite2p's `dcnv.preprocess` with the `maximin` baseline method — a sliding-window approach that computes a time-varying baseline, not a single scalar per neuron.

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

iii. CONVERSION_NOTES.md Step 6 states: "Replaced expensive moving-percentile baseline with a fast per-neuron low-percentile baseline approximation." The AI explicitly chose speed over fidelity to the reference suite2p processing. The CONVERSION_NOTES.md calls it a "fast approximation to suite2p baseline-corrected fluorescence."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `iscell[:,1] > 0.5`, keeping only ROIs classified as cells by suite2p with probability above 0.5. The reference does NOT apply this filter and includes all neurons.

ii.
```python
keep = iscell[:, 1] > 0.5
dff = compute_dff(np.asarray(F[keep], dtype=np.float32), np.asarray(Fneu[keep], dtype=np.float32), ops)
```

iii. The AI justified this based on the paper's methods ("We considered all ROIs above the default threshold of 0.5 as true cells") and the Track2p reference code default `iscell_thr = 0.50`. CONVERSION_NOTES.md Step 4 explicitly resolves this: "Use `iscell[:,1] > 0.5` to match both code default and methods text."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. The continuous recording is split into consecutive blocks starting from the beginning. No event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 120.0,
```

iii. With continuous spontaneous recordings and no stimulus events, alignment to session start is the natural choice. Both the AI and reference agree on this.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and behavioral data are averaged into non-overlapping bins of 10 frames, converting 30 Hz raw data to 3 Hz (333.33 ms bins). This matches the reference.

ii.
```python
bin_size_frames = 10
raw_fs = 30.0
time_bin_size_ms = 1000.0 * bin_size_frames / raw_fs
# ...
dff_binned = bin_time_series(dff, bin_size_frames)
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)
```

iii. CONVERSION_NOTES.md cites the paper: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from frame indices and the known frame rate (30 Hz), not from any raw timestamp file. The AI constructs frame times as `np.arange(dff.shape[1]) / 30.0`, then bins them.

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
```

iii. CONVERSION_NOTES.md Step 10 documents: "Time input bug: initial use of `tstamps.npy` produced a compressed range (~0 to 1.21). Resolved by constructing elapsed time from frame index / 30 Hz, matching the methods text." The AI initially tried using `tstamps.npy` but discovered it produced incorrect ranges and switched to frame-index-based computation.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices are divided by the frame rate (30 Hz) to get seconds, then averaged in 10-frame bins. Within each block/trial, the time values represent seconds from the start of the session (not the trial).

ii.
```python
frame_time = (np.arange(dff.shape[1], dtype=np.float32) / raw_fs)
time_binned = bin_time_series(frame_time[None, :], bin_size_frames).squeeze(0)
# Later in make_blocks:
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The reference computes time similarly but uses bin indices rather than frame indices: `t = ((s + np.arange(trial_frames)) * BIN_FRAMES / FS)`. Both approaches yield time from session start in seconds. The AI's approach bins the raw frame times, while the reference computes bin-center times directly. The AI's time values represent bin-averaged frame times (centered within each bin), while the reference uses left-edge times of each bin. The numerical difference is small (half a bin width = ~0.15s).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned because it's derived from the same frame indices as the neural data. Both are binned identically and sliced into the same blocks.

ii.
```python
# In make_blocks, same slice applied to all:
sl = slice(i * block_bins, (i + 1) * block_bins)
neural_trials.append(dff_binned[:, sl].astype(np.float32))
input_trials.append(time_binned[sl][None, :].astype(np.float32))
```

iii. The alignment is guaranteed by construction since the time array has the same length as the neural array and both are sliced with identical indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session. Unlike the reference, the AI does NOT load `interframe_int.npy` for dropped frame detection.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r').astype(np.float32)
```

iii. The AI identified this as the pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI applies: (1) truncation to minimum length across neural, motion, and timestamp arrays, (2) 10-frame bin averaging, (3) discretization into 5 bins using **global** percentile edges computed across ALL sessions (at fixed [20, 40, 60, 80] percentiles).

This differs from the reference which uses **per-session** percentile edges.

ii.
```python
# Truncation instead of interpolation:
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:T]

# Binning:
motion_binned = bin_time_series(motion[None, :], bin_size_frames).squeeze(0)

# Global discretization:
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
# ...
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. CONVERSION_NOTES.md Step 5 states: "Percentile thresholds for discretization should be computed on the full binned output distribution, then applied consistently to all sessions: This best matches the task requirement for normalized/discretized motion energy and avoids per-session label drift." The AI chose global percentiles to ensure consistent bin boundaries across sessions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global percentile edges at [20, 40, 60, 80] percentiles across all sessions' motion energy values, then uses `np.digitize` with clipping to assign labels 0-4. The reference computes per-session edges at [0, 20, 40, 60, 80, 100] percentiles and uses `np.digitize(me, bin_edges[1:-1])`.

ii.
```python
edges = np.percentile(all_motion_values, [20, 40, 60, 80]).astype(np.float32)
labels = np.digitize(x, edges, right=False).astype(np.int64)
labels = np.clip(labels, 0, 4)
```

iii. The instructions say "five equal-percentile bins, selected per session." The AI chose global percentiles instead of per-session, contradicting the explicit instruction.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy with neural data by truncating both to the minimum length: `T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])`. This discards any frames where lengths don't match. The reference instead detects dropped video frames via `interframe_int.npy` and interpolates to match the neural data length.

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
tstamps = tstamps[:T]
```

iii. The AI does not document this truncation approach explicitly in CONVERSION_NOTES.md. The reference's approach preserves more data by filling in dropped frames rather than truncating.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles length mismatches between neural and behavioral data by truncating to the shortest array length. Sessions with fewer than 2 blocks are skipped. There is no explicit handling of dropped video frames (unlike the reference which interpolates them).

ii.
```python
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
dff = dff[:, :T]
motion = motion[:T]
# ...
if len(neural_trials) < 2:
    print(f'Skipping {sess_id}: fewer than 2 blocks')
    continue
```

iii. CONVERSION_NOTES.md does not discuss dropped frame handling. The truncation approach is simpler but potentially loses data and introduces misalignment if frames are dropped in the middle of a recording (though the actual impact may be small if dropped frames are few and near the end).

## 6-a. What are the most time-consuming steps of the code?

i. The AI identified the initial moving-percentile baseline estimation as the main bottleneck and replaced it with a fast per-neuron percentile approximation. Loading `.npy` files with memory mapping is also used for efficiency.

ii.
```python
F = np.load(suite / 'F.npy', mmap_mode='r')
# ...
F0 = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 6 states: "Replaced expensive moving-percentile baseline with a fast per-neuron low-percentile baseline approximation, reducing sample conversion to <1 second for 2 sessions." The reference uses `dcnv.preprocess` which is slower but produces the correct suite2p baseline correction.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is relatively well vectorized. The main loop is over sessions which cannot be easily vectorized. The `make_blocks` function loops over blocks but uses numpy slicing within each iteration.

ii.
```python
for i in range(n_blocks):
    sl = slice(i * block_bins, (i + 1) * block_bins)
    neural_trials.append(dff_binned[:, sl].astype(np.float32))
```

iii. No explicit discussion of vectorization opportunities in CONVERSION_NOTES.md beyond the baseline computation speedup.

## 6-c. What processing does the code repeat multiple times?

i. The AI processes motion energy values twice: once during the initial session loop (storing continuous values) and again in a second pass to apply discretization. This two-pass approach is necessary because global percentile edges require seeing all data first.

ii.
```python
# First pass: collect continuous motion energy
all_motion_values.append(np.concatenate([x.ravel() for x in output_cont]))

# Second pass: discretize
all_motion_values = np.concatenate(all_motion_values)
edges = np.percentile(all_motion_values, [20, 40, 60, 80])
for sess_trials in all_output_cont:
    # ...apply edges...
```

iii. This two-pass approach is a consequence of the global discretization choice and is unavoidable given that design decision.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `tstamps.npy` for each session but does not use it for the time input (which is computed from frame indices instead). The timestamps are only used for the truncation length calculation.

ii.
```python
tstamps = np.load(move / 'tstamps.npy', mmap_mode='r').astype(np.float64)
# ...
T = min(dff.shape[1], motion.shape[0], tstamps.shape[0])
```

iii. CONVERSION_NOTES.md documents that the AI initially used `tstamps.npy` for time input but switched to frame-index-based computation after discovering incorrect ranges. The loading of `tstamps.npy` remains as vestigial code used only for length determination.
