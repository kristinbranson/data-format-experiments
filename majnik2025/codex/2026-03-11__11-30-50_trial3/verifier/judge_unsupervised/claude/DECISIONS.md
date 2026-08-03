# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by scanning the `data/` directory for subject folders (starting with "jm") and session date sub-folders. For each session, it loads `suite2p/plane0/F.npy`, `Fneu.npy`, `ops.npy` (neural data), and `move_deve/motion_energy_glob.npy`, `tstamps.npy` (behavioral data). The conversion uses a two-pass approach: Pass 1 scans all sessions' behavioral data to compute global normalization/quantile thresholds; Pass 2 processes neural data session-by-session.

ii.
```python
def discover_sessions(sample: bool) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
        session_dirs = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )
        for session_dir in session_dirs:
            sessions.append(SessionInfo(subject=subject_dir.name, session_id=f"{subject_dir.name}_{session_dir.name}", path=session_dir))
    ...

# In main(), Pass 2:
for idx, session in enumerate(sessions):
    ops = load_ops(session)
    F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
    Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md that the `data/README.md` states the `suite2p/` folder already contains only neurons tracked across all days, saved by Track2p. Therefore no additional Track2p matching step is needed. The AI loads the pre-matched Suite2p exports directly.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the top-level directory names under `data/` (e.g., `jm031`, `jm032`, ..., `jm046`). A sorted list of unique subject names is created and each session is mapped to its subject via `subject_to_idx`.

ii.
```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
converted["subject_idx"] = np.array([subject_to_idx[session.subject] for session in sessions], dtype=np.int64)
```

iii. The AI noted that the data release contains 6 subjects (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) matching the paper's description of 6 mice. Each subject folder contains session sub-folders with matched neuron data.

## 1-c. How are the data split into sessions?

i. Sessions are individual recording days within each subject folder, identified by date-format directory names (e.g., `2023-10-18_a`). Each session is treated as a separate entry in the dataset. 41 total sessions were discovered (7 sessions for 5 mice, 6 sessions for jm040).

ii.
```python
session_dirs = sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
)
for session_dir in session_dirs:
    sessions.append(SessionInfo(subject=subject_dir.name, session_id=f"{subject_dir.name}_{session_dir.name}", path=session_dir))
```

iii. The AI identified that sessions correspond to individual recording days. The paper states imaging was done daily for at least 6 consecutive days per mouse. Session counts match: jm031(7), jm032(7), jm038(7), jm039(7), jm040(6), jm046(7) = 41 total.

## 1-d. How are the data split into trials?

i. The native data has no trial structure (continuous recordings). The AI constructs pseudo-trials by splitting each session into consecutive non-overlapping 2-minute blocks. After 10-frame temporal binning (30 Hz -> 3 Hz), each 2-minute block yields 360 time bins. 20-minute sessions produce 10 trials and 30-minute sessions produce 15 trials.

ii.
```python
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)  # = 360

def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
```

iii. The AI justified this by citing the paper's methods: "splits were done on consecutive 2 minute blocks of the recording" for cross-validation in the paper's ridge regression decoder. The 2-minute block structure is directly from the paper's decoding methodology.

## 1-e. How are trials filtered based on quality controls?

i. No trials are explicitly filtered or removed. All 2-minute blocks from all sessions are included. The code raises an error if a session would have fewer than 2 trials, but this never happens. Sessions with missing camera frames are handled via interpolation rather than exclusion.

ii.
```python
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The AI noted that the paper describes no trial-level quality filtering since the recordings are continuous spontaneous sessions rather than event-driven trials. The data README mentions possible missing video frames but not trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from three Suite2p output files per session: `F.npy` (raw fluorescence traces), `Fneu.npy` (neuropil traces), and `ops.npy` (Suite2p operations dictionary containing processing parameters like `neucoeff`, `baseline`, `win_baseline`, `fs`, etc.).

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)  # loads ops.npy
```

iii. The AI documented that the paper states: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses." The AI chose to reconstruct this from F and Fneu rather than using the pre-computed `spks.npy` (deconvolved spikes).

## 2-b. How is the `neural` data processed?

i. The AI applies Suite2p-style baseline correction: (1) Neuropil subtraction: `Fc = F - neucoeff * Fneu` using per-session `neucoeff` from `ops.npy` (default 0.7). (2) Baseline correction via `suite2p.extraction.dcnv.preprocess()` with session-specific parameters. (3) Temporal binning by averaging in non-overlapping bins of 10 frames (30 Hz -> 3 Hz).

ii.
```python
def compute_suite2p_baseline_corrected(F, Fneu, ops):
    fc = F.astype(np.float32) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(np.float32)
    return dcnv.preprocess(
        fc.copy(),
        baseline=ops.get("baseline", "maximin"),
        win_baseline=float(ops.get("win_baseline", 60.0)),
        sig_baseline=float(ops.get("sig_baseline", 10.0)),
        fs=float(ops.get("fs", IMAGING_FS)),
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 100)),
        device=torch.device("cpu"),
    ).astype(np.float32)

def bin_average_2d(x, bin_size):
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2).astype(np.float32)

neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)  # 10-frame bins
```

iii. The AI explicitly noted that the Track2p GUI's `F_processing` uses `neucoeff=0.0` by default, which differs from the paper's stated use of "default Suite2p parameters" (which uses `neucoeff=0.7`). The AI chose to use Suite2p's actual `dcnv.preprocess` function with session-specific ops parameters for paper consistency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied beyond what's already in the data release. The data README states the Suite2p folders contain only neurons "present across all days" (pre-filtered by Track2p with `iscell > 0.5`). The AI trusts this pre-filtering.

ii.
```python
# No explicit neuron filtering code - all neurons from F.npy are used
# The code simply loads and processes all rows of F.npy
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
```

iii. The AI verified that `iscell.npy` values in the data release are all above 0.5, confirming pre-filtering. The AI documented: "No additional ROI filtering beyond trusting provided matched Suite2p export." This is consistent with the data README which says the data is already filtered.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The data is continuous, and trials are simply consecutive 2-minute blocks from the start of the session. The temporal alignment event is described as "start of each consecutive 2-minute block." The neural and behavioral data are aligned by imaging frame indices (the imaging clock serves as master reference).

ii.
```python
# Neural data is simply split into consecutive blocks
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)  # BLOCK_BINS = 360

converted["metadata"]["temporal_alignment_event"] = "start of each consecutive 2-minute block; input stores absolute elapsed time from session start"
converted["metadata"]["off_start"] = 0.0
converted["metadata"]["off_end"] = BLOCK_DURATION_SEC  # 120.0
```

iii. The AI justified this by noting the paper's decoding uses 2-minute blocks of continuous recording rather than event-aligned trials. There is no stimulus onset or behavioral event to align to since the experiment involves spontaneous behavior.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw data is at 30 Hz (imaging frame rate). The AI applies 10-frame averaging, yielding a 3 Hz temporal resolution (bin size = 10/30 = 0.333 seconds = 333.33 ms). The metadata stores this as `time_bin_size = 1000.0 * 10 / 30.0 = 333.33 ms`.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0

converted["metadata"]["time_bin_size"] = 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS  # 333.33 ms

neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
```

iii. The AI cited the paper's methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" and "Imaging rate was 30 Hz". This exactly matches the paper's preprocessing.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is derived from the imaging frame indices and the sampling rate (`fs`) from `ops.npy`. It is not read from any raw data file directly but computed from the frame count and known sampling rate.

ii.
```python
def make_time_input(nbins, fs, bin_size_frames):
    step = bin_size_frames / fs  # 10/30 = 0.333 seconds
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]

time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. The AI documented that the input is "elapsed time from session start in seconds" computed from frame indices at the 10-frame-binned resolution. This is the decoder input as specified in the task instructions ("Time elapsed from the beginning of the experiment").

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The elapsed time is computed as a simple arithmetic sequence: `t[i] = i * (bin_size_frames / fs)`, where `bin_size_frames = 10` and `fs = 30 Hz`. This gives time values in seconds from the start of each session, at the binned temporal resolution.

ii.
```python
def make_time_input(nbins, fs, bin_size_frames):
    step = bin_size_frames / fs  # = 0.3333 seconds
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

iii. The AI states this represents "absolute elapsed time from session start." The time input is continuous across the entire session before being split into trial blocks, so each trial's time values reflect absolute session time, not time relative to the trial start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is constructed from the same number of time bins as the neural data (`neural_binned.shape[1]`), ensuring perfect alignment. Both use the same 10-frame binning scheme. The time input is then split into the same 2-minute blocks as the neural and output data.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The AI ensured alignment by deriving the time vector from the same temporal grid as the neural data, guaranteeing a 1:1 correspondence between time bins and neural bins.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy` (global motion energy scalar per frame) and `move_deve/tstamps.npy` (timestamps for behavior camera frames, used to align motion to imaging frames when camera frames are missing).

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented that `motion_energy_glob.npy` contains pre-computed global motion energy from videography. The paper describes motion energy as "pixelwise difference of consecutive frames... squared... summed across pixels." The data release already provides this computed metric.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Align motion to imaging frames using timestamps (interpolate missing camera frames). (2) Average in 10-frame bins. (3) Global min-max normalization across all sessions. (4) Discretize into 5 equal-percentile (quintile) bins using global quantile edges.

ii.
```python
# Step 1: Align to imaging frames
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

# Step 2: 10-frame binning
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)

# Step 3: Global normalization (after collecting all sessions in Pass 1)
all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

# Step 4: Discretize per session
motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12))
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The AI justified global normalization + quintile discretization as matching the task requirement of "five equal-percentile bins" while preserving the paper's temporal processing (10-frame averaging).

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, the motion energy is discretized into 5 bins using quantile edges at the 20th, 40th, 60th, and 80th percentiles computed across all sessions' binned motion data. `np.digitize` maps values to bins 0-4.

ii.
```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

def motion_to_bins(x, quantile_edges):
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)

# Output values labeled:
converted["output_values"] = [["0-20pct", "20-40pct", "40-60pct", "60-80pct", "80-100pct"]]
```

iii. The AI documented that the task instructions require "normalized and discretized into five equal-percentile bins." The quintile approach yields exactly 20% of data in each bin globally, which was verified in the output statistics.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to the imaging frame grid using timestamps. Where camera frames are missing, linear interpolation fills gaps. After alignment, the same 10-frame binning is applied to both neural and motion data. A consistency check verifies equal lengths before splitting into trial blocks.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]

if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(f"{session.session_id}: neural/motion binned length mismatch")
```

iii. The AI cited the paper/README: "microscope acquisition acting as a trigger... simple synchronisation" and "In some recordings there might be missing video frames." The imaging frame clock is treated as the master reference.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (9 sessions affected, from 1 to 148 missing frames) are handled by mapping behavior timestamps to imaging frame indices and linearly interpolating over gaps. No sessions or trials are discarded due to missing data.

ii.
```python
def align_motion_to_imaging(motion, tstamps, nframes):
    if len(motion) == nframes:
        return motion.astype(np.float32), stats  # No alignment needed

    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)

    # Average multiple behavior samples per frame
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
    aligned[good] = summed[good] / counts[good]

    # Interpolate missing frames
    missing = np.flatnonzero(~good)
    if missing.size:
        aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
    return aligned.astype(np.float32), stats
```

iii. The AI noted the data README states that missing video frames should be handled using timestamps and inter-frame intervals. The interpolation approach avoids discarding data while maintaining temporal alignment.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p baseline correction (`dcnv.preprocess`) in Pass 2, which processes each session's full fluorescence matrix on CPU. The conversion output shows individual session processing times ranging from 0.4s (small sessions) to 2.4s (large sessions). Total conversion time was ~61 seconds for all 41 sessions.

ii.
```python
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)  # Most expensive step
```

iii. The AI documented timing estimates: "20-minute session conversion ~1.5s", "30-minute session conversion ~3.6s", total ~2 minutes. The Suite2p baseline correction dominates because it processes the full n_neurons x n_frames matrix.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop over sessions cannot easily be vectorized since sessions have different numbers of neurons. However, the `align_motion_to_imaging` function uses `np.add.at` (which is not fully vectorized) for accumulating values at frame indices. The trial splitting uses list comprehensions with slicing rather than a single reshape operation.

ii.
```python
# np.add.at is used instead of fully vectorized operations
np.add.at(summed, frame_idx, motion.astype(np.float64))
np.add.at(counts, frame_idx, 1)

# List comprehension for block splitting
return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

iii. The AI noted that the code uses vectorized numpy operations for binning (`reshape` + `mean`) and that the main bottleneck is Suite2p's baseline correction which is already implemented efficiently within Suite2p.

## 6-c. What processing does the code repeat multiple times?

i. Motion data loading and alignment is performed twice: once in Pass 1 (to compute global quantile thresholds) and again in Pass 2 (during per-session processing). The `ops.npy` file is also loaded multiple times: during session discovery (for frame count), in Pass 1, and in Pass 2.

ii.
```python
# Pass 1: loads motion and aligns
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)

# Pass 2: loads motion AGAIN and aligns AGAIN
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
# But then uses cached motion_binned from Pass 1:
motion_binned = motion_binned_by_session[session.session_id]
```

iii. The AI mitigated some repetition by caching the binned motion from Pass 1 (`motion_binned_by_session`), but still reloads and re-aligns the raw motion data in Pass 2 (used for plotting). The `ops.npy` is loaded in `discover_sessions` (sample mode) and again in the main loop.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `motion_aligned` again in Pass 2 (line 353) even though the aligned motion is only needed for plotting. The `align_stats` dictionary computed in Pass 1 for each session is only partially used (stored in metadata). The code also stores absolute elapsed time as input, but within each 2-minute trial block, the time values are non-zero offsets from session start rather than trial start, which means the decoder receives time-of-day-like information rather than relative trial timing.

ii.
```python
# motion_aligned recomputed in Pass 2 but only used in plot_processing
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])

# time input carries absolute session elapsed time into each trial
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# After splitting into blocks, trial 2's time starts at 120s, not 0s
```

iii. The AI noted the two-pass design was chosen for memory efficiency. The repeated motion alignment in Pass 2 was needed for the plotting feature but is indeed redundant for the core conversion when plotting is disabled.
