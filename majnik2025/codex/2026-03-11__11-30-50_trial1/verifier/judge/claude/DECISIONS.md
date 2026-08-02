# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` under `data/`, then discovers sessions as subdirectories whose names start with 4 digits. For each session, it loads Suite2p outputs (`F.npy`, `Fneu.npy`, `ops.npy`) from `suite2p/plane0/`, and behavioral data (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`) from `move_deve/`. All sessions are processed sequentially.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    ...
```
```python
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The AI documented in CONVERSION_NOTES.md that the directory structure follows a standard convention with subject folders containing session subfolders. It loads all available data files per session, including `ops.npy` to read saved Suite2p parameters and `tstamps.npy` for timing reconstruction.

## 1-b. How are the data split into subjects?

i. Subjects are identified as directories starting with `jm` in the data root, sorted alphabetically. A unique subject list is built and each session is mapped to its subject index.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...
subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI noted that each `jm*` directory represents one mouse. The naming convention is consistent across the dataset (6 mice: jm031, jm032, jm038, jm039, jm040, jm046).

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder whose names start with 4 digits (date format), sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

iii. The AI noted that sorting ensures a deterministic order and that each subdirectory contains the Suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. The AI splits each continuous recording session into consecutive 2-minute (120-second) non-overlapping blocks. After 10-frame temporal binning, this yields 360 bins per trial. Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120.0
BIN_FRAMES = 10
...
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)  # 360
n_trials = n_total_bins // trial_bins
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
```

iii. The AI justified 2-minute trials based on the paper's statement that "splits were done on consecutive 2 minute blocks" for decoder cross-validation. This gives 10 trials for 20-minute sessions and 15 trials for 30-minute sessions.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks from each session are included.

ii. N/A (no filtering code)

iii. The AI noted that because the recordings are continuous spontaneous behavior sessions rather than discrete trials, no trial-based curation is described in the reference paper. Sessions with fewer than 2 trials raise an error but this never occurs in practice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (processing parameters), all from `suite2p/plane0/`.

ii.
```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. The AI documented that these are standard Suite2p output files. The use of `ops.npy` to read saved parameters is justified as ensuring consistency with the original processing pipeline.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied using `neucoeff` from `ops.npy` (which is 0.7), followed by Suite2p's `dcnv.preprocess` for baseline correction. All preprocessing parameters are read from the session's saved `ops.npy` rather than hardcoded. The result is then temporally binned by averaging non-overlapping windows of 10 frames.

ii.
```python
corrected = np.array(F, dtype=np.float32, copy=True)
corrected -= np.float32(ops["neucoeff"]) * np.asarray(Fneu, dtype=np.float32)
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 2000)),
    device=torch.device("cpu"),
).astype(np.float32, copy=False)
...
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```
```python
def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32)
```

iii. The AI justified reading parameters from `ops.npy` as ensuring paper-consistent preprocessing. The 10-frame averaging was justified by the paper's statement that neural traces were "averaged in bins of 10 consecutive timestamps" before decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons present in the released Suite2p arrays are included. The AI reasoned that the released data already contain Track2p-matched cells that passed the Suite2p `iscell > 0.5` threshold.

ii. N/A (no filtering code)

iii. The AI documented in CONVERSION_NOTES.md that the bundled neural data represent the already-filtered, Track2p-matched neuron population described in the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block cut from the continuous recording session. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed.

ii.
```python
"temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SEC,  # 120.0
```

iii. The AI noted there is no stimulus event to align to. The recording is continuous, and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging non-overlapping windows of 10 raw frames. At 30 Hz, this yields a time bin size of 333.33 ms (10/30 * 1000). Each trial has 360 time bins (120 seconds / 0.3333 seconds).

ii.
```python
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ  # 333.33 ms
...
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```

iii. The AI justified the 10-frame rebinning based on the paper's decoder methodology: "averaging in bins of 10 consecutive timestamps (~333 ms)".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from frame indices and the frame rate from `ops.npy`. Specifically, it is `(frame_index + 0.5) / fs`, giving bin-center times in seconds from session start.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The AI noted that since the frame rate is constant at 30 Hz, computing time from frame indices is equivalent to using timestamps. The +0.5 offset places the time at frame centers.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The raw frame-center times are computed, then averaged in the same 10-frame bins as the neural data. The time values are absolute (from session start), so each trial retains its position within the original recording.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
...
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
```

iii. The AI justified keeping absolute session time (not resetting per trial) to give the decoder "time elapsed from the beginning of the experiment" as requested by the task instructions.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is constructed from the same frame indices as the neural data, then binned in the same 10-frame windows, and segmented at the same trial boundaries. Alignment is exact by construction.

ii.
```python
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
# Both sliced identically:
neural_trial = neural_binned[:, start:stop]
input_trial = time_binned[np.newaxis, start:stop]
```

iii. Since both streams share the same indexing, they are inherently aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect and reconstruct dropped frames. `tstamps.npy` is also loaded but used only for plotting/preview.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. The AI documented that the motion energy file contains pre-computed global motion energy from the behavioral video, and that the interframe interval file is needed to identify dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) Dropped frames are detected via interframe intervals and reconstructed by computing step sizes from the median interval, placing observed values at cumulative-step positions, and linearly interpolating gaps. (2) Motion energy is averaged in 10-frame bins matching the neural data. (3) The binned motion energy is globally min-max normalized, then discretized into 5 equal-percentile (quintile) bins computed across all sessions.

ii.
```python
def reconstruct_motion_trace(motion, interframe_int, target_len):
    median_interval = float(np.median(interframe_int))
    steps = np.rint(interframe_int / median_interval).astype(np.int64)
    steps[steps < 1] = 1
    observed_idx = np.empty(motion.shape[0], dtype=np.int64)
    observed_idx[0] = 0
    observed_idx[1:] = np.cumsum(steps)
    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    full = interpolate_nans(full)
    return full, missing_idx
...
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
...
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
```

iii. The AI justified the timing-based reconstruction as more robust than a simple threshold. Min-max normalization was chosen to satisfy the task wording. Global percentile-based discretization ensures balanced class counts.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, the normalized motion energy values are discretized into 5 classes using quintile edges (0th, 20th, 40th, 60th, 80th, 100th percentiles) computed over all binned, normalized values across all sessions. `np.digitize` maps values to categories 0-4.

ii.
```python
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
```

iii. The AI chose equal-percentile bins to produce balanced class counts (exactly 20% each class globally). `np.maximum.accumulate` prevents non-monotonic edges from duplicate quantile values.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz (camera triggered by microscope). Dropped video frames are detected from interframe intervals and reconstructed to match the neural frame count. After reconstruction, both streams are binned in the same 10-frame windows and segmented at the same trial boundaries.

ii.
```python
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
...
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
# Same trial segmentation for both
```

iii. The AI noted that imaging frames serve as the master clock, and missing behavior frames are reconstructed so both modalities are frame-aligned before binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected from interframe intervals using a median-interval-based approach. Missing frames are reinserted at their estimated positions and linearly interpolated. If reconstruction fails to match the target length, an error is raised. Remainder bins that don't fill a complete trial are implicitly discarded by integer division.

ii.
```python
def reconstruct_motion_trace(motion, interframe_int, target_len):
    ...
    if observed_idx[-1] != target_len - 1:
        raise ValueError(...)
    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    full = interpolate_nans(full)
    return full, missing_idx
```

iii. The AI documented that 9 sessions had missing behavior frames (ranging from 1 to 148 missing frames). The timing-based reconstruction successfully handled all cases.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is Suite2p's `dcnv.preprocess` baseline correction. The full conversion takes ~86 seconds for 41 sessions (~2.1 seconds per session on average).

ii. N/A (timing is printed during execution)

iii. The AI documented timing information showing that sessions with more neurons (e.g., jm039 with 746 neurons) take proportionally longer (~3.5s) compared to smaller sessions (e.g., jm031 with 221 neurons at ~0.5s).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop is over sessions, which is inherently sequential due to memory constraints. Within sessions, the AI vectorized most operations (binning via reshape-and-mean, frame reconstruction via cumulative sums). The trial segmentation loop could potentially be vectorized but has minimal overhead.

ii. N/A

iii. The AI used vectorized approaches throughout (reshape-based binning, cumulative-sum frame reconstruction), leaving little room for further vectorization.

## 6-c. What processing does the code repeat multiple times?

i. Motion energy normalization is computed twice: once globally in `build_dataset` (for computing edges) and once per session when segmenting trials. This is by design since global statistics must be computed before per-session discretization.

ii.
```python
# Global computation:
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, ...)
# Per-session:
motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
```

iii. The two-pass approach (first compute global statistics, then apply per session) is a necessary design pattern.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores preview data (neural previews, motion previews) for every session, even when `--show-processing` is not used. These previews consume memory during processing but are not saved to the output pickle.

ii.
```python
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    ...
}
```

iii. The preview data is only used for diagnostic plots. Computing it unconditionally is a minor inefficiency.
