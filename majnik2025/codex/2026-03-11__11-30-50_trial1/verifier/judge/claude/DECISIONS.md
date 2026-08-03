# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as sorted directories starting with `jm` under the data root, then discovers sessions as sorted subdirectories whose names start with a 4-digit year (e.g., `2023-10-18_a`). For each session, it loads suite2p outputs (`F.npy`, `Fneu.npy`, `ops.npy`) from `suite2p/plane0/`, and behavioral data (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`) from `move_deve/`. All sessions are processed sequentially in a single pass.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    ...

# In process_session:
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The AI followed the directory structure convention (subject folders contain session subfolders). It used `ops.npy` to get authoritative session metadata like `nframes` and preprocessing parameters. Session filtering by `name[:4].isdigit()` avoids accidentally including non-session directories.

## 1-b. How are the data split into subjects?

i. Subjects correspond to sorted directories starting with `jm` under the data root. Each directory name (e.g., `jm031`) becomes a subject identifier.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

iii. The `jm*` prefix naming convention is consistent across the dataset. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject folder, filtered to those whose name starts with a 4-digit year string (e.g., `2023-10-18_a`), sorted alphabetically.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session. The `name[:4].isdigit()` filter ensures only actual session directories are included, avoiding any non-session subdirectories.

## 1-d. How are the data split into trials?

i. There is no natural trial structure in this dataset. Trials are defined as consecutive 2-minute (120-second) non-overlapping blocks of the continuous recording. After 10-frame temporal binning, each trial contains 360 time bins. Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120.0
BIN_FRAMES = 10
...
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)  # = 360
n_trials = n_total_bins // trial_bins

for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop]
```

iii. The paper states decoder evaluation used "consecutive 2 minute blocks" for cross-validation splits. The AI matched this exactly by using 120-second trial segments, consistent with the paper's methodology.

## 1-e. How are trials filtered based on quality controls?

i. The AI checks that each session produces at least 2 trials after segmentation, raising an error if not. No other trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
```

iii. The minimum of 2 trials is necessary for decoder evaluation (train/test split). Since there are no stimulus-driven trial events, no additional trial curation is needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (preprocessing parameters) from `plane0`.

ii.
```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. These are the standard suite2p output files. The AI loads `ops.npy` to use the same preprocessing parameters that suite2p used during cell extraction.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied using the neuropil coefficient from `ops.npy` (`Fc = F - neucoeff * Fneu`), followed by suite2p's `dcnv.preprocess` baseline correction using parameters from `ops.npy` (baseline method, window size, sigma, percentile, etc.). The result is then averaged in non-overlapping 10-frame bins.

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

# Then temporal binning:
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```

iii. Using parameters from `ops.npy` ensures consistency with the original suite2p processing. The paper describes using default suite2p parameters for baseline-corrected fluorescence. The 10-frame temporal binning matches the paper's description of averaging over "10 consecutive timestamps" for decoder denoising.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond what suite2p and Track2p already performed. The released data already contains only the matched-cell population tracked across all days.

ii. N/A (no filtering code)

iii. The released suite2p arrays represent Track2p-matched cells that were already filtered by `iscell > 0.5` and matched across all days. The CONVERSION_NOTES document this reasoning explicitly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 2-minute segments of the continuous recording, no event-based alignment is needed. The metadata documents the alignment event as "Start of each consecutive 2-minute block."

ii.
```python
'temporal_alignment_event': "Start of each consecutive 2-minute block cut from a continuous recording session",
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous behavior, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz data is rebinned by averaging non-overlapping windows of 10 frames, yielding a time bin size of approximately 333.33 ms.

ii.
```python
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ  # = 333.33 ms

def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32)

neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```

iii. The paper states that "Neural and behavior traces are both slightly denoised by averaging over 10-frame bins before decoding." The AI matches this exactly. The methods.txt also mentions "averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is derived from the frame index and the frame rate from `ops.npy`. The AI computes elapsed time in seconds from the start of each session, using frame centers (index + 0.5) divided by the sampling rate.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. The +0.5 offset places the time at the center of each frame rather than the edge, which is more precise for binned data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After computing per-frame times, the time values are averaged in the same 10-frame bins as the neural and behavioral data.

ii.
```python
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. Binning the time values ensures they represent the center of each temporal bin, consistent with the neural and output data binning.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time array is constructed from the same frame indices as the neural data and undergoes the same 10-frame binning, ensuring exact alignment. Both are sliced identically during trial segmentation.

ii.
```python
# In segment_trials:
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
```

iii. Because time and neural data share the same frame-based construction and identical binning/segmentation, they are inherently aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (pre-computed global motion energy from video) and `interframe_int.npy` (camera inter-frame intervals for detecting dropped frames) in the `move_deve` subdirectory.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The motion energy file contains a pre-computed global motion energy signal from behavioral video. The inter-frame intervals are needed to detect and reconstruct dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) dropped frames are detected via median inter-frame interval analysis and reconstructed using cumulative step timing, with missing values filled by linear interpolation; (2) the reconstructed signal is averaged in 10-frame bins; (3) global min-max normalization is applied across all sessions; (4) the normalized signal is discretized into 5 equal-percentile (quintile) bins.

ii.
```python
# Missing frame reconstruction:
median_interval = float(np.median(interframe_int))
steps = np.rint(interframe_int / median_interval).astype(np.int64)
observed_idx = np.empty(motion.shape[0], dtype=np.int64)
observed_idx[0] = 0
observed_idx[1:] = np.cumsum(steps)
full = np.full(target_len, np.nan, dtype=np.float32)
full[observed_idx] = motion
full = interpolate_nans(full)

# Binning:
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]

# Global normalization and discretization:
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
```

iii. The median-based timing reconstruction is robust to outlier inter-frame intervals. Global min-max normalization removes scale differences across sessions. Quintile discretization ensures balanced class counts as required by the instructions ("five equal-percentile bins").

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, the continuous motion energy is discretized into 5 classes using global quintile edges (20th, 40th, 60th, 80th percentiles). `np.digitize` assigns each value to a bin (0-4). The classes are labeled Q1-Q5.

ii.
```python
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. Global quintile computation ensures the 5 classes are approximately equally populated across the entire dataset. The `np.maximum.accumulate` ensures bin edges are monotonically increasing.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz (camera triggered by microscope). Dropped video frames are detected by comparing inter-frame intervals to the median interval, and missing frames are reconstructed via linear interpolation before the motion trace is binned in the same 10-frame windows as the neural data. An assertion verifies the reconstructed length matches the expected frame count from `ops.npy`.

ii.
```python
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
...
# Both binned with same function:
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The median-interval approach robustly identifies frames where the camera missed a capture, even with varying inter-frame intervals. After reconstruction, both streams are frame-aligned and undergo identical 10-frame binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are the primary data issue. The AI detects them by comparing inter-frame intervals to the median interval (steps > 1 indicate gaps), reconstructs the full-length trace by placing observed values at computed indices and linearly interpolating missing positions. If reconstruction fails (last observed index doesn't match expected), an error is raised. Remainder frames/bins that don't fill a complete trial are discarded.

ii.
```python
def reconstruct_motion_trace(motion, interframe_int, target_len):
    ...
    steps = np.rint(interframe_int / median_interval).astype(np.int64)
    steps[steps < 1] = 1
    observed_idx[1:] = np.cumsum(steps)
    if observed_idx[-1] != target_len - 1:
        raise ValueError(...)
    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    full = interpolate_nans(full)
    return full, missing_idx
```

iii. The timing-based reconstruction is more robust than a fixed threshold approach. The assertion ensures any frame count mismatch is caught rather than silently producing misaligned data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `dcnv.preprocess` baseline correction, which is run on CPU. The AI reports mean processing time of ~2.10 seconds per session in the full conversion output, with total conversion taking ~86 seconds for 41 sessions.

ii. N/A

iii. The baseline correction involves sliding window operations over the full session length for every neuron. Despite using CPU (not GPU), the processing time is reasonable.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over trials doing array slicing, but each iteration creates new arrays from slices. This could potentially be done with a single reshape operation, though the current implementation is already quite efficient.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop]
    ...
```

iii. The loop performs simple slicing, so the overhead is negligible. The missing frame reconstruction is already vectorized using cumsum and np.interp.

## 6-c. What processing does the code repeat multiple times?

i. The processing plot function recomputes motion normalization and discretization for each plotted session, duplicating logic from `build_dataset` and `segment_trials`.

ii.
```python
# In plot_processing:
motion_norm_binned = (motion_binned - motion_min) / (motion_max - motion_min)
motion_classes = np.digitize(motion_norm_binned, motion_edges[1:-1], right=False)
```

iii. This duplication is minor since plotting is only done for up to 2 sessions and only when `--show-processing` is used.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores preview data (neural preprocessing previews, motion previews with raw/reconstructed traces, timestamps, inter-frame intervals) for each session, which is only used for optional diagnostic plotting but takes up memory during processing.

ii.
```python
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    "corrected": corrected[sample_neuron, :preview_len].copy(),
    "processed": processed[sample_neuron, :preview_len].copy(),
}
```

iii. This preview data is useful for debugging but unnecessary for the conversion itself. It is discarded before the final pickle output.
