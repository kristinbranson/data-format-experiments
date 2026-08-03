# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over subject directories in the data folder, then iterates over session subdirectories within each subject. For each session it loads `F.npy`, `Fneu.npy`, `ops.npy`, and `iscell.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. All available subjects and sessions are included (no filtering by subject name prefix).

ii.
```python
def session_paths(data_dir: Path, selected_subjects: list[str] | None) -> list[tuple[str, Path]]:
    available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    ...

def load_session(subject: str, session_dir: Path) -> dict:
    plane_dir = session_dir / "suite2p" / "plane0"
    move_dir = session_dir / "move_deve"

    fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
    motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. From trajectory step 34: "the shipped session matrices are already Track2p-matched cell sets; `ops.npy` carries Suite2p defaults (`neucoeff=0.7`, `baseline=maximin`, 60 s baseline window)." The AI loads `ops.npy` to read parameters dynamically rather than hardcoding them, and loads `iscell.npy` as a validation check.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all directories in the data folder, sorted alphabetically. No prefix filtering (e.g., `jm*`) is applied; all directories are treated as subjects.

ii.
```python
available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
```

iii. The AI determines subjects from all subdirectories in the data directory. Since the data directory only contains subject folders (starting with `jm`), no prefix filtering is needed in practice.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording session.

ii.
```python
for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
    out.append((subject, session_dir))
```

iii. The agent identified from the data structure that each subdirectory represents one recording day.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive 2-minute (120-second) non-overlapping blocks of the (binned) recording. After 10-frame temporal binning, each trial has `TRIAL_BINS = int(120.0 / (10/30))` = 360 time bins. Sessions that do not evenly divide into 360-bin trials raise an error.

ii.
```python
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)  # 360

n_trials = neural_binned.shape[1] // TRIAL_BINS
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
```

iii. From trajectory step 34: "cut the recording into the same consecutive 2-minute blocks described in the paper." The agent verified that binned session lengths evenly divide by 360 bins, raising an error otherwise rather than silently discarding remainder frames.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All trials from all sessions are included. The AI does validate that `iscell` probabilities are >0.5 for all cells (a neuron-level check, not trial-level).

ii.
```python
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The AI verified that Track2p-exported cells all pass the iscell threshold, treating this as a sanity check rather than a filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p processing parameters) from `suite2p/plane0/`.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. The agent recognized these as standard Suite2p outputs. Loading `ops.npy` allows reading the exact processing parameters used for each session.

## 2-b. How is the `neural` data processed?

i. Processing involves: (1) neuropil subtraction using the coefficient from `ops.npy` (`F - neucoeff * Fneu`), (2) baseline correction using the `maximin` method (Gaussian smooth, then min filter, then max filter with a window of `win_baseline * fs` frames), (3) temporal averaging into non-overlapping 10-frame bins.

ii.
```python
def suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops):
    fc = fluorescence.astype(np.float32) - float(ops["neucoeff"]) * neuropil.astype(np.float32)
    baseline = ops.get("baseline", "maximin")
    if baseline == "maximin":
        win = int(round(float(ops["win_baseline"]) * float(ops["fs"])))
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = minimum_filter1d(flow, win)
        flow = maximum_filter1d(flow, win)
    ...
    return (fc - flow).astype(np.float32)

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
```

iii. From trajectory step 12: "use Suite2p `iscell > 0.5`, baseline-corrected `dF/F`, and both neural plus behavior averaged in bins of 10 imaging frames before decoding." The AI reimplemented Suite2p's baseline correction using scipy rather than calling `dcnv.preprocess`, using parameters read from `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied. The AI validates that all cells in the Track2p export have `iscell > 0.5` but does not remove any neurons.

ii.
```python
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The agent noted (trajectory step 20): "I'm checking the saved `ops.npy` defaults and building summary stats before I lock the conversion." Since Track2p already exports only tracked cells, no further filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block within the session. There is no stimulus event; the recording is continuous and trials are artificial segments. `off_start = 0.0`, `off_end = 120.0`.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block within a session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SECONDS,  # 120.0
```

iii. The agent stated in trajectory step 34: "cut the recording into the same consecutive 2-minute blocks described in the paper."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz data is rebinned by averaging 10 consecutive frames, yielding a bin size of 10/30 = 0.333 seconds (333.33 ms).

ii.
```python
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ  # 10/30 = 0.3333
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0  # 333.33

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
```

iii. From trajectory step 12: "both neural plus behavior averaged in bins of 10 imaging frames before decoding." This follows the paper's described analysis pipeline.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed synthetically from the bin index and the bin size, representing elapsed time from session start at the center of each time bin.

ii.
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. Since the frame rate is constant and known, time can be computed from indices. The `+0.5` offset places the time at the center of each bin.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `(bin_index + 0.5) * BIN_SIZE_SECONDS`, giving the center of each temporal bin in seconds from session start.

ii.
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. Using bin centers (rather than bin edges) is a standard convention for representing time in binned data.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is directly computed from the same bin indices as the neural data, so alignment is guaranteed. Both neural and input share the same temporal grid after 10-frame binning.

ii.
```python
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. Since time is computed from indices, it is inherently aligned with the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect and interpolate dropped video frames.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The AI identified these files from the data structure and paper description.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves: (1) detecting missing camera frames via interframe interval ratios and interpolating linearly, (2) temporal averaging into 10-frame bins (same as neural), (3) per-session z-scoring (mean subtraction + std normalization), (4) global quintile-based discretization into 5 categories across all sessions.

ii.
```python
# Missing frame interpolation
missing_after = infer_missing_frames(interframe_int)
aligned[dst : dst + gap_missing] = np.linspace(
    motion_energy[src], motion_energy[next_src], int(gap_missing) + 2, dtype=np.float32
)[1:-1]

# Temporal binning
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)

# Z-scoring
motion_z = ((motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)).astype(np.float32)

# Global quintile discretization
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
```

iii. From trajectory step 34: "preprocess full-session traces first, repair camera dropouts on the frame grid, average both streams in 10-frame windows." The z-scoring normalizes across sessions before pooling for percentile computation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins using global quintile edges computed across all sessions. The edges are computed from z-scored, binned motion energy at quantiles [0.2, 0.4, 0.6, 0.8], and `np.digitize` maps values to classes 0-4.

ii.
```python
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
```

iii. The instruction specifies "five equal-percentile bins." Using quantiles at 0.2, 0.4, 0.6, 0.8 divides the distribution into quintiles. Global computation ensures balanced classes across the entire dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. After interpolating missing camera frames to match the imaging frame count, both neural and motion energy are binned into the same 10-frame windows, ensuring frame-for-frame alignment. Missing frames are detected via interframe interval ratios and linearly interpolated.

ii.
```python
def infer_missing_frames(interframe_int: np.ndarray) -> np.ndarray:
    median_interval = float(np.median(interframe_int))
    frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
    return np.clip(frame_jumps - 1, 0, None)

motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy, n_imaging_frames=n_frames, interframe_int=interframe_int,
)
```

iii. The agent recognized that camera frame drops cause the motion energy array to be shorter than the neural data. Missing frames are inferred by dividing each interframe interval by the median interval and rounding to find how many frames were skipped.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are detected via interframe interval analysis and filled by linear interpolation. Length mismatches raise errors. The AI validates that the number of inferred missing frames matches the expected count (`n_imaging_frames - motion_energy.shape[0]`). Sessions that don't evenly divide into trials also raise errors.

ii.
```python
if missing_total != expected_missing:
    raise ValueError(
        f"Could not reconcile missing motion frames: inferred {missing_total}, "
        f"expected {expected_missing}"
    )

if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)
```

iii. The AI chose to raise errors rather than silently handling mismatches, ensuring data integrity. From trajectory: the agent checked that all sessions pass these validations.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction, which applies Gaussian smoothing, minimum filtering, and maximum filtering across the full neuron-by-time matrix for each session using scipy. This is done on CPU (unlike the reference which uses GPU via dcnv.preprocess).

ii. N/A

iii. The scipy-based implementation processes all neurons across the full session length with sliding window operations. This is CPU-bound and slower than the GPU-accelerated suite2p approach.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The missing frame interpolation loop in `align_motion_to_imaging_frames` iterates frame-by-frame through the interframe intervals, inserting interpolated values one at a time. This could be vectorized by pre-allocating the output array and computing all interpolated positions at once.

ii.
```python
for gap_missing in missing_after:
    next_src = src + 1
    if gap_missing:
        aligned[dst : dst + gap_missing] = np.linspace(...)
        dst += int(gap_missing)
    aligned[dst] = motion_energy[next_src]
    dst += 1
    src = next_src
```

iii. However, the number of missing frames is very small (276 total across 41 sessions), so the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The code processes all sessions in a single pass through `load_session`, then pools motion energy for discretization, then iterates again for trial splitting. There is no significant repeated processing.

ii. N/A

iii. The two-pass structure (load+preprocess, then discretize+split) is necessary because global percentile computation requires all sessions to be loaded first.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `iscell.npy` and `ops.npy` for each session, using them only for validation and parameter reading respectively. The `iscell` data is used only as a sanity check and is not used to filter neurons. The code also tracks detailed metadata (missing frame indices, per-subject neuron counts, etc.) that is stored but not used by the decoder.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
# only used for validation:
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(...)
```

iii. Loading extra files for validation is a reasonable defensive practice, though it adds I/O overhead.
