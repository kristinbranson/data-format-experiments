# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all subdirectories in the data directory (not filtering by prefix), then iterates over session subdirectories within each subject. For each session it loads six files: `F.npy`, `Fneu.npy`, `ops.npy`, and `iscell.npy` from `suite2p/plane0/`, plus `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. All sessions are loaded in a list comprehension before any splitting or discretization.

ii.
```python
def session_paths(data_dir: Path, selected_subjects: list[str] | None) -> list[tuple[str, Path]]:
    available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    ...
    for subject in subjects:
        subject_dir = data_dir / subject
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            out.append((subject, session_dir))
    return out

def load_session(subject: str, session_dir: Path) -> dict:
    fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
    motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The AI loads ops.npy to read Suite2p processing parameters (neucoeff, baseline method, window sizes) rather than hardcoding them. It also loads iscell.npy and verifies all exported cells have classification probability > 0.5, as described in the paper. From the trajectory (step 28): "the shipped session matrices are already Track2p-matched cell sets; ops.npy carries Suite2p defaults."

## 1-b. How are the data split into subjects?

i. Subjects are identified as all subdirectories within the data directory, sorted alphabetically. Unlike the reference, the AI does not filter by a `jm` prefix -- it includes all directories.

ii.
```python
available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
```

iii. The AI relies on the assumption that only subject directories exist as subdirectories of the data folder. From the trajectory (step 12): subjects are identified from the directory structure.

## 1-c. How are the data split into sessions?

i. Sessions are identified as sorted subdirectories within each subject's folder. Each subdirectory corresponds to one daily recording session.

ii.
```python
for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
    out.append((subject, session_dir))
```

iii. Each subdirectory contains suite2p output and motion energy files for one recording session. Sorting ensures deterministic ordering by date.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive 2-minute (120-second) non-overlapping blocks of the binned recording. After 10-frame temporal binning, each trial contains 360 time bins. Sessions must be exactly divisible into these trial blocks (the code raises an error if not).

ii.
```python
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)  # 360

n_trials = neural_binned.shape[1] // TRIAL_BINS
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
```

iii. From the trajectory (step 34): "cut the recording into the same consecutive 2-minute blocks described in the paper." The paper states "splits were done on consecutive 2 minute blocks of the recording."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. However, the code requires that the total number of binned frames is exactly divisible by the trial bin count (360), raising a ValueError if not.

ii.
```python
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)
```

iii. No trial quality filtering is needed as the recordings are continuous with no stimulus-driven trial structure.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p processing parameters), all from the `suite2p/plane0/` subdirectory.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. The AI reads processing parameters from ops.npy (neucoeff, baseline method, window sizes, frame rate) rather than hardcoding them, making the pipeline more robust to variations in Suite2p settings.

## 2-b. How is the `neural` data processed?

i. The AI reimplements Suite2p's baseline correction using scipy: (1) neuropil subtraction with coefficient from ops.npy (`F - neucoeff * Fneu`), (2) maximin baseline estimation using Gaussian smoothing followed by minimum then maximum filters, (3) baseline subtraction. After baseline correction, neural traces are temporally binned by averaging 10 consecutive frames.

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

iii. From trajectory (step 28): "ops.npy carries Suite2p defaults (neucoeff=0.7, baseline=maximin, 60 s baseline window)." The AI chose to reimplement the baseline correction using scipy rather than depending on suite2p's dcnv module.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI loads `iscell.npy` and verifies that all cells in the Track2p-exported data have classification probability > 0.5, but does not actually filter any neurons out. This serves as a sanity check that the Track2p export is consistent.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The paper states "We considered all ROIs above the default threshold of 0.5 as true cells." Since Track2p's suite2p-format export already contains only tracked cells, the AI verifies this constraint rather than filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block within a session. There is no stimulus event; trials are contiguous segments of the continuous recording.

ii.
```python
'temporal_alignment_event': "start of each consecutive 2-minute block within a session",
'off_start': 0.0,
'off_end': TRIAL_DURATION_SECONDS,
```

iii. The recording is continuous spontaneous activity with no stimulus events. Trials are artificial segments, so alignment is simply the start of each block.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames, resulting in a bin size of 10/30 * 1000 = 333.33 ms. This matches the paper's description of denoising by "averaging in bins of 10 consecutive timestamps."

ii.
```python
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ  # 0.333...
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0  # 333.33...

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
```

iii. From the trajectory (step 34): "average both streams in 10-frame windows." The paper states: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed synthetically from the bin indices and the bin size.

ii.
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. With a constant bin size, computing time from bin indices is equivalent to having timestamps. The +0.5 offset places time at the center of each bin.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the bin center time in seconds from session start: `(bin_index + 0.5) * bin_size_seconds`. The +0.5 offset represents the center of each temporal bin rather than the start.

ii.
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. Using bin centers is standard practice when the data represents averages over bins. The time values increase continuously across trials within a session.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices as the neural data, so alignment is exact by construction. Both share the same temporal grid after 10-frame binning.

ii.
```python
# Both computed from same indices
neural_trials.append(neural_binned[:, start:end])
input_trials.append(elapsed_time_seconds[start:end][None, :])
```

iii. Since time is derived from the bin index of the neural data, there is no alignment issue.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (pre-computed global motion energy from behavioral video) and `interframe_int.npy` (inter-frame intervals for detecting dropped camera frames), both from the `move_deve/` subdirectory.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The motion energy file contains the pixel-wise difference metric described in the paper. The interframe interval file is needed to identify and interpolate dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves four steps: (1) detect missing camera frames using median inter-frame interval ratios, (2) linearly interpolate missing frames to align with the imaging frame grid, (3) average in 10-frame bins (matching neural data binning), (4) z-score normalize per session (subtract mean, divide by std), (5) discretize into 5 quintile bins using global quantile edges pooled across all sessions.

ii.
```python
# Missing frame detection
missing_after = infer_missing_frames(interframe_int)
# median_interval = float(np.median(interframe_int))
# frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
# return np.clip(frame_jumps - 1, 0, None)

# Binning
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)

# Z-score
motion_z = ((motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)).astype(np.float32)

# Global quintile discretization
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False)
```

iii. The paper says "we slightly denoised ... the behaviour traces by averaging in bins of 10 consecutive timestamps." Z-scoring removes session-level scale differences before pooling for global quintile computation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories (quintiles) using global quantile edges. The edges are computed from the pooled z-scored motion energy across all sessions at quantiles [0.2, 0.4, 0.6, 0.8]. Values are assigned to bins 0-4 using `np.digitize`.

ii.
```python
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)

pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
```

iii. Equal-percentile binning ensures balanced class counts across the dataset, as specified in the instructions ("five equal-percentile bins").

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy is aligned to the imaging frame grid by detecting and interpolating missing camera frames. The camera is triggered by the microscope acquisition, so frames should be 1:1. Missing frames are detected by comparing inter-frame intervals to the median interval, then linearly interpolated. After alignment, both streams are binned in 10-frame windows, ensuring synchronized temporal grids.

ii.
```python
def infer_missing_frames(interframe_int: np.ndarray) -> np.ndarray:
    median_interval = float(np.median(interframe_int))
    frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
    return np.clip(frame_jumps - 1, 0, None)

def align_motion_to_imaging_frames(motion_energy, n_imaging_frames, interframe_int):
    ...
    aligned[dst : dst + gap_missing] = np.linspace(
        motion_energy[src], motion_energy[next_src],
        int(gap_missing) + 2, dtype=np.float32,
    )[1:-1]
    ...
```

iii. The method is unit-agnostic (uses median as reference) and performs true linear interpolation between observed values at detected gap locations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are detected via inter-frame interval analysis and linearly interpolated. Shape mismatches between fluorescence and neuropil arrays raise errors. If the total number of binned frames is not divisible by the trial size, a ValueError is raised (the sessions are exactly 20 minutes = 36000 frames, so this should never trigger). A small epsilon (1e-8) is added to std for numerical stability during z-scoring.

ii.
```python
if fluorescence.shape != neuropil.shape:
    raise ValueError(f"{session_dir}: F and Fneu shapes do not match")

if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)

motion_z = ((motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8))
```

iii. The AI uses strict error checking (raising exceptions) rather than silently discarding data, which ensures problems are caught early.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction using scipy's `gaussian_filter`, `minimum_filter1d`, and `maximum_filter1d` on the full-session neural data matrix. These operate on (n_neurons x n_frames) arrays where n_frames can be ~36000.

ii.
```python
flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
flow = minimum_filter1d(flow, win)
flow = maximum_filter1d(flow, win)
```

iii. Unlike the reference which uses suite2p's GPU-accelerated dcnv.preprocess, the AI uses CPU-based scipy operations, which are slower for large matrices.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `align_motion_to_imaging_frames` function uses a sequential loop over each inter-frame gap to perform linear interpolation. This could potentially be vectorized by pre-allocating the output array and computing all interpolation targets in bulk. The trial splitting also uses a loop but is straightforward and fast.

ii.
```python
for gap_missing in missing_after:
    next_src = src + 1
    if gap_missing:
        aligned[dst : dst + gap_missing] = np.linspace(...)
        dst += int(gap_missing)
    aligned[dst] = motion_energy[next_src]
    ...
```

iii. The number of dropped frames is typically very small (a few per session at most), so the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat significant processing. All sessions are processed in a single pass, and the global quintile computation is done once after all sessions are loaded. The `build_dataset` function iterates over loaded sessions once for trial splitting.

ii. N/A

iii. The code architecture is clean with a single pass for loading and a single pass for assembly.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `iscell.npy` and verifies it but does not use it for filtering. It also computes and stores extensive metadata (tracked neuron counts, per-session info, missing frame details, quintile edges) that is not used by the decoder. The per-session z-scoring normalizes motion energy, but the downstream decoder might not need this normalization if it operates on discretized categories.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(...)

# Extensive metadata stored but unused by decoder:
"tracked_neurons_per_subject": tracked_neurons_per_subject,
"missing_motion_frames_by_session": missing_motion_by_session,
```

iii. The iscell verification is a useful sanity check even if it doesn't filter. The metadata, while unused by the decoder, documents the conversion for reproducibility.
