# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subject directories starting with `jm` in the data root, then finds session subdirectories whose names start with 4 digits (date-formatted). For each session, it loads suite2p output files (`F.npy`, `Fneu.npy`, `ops.npy`) and motion energy files (`motion_energy_glob.npy`, `interframe_int.npy`).

ii.
```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )

def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )

# Loading neural data:
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()

# Loading motion energy:
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The AI explored the data directory structure and suite2p outputs, confirming that each subject has date-named session subdirectories containing suite2p and motion energy files. It also loads `ops.npy` to read preprocessing parameters rather than hardcoding them, reasoning that this ensures the conversion matches whatever Suite2p actually used.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data root, sorted alphabetically. A mapping from subject name to index is constructed.

ii.
```python
subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}
```

iii. The `jm` prefix naming convention is consistent across the dataset. The AI confirmed 6 subjects: jm031, jm032, jm038, jm039, jm040, jm046.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a date-named subdirectory within a subject's folder. Sessions are filtered to directories whose first 4 characters are digits (e.g., `2023-10-18_a`) and sorted alphabetically.

ii.
```python
def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )
```

iii. The date-based filtering ensures only actual recording session directories are included, not any other subdirectories.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive 2-minute (120-second) non-overlapping blocks of the continuous recording. After 10-frame temporal binning, each trial has 360 time bins. Any trailing frames that don't fill a complete 2-minute block are discarded.

ii.
```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)  # 3600
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES  # 360

usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
n_trials = usable_frames // TRIAL_FRAMES
```

iii. The AI chose 2-minute blocks based on the paper's description that decoding used "consecutive 2-minute blocks" as the cross-validation split strategy. From the CONVERSION_NOTES.md: "Each session is cut into consecutive 2-minute blocks because the paper's decoder uses consecutive 2-minute splits."

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial quality filtering is applied. The only constraint is that sessions must have at least 2 trials (i.e., at least 4 minutes of recording).

ii.
```python
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. No trial quality filtering is described in the paper for this data type.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from three suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (preprocessing parameters including neucoeff, baseline method, etc.).

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The AI confirmed these are the standard suite2p output files and used ops.npy to read preprocessing parameters rather than hardcoding them.

## 2-b. How is the `neural` data processed?

i. Four processing steps: (1) neuropil subtraction using the coefficient from ops.npy, (2) suite2p's `dcnv.preprocess` for baseline estimation using parameters from ops.npy, (3) conversion to dF/F by dividing the baselined signal by the recovered baseline, (4) temporal averaging in 10-frame bins.

ii.
```python
Fcorr = F
Fcorr -= float(ops["neucoeff"]) * Fneu

dff_num = Fcorr.copy()
dff_num = dcnv.preprocess(
    F=dff_num,
    baseline=ops["baseline"],
    win_baseline=ops["win_baseline"],
    sig_baseline=ops["sig_baseline"],
    fs=ops["fs"],
    prctile_baseline=ops["prctile_baseline"],
    batch_size=ops.get("batch_size", 100),
    device=torch.device("cpu"),
)

baseline = Fcorr
baseline -= dff_num
np.maximum(baseline, 1e-3, out=baseline)
dff_num /= baseline

# Then 10-frame binning:
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. The AI justified this as matching the paper's description of "slightly denoised dF/F" and "averaging 10 consecutive timestamps." From the trajectory: the agent confirmed that dcnv.preprocess returns F - F0, so the baseline F0 is recovered as Fcorr - preprocess(Fcorr), and dF/F = (F - F0) / F0. The 1e-3 floor prevents division by zero.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron quality filtering is applied. All neurons present in the F.npy output are included.

ii. N/A (no filtering code)

iii. The bundled data contains only tracked neurons (already filtered by the Track2p pipeline). No additional filtering was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block from the continuous recording. The alignment is implicit since trials are contiguous segments.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block from a continuous recording",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,  # 120.0
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity, so trials are defined by their position within the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned from the native 30 Hz (33.3 ms per frame) to 10-frame bins (~333.3 ms per bin). Each trial has 360 time bins (120s / 0.333s).

ii.
```python
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES  # 360
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FRAME_RATE_HZ  # 333.33 ms

neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. The AI implemented 10-frame binning based on the paper's methods section which describes "averaging 10 consecutive timestamps" for the decoding analysis.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the frame/bin indices and the known frame rate.

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The frame rate is constant at 30 Hz, so time can be computed from bin indices. The time represents the center of each 10-frame bin, offset by the trial's position within the session.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each 10-frame bin in seconds from session start: `(bin_index * 10 + 5) / 30.0 + trial_index * 120.0`. This gives absolute time from the start of the session, not trial-relative time.

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :]
    for trial_index in range(n_trials)
]
```

iii. The AI noted in CONVERSION_NOTES.md: "The decoder input is not trial-relative time; it is absolute time-from-session-start in seconds."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: each time value corresponds to the center of the same 10-frame bin used for neural and motion energy data. The time array has the same number of bins (360) as the neural and output arrays.

ii. Same code as 3-a/3-b - the time array shape matches neural and output shapes by construction.

iii. Since all data streams share the same binning scheme, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect and repair dropped camera frames.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) dropped frames are detected via interframe interval analysis and repaired with linear interpolation, (2) motion energy is averaged in 10-frame bins (matching neural data), (3) global min-max normalization is applied, (4) the normalized signal is discretized into 5 quintile bins using percentile thresholds computed across all sessions.

ii.
```python
# Frame repair:
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
nominal = float(np.median(interframe))
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
# ... insert NaN at gap positions, then interpolate ...
repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask])

# 10-frame binning:
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)

# Global min-max normalization and quintile discretization:
all_motion = np.concatenate([trial for record in session_records for trial in record["motion_trials"]])
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80])
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale)
motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False)
```

iii. The AI justified this processing as matching the paper's methods. The 10-frame binning matches the neural processing. The global percentile discretization ensures balanced class counts across all sessions and subjects.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using global quintile boundaries (20th, 40th, 60th, 80th percentiles). The percentiles are computed on the globally min-max-normalized, 10-frame-binned motion energy across all sessions. `np.digitize` assigns each value to one of 5 bins (0-4).

ii.
```python
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)

motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

iii. The instruction specifies "five equal-percentile bins." Using global percentiles ensures each bin contains approximately 20% of all samples across the dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. Dropped camera frames are detected from interframe intervals and repaired via linear interpolation to match the neural frame count. Both streams are then identically binned into 10-frame averages and segmented into 2-minute trials.

ii.
```python
motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
neural_trials, time_trials, motion_trials = session_to_trials(neural_full, motion_full, n_frames)
```

iii. Frame repair ensures motion energy matches neural data frame-for-frame before binning and trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected using interframe intervals: intervals that are approximately integer multiples of the median interval indicate missed frames. Missing frames are inserted as NaN and linearly interpolated. The repair count is validated against the expected frame count difference. Trailing frames that don't fill a complete trial are discarded.

ii.
```python
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
nominal = float(np.median(interframe))
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
missing_count = int(jumps.sum())
expected_missing = n_frames - len(motion)
if missing_count != expected_missing:
    raise ValueError(...)
```

iii. The median-based approach robustly identifies dropped frames even if the actual frame rate drifts slightly from the nominal 30 Hz. A strict equality check ensures the repair accounts for all missing frames.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is `dcnv.preprocess` for baseline estimation, which involves sliding window operations over the full session for every neuron. The AI runs this on CPU (`torch.device("cpu")`), making it slower than a GPU implementation. Loading and repairing motion energy is comparatively fast.

ii.
```python
device=torch.device("cpu"),
```

iii. The CPU device choice may have been intentional for reproducibility or compatibility reasons.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame repair loop iterates over each gap position sequentially, inserting NaN values one at a time. This could be vectorized by pre-computing all insertion positions and constructing the repaired array in one pass (which the AI partially does with pre-allocation, but the loop is still sequential). The per-trial output discretization loop could also be vectorized.

ii.
```python
for gap in jumps:
    repaired[dst] = motion[src]
    src += 1
    dst += 1
    if gap:
        repaired[dst:dst + gap] = np.nan
        dst += gap
```

iii. The number of dropped frames is small (typically 0-148 per session), so the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. The code makes a single pass through all sessions to preprocess neural and motion data, then a second pass to discretize and assemble trials. This two-pass design is necessary because discretization requires global statistics.

ii. N/A

iii. The two-pass approach is the minimum needed for global percentile computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes detailed metadata (session frame counts, missing frame counts per session, min/max values, raw and normalized percentile thresholds) that are stored in the metadata dict but not used by the downstream decoder. The min-max normalization of motion energy is also redundant since the percentile-based discretization is invariant to monotonic linear transforms.

ii.
```python
"motion_energy_normalization": {
    "type": "global_min_max_after_10_frame_averaging",
    "min": motion_min,
    "max": motion_max,
},
"motion_energy_percentiles_raw": motion_quantiles_raw.tolist(),
"motion_energy_percentiles_normalized": motion_quantiles_norm.tolist(),
"session_frame_counts": [record["n_frames"] for record in session_records],
"missing_behavior_frames_by_session": missing_behavior_by_session,
```

iii. The extra metadata is useful for documentation and debugging but does not affect the decoder pipeline.
