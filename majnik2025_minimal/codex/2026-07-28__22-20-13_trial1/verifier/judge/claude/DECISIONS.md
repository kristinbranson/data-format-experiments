# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data root, and sessions as subdirectories whose names start with four digits. For each session, it loads suite2p output files (`F.npy`, `Fneu.npy`, `ops.npy`) from `suite2p/plane0/`, and motion energy from `move_deve/motion_energy_glob.npy`. Interframe intervals (`interframe_int.npy`) are loaded for dropped-frame repair.

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

# Loading data:
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The AI explored the data directory structure, identified the naming convention, and loaded the standard suite2p outputs. It also loads `ops.npy` to read preprocessing parameters rather than hardcoding them.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data root, sorted alphabetically.

ii.
```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )

subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
```

iii. Each `jm*` directory represents one mouse. The AI confirmed this by examining the data directory structure.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder whose name starts with four digits, sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )
```

iii. The AI filters session directories by requiring the name to start with 4 digits, to ensure only actual session directories are included.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive 120-second (2-minute) non-overlapping blocks of the continuous recording. After 10-frame binning (30 Hz to 3 Hz), each trial has 360 bins. Any remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)  # 3600
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES  # 360

usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
n_trials = usable_frames // TRIAL_FRAMES
```

iii. The AI noted in its trajectory that each session is "split into consecutive 2-minute blocks used in the paper's decoding." The agent chose 2-minute blocks based on the paper's methodology rather than the 60-second instruction.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. The AI raises an error if a session has fewer than 2 trials after blocking.

ii.
```python
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The AI included a minimum trial count check to ensure decoder evaluation is possible, but otherwise does not filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (processing parameters including neucoeff, baseline method, etc.), from `plane0`.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The AI explored the suite2p output directory and identified the standard files needed for calcium signal processing.

## 2-b. How is the `neural` data processed?

i. The AI applies a three-step processing pipeline: (1) neuropil subtraction using the coefficient from `ops.npy` (`Fcorr = F - neucoeff * Fneu`), (2) suite2p's `dcnv.preprocess` for baseline estimation using parameters from `ops.npy`, and (3) conversion to dF/F by dividing the preprocessed trace by the baseline (with a floor of 1e-3 to avoid division by zero). The baseline is computed as `Fcorr - preprocessed`, clamped to a minimum of 1e-3. The result is then averaged in 10-frame bins.

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
return dff_num.astype(np.float32, copy=False)
```

iii. The AI investigated the paper's methods and the suite2p codebase to determine the correct dF/F computation. The agent noted that the notebook "explicitly says to compute dF/F yourself for proper analysis" and tested multiple neural representations against decoder performance to select the best one.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the suite2p `F.npy` output are included.

ii. N/A

iii. The AI does not apply `iscell` filtering or any other neuron quality filter, relying on suite2p's cell detection pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block within the session. Since trials are contiguous segments, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'start of each consecutive 2-minute block from a continuous recording',
'off_start': 0.0,
'off_end': TRIAL_SECONDS,  # 120.0
```

iii. There is no stimulus event to align to; the recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both the neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing the 30 Hz sampling rate to 3 Hz (333.33 ms time bin). Binning is applied before motion energy discretization.

ii.
```python
BIN_FRAMES = 10
# ...
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
# ...
'time_bin_size': TIME_BIN_SIZE_MS,  # 1000.0 * 10 / 30.0 = 333.33 ms
```

iii. The AI followed the paper's methods which state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index, the bin duration, and trial index, giving seconds from the start of the session. The time value represents the bin center (adding BIN_FRAMES/2 to the frame index before dividing by frame rate).

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. Since the frame rate is constant at 30 Hz and there are no stored timestamps, computing time from bin indices is equivalent. The AI uses bin centers rather than bin left edges.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `(bin_index * BIN_FRAMES + BIN_FRAMES/2) / FRAME_RATE_HZ + trial_index * TRIAL_SECONDS`. This gives the center of each time bin in seconds from the start of the session, with continuity across trials.

ii. See 3-a code snippet.

iii. The AI chose to use bin centers rather than bin edges for the time representation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same bin indices as the neural data, so they are inherently aligned. Each time bin has a corresponding neural data bin at the same index.

ii. N/A (alignment is implicit through shared indexing)

iii. Since both time and neural data derive from the same binning structure, no explicit alignment step is needed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Interframe intervals from `interframe_int.npy` are used to detect and repair dropped video frames.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The AI identified the motion energy file as the pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped frames are detected via interframe intervals and repaired by linear interpolation, (2) the trace is averaged into 10-frame bins, (3) the binned signal is normalized using global min-max normalization across all sessions, then discretized into 5 bins using global 20th/40th/60th/80th percentile thresholds computed on the normalized values.

ii.
```python
# Dropped frame repair
nominal = float(np.median(interframe))
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
# ... inserts NaN then linearly interpolates

# Global normalization and discretization
all_motion = np.concatenate([trial for record in session_records for trial in record["motion_trials"]])
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_scale = motion_max - motion_min
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)

# Per-trial discretization
motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

iii. The AI chose global normalization and global percentile binning for consistency across sessions. The trajectory mentions the AI ran decoder benchmarks to validate the processing choices.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is first globally min-max normalized, then discretized using global quintile boundaries (20th, 40th, 60th, 80th percentiles computed across all sessions). The result is 5 bins (0-4) using `np.digitize`.

ii.
```python
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

iii. The AI chose global percentile bins rather than per-session percentile bins. The instructions state "five equal-percentile bins, selected per session."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. Dropped video frames are detected by computing the ratio of each interframe interval to the median interval, rounding, and inserting NaN values at gap positions. These NaNs are then linearly interpolated. After repair, the motion energy array matches the neural data length and is binned identically.

ii.
```python
def repair_motion_energy(session_dir: Path, n_frames: int) -> tuple[np.ndarray, int]:
    interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
    nominal = float(np.median(interframe))
    jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
    # ... fills gaps with NaN then interpolates
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask])
```

iii. The AI noted in its trajectory that the naive timestamp-to-index mapping was too drift-sensitive and switched to a gap-detection approach using median-based rounding of interframe intervals.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected and repaired through linear interpolation (see 4-d). The code validates that the repaired motion energy length matches the neural data length. Remainder frames that don't fill a complete trial are discarded. Sessions with fewer than 2 trials raise an error.

ii.
```python
missing_count = int(jumps.sum())
expected_missing = n_frames - len(motion)
if missing_count != expected_missing:
    raise ValueError(...)

# Trailing frame discard
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
```

iii. The AI verifies that the number of inferred missing frames matches the actual frame count discrepancy, raising an error if they don't match, ensuring data integrity.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `dcnv.preprocess` baseline correction, which runs on CPU (the AI explicitly forces `torch.device("cpu")`). The dF/F computation (including the copy and division) adds some overhead. Loading `.npy` files is I/O-bound but relatively fast.

ii.
```python
device=torch.device("cpu"),
```

iii. The AI forces CPU mode, which makes the baseline correction slower than it would be on GPU. The reference code uses GPU when available.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame repair loop iterates through each gap position sequentially, writing values one at a time and then using `np.interp` for NaN filling. This could potentially be vectorized. The trial assembly uses list comprehensions but the per-trial operations are already vectorized.

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

iii. The number of gaps is typically small, so the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The motion energy data is iterated over multiple times: once during session processing to collect raw trials, once to concatenate all trials for global statistics, and once more during the final assembly to normalize and discretize each trial. The global min/max and percentile computation requires a full pass over all motion data.

ii.
```python
# First: collect motion_trials per session
# Then: concatenate all
all_motion = np.concatenate([trial for record in session_records for trial in record["motion_trials"]])
# Then: per-trial normalization and discretization
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
```

iii. This multi-pass approach is necessary because the global normalization requires knowing the min/max across all sessions before discretizing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes detailed metadata (motion energy normalization stats, per-session frame counts, conversion mode, etc.) that is not used by the decoder. It also computes and stores `motion_quantiles_raw` and `motion_quantiles_norm` in metadata. The global min-max normalization step is unnecessary if per-session percentile binning were used instead.

ii.
```python
"motion_energy_normalization": {
    "type": "global_min_max_after_10_frame_averaging",
    "min": motion_min,
    "max": motion_max,
},
"motion_energy_percentiles_raw": motion_quantiles_raw.tolist(),
"motion_energy_percentiles_normalized": motion_quantiles_norm.tolist(),
```

iii. The extra metadata provides documentation value but does not affect decoder performance.
