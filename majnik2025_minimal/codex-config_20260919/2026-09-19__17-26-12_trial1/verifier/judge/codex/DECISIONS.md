# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `/app/data` for subject directories matching `jm*`, then scans each subject for session directories matching `*_a`. A session is accepted only if `F.npy`, `iscell.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy` are present. During conversion it loads those files for each session before later splitting the session into trials.

ii.
```python
def session_directories(data_root: Path) -> list[tuple[str, Path]]:
    sessions: list[tuple[str, Path]] = []
    for subject_dir in sorted(data_root.glob("jm*")):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.glob("*_a")):
            plane_dir = session_dir / "suite2p" / "plane0"
            motion_dir = session_dir / "move_deve"
            required = [
                plane_dir / "F.npy",
                plane_dir / "iscell.npy",
                plane_dir / "ops.npy",
                motion_dir / "motion_energy_glob.npy",
                motion_dir / "tstamps.npy",
            ]
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(f"incomplete session: {session_dir}")
            sessions.append((subject_dir.name, session_dir))
```

```python
F = np.load(plane_dir / "F.npy", mmap_mode="r")
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. In trajectory step 14, the agent said that every dated directory should count as a session and that missing camera frames should be reconstructed from timestamp gaps. Its earlier searches of `data/README.md` and `track2p/gui/data_management.py` led it to treat these Track2p/Suite2p files as the relevant source files.

## 1-b. How are the data split into subjects?

i. Subjects are the directory names matching `jm*`, sorted lexicographically. After collecting all sessions, the script builds a sorted unique subject list and a lookup table from subject name to subject index.

ii.
```python
for subject_dir in sorted(data_root.glob("jm*")):
    if not subject_dir.is_dir():
        continue
```

```python
subjects = sorted({subject for subject, _ in session_paths})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. In the trajectory, the agent repeatedly referred to the six `jm*` directories as the six mice and validated that the converted dataset had 6 subjects and 41 sessions.

## 1-c. How are the data split into sessions?

i. Each `*_a` directory under a subject is treated as one session. Sessions are sorted within each subject and appended in that order to the dataset.

ii.
```python
for subject_dir in sorted(data_root.glob("jm*")):
    ...
    for session_dir in sorted(subject_dir.glob("*_a")):
        ...
        sessions.append((subject_dir.name, session_dir))
```

iii. In trajectory step 14, the agent explicitly said "Every dated directory is a session." The dataset README was used to justify this, since it describes each daily recording directory as one session.

## 1-d. How are the data split into trials?

i. The script defines artificial trials as contiguous, non-overlapping 60 second windows. It requires the raw imaging frame count to be an exact multiple of `30 Hz * 60 s = 1800` frames, bins the time series into 10-frame averages, and then slices each session into `180` binned samples per trial.

ii.
```python
FRAME_RATE_HZ = 30.0
AVERAGE_FRAMES = 10
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FRAME_RATE_HZ * TRIAL_SECONDS)
TRIAL_BINS = TRIAL_FRAMES // AVERAGE_FRAMES
```

```python
if F.shape[1] % TRIAL_FRAMES:
    raise ValueError(f"session is not a whole number of 60-second trials: {session_dir}")
...
n_trials = F.shape[1] // TRIAL_FRAMES
...
for trial in range(n_trials):
    start = trial * TRIAL_BINS
    stop = start + TRIAL_BINS
    session_neural.append(neural_binned[:, start:stop])
```

iii. The agent’s header comment says the paper’s 10-frame averaging yields exactly 180 samples per 60-second trial, and in trajectory step 20 it validated that sessions produced either 20 or 30 such trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality filter. Instead, the script enforces session-level preconditions before any trials are created: required files must exist, sampling rate must be 30 Hz, `iscell` must have the expected shape and all entries must already satisfy the cell criterion, and the session length must be an exact multiple of 60 seconds.

ii.
```python
if not np.isclose(fs, FRAME_RATE_HZ):
    raise ValueError(f"unexpected sampling rate {fs} in {session_dir}")
if F.ndim != 2 or iscell.shape != (F.shape[0], 2):
    raise ValueError(f"unexpected Suite2p shapes in {session_dir}")
if not np.all((iscell[:, 0] == 1) & (iscell[:, 1] > 0.5)):
    raise ValueError(
        f"{session_dir} contains an ROI outside the paper's cell criterion"
    )
if F.shape[1] % TRIAL_FRAMES:
    raise ValueError(f"session is not a whole number of 60-second trials: {session_dir}")
```

iii. In trajectory step 14, the agent said the supplied files already contain Track2p-matched, Suite2p-classified cells, so it chose not to add another trial-level filter and instead validated those assumptions up front.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural traces are derived from `suite2p/plane0/F.npy`. The code also loads `ops.npy` to read the frame rate and `iscell.npy` to validate the session, but the numerical neural signal itself is computed only from `F.npy`.

ii.
```python
F = np.load(plane_dir / "F.npy", mmap_mode="r")
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
...
neural_binned = average_in_bins(baseline_correct_fluorescence(F, fs))
```

iii. In trajectory step 14, the agent concluded from `track2p/gui/data_management.py::F_processing` that the intended neural signal was the baseline-corrected `F` trace and that Track2p’s `dF/F0` path used `neucoeff=0.0`, so it intentionally did not use `Fneu.npy`.

## 2-b. How is the `neural` data processed?

i. Neural activity is baseline-corrected with a manual Track2p-style maximin filter: Gaussian smoothing with `sigma=10`, then a 60-second rolling minimum, then a 60-second rolling maximum, and finally subtraction from `F`. After that, the trace is averaged in non-overlapping 10-frame bins.

ii.
```python
def baseline_correct_fluorescence(F: np.ndarray, fs: float) -> np.ndarray:
    corrected = gaussian_filter1d(
        np.asarray(F, dtype=np.float32), sigma=10.0, axis=1
    )
    window = int(60.0 * fs)
    corrected = minimum_filter1d(corrected, size=window, axis=1)
    corrected = maximum_filter1d(corrected, size=window, axis=1)
    return np.subtract(F, corrected, dtype=np.float32)
```

```python
neural_binned = average_in_bins(baseline_correct_fluorescence(F, fs))
```

iii. In trajectory step 14, the agent said it would preserve the implementation in `track2p/gui/data_management.py::F_processing`, which it interpreted as the paper’s baseline-corrected fluorescence pipeline, and then apply the paper’s 10-frame averaging for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not remove a subset of neurons from `F.npy`; it assumes the saved Track2p/Suite2p outputs already contain only tracked cells. However, it performs a hard validation that every row in `iscell.npy` has `iscell[:, 0] == 1` and `iscell[:, 1] > 0.5`, and aborts the whole session if not.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
...
if not np.all((iscell[:, 0] == 1) & (iscell[:, 1] > 0.5)):
    raise ValueError(
        f"{session_dir} contains an ROI outside the paper's cell criterion"
    )
```

iii. In trajectory step 14, the agent said the files already contain only Track2p-matched, Suite2p-classified cells, so it chose validation rather than an additional filtering pass.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not align neural data to a stimulus or behavior event. Instead, it defines each trial as a contiguous 60-second block from the continuous recording and stores metadata describing the alignment event as the start of each such block.

ii.
```python
for trial in range(n_trials):
    start = trial * TRIAL_BINS
    stop = start + TRIAL_BINS
    session_neural.append(
        np.ascontiguousarray(neural_binned[:, start:stop], dtype=np.float32)
    )
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each contiguous 60-second session block",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
    ...
}
```

iii. The agent’s script header says the data should be split into exact 60-second trials after 10-frame averaging, and its later validation in trajectory step 20 treats those artificial 60-second blocks as the basic aligned unit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are rebinned into non-overlapping 10-frame means. At 30 Hz, that produces a time bin size of `10 / 30 = 0.333...` seconds, or `333.333 ms`.

ii.
```python
FRAME_RATE_HZ = 30.0
AVERAGE_FRAMES = 10
...
def average_in_bins(values: np.ndarray, bin_size: int = AVERAGE_FRAMES) -> np.ndarray:
    usable = values.shape[-1] - values.shape[-1] % bin_size
    values = values[..., :usable]
    new_shape = values.shape[:-1] + (usable // bin_size, bin_size)
    return values.reshape(new_shape).mean(axis=-1, dtype=np.float32)
```

```python
"time_bin_size": 1000.0 * AVERAGE_FRAMES / FRAME_RATE_HZ,
```

iii. In trajectory step 14, the agent said the paper averages neural and motion traces in non-overlapping 10-frame bins, and in step 21 it verified that each trial then had 180 time bins.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is not loaded from a raw timestamp file. It is synthesized from the binned sample index and the imaging frame rate `fs` read from `ops.npy`.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
fs = float(ops.get("fs", FRAME_RATE_HZ))
...
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32) * AVERAGE_FRAMES
    + (AVERAGE_FRAMES - 1) / 2
) / np.float32(fs)
```

iii. In trajectory step 21, the agent validated that the input ranged from about `0.2` to `1199.8` or `1799.8` seconds, showing that it intended time to be generated from bin index rather than a separate raw variable.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script computes the time assigned to each binned sample as the center of the 10-frame bin: `(bin_index * 10 + 4.5) / fs`, cast to `float32`. This creates a continuous time axis across the whole session, not a trial-resetting time axis.

ii.
```python
# Mean times of the ten original frames represented by each sample.
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32) * AVERAGE_FRAMES
    + (AVERAGE_FRAMES - 1) / 2
) / np.float32(fs)
```

iii. In trajectory step 21, the agent explicitly checked the first and second trial times and confirmed values such as `0.15 ... 59.816666` for trial 1 and `60.15 ... 119.816666` for trial 2, showing that this bin-center, session-continuous convention was intentional.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned to neural data by construction: it is created at the same post-binning length as `neural_binned` and then sliced into trials with the same `start:stop` indices used for neural activity.

ii.
```python
for trial in range(n_trials):
    start = trial * TRIAL_BINS
    stop = start + TRIAL_BINS
    session_neural.append(
        np.ascontiguousarray(neural_binned[:, start:stop], dtype=np.float32)
    )
    session_input.append(
        np.ascontiguousarray(elapsed_seconds[None, start:stop], dtype=np.float32)
    )
```

iii. The trajectory does not contain a separate verbal justification for this point, but the agent’s validation in step 21 checked the time values trial by trial after constructing neural and input arrays from the same slices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`. The script also loads `move_deve/tstamps.npy` so it can infer dropped camera frames and align motion samples to imaging frames.

ii.
```python
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
motion_aligned = align_motion_to_imaging(motion_raw, timestamps, F.shape[1])
```

iii. In trajectory step 14, the agent said missing camera frames would be reconstructed from timestamp gaps, based on the README note that missing frames can be found from `tstamps.npy` or `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script first aligns motion samples to imaging frames with timestamp-based interpolation when the motion stream is shorter than the imaging stream. It then averages the aligned motion trace into non-overlapping 10-frame bins and discretizes the binned values into five session-specific percentile bins.

ii.
```python
def align_motion_to_imaging(
    motion: np.ndarray, timestamps: np.ndarray, n_imaging_frames: int
) -> np.ndarray:
    ...
    if motion.size == n_imaging_frames:
        return motion
    ...
    frame_steps = np.maximum(1, np.rint(intervals / typical_interval).astype(np.int64))
    camera_frame_idx = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(frame_steps))
    )
    ...
    imaging_frame_idx = np.arange(n_imaging_frames, dtype=np.float64)
    return np.interp(imaging_frame_idx, camera_frame_idx, motion).astype(np.float32)
```

```python
motion_binned = average_in_bins(motion_aligned)
labels, thresholds = quintile_labels(motion_binned)
```

iii. In trajectory step 14, the agent justified this as reconstructing missing camera frames from doubled timestamp intervals and then applying the paper’s 10-frame averaging before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The binned motion trace for each session is discretized into five categories using that same session’s 20th, 40th, 60th, and 80th percentiles. `np.digitize` then maps each time bin to integer labels `0` through `4`.

ii.
```python
def quintile_labels(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.percentile(motion, [20, 40, 60, 80])
    labels = np.digitize(motion, thresholds, right=False).astype(np.int64)
    return labels, thresholds
```

iii. The script header says motion quintile thresholds are computed separately for each complete session after alignment and 10-frame averaging, and trajectory step 21 confirms that each class occupied exactly 20% of the samples.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by mapping camera samples onto imaging-frame indices. If the motion array already has the same length as the imaging recording, it is left unchanged. If it is shorter, timestamp gaps are converted into larger frame steps, and `np.interp` fills in the missing imaging-frame positions. Afterward, motion and neural traces are both 10-frame binned and trial-sliced with the same indices.

ii.
```python
if motion.size == n_imaging_frames:
    return motion
...
frame_steps = np.maximum(1, np.rint(intervals / typical_interval).astype(np.int64))
camera_frame_idx = np.concatenate(
    (np.array([0], dtype=np.int64), np.cumsum(frame_steps))
)
...
return np.interp(imaging_frame_idx, camera_frame_idx, motion).astype(np.float32)
```

```python
motion_aligned = align_motion_to_imaging(motion_raw, timestamps, F.shape[1])
motion_binned = average_in_bins(motion_aligned)
...
session_output.append(
    np.ascontiguousarray(labels[None, start:stop], dtype=np.int64)
)
```

iii. In trajectory step 14, the agent said that full-length streams should be treated as already one-to-one, while shorter streams should be reconstructed from timestamp gaps. The agent’s exploratory step 13 specifically checked sessions whose timestamps had anomalous gaps to support that choice.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are handled by timestamp-based interpolation when `motion_energy_glob.npy` is shorter than the imaging trace. Other irregularities are not repaired; they cause errors instead. In particular, mismatched motion/timestamp lengths, invalid timestamps, inferred frame counts that do not match imaging length, unexpected sampling rates, and sessions that are not an exact multiple of 60 seconds all cause the script to abort.

ii.
```python
if motion.size != timestamps.size:
    raise ValueError("motion_energy_glob and tstamps lengths differ")
if motion.size == n_imaging_frames:
    return motion
if motion.size < 2 or motion.size > n_imaging_frames:
    raise ValueError(
        f"cannot align {motion.size} motion samples to {n_imaging_frames} frames"
    )
...
if inferred_frames != n_imaging_frames:
    raise ValueError(
        "timestamp gaps imply "
        f"{inferred_frames} frames, expected {n_imaging_frames}"
    )
```

```python
if not np.isclose(fs, FRAME_RATE_HZ):
    raise ValueError(f"unexpected sampling rate {fs} in {session_dir}")
...
if F.shape[1] % TRIAL_FRAMES:
    raise ValueError(f"session is not a whole number of 60-second trials: {session_dir}")
```

iii. In trajectory step 14, the agent said missing camera frames should be reconstructed from timestamp gaps. The trajectory does not show any justification for the stricter hard-failure behavior on other minor irregularities.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant work is per-session preprocessing over full-length recordings: the fluorescence baseline correction (`gaussian_filter1d`, `minimum_filter1d`, `maximum_filter1d`) and the motion-alignment interpolation and binning. Trial slicing and metadata assembly are comparatively cheap.

ii.
```python
def baseline_correct_fluorescence(F: np.ndarray, fs: float) -> np.ndarray:
    corrected = gaussian_filter1d(
        np.asarray(F, dtype=np.float32), sigma=10.0, axis=1
    )
    window = int(60.0 * fs)
    corrected = minimum_filter1d(corrected, size=window, axis=1)
    corrected = maximum_filter1d(corrected, size=window, axis=1)
    return np.subtract(F, corrected, dtype=np.float32)
```

```python
motion_aligned = align_motion_to_imaging(motion_raw, timestamps, F.shape[1])
neural_binned = average_in_bins(baseline_correct_fluorescence(F, fs))
motion_binned = average_in_bins(motion_aligned)
```

iii. The trajectory does not contain an explicit runtime analysis, but the agent focused most of its validation effort on reproducing the fluorescence filter exactly and on handling motion-stream alignment, which indicates these were the substantive processing steps it considered important.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loop is the per-trial loop that repeatedly slices and copies neural, input, and output arrays into lists. Because trials are fixed-length contiguous blocks, this could have been reshaped in a more vectorized way before converting to the required list-of-trials structure.

ii.
```python
for trial in range(n_trials):
    start = trial * TRIAL_BINS
    stop = start + TRIAL_BINS
    session_neural.append(
        np.ascontiguousarray(neural_binned[:, start:stop], dtype=np.float32)
    )
    session_input.append(
        np.ascontiguousarray(elapsed_seconds[None, start:stop], dtype=np.float32)
    )
    session_output.append(
        np.ascontiguousarray(labels[None, start:stop], dtype=np.int64)
    )
```

iii. The trajectory does not discuss vectorization explicitly. This answer is inferred from the implementation the agent wrote.

## 6-c. What processing does the code repeat multiple times?

i. There is no major redundant reprocessing of the same session data. The same intended pipeline is repeated once per session: load files, validate shapes, align motion, baseline-correct fluorescence, bin both streams, discretize motion, and then slice into trials.

ii.
```python
for session_number, (subject, session_dir) in enumerate(session_paths, start=1):
    ...
    motion_aligned = align_motion_to_imaging(motion_raw, timestamps, F.shape[1])
    neural_binned = average_in_bins(baseline_correct_fluorescence(F, fs))
    motion_binned = average_in_bins(motion_aligned)
    labels, thresholds = quintile_labels(motion_binned)
    ...
```

iii. The trajectory shows the agent deliberately designing a single per-session preprocessing pipeline and then applying it uniformly to all 41 sessions; it does not mention any repeated work as a problem.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does a few extra things that the decoder does not need directly: it loads and validates `iscell.npy`, loads `ops.npy` mainly to confirm the frame rate, computes and stores per-session `motion_quintile_thresholds`, and stores verbose `session_info` metadata. Those checks and annotations are not part of the decoder input/output arrays used downstream.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
...
if not np.all((iscell[:, 0] == 1) & (iscell[:, 1] > 0.5)):
    raise ValueError(
        f"{session_dir} contains an ROI outside the paper's cell criterion"
    )
```

```python
session_info.append(
    {
        "subject": subject,
        "session": session_dir.name,
        "n_neurons": int(F.shape[0]),
        "n_trials": n_trials,
        "n_imaging_frames": int(F.shape[1]),
        "n_camera_frames": int(motion_raw.size),
        "motion_quintile_thresholds": thresholds.tolist(),
    }
)
```

iii. The trajectory does not explicitly call these unnecessary, but it does show the agent adding validation and descriptive metadata beyond the strict decoder schema.
