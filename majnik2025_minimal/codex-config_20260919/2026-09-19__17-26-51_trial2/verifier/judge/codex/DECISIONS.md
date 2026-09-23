# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sessions by globbing dated directories matching `jm*/20*` and keeping only directories that contain both `suite2p/plane0/F.npy` and `move_deve/motion_energy_glob.npy`. It then iterates session-by-session and loads fluorescence from `F.npy`, QC metadata from `iscell.npy` and `ops.npy`, and behavior from `motion_energy_glob.npy` plus `tstamps.npy`. Trials are not loaded directly from disk; they are created later by slicing each processed session into 60-second chunks.

ii.
```python
def discover_sessions(data_root: Path) -> list[Path]:
    sessions = sorted(
        p
        for p in data_root.glob("jm*/20*")
        if (p / "suite2p/plane0/F.npy").is_file()
        and (p / "move_deve/motion_energy_glob.npy").is_file()
    )
    if not sessions:
        raise FileNotFoundError(f"No complete sessions found below {data_root}")
    return sessions

for session_i, session_dir in enumerate(sessions, start=1):
    plane_dir = session_dir / "suite2p/plane0"
    motion_dir = session_dir / "move_deve"

    fluorescence = np.load(plane_dir / "F.npy")
    iscell = np.load(plane_dir / "iscell.npy")
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    ...
    raw_motion = np.load(motion_dir / "motion_energy_glob.npy")
    timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. In the trajectory, the agent said it would inspect the raw data layout, repository code, and array metadata before implementing the converter. It later concluded that the source data were already Track2p-curated and that the key remaining choices were the neural representation, camera-frame handling, temporal binning, and 60-second trial segmentation.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the parent directory names of the discovered session paths. The unique mouse IDs are sorted alphabetically, and a mapping from subject name to subject index is created.

ii.
```python
sessions = discover_sessions(data_root)
subjects = sorted({session.parent.name for session in sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
```

iii. The trajectory does not contain a separate explicit justification for subject splitting beyond the agent's use of the folder structure and its later report that it found 6 mice across 41 sessions.

## 1-c. How are the data split into sessions?

i. Each dated recording directory under a mouse directory is treated as one session. Sessions are globally sorted as paths, then later grouped back to subjects through `session.parent.name`.

ii.
```python
def discover_sessions(data_root: Path) -> list[Path]:
    sessions = sorted(
        p
        for p in data_root.glob("jm*/20*")
        if (p / "suite2p/plane0/F.npy").is_file()
        and (p / "move_deve/motion_energy_glob.npy").is_file()
    )
```

iii. The code docstring states, "Every dated directory is a session." The trajectory also shows the agent validating the session inventory and reporting counts for each discovered session during conversion.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive, non-overlapping 60-second segments of each session. The code computes `trial_frames = 60 * 30 = 1800` raw imaging frames per trial and `trial_bins = 180` downsampled bins per trial. It requires the raw session length to be an exact multiple of 60 seconds; if not, it raises an error rather than discarding leftover data.

ii.
```python
trial_frames = int(TRIAL_SECONDS * FS_HZ)
trial_bins = trial_frames // AVERAGE_FRAMES
...
n_neurons, n_frames = fluorescence.shape
if n_frames % trial_frames:
    raise ValueError(
        f"{session_dir} has {n_frames} frames, not an integer number of trials"
    )
...
n_trials = n_frames // trial_frames
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
    session_input.append(elapsed_seconds[None, start:stop])
    session_output.append(motion_classes[None, start:stop])
```

iii. In the trajectory, the agent said one of the "key remaining choices" was the exact 60-second trial boundaries. The final code docstring then states that sessions are cut into consecutive, non-overlapping 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply any per-trial quality-control filtering. All computed 60-second trial slices are kept. The only related guard is a session-level check that the session length is an integer number of trials.

ii.
```python
if n_frames % trial_frames:
    raise ValueError(
        f"{session_dir} has {n_frames} frames, not an integer number of trials"
    )
...
for trial in range(n_trials):
    ...
    session_neural.append(neural_binned[:, start:stop])
    session_input.append(elapsed_seconds[None, start:stop])
    session_output.append(motion_classes[None, start:stop])
```

iii. The trajectory does not mention any intended trial rejection rule. The agent treated the recordings as complete continuous sessions that could simply be partitioned into 60-second chunks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives neural data from `suite2p/plane0/F.npy` only. It also loads `iscell.npy` and `ops.npy` for sanity checks, but it does not use `Fneu.npy` in the conversion.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy")
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
...
neural_binned = average_consecutive(
    baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
).astype(np.float32, copy=False)
```

iii. In the trajectory, the agent said it had found the repository's "actual dF/F0 implementation" and concluded that its call site used a neuropil coefficient of zero. Based on that interpretation, it chose to use fluorescence alone and not apply any `Fneu` subtraction.

## 2-b. How is the `neural` data processed?

i. The agent applies a Track2p-style baseline subtraction to `F.npy`: Gaussian smoothing with sigma 10 frames, then a 60-second minimum filter followed by a 60-second maximum filter, and finally `F - baseline`. After that baseline correction, it averages non-overlapping groups of 10 frames, producing one neural time bin every 1/3 second.

ii.
```python
def baseline_correct_fluorescence(fluorescence: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(
        fluorescence, sigma=(0.0, BASELINE_SIGMA_FRAMES), mode="reflect"
    )
    window = int(BASELINE_WINDOW_SECONDS * FS_HZ)
    baseline = minimum_filter1d(smoothed, size=window, axis=1, mode="reflect")
    baseline = maximum_filter1d(baseline, size=window, axis=1, mode="reflect")
    return fluorescence - baseline

neural_binned = average_consecutive(
    baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
).astype(np.float32, copy=False)
```

iii. The trajectory explicitly says the agent found the Track2p `dF/F0` implementation and intended to "preserve that implementation exactly." It justified the choice as reproducing the repository's baseline correction before the paper's 10-frame averaging step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not filter neurons out of the saved dataset. Instead, it asserts that every ROI in `iscell.npy` is already marked as a cell and that the Track2p output is prefiltered. If those checks pass, it keeps all rows of `F.npy`.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
...
if len(iscell) != fluorescence.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(
        f"{session_dir} is not the expected prefiltered Track2p output"
    )
...
brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The trajectory states that "the source data are already the Track2p-curated cells" and that within each mouse the neuron rows are already matched across days. The code docstring echoes that no additional ROI filter or rematching is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural preprocessing is run on the full session before any trial cutting, so baseline filtering is continuous across the whole recording. The saved trial matrices are then just consecutive slices of the binned session. In metadata, the agent describes the alignment event as the "start of each consecutive 60-second trial," with `off_start = 0.0` and `off_end = 60.0`.

ii.
```python
# Process before splitting, so the baseline filter is continuous across
# artificial trial boundaries, exactly as for the full paper sessions.
neural_binned = average_consecutive(
    baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
).astype(np.float32, copy=False)
...
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
...
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 60-second trial",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
```

iii. In the trajectory, the agent emphasized preserving continuous preprocessing across artificial trial boundaries and later described the final dataset as having usable neural-motion alignment after its trial slicing and timestamp interpolation choices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping temporal averages at a 30 Hz source rate, so each saved time bin is `10 / 30 = 1/3` second, or 333.33 ms. This rebinning is applied to both neural activity and motion energy before trial splitting.

ii.
```python
FS_HZ = 30.0
AVERAGE_FRAMES = 10
...
neural_binned = average_consecutive(
    baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
).astype(np.float32, copy=False)
motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)
...
"time_bin_size": 1000.0 * AVERAGE_FRAMES / FS_HZ,
```

iii. The trajectory explicitly says the paper averages both neural and behavior traces in 10-frame bins and that the resulting time bin is 1/3 second.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not read from a raw data file. It is synthesized from the binned sample index together with the fixed sampling rate (`FS_HZ = 30`) and the 10-frame averaging factor.

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32)
    * (AVERAGE_FRAMES / FS_HZ)
)
...
session_input.append(elapsed_seconds[None, start:stop])
```

iii. The trajectory does not give a separate justification beyond following the decoder requirement for "time elapsed from the beginning of the session in seconds" and using the same binned timeline as the neural data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code computes a continuous per-session elapsed-time vector after neural and motion traces have already been binned. The vector is float32, advances in 1/3-second steps, and is then sliced into per-trial segments without resetting at each trial.

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32)
    * (AVERAGE_FRAMES / FS_HZ)
)
...
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_input.append(elapsed_seconds[None, start:stop])
```

iii. The trajectory contains no extended discussion of this step; it appears to have been treated as a straightforward consequence of the fixed frame rate and the 10-frame binning.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated on the same binned session timeline as the neural data and then sliced using the same `start` and `stop` indices for each trial.

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32)
    * (AVERAGE_FRAMES / FS_HZ)
)
...
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
    session_input.append(elapsed_seconds[None, start:stop])
```

iii. The trajectory does not separately justify this beyond the general goal of keeping neural and behavioral streams on a shared binned clock.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`, and the alignment information comes from `move_deve/tstamps.npy`. The agent does not use `interframe_int.npy`.

ii.
```python
raw_motion = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
aligned_motion, n_missing_camera_frames = align_motion_to_imaging(
    raw_motion, timestamps, n_frames
)
```

iii. In the trajectory, the agent said it would "interpolate camera motion onto the 30 Hz imaging clock using camera timestamps." It later justified this as handling documented missing camera frames while retaining the full neural recording.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent first validates the camera timestamp vector, estimates the nominal camera frame interval from the median timestamp difference, infers how many camera frames were missed, and linearly interpolates the motion-energy signal onto a regular imaging-frame clock of length `n_frames`. It then averages the aligned motion energy in non-overlapping 10-frame bins and later discretizes those bins into quintiles.

ii.
```python
def align_motion_to_imaging(
    motion: np.ndarray, timestamps: np.ndarray, n_imaging_frames: int
) -> tuple[np.ndarray, int]:
    motion = np.asarray(motion, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if motion.ndim != 1 or timestamps.ndim != 1 or len(motion) != len(timestamps):
        raise ValueError("Motion energy and timestamps must be equal-length 1-D arrays")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Camera timestamps must be strictly increasing")

    intervals = np.diff(timestamps)
    frame_step = float(np.median(intervals))
    missing_frames = int(
        np.maximum(np.rint(intervals / frame_step).astype(np.int64) - 1, 0).sum()
    )
    imaging_clock = timestamps[0] + np.arange(n_imaging_frames) * frame_step
    aligned = np.interp(imaging_clock, timestamps, motion)
    return aligned, missing_frames

motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)
```

iii. The trajectory explicitly says the agent wanted timestamp-based interpolation because it handled the missing camera frames described in the data materials. It later pointed to recovered missing-frame counts as evidence that the alignment was working.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After timestamp alignment and 10-frame averaging, the agent computes session-specific quintile edges at the 20th, 40th, 60th, and 80th percentiles. It then assigns each binned motion value to one of five categories using `np.searchsorted(..., side="right")`.

ii.
```python
quintile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
motion_classes = np.searchsorted(
    quintile_edges, motion_binned, side="right"
).astype(np.int64)
```

iii. In the trajectory, the agent said it would "apply session-specific quintiles only after the 10-frame averaging." The final summary also says it used per-session motion-energy quintiles.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent aligns motion energy to the neural recording by interpolating camera samples onto a regular imaging-frame clock with exactly `n_frames` samples, where `n_frames` is the number of fluorescence frames. It then applies the same 10-frame averaging and the same per-trial slicing indices to the motion labels and the neural data.

ii.
```python
n_neurons, n_frames = fluorescence.shape
...
aligned_motion, n_missing_camera_frames = align_motion_to_imaging(
    raw_motion, timestamps, n_frames
)
motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)
...
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
    session_output.append(motion_classes[None, start:stop])
```

iii. The trajectory says the timestamp alignment "recovered the documented dropped-camera-frame patterns ... without discarding neural data," and the agent later used decoder performance as a sanity check that neural-motion alignment was sensible.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing camera samples by interpolating the motion-energy trace from camera timestamps onto the imaging timeline. For other anomalies, it fails fast: it raises errors if timestamps are malformed, if `iscell.npy` does not indicate all ROIs are cells, if the imaging frame rate differs from 30 Hz, or if the session length is not an integer multiple of 60 seconds.

ii.
```python
if motion.ndim != 1 or timestamps.ndim != 1 or len(motion) != len(timestamps):
    raise ValueError("Motion energy and timestamps must be equal-length 1-D arrays")
if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
    raise ValueError("Camera timestamps must be strictly increasing")
...
if len(iscell) != fluorescence.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(
        f"{session_dir} is not the expected prefiltered Track2p output"
    )
if not np.isclose(float(ops["fs"]), FS_HZ):
    raise ValueError(f"Unexpected sampling rate {ops['fs']} in {session_dir}")
if n_frames % trial_frames:
    raise ValueError(
        f"{session_dir} has {n_frames} frames, not an integer number of trials"
    )
```

iii. In the trajectory, the agent mainly justified this through the timestamp-interpolation choice, saying it wanted to retain the complete neural recording despite missing camera frames. It did not give a separate justification for the stricter fail-fast checks.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work in this code is likely the per-session full-trace neural preprocessing: Gaussian smoothing plus 60-second min/max filtering over every neuron's fluorescence trace, followed by 10-frame averaging. Timestamp interpolation over every camera sample is also nontrivial, but the neural baseline correction is the dominant heavy operation.

ii.
```python
smoothed = gaussian_filter(
    fluorescence, sigma=(0.0, BASELINE_SIGMA_FRAMES), mode="reflect"
)
window = int(BASELINE_WINDOW_SECONDS * FS_HZ)
baseline = minimum_filter1d(smoothed, size=window, axis=1, mode="reflect")
baseline = maximum_filter1d(baseline, size=window, axis=1, mode="reflect")
...
aligned = np.interp(imaging_clock, timestamps, motion)
```

iii. The trajectory does not explicitly profile runtime, so this is inferred from the implemented operations and from the fact that these filters and interpolations are run over entire sessions before trial splitting.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loop is the per-trial assembly loop that appends session slices into Python lists. It could be partially vectorized by reshaping the already binned session arrays into `(n_trials, ...)` blocks before converting them to the required list-of-trials format, although the final output format still needs Python lists. The outer session loop is also sequential.

ii.
```python
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
    session_input.append(elapsed_seconds[None, start:stop])
    session_output.append(motion_classes[None, start:stop])
```

iii. The trajectory does not mention efficiency-oriented vectorization. The agent instead focused on correctness of preprocessing and alignment.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same preprocessing pipeline independently for every session: load arrays, validate metadata, baseline-correct fluorescence, align motion to imaging timestamps, average both streams into 10-frame bins, discretize motion by session, and then split into trials. This repetition is session-by-session rather than redundant repeated work on the same data.

ii.
```python
for session_i, session_dir in enumerate(sessions, start=1):
    ...
    neural_binned = average_consecutive(
        baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
    ).astype(np.float32, copy=False)
    ...
    aligned_motion, n_missing_camera_frames = align_motion_to_imaging(
        raw_motion, timestamps, n_frames
    )
    motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)
    ...
    quintile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
```

iii. The trajectory treats this as the intended per-session workflow and does not identify it as unnecessary repetition.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra work that is not needed by the downstream decoder itself: it loads `iscell.npy` and `ops.npy` only for validation checks, computes `n_missing_camera_frames` only for logging/metadata, stores `motion_energy_quintile_edges` and other `session_info` metadata that the decoder does not consume, and keeps verbose provenance strings in `metadata`.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
...
aligned_motion, n_missing_camera_frames = align_motion_to_imaging(
    raw_motion, timestamps, n_frames
)
...
session_info.append(
    {
        "subject": subject,
        "session": session_dir.name,
        "n_neurons": int(n_neurons),
        "n_trials": int(n_trials),
        "source_frames": int(n_frames),
        "camera_samples": int(len(raw_motion)),
        "inferred_missing_camera_frames": n_missing_camera_frames,
        "motion_energy_quintile_edges": quintile_edges.tolist(),
    }
)
```

iii. The trajectory does not call these pieces out as unnecessary. They appear to have been added for validation, reporting, and provenance rather than for the downstream decoder.
