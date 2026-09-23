# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all usable sessions by scanning `/app/data` for directories matching `*/20*_*` that contain both `suite2p/plane0/F.npy` and `move_deve/motion_energy_glob.npy`. It then processes each session one at a time, and only later groups session outputs into the final dataset. Within each session it loads `ops.npy`, `iscell.npy`, `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy`.

ii.
```python
def find_sessions(data_dir: Path) -> list[Path]:
    sessions = sorted(
        path
        for path in data_dir.glob("*/20*_*")
        if (path / "suite2p" / "plane0" / "F.npy").is_file()
        and (path / "move_deve" / "motion_energy_glob.npy").is_file()
    )
    if not sessions:
        raise FileNotFoundError(f"No complete sessions found below {data_dir}")
    return sessions

ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
fluorescence_all = np.load(plane_dir / "F.npy")
neuropil_all = np.load(plane_dir / "Fneu.npy")
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. In the trajectory, the AI said it would "check the supplied arrays and notebook code to determine the exact dF/F implementation, session coverage, and whether timestamps require interpolation or direct frame pairing" (step 9). It justified the session scan as following the paper/repository layout and then processing the resulting sessions in stable order.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each discovered session, then deduplicated and sorted lexicographically. `subject_idx` is built by mapping each processed session back to its subject name.

ii.
```python
session_dirs = find_sessions(data_dir)
subjects = sorted({path.parent.name for path in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}

for session_number, session_dir in enumerate(session_dirs, start=1):
    ...
    subject_idx.append(subject_to_index[session_dir.parent.name])
```

iii. The trajectory indicates the AI believed subject/date directory names were already sufficient to reconstruct the dataset organization, and that "sessions and subjects are sorted chronologically/lexicographically" (module docstring and final summary).

## 1-c. How are the data split into sessions?

i. Each session is one directory matching `subject/date_suffix`, and each such directory becomes one entry in `data["neural"]`, `data["input"]`, and `data["output"]`. Sessions are sorted globally by subject and date because `Path.glob(... )` results are wrapped in `sorted(...)`.

ii.
```python
def find_sessions(data_dir: Path) -> list[Path]:
    sessions = sorted(
        path
        for path in data_dir.glob("*/20*_*")
        if (path / "suite2p" / "plane0" / "F.npy").is_file()
        and (path / "move_deve" / "motion_energy_glob.npy").is_file()
    )
    ...

for session_number, session_dir in enumerate(session_dirs, start=1):
    converted, info = convert_session(session_dir)
    neural.append(converted["neural"])
    decoder_input.append(converted["input"])
    output.append(converted["output"])
```

iii. The AI’s rationale was that the source layout already defines one daily recording per session directory, so it preserved that organization directly.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous and splits each session into non-overlapping 60-second trials after 10-frame temporal averaging. It computes the number of binned samples per trial from the sampling rate and bin size, keeps only complete trials, and discards any trailing partial trial. It also requires at least two complete trials per session.

ii.
```python
binned_rate_hz = sampling_rate_hz / SOURCE_FRAMES_PER_BIN
bins_per_trial = int(round(TRIAL_SECONDS * binned_rate_hz))
...
trial_count = binned_time.size // bins_per_trial
if trial_count < 2:
    raise ValueError(f"Fewer than two complete trials in {session_dir}")
used_bins = trial_count * bins_per_trial

for trial_index in range(trial_count):
    start = trial_index * bins_per_trial
    stop = start + bins_per_trial
    neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
    input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
    output_trials.append(np.ascontiguousarray(motion_labels[None, start:stop]))
```

iii. In the trajectory the AI repeatedly described the dataset as "continuous recordings" split into "non-overlapping 60-second trials" (steps 26, 38, 46), consistent with the task instruction to make 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a content-based trial quality filter. The only trial-level exclusions are implicit: incomplete trailing trial fragments are dropped, and any session with fewer than two complete trials raises an error and is excluded.

ii.
```python
trial_count = binned_time.size // bins_per_trial
if trial_count < 2:
    raise ValueError(f"Fewer than two complete trials in {session_dir}")
used_bins = trial_count * bins_per_trial
...
"discarded_binned_samples": int(binned_time.size - used_bins),
```

iii. The trajectory does not mention any trial-quality heuristic beyond preserving complete trials; the AI emphasized preserving sessions by interpolating dropped motion frames rather than losing a trial because of a small camera dropout (step 16).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from Suite2p fluorescence traces `F.npy`, neuropil traces `Fneu.npy`, ROI classification scores `iscell.npy`, and the sampling rate in `ops.npy`.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
fluorescence_all = np.load(plane_dir / "F.npy")
neuropil_all = np.load(plane_dir / "Fneu.npy")
sampling_rate_hz = float(ops["fs"])
```

iii. The AI’s trajectory states that it confirmed "baseline-corrected fluorescence (not raw `F`), 30 Hz acquisition" and the paper’s cell classification rule before implementing the converter (step 9).

## 2-b. How is the `neural` data processed?

i. The AI applies a Track2p-style baseline correction rather than the reference’s Suite2p `dcnv.preprocess`. Specifically, it subtracts neuropil with `neucoeff = 0.0`, smooths with a Gaussian filter (`sigma_frames = 10`), applies a 60-second minimum filter followed by a maximum filter, subtracts that baseline, and then averages the trace in non-overlapping 10-frame bins.

ii.
```python
def baseline_correct_fluorescence(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    neucoeff = 0.0
    sigma_frames = 10.0
    baseline_window_seconds = 60.0

    corrected = fluorescence - neucoeff * neuropil
    baseline = gaussian_filter(corrected, [0.0, sigma_frames])
    window_frames = int(baseline_window_seconds * sampling_rate_hz)
    baseline = minimum_filter1d(baseline, window_frames)
    baseline = maximum_filter1d(baseline, window_frames)
    return corrected - baseline

corrected = baseline_correct_fluorescence(
    fluorescence,
    neuropil,
    sampling_rate_hz,
)
binned_neural = average_consecutive_bins(
    corrected,
    SOURCE_FRAMES_PER_BIN,
).astype(np.float32, copy=False)
```

iii. The AI explicitly justified this choice in the trajectory: it said it was using "the repository’s exact fluorescence routine: `F - F0` using the maximin baseline (Gaussian σ=10 frames, 60-second min/max window, and the code’s explicit neuropil coefficient of 0)" and would not add z-scoring or other processing (step 26).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters ROIs by `iscell[:, 1] > 0.5`. It checks that at least one ROI survives. It also contains an optimization: if all ROIs already pass, it skips boolean indexing and uses the original `F` and `Fneu` arrays directly.

ii.
```python
ISCELL_THRESHOLD = 0.5
...
iscell = np.load(plane_dir / "iscell.npy")
cell_mask = iscell[:, 1] > ISCELL_THRESHOLD
if not np.any(cell_mask):
    raise ValueError(f"No ROIs pass iscell>{ISCELL_THRESHOLD} in {session_dir}")

if np.all(cell_mask):
    fluorescence = fluorescence_all
    neuropil = neuropil_all
else:
    fluorescence = fluorescence_all[cell_mask]
    neuropil = neuropil_all[cell_mask]
```

iii. The AI justified this from the paper’s stated cell rule and from inspection of the actual data: it reported that "every retained ROI passes the paper’s `iscell > 0.5` rule" (step 16) and later summarized processing as following "`iscell > 0.5`" (step 46).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats each non-overlapping 60-second trial start as the alignment event for the trialized neural matrices. It slices neural, input, and output arrays with identical trial boundaries. However, the time input remains elapsed time from session start rather than resetting at each trial start.

ii.
```python
for trial_index in range(trial_count):
    start = trial_index * bins_per_trial
    stop = start + bins_per_trial
    neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
    input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
    output_trials.append(np.ascontiguousarray(motion_labels[None, start:stop]))
...
"temporal_alignment_event": (
    "start of each non-overlapping 60-second trial; decoder time input "
    "remains elapsed time from session start"
),
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The AI’s final summary described the output as 60-second trials and explicitly said the "decoder time input remains elapsed time from session start" (metadata and final file contents). That is the rationale it encoded for alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion streams by averaging non-overlapping 10-frame blocks from a 30 Hz source rate, producing 3 Hz data with 333.33 ms bins. No further temporal resampling is applied.

ii.
```python
SOURCE_FRAMES_PER_BIN = 10
EXPECTED_SOURCE_FS = 30.0
...
def average_consecutive_bins(values: np.ndarray, bin_frames: int) -> np.ndarray:
    complete_frames = values.shape[-1] // bin_frames * bin_frames
    if complete_frames != values.shape[-1]:
        values = values[..., :complete_frames]
    new_shape = values.shape[:-1] + (complete_frames // bin_frames, bin_frames)
    return values.reshape(new_shape).mean(axis=-1)

time_bin_ms = 1000.0 * SOURCE_FRAMES_PER_BIN / EXPECTED_SOURCE_FS
```

iii. The trajectory says the AI followed the paper’s "10-frame averaging for decoding" (steps 9, 26, 46) and validated that the result had "333.3 ms bins" (step 38).

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is not loaded from a recorded timestamp vector. It is computed from the neural frame index and the session sampling rate `ops["fs"]`.

ii.
```python
sampling_rate_hz = float(ops["fs"])
...
frame_times = np.arange(neural_frame_count, dtype=np.float64) / sampling_rate_hz
binned_time = average_consecutive_bins(
    frame_times,
    SOURCE_FRAMES_PER_BIN,
).astype(np.float32, copy=False)
```

iii. The AI’s trajectory says it wanted "elapsed-time bins" tied to the 30 Hz acquisition rather than a separate behavioral clock (steps 9 and 31).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI builds a per-frame time axis from `np.arange(neural_frame_count) / sampling_rate_hz`, then averages those frame times within each 10-frame bin. This means the time values are bin centers, not bin starts, and they continue increasing across trials within a session.

ii.
```python
frame_times = np.arange(neural_frame_count, dtype=np.float64) / sampling_rate_hz
binned_time = average_consecutive_bins(
    frame_times,
    SOURCE_FRAMES_PER_BIN,
).astype(np.float32, copy=False)
...
input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
```

iii. The AI justified this choice in code comments and trajectory: the code comment says "Mean raw-frame times make each decoder input the temporal center of its 10-frame bin," and step 31 says the session check produced "correctly centered elapsed-time bins."

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time with neural activity by deriving both from the same neural frame count, applying the same 10-frame averaging, checking that all binned streams have equal length, and then slicing them with the same trial boundaries.

ii.
```python
binned_neural = average_consecutive_bins(
    corrected,
    SOURCE_FRAMES_PER_BIN,
).astype(np.float32, copy=False)
...
binned_time = average_consecutive_bins(
    frame_times,
    SOURCE_FRAMES_PER_BIN,
).astype(np.float32, copy=False)

if not (
    binned_neural.shape[1] == binned_motion.size == motion_labels.size == binned_time.size
):
    raise ValueError(f"Binned streams do not align in {session_dir}")

for trial_index in range(trial_count):
    ...
    neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
    input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
```

iii. The AI’s rationale was that all decoder streams should share one common binned index after motion interpolation and 10-frame averaging; that is implicit in steps 26, 31, and 38.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, and its alignment/reconstruction uses `move_deve/tstamps.npy` to infer missing triggered camera frames.

ii.
```python
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
aligned_motion, interpolated_frames = align_motion_to_neural_frames(
    motion_raw,
    timestamps,
    neural_frame_count,
)
```

iii. In the trajectory, the AI said the "few shorter motion arrays are camera-frame drops" and that the "timestamp gaps exactly equal the missing-frame count," so it chose timestamp-based reconstruction rather than dropping data (step 16).

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first aligns the motion energy trace to neural frames by recovering microscope trigger indices from the timestamp intervals and filling missing camera samples with linear interpolation via `np.interp`. It then averages the aligned motion trace in 10-frame bins and discretizes the binned values within each session into five percentile-based categories.

ii.
```python
intervals = np.diff(timestamps)
nominal_interval = np.median(intervals)
missing_after = np.maximum(
    np.rint(intervals / nominal_interval).astype(np.int64) - 1,
    0,
)
trigger_indices = np.arange(motion.size, dtype=np.int64)
trigger_indices[1:] += np.cumsum(missing_after)
...
aligned = np.interp(
    np.arange(neural_frame_count, dtype=np.float64),
    trigger_indices,
    motion,
)

binned_motion = average_consecutive_bins(
    aligned_motion,
    SOURCE_FRAMES_PER_BIN,
)
motion_labels, percentile_boundaries = motion_quintile_labels(binned_motion)
```

iii. The AI explicitly justified this in steps 16 and 26: it said timestamp-based interpolation preserves complete trials and that motion quintiles should be computed after the same 10-frame averaging used for decoding.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes within-session motion thresholds at the 20th, 40th, 60th, and 80th percentiles of the binned motion trace, then uses `np.digitize` to assign each binned time point to quintile labels `0` through `4`.

ii.
```python
MOTION_PERCENTILES = (20.0, 40.0, 60.0, 80.0)
...
def motion_quintile_labels(
    binned_motion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.percentile(binned_motion, MOTION_PERCENTILES)
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError(f"Non-unique motion percentile boundaries: {boundaries}")
    labels = np.digitize(binned_motion, boundaries, right=False).astype(np.int64)
    return labels, boundaries
```

iii. The trajectory says the AI would compute "motion quintile cut points ... separately for each session" (step 26), and it later reported "exactly balanced quintile labels" (step 31).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data in frame space before any binning or trial splitting. If motion and neural lengths already match, it keeps the original motion vector. Otherwise it infers the missing camera-trigger positions from `tstamps.npy`, interpolates the missing values onto the full neural frame index, and only then bins and trializes the aligned stream together with neural data.

ii.
```python
if motion.size == neural_frame_count:
    return motion, 0
...
trigger_indices = np.arange(motion.size, dtype=np.int64)
trigger_indices[1:] += np.cumsum(missing_after)
...
if int(missing_after.sum()) != missing_count:
    raise ValueError(
        "Timestamp-inferred camera drops do not match neural/motion length "
        f"difference ({int(missing_after.sum())} vs {missing_count})"
    )
...
aligned = np.interp(
    np.arange(neural_frame_count, dtype=np.float64),
    trigger_indices,
    motion,
)
```

iii. The AI’s justification in the trajectory was that camera-frame drops were small, detectable from timestamp gaps, and should be interpolated so sessions would not lose a full 60-second trial (step 16).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing motion samples by timestamp-based interpolation. It also raises explicit errors if motion/timestamp lengths disagree, timestamps are not strictly increasing, inferred drop counts do not equal the neural-motion length difference, no cells pass threshold, the sampling rate is unexpected, or percentile boundaries collapse. It trims incomplete trailing bins and records how many binned samples were discarded.

ii.
```python
if motion.size != timestamps.size:
    raise ValueError(
        f"Motion/timestamp length mismatch: {motion.size} vs {timestamps.size}"
    )
...
if motion.size < 2 or np.any(np.diff(timestamps) <= 0):
    raise ValueError("Motion timestamps must be strictly increasing")
...
if int(missing_after.sum()) != missing_count:
    raise ValueError(
        "Timestamp-inferred camera drops do not match neural/motion length "
        f"difference ({int(missing_after.sum())} vs {missing_count})"
    )
...
if np.any(np.diff(boundaries) <= 0):
    raise ValueError(f"Non-unique motion percentile boundaries: {boundaries}")
...
"discarded_binned_samples": int(binned_time.size - used_bins),
```

iii. The trajectory emphasizes one such correction explicitly: missing motion samples were interpolated because they were attributable to camera-frame drops rather than unusable sessions (step 16). The rest of the handling is visible in code rather than spelled out in the trajectory.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are loading the full fluorescence and neuropil arrays for each session, running the Gaussian/minimum/maximum-filter baseline correction over every neuron and frame, and storing per-trial copies with `np.ascontiguousarray`. Motion interpolation and percentile calculation are lighter by comparison.

ii.
```python
fluorescence_all = np.load(plane_dir / "F.npy")
neuropil_all = np.load(plane_dir / "Fneu.npy")
...
baseline = gaussian_filter(corrected, [0.0, sigma_frames])
window_frames = int(baseline_window_seconds * sampling_rate_hz)
baseline = minimum_filter1d(baseline, window_frames)
baseline = maximum_filter1d(baseline, window_frames)
...
for trial_index in range(trial_count):
    ...
    neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
```

iii. The trajectory does not explicitly discuss performance hotspots, but it does emphasize the custom fluorescence preprocessing as the central processing step (step 26), which is also the most computationally intensive section of the code.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest obvious remaining Python loop is the per-trial append loop that slices each session into 60-second trial arrays. That could be vectorized with a reshape or split-based approach before converting to the nested list structure. Relative to the reference solution, the AI already removed the more expensive repeated `np.insert` loop for missing motion frames by using one `np.interp` call.

ii.
```python
for trial_index in range(trial_count):
    start = trial_index * bins_per_trial
    stop = start + bins_per_trial
    neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
    input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
    output_trials.append(np.ascontiguousarray(motion_labels[None, start:stop]))
```

iii. The trajectory does not call out this optimization explicitly. Its only relevant stated preference was to interpolate dropped motion frames directly from timestamps (step 16), which is already a vectorized choice.

## 6-c. What processing does the code repeat multiple times?

i. The same per-session processing pipeline is repeated independently for every session: load arrays, threshold ROIs, baseline-correct fluorescence, align motion to neural frames, average into 10-frame bins, discretize motion, compute binned time, and then slice into trials. There is no major accidental duplicate computation across sessions, although time binning, neural binning, and motion binning each call the same averaging helper separately.

ii.
```python
for session_number, session_dir in enumerate(session_dirs, start=1):
    ...
    converted, info = convert_session(session_dir)

corrected = baseline_correct_fluorescence(...)
binned_neural = average_consecutive_bins(corrected, SOURCE_FRAMES_PER_BIN)
...
binned_motion = average_consecutive_bins(aligned_motion, SOURCE_FRAMES_PER_BIN)
...
binned_time = average_consecutive_bins(frame_times, SOURCE_FRAMES_PER_BIN)
```

iii. The trajectory frames this as deliberate uniform per-session processing rather than a redundancy problem; it does not claim a separate caching or reuse strategy.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary processing is that it loads and passes `Fneu.npy` through the code path even though `neucoeff = 0.0`, so neuropil values do not affect the output neural traces. It also computes and stores extensive `session_info` metadata, including motion quintile boundaries and frame counts, which the downstream decoder does not use. In the common case where all cells already pass threshold, it still loads `iscell.npy` just to confirm that fact.

ii.
```python
neucoeff = 0.0
...
corrected = fluorescence - neucoeff * neuropil

neuropil_all = np.load(plane_dir / "Fneu.npy")
...
session_info = {
    "session_id": session_id,
    "subject": session_dir.parent.name,
    "source_sampling_rate_hz": sampling_rate_hz,
    "source_neural_frames": int(neural_frame_count),
    "source_motion_frames": int(motion_raw.size),
    "interpolated_motion_frames": int(interpolated_frames),
    "retained_neurons": int(cell_mask.sum()),
    "source_rois": int(iscell.shape[0]),
    "complete_trials": int(trial_count),
    "discarded_binned_samples": int(binned_time.size - used_bins),
    "motion_quintile_boundaries": percentile_boundaries.tolist(),
}
```

iii. The trajectory does not explicitly acknowledge these costs. They follow from the AI’s decision to mirror the repository’s fluorescence routine and to save rich provenance in metadata.
