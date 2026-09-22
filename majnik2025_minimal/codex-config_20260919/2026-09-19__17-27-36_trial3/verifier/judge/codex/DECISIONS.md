# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively discovers complete subject/date session directories under the data root, requiring both neural fluorescence and motion-energy files. For every discovered session it loads Suite2p `ops.npy`, `iscell.npy`, `F.npy`, and `Fneu.npy`, plus `motion_energy_glob.npy` and `tstamps.npy`, then converts every session.

ii.
```python
sessions = sorted(
    path
    for path in data_dir.glob("*/20*_*")
    if (path / "suite2p" / "plane0" / "F.npy").is_file()
    and (path / "move_deve" / "motion_energy_glob.npy").is_file()
)
...
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
fluorescence_all = np.load(plane_dir / "F.npy")
neuropil_all = np.load(plane_dir / "Fneu.npy")
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. The trajectory says the AI first inspected the actual file layout, paper, methods, notebook, and repository. It chose the glob because supplied sessions follow subject/date layout, and completeness checks prevent partially populated directories from being treated as sessions.

## 1-b. How are the data split into subjects?

i. A subject is the parent directory name of a discovered session. Unique names are sorted and mapped to integer indices; every converted session gets its parent's index.

ii.
```python
subjects = sorted({path.parent.name for path in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_index[session_dir.parent.name])
```

iii. The AI observed that the `jm*` directories represented mice and that each mouse's sessions contained a consistent Track2p-matched population. Sorting gives deterministic subject order.

## 1-c. How are the data split into sessions?

i. Each dated directory matching `*/20*_*` and containing the required neural and motion files is one session. The full paths are sorted, giving subject then date/name order.

ii.
```python
def find_sessions(data_dir: Path) -> list[Path]:
    sessions = sorted(
        path for path in data_dir.glob("*/20*_*")
        if (path / "suite2p" / "plane0" / "F.npy").is_file()
        and (path / "move_deve" / "motion_energy_glob.npy").is_file()
    )
```

iii. The trajectory reports checking every supplied session and using stable chronological/lexicographic ordering. Each dated directory corresponds to a daily recording.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping 60-second trials. At 30 Hz with 10-frame bins, each trial contains 180 bins. Only complete trials are retained; trailing bins are discarded, and sessions with fewer than two complete trials cause an error.

ii.
```python
binned_rate_hz = sampling_rate_hz / SOURCE_FRAMES_PER_BIN
bins_per_trial = int(round(TRIAL_SECONDS * binned_rate_hz))
trial_count = binned_time.size // bins_per_trial
if trial_count < 2:
    raise ValueError(f"Fewer than two complete trials in {session_dir}")
for trial_index in range(trial_count):
    start = trial_index * bins_per_trial
    stop = start + bins_per_trial
```

iii. The instructions explicitly require 60-second trials and at least two trials per session. Because the recordings are continuous rather than naturally trial-based, the AI chose fixed, complete, non-overlapping windows.

## 1-e. How are trials filtered based on quality controls?

i. No individual trial quality metric is applied. Incomplete trailing trials are discarded, and a whole session is rejected if it yields fewer than two complete trials. Stream-length and sampling-rate checks must also pass before trials are formed.

ii.
```python
trial_count = binned_time.size // bins_per_trial
if trial_count < 2:
    raise ValueError(f"Fewer than two complete trials in {session_dir}")
used_bins = trial_count * bins_per_trial
```

iii. The trajectory does not identify a paper-defined trial QC because these are artificial windows. The two-trial check directly enforces the downstream decoder requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` fluorescence and `Fneu.npy` neuropil traces. `iscell.npy` supplies the ROI probability mask, and `ops.npy["fs"]` supplies the sampling rate used by baseline processing.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
cell_mask = iscell[:, 1] > ISCELL_THRESHOLD
fluorescence_all = np.load(plane_dir / "F.npy")
neuropil_all = np.load(plane_dir / "Fneu.npy")
```

iii. The AI traced these variables through the supplied Track2p GUI code and inspected the arrays and Suite2p settings before implementation.

## 2-b. How is the `neural` data processed?

i. ROIs are masked first. The AI then reproduces the Track2p GUI `F_processing` maximin baseline: neuropil coefficient 0.0, Gaussian smoothing with sigma 10 frames, a 60-second minimum then maximum filter, and subtraction of that baseline. It does not divide by baseline, deconvolve, or z-score. Finally, it averages non-overlapping groups of 10 frames and casts to `float32`.

ii.
```python
neucoeff = 0.0
corrected = fluorescence - neucoeff * neuropil
baseline = gaussian_filter(corrected, [0.0, sigma_frames])
window_frames = int(baseline_window_seconds * sampling_rate_hz)
baseline = minimum_filter1d(baseline, window_frames)
baseline = maximum_filter1d(baseline, window_frames)
return corrected - baseline
...
binned_neural = average_consecutive_bins(
    corrected, SOURCE_FRAMES_PER_BIN,
).astype(np.float32, copy=False)
```

iii. The trajectory explicitly says the AI selected the repository's “exact fluorescence routine,” interpreting its default as `F-F0` with `neucoeff=0.0`, and intentionally avoided z-scoring and deconvolution. It considered this the applicable implementation behind the paper's processing description.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are retained when the second `iscell.npy` column (cell probability) is strictly greater than 0.5. Shape consistency and presence of at least one retained ROI are checked. In the supplied data every ROI passes, so the original arrays are retained without an indexing copy.

ii.
```python
cell_mask = iscell[:, 1] > ISCELL_THRESHOLD
if not np.any(cell_mask):
    raise ValueError(...)
...
if np.all(cell_mask):
    fluorescence = fluorescence_all
    neuropil = neuropil_all
else:
    fluorescence = fluorescence_all[cell_mask]
    neuropil = neuropil_all[cell_mask]
```

iii. The AI states that the paper uses Suite2p's default cell classification and confirmed empirically that all distributed, already Track2p-reduced ROIs pass probability 0.5. The conditional avoids an unnecessary full-array copy in this dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins are segmented from session start into consecutive 60-second windows. Each returned trial begins at its own artificial trial-start event, covers offsets 0–60 seconds, and uses the identical slice boundaries as input and output. The elapsed-time covariate itself remains session-relative.

ii.
```python
start = trial_index * bins_per_trial
stop = start + bins_per_trial
neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
...
"temporal_alignment_event": (
    "start of each non-overlapping 60-second trial; decoder time input "
    "remains elapsed time from session start"
),
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. There is no experimental stimulus event in the continuous recording. The AI therefore treats the start of each instruction-required artificial window as the alignment event and records that convention in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Source data at 30 Hz is averaged over non-overlapping groups of 10 frames, producing 3 Hz data and 333.333 ms bins. Any final source frames insufficient for a complete 10-frame bin are dropped.

ii.
```python
SOURCE_FRAMES_PER_BIN = 10
...
complete_frames = values.shape[-1] // bin_frames * bin_frames
return values.reshape(new_shape).mean(axis=-1)
...
time_bin_ms = 1000.0 * SOURCE_FRAMES_PER_BIN / EXPECTED_SOURCE_FS
```

iii. The trajectory cites the paper's decoding method: neural and behavior traces are slightly denoised by averaging 10 consecutive timestamps. Applying identical bins preserves alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from neural frame indices and the per-session sampling rate in `ops.npy`; no recorded timestamp stream is used for the input.

ii.
```python
sampling_rate_hz = float(ops["fs"])
frame_times = np.arange(neural_frame_count, dtype=np.float64) / sampling_rate_hz
```

iii. The AI verified a stable 30 Hz neural acquisition rate. With uniform acquisition, frame number divided by sampling rate gives elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It computes the time of every raw neural frame, averages each 10-frame group, casts to `float32`, and reshapes each trial to `(1, 180)`. Thus values represent bin centers (the first is 0.15 s) and continue increasing across trial boundaries.

ii.
```python
frame_times = np.arange(neural_frame_count, dtype=np.float64) / sampling_rate_hz
binned_time = average_consecutive_bins(
    frame_times, SOURCE_FRAMES_PER_BIN,
).astype(np.float32, copy=False)
...
input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
```

iii. The code comment explains that mean raw-frame times represent the temporal center of each 10-frame bin, while retaining elapsed time from session start rather than resetting each trial.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is built from the same number of neural frames, binned with the same 10-frame function, checked to have the same binned length, and sliced with the same `start:stop` indices as neural data.

ii.
```python
if not (
    binned_neural.shape[1] == binned_motion.size
    == motion_labels.size == binned_time.size
):
    raise ValueError(...)
...
neural_trials.append(...binned_neural[:, start:stop])
input_trials.append(...binned_time[None, start:stop])
```

iii. The trajectory reports validating trial shapes and centered time endpoints in a single-session test and then validating all sessions.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from `move_deve/motion_energy_glob.npy`. `move_deve/tstamps.npy` is used to recover missing camera-trigger positions and align motion samples to neural frames.

ii.
```python
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
aligned_motion, interpolated_frames = align_motion_to_neural_frames(
    motion_raw, timestamps, neural_frame_count,
)
```

iii. The AI identified motion energy as the paper's precomputed global consecutive-frame squared-pixel-difference signal. It inspected timestamp gaps and found that they exactly accounted for the neural/motion length deficits.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion and timestamps are flattened and validated. If camera samples are missing, timestamp interval multiples determine their trigger indices and `np.interp` linearly fills the missing values. The aligned continuous trace is averaged in 10-frame bins, and those binned values are converted to categorical quintile labels.

ii.
```python
intervals = np.diff(timestamps)
nominal_interval = np.median(intervals)
missing_after = np.maximum(
    np.rint(intervals / nominal_interval).astype(np.int64) - 1, 0,
)
...
aligned = np.interp(
    np.arange(neural_frame_count, dtype=np.float64), trigger_indices, motion,
)
...
binned_motion = average_consecutive_bins(aligned_motion, SOURCE_FRAMES_PER_BIN)
motion_labels, percentile_boundaries = motion_quintile_labels(binned_motion)
```

iii. The AI chose interpolation to preserve complete 60-second trials rather than lose a trial due to a few dropped frames. It bins before categorization because averaging categorical labels would not represent average motion energy.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four boundaries—the 20th, 40th, 60th, and 80th percentiles—are computed separately from each session's binned motion trace. `np.digitize(..., right=False)` produces integer categories 0–4. Non-unique boundaries are rejected.

ii.
```python
boundaries = np.percentile(binned_motion, MOTION_PERCENTILES)
if np.any(np.diff(boundaries) <= 0):
    raise ValueError(...)
labels = np.digitize(binned_motion, boundaries, right=False).astype(np.int64)
```

iii. The instructions demand five equal-percentile bins selected per session. The trajectory confirms exactly balanced session-specific quintiles after conversion.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera timestamps identify each observed sample's microscope-trigger index. Missing trigger positions are interpolated to reach exactly the neural frame count. Motion and neural signals then undergo identical 10-frame binning and identical trial slicing, with equality checks before slicing.

ii.
```python
trigger_indices = np.arange(motion.size, dtype=np.int64)
trigger_indices[1:] += np.cumsum(missing_after)
...
aligned = np.interp(np.arange(neural_frame_count), trigger_indices, motion)
...
output_trials.append(np.ascontiguousarray(motion_labels[None, start:stop]))
```

iii. The trajectory says the camera was microscope-triggered and timestamp gaps exactly matched all absent frames. Timestamp-derived interpolation was selected as a more precise alignment than truncating streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are inferred from timestamp gaps and linearly interpolated. The code rejects mismatched motion/timestamp lengths, extra motion frames, invalid timestamps, unexplained deficits, wrong final trigger positions, unexpected sampling rates, array-shape mismatches, empty ROI selections, degenerate percentiles, and post-binning misalignment. Incomplete final bins/trials are dropped.

ii.
```python
if int(missing_after.sum()) != missing_count:
    raise ValueError(...)
if trigger_indices[-1] != neural_frame_count - 1:
    raise ValueError(...)
...
if fluorescence_all.shape != neuropil_all.shape:
    raise ValueError(...)
...
complete_frames = values.shape[-1] // bin_frames * bin_frames
```

iii. The AI examined all affected sessions and verified that timestamp gaps exactly explained missing camera frames. It favored explicit failures for unexplained corruption and interpolation only where supported by synchronization data.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session maximin baseline correction is the dominant computation: Gaussian filtering followed by large-window minimum and maximum filters over every neuron's complete trace. Loading the large fluorescence arrays and serializing the roughly 2 GB converted pickle are also substantial I/O costs.

ii.
```python
baseline = gaussian_filter(corrected, [0.0, sigma_frames])
baseline = minimum_filter1d(baseline, window_frames)
baseline = maximum_filter1d(baseline, window_frames)
...
pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory anticipated baseline processing as the expensive stage, ran the full 41-session conversion as a long process, and performed integrity and decoder checks on the resulting large artifact.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python trial loop could be replaced by reshaping/transposing the complete-trial prefix and generating trial arrays from views or a batched array. The outer session loop is naturally serial in this implementation but could be parallelized. Missing-frame interpolation and temporal binning are already vectorized.

ii.
```python
for trial_index in range(trial_count):
    start = trial_index * bins_per_trial
    stop = start + bins_per_trial
    neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
    input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
    output_trials.append(np.ascontiguousarray(motion_labels[None, start:stop]))
```

iii. The trajectory does not explicitly discuss vectorizing this loop. Its optimization effort instead avoided a full boolean-index copy when every cell passed and used vectorized `np.interp` for all missing frames.

## 6-c. What processing does the code repeat multiple times?

i. Every session independently repeats file loading, ROI checks, maximin filtering, motion alignment, 10-frame averaging, percentile computation, and trial construction. The same binning helper is intentionally applied separately to neural, motion, and time streams. No session is processed twice.

ii.
```python
for session_number, session_dir in enumerate(session_dirs, start=1):
    converted, info = convert_session(session_dir)
...
binned_neural = average_consecutive_bins(corrected, SOURCE_FRAMES_PER_BIN)
binned_motion = average_consecutive_bins(aligned_motion, SOURCE_FRAMES_PER_BIN)
binned_time = average_consecutive_bins(frame_times, SOURCE_FRAMES_PER_BIN)
```

iii. The repeated per-stream binning is deliberate to use identical complete-frame boundaries. The trajectory describes a single preprocessing pass across all sessions, followed by validation rather than redundant conversion work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Because `neucoeff=0.0`, loading, potentially masking, and shape-checking the entire `Fneu.npy` array does not affect the neural result. The continuous binned motion values are discarded after labels and percentile metadata are produced. Contiguously copying every trial increases work and memory, though it makes stored trial arrays self-contained. Rich `session_info` metadata is not consumed by the decoder.

ii.
```python
neuropil_all = np.load(plane_dir / "Fneu.npy")
...
corrected = fluorescence - neucoeff * neuropil  # neucoeff = 0.0
...
motion_labels, percentile_boundaries = motion_quintile_labels(binned_motion)
...
neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
```

iii. The trajectory does not call these steps unnecessary. It loaded neuropil to reproduce and document the repository routine faithfully, retained diagnostic metadata for auditability, and used contiguous arrays as a conservative serialization/downstream-compatibility choice.
