# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all subject folders under `/app/data` whose names start with `jm`, sorts them, then iterates through sorted session subdirectories whose names begin with digits. For each session it loads neural traces (`F.npy`, `Fneu.npy`), QC/metadata files (`iscell.npy`, `ops.npy`), and behavior files (`motion_energy_glob.npy`, `interframe_int.npy`). Trials are not loaded directly; they are created later by splitting the full-session arrays.

ii.
```python
DATA_ROOT = Path("/app/data")
...
subjects = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")])
...
session_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
...
F = np.load(plane_dir / "F.npy", allow_pickle=True)
Fneu = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
motion = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. In the trajectory, the AI said the dataset was "organized as Suite2p outputs plus `move_deve` motion-energy traces per session" and that it was inspecting the methods and repository to "match the original processing" before implementing the converter.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories under `/app/data` whose names start with `jm`, sorted lexicographically. A `subject_to_idx` mapping is then built from that ordered list.

ii.
```python
subjects = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The trajectory shows the AI relying on the provided data organization notes, which describe one folder per subject and use `jm031`, `jm032`, etc. as mouse IDs.

## 1-c. How are the data split into sessions?

i. Sessions are split as sorted subdirectories within each subject folder, restricted to directory names whose first four characters are digits. Each such folder is treated as one session.

ii.
```python
for subject in subjects:
    subject_dir = DATA_ROOT / subject
    session_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    for session_dir in session_dirs:
        ...
```

iii. In the trajectory, the AI inspected the data README and session folder names, then reported it was using "all 41 Track2p-exported sessions across 6 mice."

## 1-d. How are the data split into trials?

i. The AI treats each session as a continuous recording, bins it in 10-frame steps, then splits the binned arrays into contiguous non-overlapping 60-second trials. At 30 Hz with 10-frame bins, each trial is 180 bins long. Unlike the reference solution, this implementation requires exact divisibility and raises an error if a session cannot be evenly partitioned.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
TRIAL_SECONDS = 60
...
trial_bins = int(TRIAL_SECONDS * FS / BIN_FRAMES)
...
def split_trials(x: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    if x.shape[-1] % trial_bins != 0:
        raise ValueError(f"Time axis length {x.shape[-1]} is not divisible by trial length {trial_bins}.")
    ntrials = x.shape[-1] // trial_bins
    return [x[..., i * trial_bins:(i + 1) * trial_bins] for i in range(ntrials)]
```

iii. The AI stated in the trajectory that it was implementing "10-frame binning, 60-second trial splitting" and later summarized that it "split each session into contiguous 60-second trials."

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial-level quality-control filter is applied. The AI does not discard trials based on behavior, signal quality, or motion. It only requires that the full session length be divisible by the chosen trial length after binning.

ii.
```python
def split_trials(x: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    if x.shape[-1] % trial_bins != 0:
        raise ValueError(f"Time axis length {x.shape[-1]} is not divisible by trial length {trial_bins}.")
    ntrials = x.shape[-1] // trial_bins
    return [x[..., i * trial_bins:(i + 1) * trial_bins] for i in range(ntrials)]
```

iii. The trajectory does not mention any trial exclusion criteria. The AI framed the sessions as continuous recordings that were simply segmented into fixed windows for the downstream decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the Suite2p fluorescence arrays `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii.
```python
F = np.load(plane_dir / "F.npy", allow_pickle=True)
Fneu = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
...
neural = suite2p_style_baseline_correct(F, Fneu, fs)
```

iii. In the trajectory, the AI said it would use "baseline-corrected `dF/F` rather than deconvolved spikes" and later summarized that it computed the neural signal from `F.npy` and `Fneu.npy`.

## 2-b. How is the `neural` data processed?

i. The AI computes a baseline-corrected fluorescence trace by subtracting `NEUCOEFF * Fneu` from `F`, with `NEUCOEFF = 0.0`, then applying a manual "maximin" baseline subtraction using a Gaussian filter followed by minimum and maximum filters over a 60-second window. It does not call Suite2p's `dcnv.preprocess`; instead it reimplements the filtering with SciPy.

ii.
```python
NEUCOEFF = 0.0
BASELINE = "maximin"
SIG_BASELINE = 10.0
WIN_BASELINE = 60.0
...
def suite2p_style_baseline_correct(F: np.ndarray, Fneu: np.ndarray, fs: float) -> np.ndarray:
    """Match the paper/repo preprocessing: neuropil coefficient 0 and maximin baseline subtraction."""
    Fc = F.astype(np.float32, copy=False) - NEUCOEFF * Fneu.astype(np.float32, copy=False)
    ...
    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    return (Fc - Flow).astype(np.float32, copy=False)
```

iii. The AI explicitly justified this in the trajectory: it said it found a bundled Track2p GUI helper that "matches Suite2p’s baseline correction with `neucoeff=0`, `baseline='maximin'`, `sig_baseline=10`, and a 60 s window," and later reiterated that it used "the same Suite2p-style baseline correction used in the repo."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out of the saved arrays, but it does impose QC assertions: every ROI must have `iscell[:, 0] == 1` and `iscell[:, 1] > 0.5`. If any ROI violates those conditions, the script raises an error rather than saving the session.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
if not np.all(iscell[:, 0] == 1):
    raise ValueError(f"Found non-cell ROIs in tracked output for {session_dir}")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"Found tracked ROIs below the paper's iscell threshold in {session_dir}")
```

iii. The AI first concluded that Track2p outputs already contained tracked cells, then later made the decision explicit: "Track2p already saved only `iscell > 0.5` tracked neurons, so I’m asserting that instead of silently ignoring the `iscell` array."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to session start. There is no event-triggered alignment; the full-session trace is binned from the start of the recording and then chopped into contiguous 60-second windows.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "session start",
    "off_start": None,
    "off_end": None,
    ...
}
```

iii. The AI described the decoder input as "session-start time bin centers" and summarized that the trials were contiguous session segments rather than stimulus-aligned epochs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral traces into non-overlapping bins of 10 imaging frames. At 30 Hz this gives a bin size of 333.33 ms. The rebinning is an averaging operation.

ii.
```python
BIN_FRAMES = 10
FS = 30.0
...
def mean_bin_time_series(x: np.ndarray, bin_frames: int) -> np.ndarray:
    if x.shape[-1] % bin_frames != 0:
        raise ValueError(f"Time axis length {x.shape[-1]} is not divisible by bin size {bin_frames}.")
    new_shape = x.shape[:-1] + (x.shape[-1] // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
...
time_bin_size_ms = 1000.0 * BIN_FRAMES / FS
...
"time_bin_size": float(time_bin_size_ms),
```

iii. The AI said in the trajectory that the methods pointed to "10-frame averaging before decoding" and later summarized that it "matched the paper’s decoder smoothing by averaging neural and motion signals in 10-frame bins."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not use a stored timestamp vector. It derives time from the frame index `np.arange(nframes)` and the imaging sampling rate `ops["fs"]`, with session length taken from `ops["nframes"]`.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
...
nframes = int(ops["nframes"])
fs = float(ops["fs"])
...
frame_times = np.arange(nframes, dtype=np.float32) / fs
```

iii. The trajectory shows the AI inspecting `ops["fs"]`, `ops["nframes"]`, and the decoder requirements, then deciding to use "session-start time bin centers as the decoder input."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI constructs per-frame time stamps in seconds from session start, then averages those frame times in the same 10-frame bins used for neural and motion data. This yields bin-center times such as 0.15 s, 0.4833 s, etc.

ii.
```python
frame_times = np.arange(nframes, dtype=np.float32) / fs
time_binned = mean_bin_time_series(frame_times[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)
...
input_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(time_binned[np.newaxis, :], trial_bins)]
```

iii. The AI explicitly described this choice in the trajectory as using "session-start time bin centers as the decoder input." It also noted an earlier formatting bug where it had accidentally added one axis too many to the time input, then fixed it.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: the AI generates one frame-time value per imaging frame, bins it with the same `mean_bin_time_series` function used for neural and motion data, and then splits the binned time array into the same trial boundaries as the neural data.

ii.
```python
neural_binned = mean_bin_time_series(neural, BIN_FRAMES).astype(np.float32, copy=False)
...
time_binned = mean_bin_time_series(frame_times[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)
...
neural_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(neural_binned, trial_bins)]
input_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(time_binned[np.newaxis, :], trial_bins)]
```

iii. The trajectory does not spell this out in detail, but the AI consistently described the input as session-start bin times matched to the same 10-frame bins and 60-second trial segmentation as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy output is derived from `motion_energy_glob.npy`, with `interframe_int.npy` used to infer missing camera frames before alignment to imaging.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
...
motion_aligned = repair_motion_trace(motion, interframe_int, nframes)
```

iii. The AI said it was checking "how the motion timestamps encode missing camera frames" and later summarized that it repaired dropped video frames using `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first repairs any dropped camera frames by reconstructing a frame-aligned trace from inter-frame intervals, filling missing positions with NaNs and linearly interpolating them. It then averages the repaired motion trace into 10-frame bins and later discretizes the binned trace into per-session quintiles.

ii.
```python
def repair_motion_trace(motion: np.ndarray, interframe_int: np.ndarray, target_len: int) -> np.ndarray:
    motion = motion.astype(np.float32, copy=False)
    if len(motion) == target_len:
        return motion

    median_ifi = float(np.median(interframe_int))
    gap_sizes = np.rint(interframe_int / median_ifi).astype(int)
    ...
    repaired = [float(motion[0])]
    for i, gap in enumerate(gap_sizes):
        if gap > 1:
            repaired.extend([np.nan] * (gap - 1))
        repaired.append(float(motion[i + 1]))
    ...
    nan_mask = np.isnan(repaired)
    if nan_mask.any():
        valid_idx = np.flatnonzero(~nan_mask)
        repaired[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid_idx, repaired[valid_idx]).astype(np.float32)
    ...
motion_binned = mean_bin_time_series(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)
motion_labels = motion_to_quintiles(motion_binned)
```

iii. The AI justified this in several trajectory messages: it said it was implementing "timestamp-based motion interpolation," later summarized that it "repaired only sessions with dropped video frames by using `interframe_int.npy` to insert missing positions and linearly interpolate the motion trace back to the imaging frame count," and said 10-frame averaging should occur before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds the 10-frame-binned motion trace into five per-session equal-quantile bins. It uses the 20th, 40th, 60th, and 80th percentiles as edges and assigns labels 0 through 4 with `np.digitize`.

ii.
```python
def motion_to_quintiles(motion_binned: np.ndarray) -> np.ndarray:
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(motion_binned, edges, right=False).astype(np.int64)
    return labels[np.newaxis, :]
```

iii. The AI repeatedly described the output as "per-session quintiles" and summarized that it "discretized the 10-frame-averaged motion energy into 5 per-session quintiles as the decoder output."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data by repairing the motion trace up to the imaging frame count `nframes`, then averaging neural and motion traces with the same 10-frame binning function and splitting them with the same trial boundaries.

ii.
```python
nframes = int(ops["nframes"])
...
motion_aligned = repair_motion_trace(motion, interframe_int, nframes)
...
neural_binned = mean_bin_time_series(neural, BIN_FRAMES).astype(np.float32, copy=False)
motion_binned = mean_bin_time_series(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)
...
output_trials = [trial.astype(np.int64, copy=False) for trial in split_trials(motion_labels, trial_bins)]
```

iii. In the trajectory, the AI said it was checking timestamp irregularities "to decide the alignment," then summarized that it interpolated motion back to the imaging frame count and used the same 10-frame bins.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI only explicitly handles missing behavior frames. If the motion trace is shorter than the imaging trace, it uses `interframe_int.npy` to infer gap sizes, inserts missing positions, linearly interpolates NaNs, and pads or truncates the repaired trace to the target length if needed. It raises errors for invalid gap ratios, unexpected frame-rate mismatches, neural shape mismatches, and session lengths that are not divisible by the chosen bin or trial sizes.

ii.
```python
if len(motion) == target_len:
    return motion
...
gap_sizes = np.rint(interframe_int / median_ifi).astype(int)
if np.any(gap_sizes < 1):
    raise ValueError("Found invalid inter-frame interval ratio while repairing motion trace.")
...
if repaired.size < target_len:
    repaired = np.pad(repaired, (0, target_len - repaired.size), constant_values=np.nan)
elif repaired.size > target_len:
    repaired = repaired[:target_len]
...
if repaired.size != target_len:
    raise ValueError(f"Repaired motion length {repaired.size} does not match target length {target_len}.")
...
if x.shape[-1] % trial_bins != 0:
    raise ValueError(f"Time axis length {x.shape[-1]} is not divisible by trial length {trial_bins}.")
```

iii. The AI justified the missing-frame handling from the README and trajectory notes about "timestamp-based motion interpolation" and "repair[ing] only sessions with dropped video frames." It did not give a separate explicit justification for the pad/truncate fallback or the divisibility errors.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the manual maximin baseline correction applied to every neuron trace in every session. Loading the large `.npy` arrays is also nontrivial, but the main computational work is the Gaussian/minimum/maximum filtering over full-session calcium traces.

ii.
```python
Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
```

iii. The AI explicitly said in the trajectory: "The baseline correction is the expensive part because it runs a 60-second maximin filter over every neuron trace for all 41 sessions."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidate is the loop in `repair_motion_trace`, which appends one segment at a time to a Python list while reconstructing the motion trace from inter-frame intervals. The list-comprehension based `split_trials` path is also loop-based, though less costly.

ii.
```python
repaired = [float(motion[0])]
for i, gap in enumerate(gap_sizes):
    if gap > 1:
        repaired.extend([np.nan] * (gap - 1))
    repaired.append(float(motion[i + 1]))
...
return [x[..., i * trial_bins:(i + 1) * trial_bins] for i in range(ntrials)]
```

iii. The trajectory does not discuss vectorization opportunities directly. This is inferred from the final code structure.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same `split_trials` logic separately for neural, input time, and output motion arrays. It also repeats per-trial `astype(..., copy=False)` conversions across those three streams after splitting.

ii.
```python
neural_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(neural_binned, trial_bins)]
input_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(time_binned[np.newaxis, :], trial_bins)]
output_trials = [trial.astype(np.int64, copy=False) for trial in split_trials(motion_labels, trial_bins)]
```

iii. The trajectory does not explicitly mention this. It is inferred from the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `iscell.npy` and performs assertions on it but does not otherwise use it in the saved dataset. It also constructs a fairly detailed `session_info` metadata structure, including session date and raw motion-frame counts, which is not used by the decoder itself.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
if not np.all(iscell[:, 0] == 1):
    raise ValueError(...)
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(...)
...
session_info.append(
    {
        "subject": subject,
        "session": session_dir.name,
        "date": session_dir.name.split("_")[0],
        "n_neurons": int(neural.shape[0]),
        "n_frames_imaging": nframes,
        "n_frames_motion_raw": int(len(motion)),
        "n_frames_motion_missing": int(nframes - len(motion)),
        "duration_sec": float(nframes / fs),
        "n_trials": len(neural_trials),
    }
)
```

iii. The trajectory only partially justifies this. The `iscell` assertions were added because the AI wanted to make its tracked-cell assumption explicit; the extra metadata appears to have been added for bookkeeping rather than for downstream decoder use.
