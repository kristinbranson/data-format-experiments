# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing for `jm*/20*` directories under the data root, filtering to only those that have both `suite2p/plane0/F.npy` and `move_deve/motion_energy_glob.npy`. For each session it loads the fluorescence (`F.npy`), `iscell.npy`, `ops.npy`, motion energy (`motion_energy_glob.npy`), and timestamps (`tstamps.npy`).

ii.
```python
def discover_sessions(data_root: Path) -> list[Path]:
    sessions = sorted(
        p
        for p in data_root.glob("jm*/20*")
        if (p / "suite2p/plane0/F.npy").is_file()
        and (p / "move_deve/motion_energy_glob.npy").is_file()
    )
    ...
    return sessions

# Per session:
fluorescence = np.load(plane_dir / "F.npy")
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
raw_motion = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. The agent examined the data directory structure and identified the `jm*/20*` pattern for subject/session directories. It verified sessions have both neural and behavioral data before including them. The agent noted it uses `tstamps.npy` (not `interframe_int.npy`) for motion alignment.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the parent directory names of discovered sessions, sorted alphabetically. A mapping from subject name to index is constructed.

ii.
```python
subjects = sorted({session.parent.name for session in sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
```

iii. Each `jm*` directory represents one mouse. The AI extracts subject names from the session paths rather than scanning the top-level directory, but the result is equivalent.

## 1-c. How are the data split into sessions?

i. Each dated subdirectory (`20*`) within a subject folder is treated as one session, provided it contains both suite2p and motion energy data files.

ii.
```python
sessions = sorted(
    p for p in data_root.glob("jm*/20*")
    if (p / "suite2p/plane0/F.npy").is_file()
    and (p / "move_deve/motion_energy_glob.npy").is_file()
)
```

iii. The agent identified from the data README and paper that each subdirectory corresponds to one daily recording session. Sorting ensures deterministic ordering.

## 1-d. How are the data split into trials?

i. Sessions are split into contiguous, non-overlapping 60-second trials. At the native 30 Hz rate, each trial is 1800 frames, which after 10-frame binning becomes 180 bins per trial. The AI raises a ValueError if the session length is not an exact multiple of the trial length (1800 frames).

ii.
```python
trial_frames = int(TRIAL_SECONDS * FS_HZ)  # 1800
trial_bins = trial_frames // AVERAGE_FRAMES  # 180

n_neurons, n_frames = fluorescence.shape
if n_frames % trial_frames:
    raise ValueError(
        f"{session_dir} has {n_frames} frames, not an integer number of trials"
    )

for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
```

iii. The agent followed the instruction to "split sessions into 60-second trials." It validates that all sessions divide evenly into 60-second segments rather than discarding remainders.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All trials from valid sessions are included.

ii. N/A

iii. The agent did not implement any trial filtering. The data is continuous without a natural trial structure, so there is no basis for trial-level quality control beyond ensuring complete 60-second segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived solely from `F.npy` (raw fluorescence). Unlike the reference, `Fneu.npy` is not loaded because the AI uses a neuropil coefficient of 0.0 (matching the Track2p code's default).

ii.
```python
fluorescence = np.load(plane_dir / "F.npy")
```

iii. The agent examined Track2p's `F_processing` function in `data_management.py` and found that the default `neucoeff` parameter is 0.0. It explicitly chose to follow the Track2p default rather than the suite2p default of 0.7.

## 2-b. How is the `neural` data processed?

i. The AI reimplements Track2p's `F_processing` function: Gaussian smoothing (sigma=10 frames along time), followed by minimum filter (60s window), then maximum filter (60s window), then subtraction of this baseline from the raw fluorescence. No neuropil subtraction is applied (neucoeff=0). The result is then averaged in non-overlapping 10-frame bins.

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

iii. The agent stated: "I found the repository's actual dF/F0 implementation. It baseline-corrects fluorescence with the Suite2p-style maximin baseline (Gaussian sigma=10 frames, 60-second min/max window) and subtracts that baseline." It chose to reimplement this directly using scipy rather than calling suite2p's `dcnv.preprocess`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI validates that all neurons in `iscell.npy` have `iscell[:, 0] == 1` (confirming they are pre-filtered Track2p output), but does not apply additional filtering. All neurons from `F.npy` are included.

ii.
```python
if len(iscell) != fluorescence.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(
        f"{session_dir} is not the expected prefiltered Track2p output"
    )
```

iii. The agent recognized that the Track2p output already contains only tracked neurons (those present across all days), so no additional ROI filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 60-second trial within the session. Since trials are artificial segments of a continuous recording, no event-based alignment is applied. The metadata records `off_start=0.0` and `off_end=60.0`.

ii.
```python
'temporal_alignment_event': 'start of each consecutive 60-second trial',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The agent noted there is no stimulus event to align to. Trials are contiguous segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy data are averaged in non-overlapping bins of 10 consecutive frames, converting from 30 Hz to 3 Hz (333.33 ms bins). This is applied before discretization of motion energy.

ii.
```python
AVERAGE_FRAMES = 10
FS_HZ = 30.0

neural_binned = average_consecutive(
    baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
)
motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)

'time_bin_size': 1000.0 * AVERAGE_FRAMES / FS_HZ,  # 333.33 ms
```

iii. The agent cited the paper's methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and the bin duration (AVERAGE_FRAMES / FS_HZ = 1/3 second per bin).

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32)
    * (AVERAGE_FRAMES / FS_HZ)
)
```

iii. Since the frame rate is a constant 30 Hz and there are no explicit timestamps in the neural data, computing time from bin indices is straightforward and accurate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Simple multiplication of bin index by bin duration. No additional processing.

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32)
    * (AVERAGE_FRAMES / FS_HZ)
)
session_input.append(elapsed_seconds[None, start:stop])
```

iii. N/A

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time array has the same length as the binned neural data (they share the same bin index), so alignment is inherent. Each trial slice uses the same start:stop indices for both neural and input arrays.

ii.
```python
session_neural.append(neural_binned[:, start:stop])
session_input.append(elapsed_seconds[None, start:stop])
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Camera timestamps from `tstamps.npy` are used to interpolate the motion energy onto the imaging clock.

ii.
```python
raw_motion = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. The agent identified from the data README that `motion_energy_glob.npy` contains pre-computed global motion energy and that `tstamps.npy` provides the camera frame timestamps needed for alignment.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) motion energy is interpolated from camera timestamps onto the regular 30 Hz imaging clock using `np.interp`; (2) the interpolated signal is averaged in 10-frame bins; (3) the binned signal is discretized into 5 equal-percentile bins using per-session quintile edges via `np.searchsorted`.

ii.
```python
aligned_motion, n_missing_camera_frames = align_motion_to_imaging(
    raw_motion, timestamps, n_frames
)
motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)

quintile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
motion_classes = np.searchsorted(
    quintile_edges, motion_binned, side="right"
).astype(np.int64)
```

iii. The agent chose `np.interp` with timestamps rather than insertion-based interpolation using `interframe_int.npy`, arguing this handles dropped frames more robustly. Discretization uses 4 inner quintile edges with `searchsorted(side="right")` to produce labels 0-4.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories using session-specific quintile edges (20th, 40th, 60th, 80th percentiles). `np.searchsorted` with `side="right"` assigns values to bins 0-4.

ii.
```python
quintile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
motion_classes = np.searchsorted(
    quintile_edges, motion_binned, side="right"
).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins, selected per session." The AI computes quintiles per session and uses searchsorted for consistent bin assignment.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy from the camera is interpolated onto the regular 30 Hz imaging clock using `np.interp` with camera timestamps. This produces a motion energy value for every imaging frame. After 10-frame binning, the binned motion energy array has the same length as the binned neural data, ensuring frame-for-frame alignment.

ii.
```python
def align_motion_to_imaging(motion, timestamps, n_imaging_frames):
    intervals = np.diff(timestamps)
    frame_step = float(np.median(intervals))
    imaging_clock = timestamps[0] + np.arange(n_imaging_frames) * frame_step
    aligned = np.interp(imaging_clock, timestamps, motion)
    return aligned, missing_frames
```

iii. The agent noted: "Camera motion is interpolated from its recorded timestamps onto the regular two-photon clock. This handles the missing camera frames documented in the data README while retaining the complete neural recording." The data README states the microscope triggers camera acquisition, so timestamps provide the synchronization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (dropped frames) are handled via timestamp-based interpolation using `np.interp`, which naturally fills gaps. The code also validates that `iscell` is all 1s, that the sampling rate matches 30 Hz, and that session length is an exact multiple of the trial length. Sessions that fail these checks raise errors.

ii.
```python
if len(iscell) != fluorescence.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(...)
if not np.isclose(float(ops["fs"]), FS_HZ):
    raise ValueError(...)
if n_frames % trial_frames:
    raise ValueError(...)
```

iii. The agent chose strict validation (raising errors) rather than graceful handling for unexpected data issues, while using interpolation for the expected issue of dropped camera frames.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction (`baseline_correct_fluorescence`), which applies Gaussian filtering and min/max filtering over the full session length for every neuron using scipy. Loading `.npy` files is also I/O-intensive.

ii. N/A

iii. The scipy-based baseline correction operates on the full fluorescence matrix (n_neurons x n_frames) with large filter windows.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop is sequential but minimal since it just creates array views. The main processing (baseline correction, binning, interpolation) is already vectorized.

ii. N/A

iii. The code is well-vectorized overall. The trial-splitting loop only creates slices/views, which is inherently sequential but very fast.

## 6-c. What processing does the code repeat multiple times?

i. N/A. Each processing step is performed once per session.

ii. N/A

iii. The code processes each session in a single pass without redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and validates `iscell.npy` and `ops.npy` for sanity checks, but these files are not used in the actual neural processing (since neucoeff=0, Fneu is not needed). The `session_info` metadata includes detailed per-session statistics that may not be used downstream.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. These validations serve as safety checks rather than being necessary for the processing pipeline.
