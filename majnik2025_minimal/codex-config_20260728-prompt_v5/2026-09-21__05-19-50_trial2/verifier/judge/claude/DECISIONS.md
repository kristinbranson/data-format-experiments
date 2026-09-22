# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories starting with `jm` in the data directory. Sessions are subdirectories within each subject folder filtered to those whose name starts with 4 digits (date format). Neural data is loaded from suite2p output files (`F.npy`, `Fneu.npy`), along with `iscell.npy` and `ops.npy` for validation. Motion energy is loaded from `motion_energy_glob.npy` and `interframe_int.npy` for dropped-frame repair.

ii.
```python
subjects = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")])
# ...
session_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
# ...
F = np.load(plane_dir / "F.npy", allow_pickle=True)
Fneu = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
motion = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The AI explored the data directory structure and identified the standard organization: subject folders (`jm*`) containing session subfolders, each with suite2p output and motion energy files. The session filter (`name[:4].isdigit()`) ensures only date-formatted session directories are included.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data root, sorted alphabetically.

ii.
```python
subjects = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Each `jm*` directory represents one mouse, consistent with the data README which states folder names correspond to subject IDs.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder whose name starts with 4 digits (a date), sorted alphabetically. The session count and frame rate are validated using `ops.npy`.

ii.
```python
session_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
# ...
nframes = int(ops["nframes"])
fs = float(ops["fs"])
if fs != FS:
    raise ValueError(f"Unexpected sampling rate {fs} in {session_dir}")
```

iii. The AI noted that session folder names follow YYYY-MM-DD format and used the first 4 characters being digits as a filter. The `ops.npy` file is used to validate frame counts and sampling rate.

## 1-d. How are the data split into trials?

i. No natural trial structure exists in this dataset. Trials are defined as 60-second non-overlapping segments. At 30 Hz with 10-frame binning, each trial is 180 bins. The code requires exact divisibility of session length by trial length (raises an error if not).

ii.
```python
trial_bins = int(TRIAL_SECONDS * FS / BIN_FRAMES)  # 60 * 30 / 10 = 180

def split_trials(x: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    if x.shape[-1] % trial_bins != 0:
        raise ValueError(f"Time axis length {x.shape[-1]} is not divisible by trial length {trial_bins}.")
    ntrials = x.shape[-1] // trial_bins
    return [x[..., i * trial_bins:(i + 1) * trial_bins] for i in range(ntrials)]
```

iii. Per the instructions, trials are defined as 60-second segments. Sessions are either 1200s (20 min) or 1800s (30 min), both exactly divisible by 60s, so no remainder frames need to be discarded.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are included.

ii. N/A

iii. There is no stimulus-driven trial structure or trial-level quality metric to filter on.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from the `suite2p/plane0` subdirectory.

ii.
```python
F = np.load(plane_dir / "F.npy", allow_pickle=True)
Fneu = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces. The AI identified these as the correct source from the paper's methods section.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction with coefficient 0.0 (i.e., no neuropil subtraction: `Fc = F - 0.0 * Fneu`), followed by a manual reimplementation of suite2p's maximin baseline correction: gaussian smoothing (sigma=10), then minimum filter (60s window), then maximum filter (60s window), and subtraction of the resulting baseline.

ii.
```python
NEUCOEFF = 0.0
# ...
def suite2p_style_baseline_correct(F: np.ndarray, Fneu: np.ndarray, fs: float) -> np.ndarray:
    Fc = F.astype(np.float32, copy=False) - NEUCOEFF * Fneu.astype(np.float32, copy=False)
    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    return (Fc - Flow).astype(np.float32, copy=False)
```

iii. The AI found the Track2p GUI's `F_processing` function which uses `neucoeff=0.0` and manually implements the maximin baseline correction. The AI stated: "I found the bundled `dF/F` helper used by the Track2p GUI: it matches Suite2p's baseline correction with `neucoeff=0`, `baseline='maximin'`, `sig_baseline=10`, and a 60 s window." The AI chose to follow the Track2p repository code rather than suite2p defaults.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. The AI loads `iscell.npy` and asserts that all tracked ROIs are classified as cells (iscell == 1) with probability > 0.5, but does not remove any. The reasoning is that Track2p's output already contains only successfully tracked neurons that passed the iscell threshold.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
if not np.all(iscell[:, 0] == 1):
    raise ValueError(f"Found non-cell ROIs in tracked output for {session_dir}")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"Found tracked ROIs below the paper's iscell threshold in {session_dir}")
```

iii. The AI stated: "Track2p already saved only `iscell > 0.5` tracked neurons, so I'm asserting that instead of silently ignoring the `iscell` array." The paper's methods confirm that ROIs above the 0.5 threshold are considered true cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. The recording is continuous and trials are contiguous 60-second segments, so no event-based alignment is needed.

ii.
```python
"temporal_alignment_event": "session start",
"off_start": None,
"off_end": None,
```

iii. There is no stimulus event to align to. The continuous recording is segmented from its beginning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms time bin). Binning is applied before motion energy discretization.

ii.
```python
BIN_FRAMES = 10
# ...
neural_binned = mean_bin_time_series(neural, BIN_FRAMES)
motion_binned = mean_bin_time_series(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
# ...
time_bin_size_ms = 1000.0 * BIN_FRAMES / FS  # 333.33 ms
```

iii. The paper's methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI applied this correctly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from frame indices and the known sampling rate (30 Hz), then averaged into 10-frame bins.

ii.
```python
frame_times = np.arange(nframes, dtype=np.float32) / fs
time_binned = mean_bin_time_series(frame_times[np.newaxis, :], BIN_FRAMES)[0]
```

iii. Since the frame rate is constant at 30 Hz and there are no separate timestamps stored with the neural data, computing time from frame indices and sampling rate is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices are divided by the sampling rate to get frame-level times in seconds, then averaged in 10-frame bins using the same binning function applied to neural and motion data. This produces bin-center times rather than bin-edge times.

ii.
```python
frame_times = np.arange(nframes, dtype=np.float32) / fs
time_binned = mean_bin_time_series(frame_times[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The same binning function is reused for consistency across all time series streams.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same frame indices as the neural data and binned identically, so alignment is inherent. The time values represent bin centers (e.g., first bin = 0.15s for frames 0-9 at 30 Hz).

ii.
```python
# Time is split into trials with the same split_trials function
input_trials = [trial.astype(np.float32, copy=False) for trial in split_trials(time_binned[np.newaxis, :], trial_bins)]
```

iii. Using the same frame indexing and binning guarantees alignment without any additional processing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. `interframe_int.npy` is used to detect and interpolate dropped camera frames.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The data README describes these files. The motion energy is pre-computed from videography.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) dropped frames are detected via median inter-frame interval ratios and filled by linear interpolation, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 quintile bins computed within each session.

ii.
```python
# Step 1: repair
motion_aligned = repair_motion_trace(motion, interframe_int, nframes)

# Step 2: bin
motion_binned = mean_bin_time_series(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]

# Step 3: discretize
def motion_to_quintiles(motion_binned: np.ndarray) -> np.ndarray:
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(motion_binned, edges, right=False).astype(np.int64)
    return labels[np.newaxis, :]
```

iii. The AI followed the paper's methodology for denoising via binning and the instructions for discretizing into 5 equal-percentile bins per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 quintile bins using `np.quantile` at [0.2, 0.4, 0.6, 0.8] and `np.digitize`. Bin edges are computed per session. Labels range from 0 to 4.

ii.
```python
def motion_to_quintiles(motion_binned: np.ndarray) -> np.ndarray:
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(motion_binned, edges, right=False).astype(np.int64)
    return labels[np.newaxis, :]
```

iii. The instructions specify "five equal-percentile bins, selected per session." The AI computes quintile boundaries within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz (the microscope triggers camera frames). Dropped camera frames are detected using inter-frame intervals and filled by linear interpolation to match the neural data length. After repair, both streams have the same number of frames and are binned identically.

ii.
```python
def repair_motion_trace(motion, interframe_int, target_len):
    # ...
    median_ifi = float(np.median(interframe_int))
    gap_sizes = np.rint(interframe_int / median_ifi).astype(int)
    # Insert NaNs at gaps, then linearly interpolate
    # ...
    if repaired.size != target_len:
        raise ValueError(...)
    return repaired
```

iii. The data README notes that missing camera frames can be identified from timestamps/interframe intervals and should be interpolated. The AI uses median IFI to identify gaps rather than a fixed threshold.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected via inter-frame interval ratios and linearly interpolated. The repaired trace length is validated to match the neural data. Various assertions check consistency (F/Fneu shape match, frame count matches ops, iscell validity, sampling rate). Session lengths are required to be exactly divisible by the bin size and trial length.

ii.
```python
if repaired.size != target_len:
    raise ValueError(f"Repaired motion length {repaired.size} does not match target length {target_len}.")
if F.shape != Fneu.shape:
    raise ValueError(f"F/Fneu shape mismatch in {session_dir}")
if F.shape[1] != nframes:
    raise ValueError(f"Neural frame count mismatch in {session_dir}")
```

iii. The AI chose a strict approach: assertions and errors rather than silent handling. This catches misalignment early.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the maximin baseline correction, which applies gaussian smoothing, minimum filtering, and maximum filtering with a 1800-frame window over every neuron trace for all 41 sessions. The AI noted: "The baseline correction is the expensive part because it runs a 60-second maximin filter over every neuron trace for all 41 sessions."

ii. N/A

iii. The large filter window (1800 frames) applied to hundreds of neurons across long recordings dominates computation time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `repair_motion_trace` function builds the repaired array element-by-element in a Python list, then converts to numpy and interpolates NaNs. This could be vectorized by pre-computing insertion indices and using numpy array operations.

ii.
```python
repaired = [float(motion[0])]
for i, gap in enumerate(gap_sizes):
    if gap > 1:
        repaired.extend([np.nan] * (gap - 1))
    repaired.append(float(motion[i + 1]))
```

iii. The number of dropped frames is typically very small (0-148 per session), so the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once in a single loop.

ii. N/A

iii. The code follows a clean single-pass design.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `iscell.npy`, `ops.npy`, and `interframe_int.npy` for validation and metadata but some of the computed session_info metadata (like `n_frames_motion_missing`, `duration_sec`) is not used by the downstream decoder.

ii.
```python
session_info.append({
    "subject": subject,
    "session": session_dir.name,
    "date": session_dir.name.split("_")[0],
    "n_neurons": int(neural.shape[0]),
    "n_frames_imaging": nframes,
    "n_frames_motion_raw": int(len(motion)),
    "n_frames_motion_missing": int(nframes - len(motion)),
    "duration_sec": float(nframes / fs),
    "n_trials": len(neural_trials),
})
```

iii. The extra metadata is useful for documentation and debugging but is not consumed by the decoder.
