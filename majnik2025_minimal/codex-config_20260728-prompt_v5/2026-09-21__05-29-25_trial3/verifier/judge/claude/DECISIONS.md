# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as all subdirectories in the data root (sorted). Sessions are subdirectories within each subject folder (sorted). For each session, neural data is loaded from `suite2p/plane0/F.npy`, `Fneu.npy`, and `ops.npy`. Motion energy is loaded from `move_deve/motion_energy_glob.npy`, and timestamps from `move_deve/tstamps.npy` are used for dropped-frame repair.

ii.
```python
def list_subjects(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())

def list_sessions(subject_dir: Path) -> list[Path]:
    return sorted(path for path in subject_dir.iterdir() if path.is_dir())

# In preprocess_neural:
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)

# In repair_motion_energy:
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
tstamps = np.load(move_dir / "tstamps.npy")
```

iii. The agent explored the directory structure, read the data README, and followed the convention that subject folders contain session subfolders. It also loads `ops.npy` to read preprocessing parameters from the saved Suite2p run rather than hard-coding them.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all subdirectories in the data root, sorted alphabetically. Unlike the reference which filters for `jm*` prefix, the AI takes all directories.

ii.
```python
def list_subjects(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())

subjects = [subject_dir.name for subject_dir in list_subjects(data_root)]
```

iii. The agent assumed all directories under the data root are subject folders. In this dataset all subject directories start with `jm`, so the result is the same.

## 1-c. How are the data split into sessions?

i. Sessions are all subdirectories within each subject folder, sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
def list_sessions(subject_dir: Path) -> list[Path]:
    return sorted(path for path in subject_dir.iterdir() if path.is_dir())
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session. Sorting ensures deterministic chronological order.

## 1-d. How are the data split into trials?

i. Trials are 60-second non-overlapping segments. The number of bins per trial is computed as `TRIAL_DURATION_SEC / (BIN_SIZE_FRAMES / fs)` = 180 bins. Any leftover bins are discarded.

ii.
```python
bins_per_trial = int(round(TRIAL_DURATION_SEC / (BIN_SIZE_FRAMES / fs)))
# In split_into_trials:
n_trials = total_bins // bins_per_trial
usable = n_trials * bins_per_trial
for trial_idx in range(n_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
```

iii. Per instruction, trials are 60-second non-overlapping segments. Since no natural trial structure exists, fixed-length segmentation is used.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are included.

ii. N/A

iii. No quality control criteria for trials are described in the paper or instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p parameters), from `suite2p/plane0/`.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The agent confirmed these are the standard Suite2p outputs. It additionally loads `ops.npy` to read the preprocessing parameters used during the original Suite2p run.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied using the coefficient from `ops.npy` (defaulting to 0.7), followed by Suite2p's `dcnv.preprocess` with parameters read from `ops.npy` (baseline='maximin', win_baseline=60.0, sig_baseline=10.0, prctile_baseline=8.0). The result is then averaged into 10-frame bins.

ii.
```python
fs = float(ops["fs"])
neuropil_corrected = F - float(ops.get("neucoeff", 0.7)) * Fneu
baseline_corrected = dcnv.preprocess(
    neuropil_corrected.copy(),
    baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=fs,
    prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
    batch_size=64,
    device=torch.device("cpu"),
)
# Then binning:
neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
```

iii. The agent confirmed from the paper methods that baseline-corrected fluorescence was used for decoding. Reading parameters from `ops.npy` rather than hard-coding ensures fidelity to the original Suite2p run.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the tracked `F.npy` output are included (these are already tracked cells present across all days).

ii. N/A

iii. Suite2p's cell detection and Track2p's cross-session tracking already filtered cells. No further filtering (e.g., by `iscell`) was applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments with no stimulus events, no event-based alignment is needed.

ii.
```python
"temporal_alignment_event": "session start",
"off_start": 0.0,
"off_end": float(TRIAL_DURATION_SEC),
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 frames, converting 30 Hz to 3 Hz (333.33 ms bins). Binning is applied before discretization of motion energy.

ii.
```python
BIN_SIZE_FRAMES = 10
neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
# ...
time_bin_size_ms = 1000.0 * BIN_SIZE_FRAMES / float(common_fs)
```

iii. The paper methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from frame indices and the sampling rate. The raw frame timestamps (`np.arange(n_frames) / fs`) are created, then averaged in 10-frame bins.

ii.
```python
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
```

iii. Since the frame rate is constant at 30 Hz, computing time from frame indices is equivalent to reading timestamps from the data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame-level timestamps are created as `np.arange(n_frames) / fs`, then averaged in 10-frame bins using `mean_bin_1d`. This gives the center of each time bin rather than the left edge. The time values are then sliced per trial.

ii.
```python
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
# In split_into_trials:
input_trials.append(time_binned_sec[start:end][None, :].astype(np.float32, copy=False))
```

iii. The time values represent bin centers because frame-level timestamps are averaged within each bin.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time, neural, and motion energy data are all binned from the same frame-indexed arrays using the same bin boundaries, so they are inherently aligned.

ii.
```python
neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
```

iii. All three streams share the same frame-to-bin mapping and trial splitting.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used to detect and repair dropped camera frames.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
tstamps = np.load(move_dir / "tstamps.npy")
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video. The timestamps are needed to identify dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) Dropped frames are detected via timestamp gaps and filled by linear interpolation using `np.interp`; (2) The trace is averaged into 10-frame bins; (3) The binned signal is discretized into 5 quintile-based bins using session-specific thresholds at quantiles [0.2, 0.4, 0.6, 0.8].

ii.
```python
# Repair:
diffs = np.diff(tstamps)
frame_dt = float(np.median(diffs))
missing_counts = np.maximum(np.round(diffs / frame_dt).astype(int) - 1, 0)
repaired[missing_mask] = np.interp(...)

# Binning:
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)

# Discretization:
def session_motion_bins(motion_binned):
    thresholds = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(motion_binned, thresholds, right=False).astype(np.int64)
```

iii. Dropped frame repair ensures frame-level alignment with neural data. Binning before discretization is necessary because averaging class labels would be meaningless.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins per session using `np.quantile` at [0.2, 0.4, 0.6, 0.8], yielding equal-frequency quintiles (classes 0-4). Output values are labeled ["lowest", "low", "middle", "high", "highest"].

ii.
```python
def session_motion_bins(motion_binned: np.ndarray) -> np.ndarray:
    thresholds = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(motion_binned, thresholds, right=False).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins, selected per session", which is what quintile-based discretization achieves.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Video and neural data are acquired synchronously at 30 Hz. When camera frames are missing, timestamp-based repair inserts NaN values at the correct positions and interpolates them. After repair, lengths match. Both streams are then binned identically and split into trials from the same indices.

ii.
```python
motion_trace, n_motion_repaired = repair_motion_energy(session_dir, n_frames)
# After repair, motion_trace has same length as neural_trace
neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
```

iii. The timestamp-based repair ensures frame-for-frame alignment before any binning or trial splitting.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are detected via timestamp gaps in `tstamps.npy`. NaN values are inserted at the detected positions and then interpolated using `np.interp`. A ValueError is raised if the repaired length doesn't match the expected neural frame count. Remainder bins at session end that don't fill a complete trial are discarded.

ii.
```python
expected_len = int(len(motion) + missing_counts.sum())
if expected_len != n_frames:
    raise ValueError(...)
repaired = np.full(n_frames, np.nan, dtype=np.float32)
repaired[positions] = motion
repaired[missing_mask] = np.interp(...)
```

iii. The ValueError ensures mismatches are caught rather than producing silent misalignment. Discarding remainder frames loses at most 59 seconds per session.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is Suite2p's `dcnv.preprocess` baseline correction, run per session. The agent explicitly noted waiting for this step during execution.

ii. N/A

iii. The baseline correction involves sliding window operations over the full session length for every neuron.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates per trial, slicing arrays. This could potentially be replaced with array reshaping, though the performance impact is minimal.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end])
```

iii. The number of trials per session is small (~30), so loop overhead is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each session is processed once in a single pass.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code creates detailed `session_info` metadata for each session (n_neurons, n_frames_raw, n_motion_frames_repaired, etc.) which is informational but not used by the decoder.

ii.
```python
session_info.append({
    "subject": subject_dir.name,
    "session": session_dir.name,
    "n_neurons": int(n_neurons),
    "n_frames_raw": int(n_frames),
    "n_motion_frames_repaired": int(n_motion_repaired),
    ...
})
```

iii. This metadata is useful for debugging but not consumed by the downstream decoder.
