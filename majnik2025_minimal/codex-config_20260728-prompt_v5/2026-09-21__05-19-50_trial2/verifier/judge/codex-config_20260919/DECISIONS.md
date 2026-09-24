# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for sorted `jm*` subject directories, then sorted date-like session directories. For every session it loads `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. The resulting 41 sessions from six mice are processed and stored.

ii.
```python
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

iii. The trajectory says the raw data are Suite2p outputs plus `move_deve` motion traces and that all 41 Track2p-exported sessions across six mice should be used. Sorting gives deterministic ordering; the date-prefix restriction was used to identify actual recording directories.

## 1-b. How are the data split into subjects?

i. Each sorted directory whose name begins with `jm` is one subject. A name-to-index map assigns each session its subject index.

ii.
```python
subjects = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[subject])
```

iii. The AI inferred from the directory convention and inspection that each `jm*` directory is a mouse; its final report confirms six mice.

## 1-c. How are the data split into sessions?

i. Each date-like subdirectory of a subject is treated as one daily session, sorted lexicographically and appended as a separate entry in the session lists.

ii.
```python
session_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
for session_dir in session_dirs:
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
```

iii. The AI described these as the 41 Track2p-exported sessions and preserved each recording as a decoder session rather than combining recordings.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second trials after 10-frame binning. At 30 Hz this is 180 binned samples per trial. The helper requires exact divisibility rather than dropping a partial tail.

ii.
```python
trial_bins = int(TRIAL_SECONDS * FS / BIN_FRAMES)
...
def split_trials(x: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    if x.shape[-1] % trial_bins != 0:
        raise ValueError(...)
    ntrials = x.shape[-1] // trial_bins
    return [x[..., i * trial_bins:(i + 1) * trial_bins] for i in range(ntrials)]
```

iii. The instruction explicitly requests 60-second trials. The trajectory says trialization is a downstream-required deviation from the continuous paper data. All supplied sessions are exactly divisible, yielding 1,090 trials.

## 1-e. How are trials filtered based on quality controls?

i. No completed trial is filtered. Instead, malformed sessions cause an exception during sampling-rate, shape, ROI-quality, bin-divisibility, or trial-divisibility checks.

ii.
```python
if fs != FS:
    raise ValueError(...)
if F.shape != Fneu.shape:
    raise ValueError(...)
...
if x.shape[-1] % trial_bins != 0:
    raise ValueError(...)
```

iii. The trajectory identified no trial-level quality rule in the source. The AI therefore retained every valid fixed-duration segment and used validation to prevent silently misaligned trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from Suite2p `F.npy` and `Fneu.npy`; `ops.npy` supplies frame count and sampling rate, and `iscell.npy` is used only for validation.

ii.
```python
F = np.load(plane_dir / "F.npy", allow_pickle=True)
Fneu = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
neural = suite2p_style_baseline_correct(F, Fneu, fs)
```

iii. The AI found that the repository helper uses fluorescence with Suite2p-style baseline correction rather than deconvolved spikes.

## 2-b. How is the `neural` data processed?

i. The AI casts fluorescence to `float32`, computes `F - 0.0*Fneu`, estimates a maximin baseline by Gaussian smoothing (`sigma=10` frames), a 60-second minimum filter, and a 60-second maximum filter, subtracts that baseline, and finally averages every 10 frames. Thus, despite loading `Fneu`, it performs no effective neuropil subtraction.

ii.
```python
NEUCOEFF = 0.0
...
Fc = F.astype(np.float32, copy=False) - NEUCOEFF * Fneu.astype(np.float32, copy=False)
win = int(WIN_BASELINE * fs)
Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
return (Fc - Flow).astype(np.float32, copy=False)
...
neural_binned = mean_bin_time_series(neural, BIN_FRAMES)
```

iii. The trajectory says the AI found a bundled Track2p GUI helper using `neucoeff=0`, `baseline='maximin'`, `sig_baseline=10`, and a 60-second window, and chose to follow that literally. It also cites the methods’ instruction to average decoder traces over 10 timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROIs are removed in the converter. The AI asserts that every already-exported ROI has `iscell[:,0] == 1` and probability `iscell[:,1] > 0.5`; otherwise it aborts the session.

ii.
```python
if not np.all(iscell[:, 0] == 1):
    raise ValueError(...)
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(...)
```

iii. The trajectory states that Track2p had already saved only cross-day tracked cells satisfying the paper’s `iscell > 0.5` criterion, so the AI encoded this assumption as an assertion instead of filtering again.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus event. Binned neural samples remain in session order and are split into contiguous blocks measured from session start; metadata names session start as the alignment event.

ii.
```python
neural_trials = [trial.astype(np.float32, copy=False)
                 for trial in split_trials(neural_binned, trial_bins)]
...
"temporal_alignment_event": "session start",
"off_start": None,
"off_end": None,
```

iii. The AI treated trialization as segmentation of continuous spontaneous-behavior recordings, so session start is the only meaningful common origin and no event-triggered shift is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged into non-overlapping bins, producing 3-Hz data and a `333.333...` ms bin size. Neural, motion, and time streams receive the same binning.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
...
return x.reshape(new_shape).mean(axis=-1)
...
time_bin_size_ms = 1000.0 * BIN_FRAMES / FS
```

iii. The trajectory explicitly cites the methods’ decoder denoising by averaging neural and behavior traces over 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the neural frame index and the session sampling rate rather than loaded from a timestamp file.

ii.
```python
frame_times = np.arange(nframes, dtype=np.float32) / fs
```

iii. The AI used imaging frames as the master clock because neural and repaired motion data are frame-aligned at 30 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices are divided by sampling rate to obtain seconds, then each group of 10 frame times is averaged. Consequently values are bin centers (`0.15, 0.4833, ...`), not left edges.

ii.
```python
frame_times = np.arange(nframes, dtype=np.float32) / fs
time_binned = mean_bin_time_series(frame_times[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The final trajectory describes these as “session-start time bin centers.” Applying the same averaging operation as other streams supplies one representative time per output bin.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is generated at every neural frame, binned with identical 10-frame boundaries, and split with the identical 180-bin trial boundaries, so each input value is the center time of its corresponding neural bin.

ii.
```python
neural_binned = mean_bin_time_series(neural, BIN_FRAMES)
time_binned = mean_bin_time_series(frame_times[np.newaxis, :], BIN_FRAMES)[0]
...
input_trials = [trial.astype(np.float32, copy=False)
                for trial in split_trials(time_binned[np.newaxis, :], trial_bins)]
```

iii. The AI intentionally used a shared frame grid and shared bin/trial slicing to avoid independent-clock drift.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output comes from `move_deve/motion_energy_glob.npy`. `interframe_int.npy` supplies camera gap information used to repair missing positions before binning.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
motion_aligned = repair_motion_trace(motion, interframe_int, nframes)
```

iii. The trajectory says the global motion trace is the requested behavior signal and the timestamp-derived intervals expose dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If necessary, missing camera positions are inferred by rounding each interframe interval relative to the median interval. Missing values are inserted as NaNs, the repaired trace is padded or truncated to imaging length, and NaNs are linearly interpolated. The aligned trace is then averaged in 10-frame bins and converted to per-session quintile labels.

ii.
```python
median_ifi = float(np.median(interframe_int))
gap_sizes = np.rint(interframe_int / median_ifi).astype(int)
...
if gap > 1:
    repaired.extend([np.nan] * (gap - 1))
...
repaired[nan_mask] = np.interp(...)
...
motion_binned = mean_bin_time_series(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
motion_labels = motion_to_quintiles(motion_binned)
```

iii. The AI found that no extra behavioral transform was used beyond 10-frame averaging. It repaired gaps because the validator rejects NaNs and because omission would shift the behavior stream relative to imaging.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four boundaries at the 20th, 40th, 60th, and 80th percentiles of each session’s binned motion trace create integer classes 0–4. Ties are assigned by `np.digitize(..., right=False)`.

ii.
```python
def motion_to_quintiles(motion_binned: np.ndarray) -> np.ndarray:
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(motion_binned, edges, right=False).astype(np.int64)
    return labels[np.newaxis, :]
```

iii. This directly implements the instruction’s five equal-percentile bins selected per session. The trajectory notes that discretization follows, rather than precedes, 10-frame averaging.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera gaps are reconstructed to make motion length equal imaging frame count. Motion and neural signals are then binned over the same consecutive groups of 10 and split over the same consecutive 180-bin trials.

ii.
```python
motion_aligned = repair_motion_trace(motion, interframe_int, nframes)
neural_binned = mean_bin_time_series(neural, BIN_FRAMES)
motion_binned = mean_bin_time_series(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
...
output_trials = [trial.astype(np.int64, copy=False)
                 for trial in split_trials(motion_labels, trial_bins)]
```

iii. The AI investigated timestamp units and gaps, then chose interpolation on the imaging-time grid specifically to prevent a dropped camera frame from shifting all subsequent behavior samples.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are reconstructed and linearly interpolated. The repair also pads or truncates to the target frame count. Unexpected sampling rates, neural shapes, ROI flags, invalid interval ratios, and non-divisible lengths raise errors rather than being silently accepted.

ii.
```python
if repaired.size < target_len:
    repaired = np.pad(repaired, (0, target_len - repaired.size), constant_values=np.nan)
elif repaired.size > target_len:
    repaired = repaired[:target_len]
...
if nan_mask.any():
    repaired[nan_mask] = np.interp(...)
...
if x.shape[-1] % bin_frames != 0:
    raise ValueError(...)
```

iii. The trajectory emphasizes that missing camera frames occur in several sessions, that NaNs are rejected downstream, and that interpolation preserves frame alignment. Assertions were added to expose unexpected data rather than conceal it.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session maximin baseline correction is the principal cost: Gaussian, minimum, and maximum filters operate across every time sample for every neuron in all 41 sessions. Loading and serializing the large arrays are secondary I/O costs.

ii.
```python
Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
```

iii. The trajectory explicitly identifies the 60-second maximin filter over every neuron trace and all sessions as the expensive conversion step; the observed conversion took tens of seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. `repair_motion_trace` loops over every interframe interval while appending Python scalars/NaNs, and the per-trial list comprehensions slice trials one at a time. The repair loop is the clearest vectorization/preallocation candidate; session and subject loops are appropriate because files and neuron counts differ.

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

iii. The trajectory does not discuss vectorization. Its implementation prioritizes explicit reconstruction of irregular gaps; because dropped frames are rare, that loop is unlikely to dominate the baseline filters.

## 6-c. What processing does the code repeat multiple times?

i. Each session independently repeats file loading, validation, baseline correction, motion repair, 10-frame binning, time-vector construction, quintile calculation, and trial splitting. `split_trials` separately traverses neural, input, and output arrays with identical boundaries.

ii.
```python
for subject in subjects:
    ...
    for session_dir in session_dirs:
        ...
        neural_trials = [... split_trials(neural_binned, trial_bins)]
        input_trials = [... split_trials(time_binned[np.newaxis, :], trial_bins)]
        output_trials = [... split_trials(motion_labels, trial_bins)]
```

iii. The trajectory does not identify repeated work. Most repetition is necessary per session; the three trial splits could share precomputed slice boundaries, although their cost is small.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. With `NEUCOEFF = 0.0`, loading/casting `Fneu` and multiplying it by zero do not change neural data. `iscell` and much of `ops` are loaded only for assertions/metadata, while extensive `session_info` is saved but not consumed by decoder training. These checks and provenance are useful, but not required for the decoder arrays.

ii.
```python
NEUCOEFF = 0.0
Fc = F.astype(np.float32, copy=False) - NEUCOEFF * Fneu.astype(np.float32, copy=False)
...
iscell = np.load(...)
ops = np.load(...).item()
...
"session_info": session_info,
```

iii. The trajectory justifies `iscell` as an explicit assertion of Track2p’s prior filtering and uses `ops` to verify sampling/frame counts. It does not discuss the zero-valued `Fneu` operation or unused metadata as efficiency issues.
