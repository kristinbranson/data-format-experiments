# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans sorted `jm*` subject directories and sorted `*_a` session directories. It requires the neural, cell-label, acquisition-options, motion-energy, and camera-timestamp files, then loads each session's `F`, `iscell`, `ops`, motion energy, and timestamps. `F` is memory-mapped; the other arrays are loaded normally.

ii.
```python
for subject_dir in sorted(data_root.glob("jm*")):
    ...
    for session_dir in sorted(subject_dir.glob("*_a")):
        required = [
            plane_dir / "F.npy", plane_dir / "iscell.npy",
            plane_dir / "ops.npy", motion_dir / "motion_energy_glob.npy",
            motion_dir / "tstamps.npy",
        ]
...
F = np.load(plane_dir / "F.npy", mmap_mode="r")
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. The trajectory says the agent first inspected the release layout, paper, methods, notebook, and Track2p code. It concluded that dated directories are sessions and that Suite2p neural outputs and camera motion are the required streams. It also chose memory mapping for the large fluorescence array.

## 1-b. How are the data split into subjects?

i. Each directory matching `jm*` is a mouse. Subject names are sorted and mapped to integer indices; each session receives the index of its parent directory.

ii.
```python
subjects = sorted({subject for subject, _ in session_paths})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[subject])
```

iii. The agent followed the release's naming convention and reported six mice after conversion.

## 1-c. How are the data split into sessions?

i. Every sorted `*_a` child directory of a mouse is one daily recording session. Each becomes one outer-list element in `neural`, `input`, and `output`.

ii.
```python
for session_dir in sorted(subject_dir.glob("*_a")):
    sessions.append((subject_dir.name, session_dir))
...
neural.append(session_neural)
decoder_input.append(session_input)
output.append(session_output)
```

iii. The agent stated that every dated directory is a session and validated that this yielded 41 sessions.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into contiguous, non-overlapping 60-second trials. At 30 Hz this is 1,800 original frames or 180 ten-frame bins. Unlike the reference's general remainder-discard policy, the agent requires every session to contain an exact number of trials.

ii.
```python
TRIAL_FRAMES = int(FRAME_RATE_HZ * TRIAL_SECONDS)
TRIAL_BINS = TRIAL_FRAMES // AVERAGE_FRAMES
if F.shape[1] % TRIAL_FRAMES:
    raise ValueError("session is not a whole number of 60-second trials")
...
for trial in range(n_trials):
    start = trial * TRIAL_BINS
    stop = start + TRIAL_BINS
```

iii. The agent justified this from the requested 60-second artificial trials and confirmed that the released 20- and 30-minute recordings produce 20 and 30 trials respectively.

## 1-e. How are trials filtered based on quality controls?

i. No individual trials are filtered. A whole session is rejected if its sampling rate, array shapes, ROI criterion, or exact 60-second divisibility fails.

ii.
```python
if not np.isclose(fs, FRAME_RATE_HZ):
    raise ValueError(...)
if not np.all((iscell[:, 0] == 1) & (iscell[:, 1] > 0.5)):
    raise ValueError(...)
if F.shape[1] % TRIAL_FRAMES:
    raise ValueError(...)
```

iii. The trajectory found that the distributed files already contain selected/tracked cells and whole-duration sessions, so it applied validation rather than trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived only from Suite2p `plane0/F.npy`. `iscell.npy` is used for validation and `ops.npy` for the sampling rate, but `Fneu.npy` is not loaded or used.

ii.
```python
F = np.load(plane_dir / "F.npy", mmap_mode="r")
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. The agent reasoned from Track2p's `F_processing` call/default that the released `F` should be baseline-corrected without additional neuropil subtraction (`neucoeff=0.0`). This conflicts with the human reference, which explicitly uses `F - 0.7 * Fneu`.

## 2-b. How is the `neural` data processed?

i. `F` is converted to float32, Gaussian-smoothed along time with sigma 10 frames, passed through 60-second minimum then maximum filters, and the resulting baseline is subtracted. It is then averaged in non-overlapping ten-frame bins and sliced into trials.

ii.
```python
corrected = gaussian_filter1d(np.asarray(F, dtype=np.float32), sigma=10.0, axis=1)
window = int(60.0 * fs)
corrected = minimum_filter1d(corrected, size=window, axis=1)
corrected = maximum_filter1d(corrected, size=window, axis=1)
return np.subtract(F, corrected, dtype=np.float32)
...
neural_binned = average_in_bins(baseline_correct_fluorescence(F, fs))
```

iii. The agent says this directly reproduces Track2p's Suite2p-style maximin baseline correction and numerically checked its filter against repository code. Its justification for omitting neuropil subtraction was the Track2p function default, despite the human reference's 0.7 subtraction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROIs are removed during conversion. The code asserts that every supplied ROI already has `iscell[:,0] == 1` and probability above 0.5; otherwise it rejects the session.

ii.
```python
if not np.all((iscell[:, 0] == 1) & (iscell[:, 1] > 0.5)):
    raise ValueError(
        f"{session_dir} contains an ROI outside the paper's cell criterion"
    )
```

iii. Inspection led the agent to conclude the release already contains only Suite2p-classified cells successfully tracked across days, so a second filter would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Trials are consecutive blocks beginning at session time 0, and each neural trial is a slice `[start:stop]`. Metadata describes alignment as the start of each contiguous 60-second block, with offsets 0 to 60 seconds.

ii.
```python
session_neural.append(neural_binned[:, start:stop])
...
"temporal_alignment_event": "start of each contiguous 60-second session block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The agent treated artificial block onset as the only meaningful event because this is continuous spontaneous-behavior data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged into each output sample, producing 3 Hz data and 333.333 ms bins. Remainders smaller than ten frames would be truncated by the binning helper, though exact trial divisibility is separately required.

ii.
```python
AVERAGE_FRAMES = 10
return values.reshape(new_shape).mean(axis=-1, dtype=np.float32)
...
"time_bin_size": 1000.0 * AVERAGE_FRAMES / FRAME_RATE_HZ,
```

iii. The paper says decoding analyses denoise fluorescence and behavior by averaging ten timestamps, so the agent applies identical binning to both streams before discretization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from the binned sample index, the ten-frame bin width, and the 30-Hz sampling rate, rather than read from a raw timestamp file. It represents elapsed session time at the mean time of the ten frames in a bin.

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32) * AVERAGE_FRAMES
    + (AVERAGE_FRAMES - 1) / 2
) / np.float32(fs)
```

iii. The agent chose bin-center timing because each converted point is an average of ten original frames. The reference instead uses each bin's left edge.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For bin `k`, the code computes `(10*k + 4.5) / 30` seconds in float32, adds a singleton variable dimension, and makes each trial slice contiguous. Time continues across trial boundaries within a session.

ii.
```python
elapsed_seconds = (np.arange(...) * AVERAGE_FRAMES + 4.5) / np.float32(fs)
session_input.append(
    np.ascontiguousarray(elapsed_seconds[None, start:stop], dtype=np.float32)
)
```

iii. The trajectory does not separately discuss this choice; the code comment says it stores the mean times of the ten represented frames.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is generated with exactly one entry per binned neural sample, then sliced with the identical `start` and `stop` indices used for neural trials.

ii.
```python
start = trial * TRIAL_BINS
stop = start + TRIAL_BINS
session_neural.append(neural_binned[:, start:stop])
session_input.append(elapsed_seconds[None, start:stop])
```

iii. The shared binned index makes the streams one-to-one; no separate synchronization is needed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from `move_deve/motion_energy_glob.npy`. `move_deve/tstamps.npy` supplies camera timing used to locate missing frames.

ii.
```python
motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
motion_aligned = align_motion_to_imaging(motion_raw, timestamps, F.shape[1])
```

iii. The agent found the separate camera stream and chose actual timestamps rather than the reference's `interframe_int.npy`, citing the release documentation and doubled timestamp gaps.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing samples are interpolated onto imaging-frame indices, the aligned signal is averaged over ten-frame bins, and session-specific 20th/40th/60th/80th percentiles are computed to form categorical labels.

ii.
```python
motion_aligned = align_motion_to_imaging(motion_raw, timestamps, F.shape[1])
motion_binned = average_in_bins(motion_aligned)
labels, thresholds = quintile_labels(motion_binned)
```

iii. This ordering preserves neural/motion synchronization, applies the paper's behavior denoising, and avoids averaging already categorical labels.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds are calculated separately for every complete session at the 20th, 40th, 60th, and 80th percentiles of binned motion. `np.digitize(..., right=False)` produces integer classes 0 through 4.

ii.
```python
thresholds = np.percentile(motion, [20, 40, 60, 80])
labels = np.digitize(motion, thresholds, right=False).astype(np.int64)
```

iii. The instructions require five equal-percentile bins selected per session. The agent also validated approximately 20% occupancy for each class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If motion already has the imaging frame count, it is left unchanged. Otherwise, median camera spacing defines normal steps, rounded timestamp-gap ratios locate dropped frames, and `np.interp` reconstructs motion on every imaging-frame index. After identical ten-frame binning, equal lengths are checked and neural/output use identical trial slices.

ii.
```python
frame_steps = np.maximum(1, np.rint(intervals / typical_interval).astype(np.int64))
camera_frame_idx = np.concatenate((np.array([0]), np.cumsum(frame_steps)))
return np.interp(np.arange(n_imaging_frames), camera_frame_idx, motion)
...
if neural_binned.shape[1] != motion_binned.size:
    raise RuntimeError(...)
session_output.append(labels[None, start:stop])
```

iii. The trajectory says interpolation is used only for streams shorter than imaging, because occasional long intervals in already full-length streams do not prove an absent array element. It confirmed inferred frame counts against all affected sessions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are reconstructed by timestamp-aware linear interpolation. Invalid lengths, timestamps, sampling rates, shapes, cell criteria, incomplete files, or non-whole trial durations raise explicit errors. The code does not silently discard partial final trials.

ii.
```python
if inferred_frames != n_imaging_frames:
    raise ValueError(...)
return np.interp(imaging_frame_idx, camera_frame_idx, motion).astype(np.float32)
...
if F.shape[1] % TRIAL_FRAMES:
    raise ValueError(...)
```

iii. The agent relied on release documentation for dropped-camera-frame repair and deliberately used fail-fast checks to prevent silent neural/behavior misalignment.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant computation is full-session baseline correction for every neuron, especially the Gaussian and 1,800-frame minimum/maximum filters. Loading large arrays and serializing the full result are secondary I/O costs.

ii.
```python
corrected = gaussian_filter1d(..., sigma=10.0, axis=1)
corrected = minimum_filter1d(corrected, size=window, axis=1)
corrected = maximum_filter1d(corrected, size=window, axis=1)
```

iii. The trajectory explicitly focused on reproducing and numerically checking this full-array filter, and the agent waited through it during conversion. Unlike the reference, its SciPy implementation runs on CPU.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Binning, interpolation, filtering, and label creation are already vectorized. The remaining Python loops traverse sessions and create per-trial list objects; trial slicing could be reshaped/batched, but the required nested list format still needs per-trial objects. The agent avoids the reference's repeated `np.insert` loop entirely via `np.interp`.

ii.
```python
for session_number, (subject, session_dir) in enumerate(session_paths, start=1):
    ...
    for trial in range(n_trials):
        session_neural.append(neural_binned[:, start:stop])
```

iii. The trajectory gives no explicit loop-performance justification. The code design nevertheless vectorizes the expensive numerical operations, leaving only organizational loops.

## 6-c. What processing does the code repeat multiple times?

i. Loading, validation, baseline filtering, binning, percentile calculation, and elapsed-time construction are repeated once per session. Contiguous conversion is repeated for each trial. There is no repeated computation of the same session result.

ii.
```python
for session_number, (subject, session_dir) in enumerate(session_paths, start=1):
    neural_binned = average_in_bins(baseline_correct_fluorescence(F, fs))
    motion_binned = average_in_bins(motion_aligned)
    labels, thresholds = quintile_labels(motion_binned)
```

iii. These repetitions are necessary because neuron counts, missing frames, and percentile thresholds are session-specific; the trajectory does not identify avoidable repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No major numerical result is computed and then discarded. `iscell` and most `ops` content are loaded only for validation, percentile thresholds are retained only as metadata, and repeated contiguity copies primarily support stable output layout rather than decoder mathematics.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
...
"motion_quintile_thresholds": thresholds.tolist(),
```

iii. The trajectory emphasizes validation and reproducibility; it does not report intentionally discarded processing.
