# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted subject/session directories beneath `/app/data`, retaining directories whose session contains `suite2p/plane0/F.npy`. For each selected session it loads `F.npy`, `Fneu.npy`, and `ops.npy`, plus motion energy, timestamps, and stored interframe intervals. Full mode includes all 41 discovered sessions, but only the first 36,000 frames (20 minutes) of each session enter the converted data.

ii.
```python
for subject_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        if (session_dir / "suite2p" / "plane0" / "F.npy").exists():
            sessions.append((subject_dir.name, session_dir))

fluorescence = np.load(plane_dir / "F.npy", mmap_mode="r")
neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals_stored = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. The agent states that deterministic discovery captures 6 subjects and 41 sessions. It chose a uniform first-20-minute window because the paper analyzed 20 minutes and because equal session lengths yield 20 equal trials, despite longer source recordings being available.

## 1-b. How are the data split into subjects?

i. The subject is the parent directory name of each discovered session. Unique selected names are sorted, and each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({subject for subject, _ in selected})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
"subject_idx": np.asarray(
    [subject_lookup[subject] for subject, _ in selected], dtype=np.int64
),
```

iii. The notes identify each `jm*` directory as a mouse and report six mice with session counts 7/7/7/7/6/7.

## 1-c. How are the data split into sessions?

i. Every sorted recording subdirectory containing `F.npy` becomes one output session. Each session is processed independently and appended once to each session-level list.

ii.
```python
for session_index, (subject, session_dir) in enumerate(selected):
    ...
    data["neural"].append(neural_trials)
    data["input"].append(input_trials)
    data["output"].append(output_trials)
```

iii. The agent treats each dated folder as one daily recording and preserves deterministic subject/date ordering.

## 1-d. How are the data split into trials?

i. After restricting every session to 36,000 raw frames and averaging groups of 10, the resulting 3,600 bins are divided into exactly 20 consecutive, non-overlapping 60-second trials of 180 bins each.

ii.
```python
TRIAL_BINS = int(TRIAL_SECONDS * BINNED_FS_HZ)
N_TRIALS = ANALYSIS_FRAMES // (FRAMES_PER_BIN * TRIAL_BINS)
neural_trials = [x.copy() for x in np.split(neural_binned, N_TRIALS, axis=1)]
```

iii. The notes justify fixed 60-second windows from the explicit decoder task and the 20-minute crop from the paper’s analysis window, yielding equal dimensions and no partial trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. All 20 constructed trials are kept if session-wide shape and finite-value checks pass.

ii.
```python
validate_session_arrays(neural_binned, elapsed_time, labels)
if not (len(neural_trials) == len(input_trials) == len(output_trials) == N_TRIALS):
    raise AssertionError("Trial count mismatch")
```

iii. The source is continuous rather than naturally trialized; the agent found no paper-defined bad-trial criterion and retained all complete constructed windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from Suite2p `F.npy` and `Fneu.npy`; preprocessing parameters are read from `ops.npy`.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy", mmap_mode="r")
neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. The agent concluded that these are the already longitudinally matched, curated ROI rows distributed by the study and that baseline-corrected fluorescence—not `spks`—matches the paper decoder.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil (`F - neucoeff*Fneu`), performs Suite2p-style maximin baseline correction using Gaussian, minimum, and maximum filters, subtracts that baseline, retains the first 36,000 frames, then averages non-overlapping groups of 10 frames. Processing occurs in 64-neuron chunks.

ii.
```python
corrected_neuropil = raw_f - np.float32(neucoeff) * raw_fneu
flow = gaussian_filter1d(corrected_neuropil, sigma=sig_baseline, axis=1, mode="reflect")
flow = minimum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
flow = maximum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
baseline_corrected = corrected_neuropil - flow
retained = baseline_corrected[:, :ANALYSIS_FRAMES]
binned[start:stop] = retained.reshape(stop - start, n_bins, FRAMES_PER_BIN).mean(axis=2)
```

iii. The notes say this reproduces Suite2p’s configured `maximin` preprocessing and the paper’s 10-timestamp denoising. The full source trace is baseline-filtered before cropping so later context contributes near the crop boundary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional ROI filtering is applied. All rows in the released fluorescence files are retained, subject to shape, sampling-rate, baseline-mode, and finite-value validation.

ii.
```python
if fluorescence.shape != neuropil.shape or fluorescence.ndim != 2:
    raise ValueError(...)
if not np.isfinite(binned).all():
    raise ValueError(...)
```

iii. The agent verified that the release already contains all-day-complete Track2p matches passing `iscell > 0.5`; reapplying a filter would not remove rows and could not reconstruct paper-version cells absent from the release.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no physiological event alignment. The alignment event is session start, and trials are consecutive 60-second windows beginning there; `off_start` and `off_end` are `None`.

ii.
```python
"temporal_alignment_event": (
    "Session start; trials are consecutive non-overlapping 60-second windows."
),
"off_start": None,
"off_end": None,
```

iii. The agent explains that the recordings are continuous and spontaneous, so session start is the meaningful common origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are averaged per bin, producing a 3-Hz stream and a 333.333-ms bin size for neural and motion data.

ii.
```python
FRAMES_PER_BIN = 10
BINNED_FS_HZ = RAW_FS_HZ / FRAMES_PER_BIN
"time_bin_size": 1000.0 * FRAMES_PER_BIN / RAW_FS_HZ,
```

iii. This follows the paper statement that neural and behavior traces were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from raw imaging-frame ordinals and the fixed 30-Hz sampling rate; it is not read from a timestamp file.

ii.
```python
elapsed_time = (
    np.arange(ANALYSIS_FRAMES, dtype=np.float64)
    .reshape(-1, FRAMES_PER_BIN).mean(axis=1) / RAW_FS_HZ
).astype(np.float32)
```

iii. The agent states that each bin’s time should represent the mean time of its 10 constituent imaging frames.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices 0–35,999 are reshaped into groups of 10, averaged, divided by 30 Hz, converted to float32, and split into trials. Thus values are bin centers from 0.15 to 1199.8167 seconds.

ii.
```python
elapsed_time = (...reshape(-1, FRAMES_PER_BIN).mean(axis=1) / RAW_FS_HZ).astype(np.float32)
input_trials = [x[np.newaxis, :].copy() for x in np.split(elapsed_time, N_TRIALS)]
```

iii. The notes explicitly prefer bin centers over left edges because every neural/output sample is a ten-frame mean.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time array uses exactly the same raw-frame groups and trial boundaries as neural data and is validated to have the same 3,600-bin session length.

ii.
```python
if elapsed_time.shape != (expected_bins,) or labels.shape != (expected_bins,):
    raise AssertionError("Input/output binned lengths do not match neural data")
```

iii. Independent checks documented by the agent reproduced every converted time and confirmed continuous one-third-second steps across trial boundaries.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses `move_deve/motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals_stored = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. Motion energy is the released whole-body behavior measure; timestamps and intervals expose missed camera triggers needed for neural alignment.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Stored intervals are checked against timestamp differences. Interval-to-median ratios are rounded to infer trigger steps, motion is linearly interpolated onto imaging-trigger ordinals for the first 36,000 frames, and groups of 10 aligned samples are averaged.

ii.
```python
trigger_steps = np.rint(intervals_stored / median_interval).astype(np.int64)
trigger_positions = np.concatenate([np.array([0]), np.cumsum(trigger_steps)])
aligned = np.interp(np.arange(ANALYSIS_FRAMES), trigger_positions, motion)
binned = aligned_motion.reshape(-1, FRAMES_PER_BIN).mean(axis=1)
```

iii. The agent reasons that timestamp scale is arbitrary but ratios reveal skipped imaging-triggered camera samples; interpolation follows the dataset README, and averaging matches the paper.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Within each session, 20th, 40th, 60th, and 80th percentiles of the aligned, 10-frame-averaged motion are computed. `searchsorted(..., side="right")` assigns integer labels 0–4.

ii.
```python
edges = np.quantile(binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, binned, side="right").astype(np.int64)
```

iii. This directly implements five equal-percentile bins selected separately per session and gave exactly 720 samples per class in each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera samples are mapped to reconstructed imaging-trigger ordinals, missing ordinals are linearly interpolated, and the retained aligned samples are averaged in the identical ten-frame bins used for neural data.

ii.
```python
target_positions = np.arange(ANALYSIS_FRAMES, dtype=np.float64)
aligned = np.interp(target_positions, trigger_positions, motion)
validate_session_arrays(neural_binned, elapsed_time, labels)
```

iii. The notes cite imaging-triggered camera acquisition and report independent reconstruction of all 282 missing triggers, including multi-frame gaps and gaps where raw array lengths alone would not reveal the problem.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are inferred from timestamp gaps and linearly interpolated. Malformed shapes, inconsistent intervals, inadequate coverage, unexpected sampling/baseline settings, and non-finite values raise errors rather than being silently repaired. Longer neural traces are deliberately cropped.

ii.
```python
if not np.allclose(intervals_stored, np.diff(timestamps), rtol=1e-8, atol=1e-12):
    raise ValueError(...)
if trigger_positions[-1] < ANALYSIS_FRAMES - 1:
    raise ValueError(...)
aligned = np.interp(target_positions, trigger_positions, motion)
```

iii. The agent describes interpolation as the documented remedy for dropped video frames and uses fail-fast validation for other anomalies.

## 6-a. What are the most time-consuming steps of the code?

i. Neural maximin baseline correction—the Gaussian and large-window minimum/maximum filtering over every neuron and full trace—is the dominant computation. Loading large fluorescence arrays and pickle serialization are secondary costs.

ii.
```python
flow = gaussian_filter1d(...)
flow = minimum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
flow = maximum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
```

iii. The notes report roughly 2.15 seconds per large sample session and 33.68 seconds for the full conversion, attributing most work to baseline correction.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer session loop and 64-neuron chunk loop remain. Sessions could be parallelized, while processing every neuron simultaneously would remove the chunk loop but greatly increase peak memory. Trial creation uses list comprehensions around already vectorized `np.split`; plotting loops are diagnostic-only.

ii.
```python
for session_index, (subject, session_dir) in enumerate(selected):
    ...
for start in range(0, n_neurons, CHUNK_NEURONS):
    ...
neural_trials = [x.copy() for x in np.split(neural_binned, N_TRIALS, axis=1)]
```

iii. The agent intentionally keeps chunking as a memory optimization and says the expensive filters and 10-frame means are vectorized within each chunk.

## 6-c. What processing does the code repeat multiple times?

i. The same elapsed-time array is copied and split for every session; identical validation, filtering, binning, class counting, and metadata construction repeat per session. With `--show-processing`, a small neural subset is averaged again for display and raw motion is loaded a second time.

ii.
```python
input_trials = [x[np.newaxis, :].copy() for x in np.split(elapsed_time, N_TRIALS)]
raw_motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy")
neural_binned = neural_small.reshape(...).mean(2)
```

iii. The notes emphasize avoiding full diagnostic recomputation; only limited plotting intermediates are retained. Repeated per-session work is mostly required because neural data and thresholds differ by session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. For recordings longer than 20 minutes, baseline filtering is performed over the full trace even though frames after 36,000 are discarded. Motion is loaded as float64 and several provenance statistics are computed though the decoder uses only labels. Optional plots also recompute/display intermediates not stored in the decoder dataset.

ii.
```python
raw_f = np.asarray(fluorescence[start:stop], dtype=np.float32)
...
retained = baseline_corrected[:, :ANALYSIS_FRAMES]
counts = np.bincount(labels, minlength=5)
```

iii. Full-trace baseline processing is intentional: the agent argues later context makes baseline estimation near the 20-minute boundary faithful to Suite2p. The extra metadata and optional plots support auditability rather than decoding.
