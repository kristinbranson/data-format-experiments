# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by scanning every top-level directory in `/app/data`, then every subdirectory inside each subject directory, and keeps only subdirectories that contain `suite2p/plane0/F.npy`. For each retained session it loads fluorescence, neuropil, Suite2p `ops`, motion energy, timestamps, and stored inter-frame intervals. Trials are not loaded from raw data; they are created later from the processed session-long arrays.

ii.
```python
def discover_sessions() -> list[tuple[str, Path]]:
    sessions: list[tuple[str, Path]] = []
    for subject_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            if (session_dir / "suite2p" / "plane0" / "F.npy").exists():
                sessions.append((subject_dir.name, session_dir))
    return sessions

motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals_stored = np.load(move_dir / "interframe_int.npy").astype(np.float64)

fluorescence = np.load(plane_dir / "F.npy", mmap_mode="r")
neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI says the release already contains curated Track2p-matched Suite2p outputs plus motion-energy files, so it should discover sessions directly from the released directory structure and load those arrays without re-running Track2p.

## 1-b. How are the data split into subjects?

i. Subjects are the names of the top-level directories returned by `discover_sessions()`. In the final dataset they are deduplicated and sorted, and each session gets a `subject_idx` via a lookup table.

ii.
```python
selected = select_sessions(discover_sessions(), sample=sample)
subjects = sorted({subject for subject, _ in selected})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}

"subject_idx": np.asarray(
    [subject_lookup[subject] for subject, _ in selected], dtype=np.int64
),
```

iii. Step 2 of the notes says the release has six subject directories (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), so the AI treated those directory names as mouse identifiers.

## 1-c. How are the data split into sessions?

i. Each session is one session directory under a subject directory. Sessions are sorted lexicographically within each subject and represented as one entry in `data['neural']`, `data['input']`, `data['output']`, and `data['brain_region_idx']`.

ii.
```python
for subject_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        if (session_dir / "suite2p" / "plane0" / "F.npy").exists():
            sessions.append((subject_dir.name, session_dir))

for session_index, (subject, session_dir) in enumerate(selected):
    ...
    data["neural"].append(neural_trials)
    data["input"].append(input_trials)
    data["output"].append(output_trials)
```

iii. The notes describe each daily recording folder as one session and emphasize deterministic subject/date ordering.

## 1-d. How are the data split into trials?

i. The AI does not use the entire session. It first truncates each session to `ANALYSIS_FRAMES = 36_000` raw frames, i.e. the first 20 minutes. It then bins by 10 frames, giving 3,600 bins per session, and splits those into `N_TRIALS = 20` consecutive non-overlapping 60-second trials of `TRIAL_BINS = 180` bins each.

ii.
```python
ANALYSIS_FRAMES = 36_000  # first 20 min, matching the paper
FRAMES_PER_BIN = 10
TRIAL_SECONDS = 60
BINNED_FS_HZ = RAW_FS_HZ / FRAMES_PER_BIN
TRIAL_BINS = int(TRIAL_SECONDS * BINNED_FS_HZ)
N_TRIALS = ANALYSIS_FRAMES // (FRAMES_PER_BIN * TRIAL_BINS)

neural_trials = [x.copy() for x in np.split(neural_binned, N_TRIALS, axis=1)]
input_trials = [x[np.newaxis, :].copy() for x in np.split(elapsed_time, N_TRIALS)]
output_trials = [x[np.newaxis, :].copy() for x in np.split(labels, N_TRIALS)]
```

iii. In Step 4 and Step 5, the AI justified this by saying the paper analyzes 20-minute sessions even though many released arrays are 30 minutes, so it restricted all sessions to the first 20 minutes to match the paper and then imposed the requested 60-second decoder trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-specific quality-control filtering after trial creation. All 20 generated trials per session are kept as long as the session passes array-shape and alignment checks. The effective curation is upstream: only the first 20 minutes are retained, and malformed sessions would raise errors rather than be partially filtered.

ii.
```python
def validate_session_arrays(
    neural_binned: np.ndarray, elapsed_time: np.ndarray, labels: np.ndarray
) -> None:
    expected_bins = N_TRIALS * TRIAL_BINS
    if neural_binned.shape[1] != expected_bins:
        raise AssertionError(...)
    if elapsed_time.shape != (expected_bins,) or labels.shape != (expected_bins,):
        raise AssertionError(...)

if not (
    len(neural_trials) == len(input_trials) == len(output_trials) == N_TRIALS
):
    raise AssertionError("Trial count mismatch")
```

iii. The notes say every retained session yields 20 complete trials and that the release already contains curated neural rows, so no additional trial rejection was necessary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`. It also reads `ops.npy` to get processing parameters such as `fs`, `neucoeff`, `baseline`, `sig_baseline`, and `win_baseline`.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy", mmap_mode="r")
neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. In Step 3 through Step 5, the notes say the paper’s decoder used baseline-corrected fluorescence rather than `spks.npy`, and that the released `ops.npy` files provide the Suite2p-default parameters needed to reconstruct that signal.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction and then manually reimplements the Suite2p maximin baseline procedure with SciPy filters: Gaussian smoothing, minimum filter, maximum filter, and subtraction of the estimated baseline. It performs that on the full source trace for each neuron chunk, retains only the first 36,000 frames, and averages non-overlapping groups of 10 frames.

ii.
```python
neucoeff = float(ops.get("neucoeff", 0.7))
baseline = ops.get("baseline", "maximin")
sig_baseline = float(ops.get("sig_baseline", 10.0))
win_baseline_s = float(ops.get("win_baseline", 60.0))
...
corrected_neuropil = raw_f - np.float32(neucoeff) * raw_fneu
flow = gaussian_filter1d(corrected_neuropil, sigma=sig_baseline, axis=1, mode="reflect")
flow = minimum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
flow = maximum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
baseline_corrected = corrected_neuropil - flow
retained = baseline_corrected[:, :ANALYSIS_FRAMES]
binned[start:stop] = retained.reshape(stop - start, n_bins, FRAMES_PER_BIN).mean(axis=2)
```

iii. The notes argue that this matches the paper’s “Suite2p-default baseline-corrected fluorescence,” that `ops.npy` resolves the parameters, and that processing the full trace before cropping preserves baseline context for 30-minute sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply additional neuron filtering in `convert_data.py`. It uses all rows present in the released `F.npy`/`Fneu.npy` arrays for each session and assumes those rows are already the curated all-day-complete Track2p cells.

ii.
```python
n_neurons, source_frames = fluorescence.shape
binned = np.empty((n_neurons, n_bins), dtype=np.float32)
...
data["brain_region_idx"].append(
    np.zeros(neural_binned.shape[0], dtype=np.int64)
)
```

iii. Step 2 and Step 4 of the notes explicitly state that the distributed arrays already contain the final matched subset and that re-filtering with `iscell` would be redundant or unjustified.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to session start. The AI treats the task as continuous spontaneous behavior, so each trial is simply the next consecutive 60-second window after session onset within the retained 20-minute window.

ii.
```python
"temporal_alignment_event": (
    "Session start; trials are consecutive non-overlapping 60-second windows."
),
"off_start": None,
"off_end": None,
```

iii. The notes say there is no stimulus/event structure in the raw data, so session start is the only meaningful alignment event for the imposed fixed-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are rebinned from 30 Hz to 3 Hz by averaging every 10 consecutive raw frames, giving a 333.33 ms bin size. This same rebinning is applied to neural data and motion energy.

ii.
```python
FRAMES_PER_BIN = 10
...
"time_bin_size": 1000.0 * FRAMES_PER_BIN / RAW_FS_HZ,
...
binned[start:stop] = retained.reshape(stop - start, n_bins, FRAMES_PER_BIN).mean(axis=2)
binned = aligned_motion.reshape(-1, FRAMES_PER_BIN).mean(axis=1)
```

iii. The notes cite the paper’s decoding procedure, which averaged both fluorescence and behavior in non-overlapping groups of 10 timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not read a stored time-from-start variable. It derives elapsed time from imaging frame indices `0..35999`, the assumed raw sampling rate `RAW_FS_HZ = 30`, and the 10-frame binning scheme.

ii.
```python
elapsed_time = (
    np.arange(ANALYSIS_FRAMES, dtype=np.float64)
    .reshape(-1, FRAMES_PER_BIN)
    .mean(axis=1)
    / RAW_FS_HZ
).astype(np.float32)
```

iii. In Step 5, the notes say elapsed time is a required decoder input rather than a native dataset variable, so it should be constructed directly from the imaging clock.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes the mean time of each 10-frame bin, i.e. the bin center rather than the left edge. It first creates raw frame indices for the retained 20-minute window, reshapes them into 10-frame groups, averages within each group, and converts the result to seconds.

ii.
```python
elapsed_time = (
    np.arange(ANALYSIS_FRAMES, dtype=np.float64)
    .reshape(-1, FRAMES_PER_BIN)
    .mean(axis=1)
    / RAW_FS_HZ
).astype(np.float32)
```

iii. Step 5 explicitly justifies bin-center time: each neural/output sample is itself a 10-frame average, so the corresponding time input should represent the mean time of those 10 raw frames.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time series is aligned by construction. It has one value per 10-frame bin across the retained 20-minute window, exactly matching the binned neural data length, and is split into the same 20 consecutive trials.

ii.
```python
validate_session_arrays(neural_binned, elapsed_time, labels)

input_trials = [
    x[np.newaxis, :].copy()
    for x in np.split(elapsed_time, N_TRIALS)
]
```

iii. The notes describe this as a direct consequence of constructing time on the same 3 Hz bin grid used for neural and behavioral signals.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output starts from `move_deve/motion_energy_glob.npy`. The AI also uses `tstamps.npy` and `interframe_int.npy` to reconstruct the camera timing and infer dropped frames before alignment.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals_stored = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. In Step 2 and Step 4, the notes argue that timestamps are the authoritative behavior clock and that the stored inter-frame intervals should agree with `np.diff(tstamps)`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI verifies timestamp consistency, infers integer camera-trigger steps by dividing each inter-frame interval by the median interval and rounding, reconstructs trigger positions, linearly interpolates motion onto raw imaging frame indices `0..35999`, averages non-overlapping 10-frame bins, and then discretizes the binned trace into session-specific quintiles.

ii.
```python
if not np.allclose(intervals_stored, np.diff(timestamps), rtol=1e-8, atol=1e-12):
    raise ValueError(...)

median_interval = float(np.median(intervals_stored))
trigger_steps = np.rint(intervals_stored / median_interval).astype(np.int64)
trigger_positions = np.concatenate(
    [np.array([0], dtype=np.int64), np.cumsum(trigger_steps, dtype=np.int64)]
)
target_positions = np.arange(ANALYSIS_FRAMES, dtype=np.float64)
aligned = np.interp(target_positions, trigger_positions, motion)

binned = aligned_motion.reshape(-1, FRAMES_PER_BIN).mean(axis=1)
edges = np.quantile(binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, binned, side="right").astype(np.int64)
```

iii. The notes justify this as a more faithful implementation of the paper/README synchronization story: microscope-triggered video, missing camera frames, and interpolation onto the imaging frame grid.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is converted into five per-session categories using the 20th, 40th, 60th, and 80th percentiles of the session’s 10-frame-averaged motion trace. Labels are assigned with `np.searchsorted(..., side="right")`, yielding class IDs 0 through 4.

ii.
```python
edges = np.quantile(binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, binned, side="right").astype(np.int64)

"output_values": [["lowest", "low", "middle", "high", "highest"]],
```

iii. This is documented in Step 5 as the explicit downstream-task override required by the instructions: five equal-percentile bins chosen separately within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to the neural data on the raw 30 Hz imaging frame grid before any binning. It reconstructs the expected camera trigger ordinals from timestamp ratios, interpolates the motion signal onto frame indices `0..35999`, verifies the retained window is covered, and only then bins to 3 Hz.

ii.
```python
trigger_steps = np.rint(intervals_stored / median_interval).astype(np.int64)
trigger_positions = np.concatenate(
    [np.array([0], dtype=np.int64), np.cumsum(trigger_steps, dtype=np.int64)]
)
target_positions = np.arange(ANALYSIS_FRAMES, dtype=np.float64)
if trigger_positions[-1] < ANALYSIS_FRAMES - 1:
    raise ValueError(...)
aligned = np.interp(target_positions, trigger_positions, motion)
```

iii. Step 4 states that timestamps remain the authoritative behavior clock and that interpolation onto imaging-frame indices is needed because equal-length vectors can still contain timestamp gaps.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or inconsistent camera samples by checking that `interframe_int.npy` matches timestamp differences, inferring skipped trigger steps from unusually long intervals, and interpolating the motion trace across the missing positions. It also performs defensive shape/rate checks and raises hard errors if required invariants fail. For heterogeneous session durations, it standardizes all sessions by keeping only the first 20 minutes.

ii.
```python
if len(motion) != len(timestamps):
    raise ValueError(...)
if len(intervals_stored) != len(timestamps) - 1:
    raise ValueError(...)
if not np.allclose(intervals_stored, np.diff(timestamps), rtol=1e-8, atol=1e-12):
    raise ValueError(...)

trigger_steps = np.rint(intervals_stored / median_interval).astype(np.int64)
aligned = np.interp(target_positions, trigger_positions, motion)

if fluorescence.shape[1] < ANALYSIS_FRAMES:
    raise ValueError(...)
```

iii. The notes repeatedly justify this as a consequence of the release’s dropped-camera-frame irregularities and the paper-level desire for a common 20-minute analysis window.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s most expensive step is fluorescence processing: neuropil subtraction, Gaussian smoothing, min/max baseline filtering, and 10-frame averaging across all neurons in each session. Optional diagnostic plotting is extra work, but only for up to two sessions.

ii.
```python
for start in range(0, n_neurons, CHUNK_NEURONS):
    ...
    flow = gaussian_filter1d(...)
    flow = minimum_filter1d(...)
    flow = maximum_filter1d(...)
    baseline_corrected = corrected_neuropil - flow
    binned[start:stop] = retained.reshape(...).mean(axis=2)
```

iii. Step 6 says the code was designed around that cost: memory-mapped arrays, chunked processing, and avoiding redundant full-array passes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining explicit loop is the chunk-wise loop over neurons in `process_fluorescence`. That loop could be removed by processing all neurons at once on sufficiently large hardware, but the AI deliberately kept it to bound memory. There are also session-level and trial-splitting Python loops/list comprehensions, though those are not the dominant cost.

ii.
```python
for start in range(0, n_neurons, CHUNK_NEURONS):
    stop = min(start + CHUNK_NEURONS, n_neurons)
    ...

neural_trials = [x.copy() for x in np.split(neural_binned, N_TRIALS, axis=1)]
input_trials = [x[np.newaxis, :].copy() for x in np.split(elapsed_time, N_TRIALS)]
output_trials = [x[np.newaxis, :].copy() for x in np.split(labels, N_TRIALS)]
```

iii. Step 6 explicitly frames chunking as a speed/memory tradeoff rather than a mistake: full-session vectorization would increase temporary-array memory substantially.

## 6-c. What processing does the code repeat multiple times?

i. The AI avoids most repeated heavy processing. The notable repeated work is only in optional debugging: when plotting is enabled it loads `motion_energy_glob.npy` a second time for display, even though aligned/binned motion has already been computed.

ii.
```python
aligned_motion, motion_info = load_and_align_motion(session_dir)
...
if want_debug:
    raw_motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy")
    plot_path = make_processing_plot(...)
```

iii. In Step 6, the notes explicitly say redundant diagnostic recomputation was avoided; only lightweight extra loading for plots remains.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest discarded computation is that the AI processes full 30-minute fluorescence traces for long sessions, but then throws away everything after the first 20 minutes. It likewise loads full motion/timestamp arrays and only retains the first 36,000 aligned frames. Optional debug intermediates and plots are also discarded from the saved dataset.

ii.
```python
ANALYSIS_FRAMES = 36_000  # first 20 min, matching the paper
...
baseline_corrected = corrected_neuropil - flow
retained = baseline_corrected[:, :ANALYSIS_FRAMES]
...
target_positions = np.arange(ANALYSIS_FRAMES, dtype=np.float64)
aligned = np.interp(target_positions, trigger_positions, motion)
```

iii. The notes acknowledge this directly in Step 5: the AI intentionally processes full traces before cropping because it wanted baseline estimation to use the full acquired context, even though the downstream saved dataset only keeps the first 20 minutes.
