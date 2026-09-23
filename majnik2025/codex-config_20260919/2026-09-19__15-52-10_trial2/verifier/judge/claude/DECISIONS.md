# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by iterating sorted subject directories under `/app/data`, then sorted session subdirectories within each subject. It filters to only include session directories that contain `suite2p/plane0/F.npy`. For each session, it loads `F.npy` and `Fneu.npy` (fluorescence and neuropil) via memory-mapped `np.load`, `ops.npy` for processing parameters, and `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy` from the `move_deve` subdirectory for behavioral data.

ii.
```python
def discover_sessions() -> list[tuple[str, Path]]:
    sessions: list[tuple[str, Path]] = []
    for subject_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            if (session_dir / "suite2p" / "plane0" / "F.npy").exists():
                sessions.append((subject_dir.name, session_dir))
    return sessions
```

Loading neural data:
```python
fluorescence = np.load(plane_dir / "F.npy", mmap_mode="r")
neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

Loading behavioral data:
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals_stored = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. The AI documented in CONVERSION_NOTES Steps 1-2 that the data follows subject/session directory structure with suite2p outputs and behavioral recordings. Loading all three behavioral files (motion, timestamps, interframe intervals) enables robust dropped-frame detection.

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted top-level directories in the data root. The AI builds a sorted unique list of subject names from all discovered sessions.

ii.
```python
subjects = sorted({subject for subject, _ in selected})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
```

iii. CONVERSION_NOTES Step 2 documents 6 subjects (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), matching the paper's reported count.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject directory. Only directories containing `suite2p/plane0/F.npy` are included. Each session represents one daily recording.

ii.
```python
for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
    if (session_dir / "suite2p" / "plane0" / "F.npy").exists():
        sessions.append((subject_dir.name, session_dir))
```

iii. CONVERSION_NOTES Step 2 documents 41 total sessions (7,7,7,7,6,7 per subject), matching the data files.

## 1-d. How are the data split into trials?

i. The AI restricts each session to the first 36,000 frames (20 minutes at 30 Hz), matching the paper's stated analysis window. After 10-frame binning, this yields 3,600 bins. Trials are defined as 20 consecutive non-overlapping 60-second segments (180 bins each), with no remainder. This results in exactly 20 trials per session and 820 total trials.

ii.
```python
ANALYSIS_FRAMES = 36_000  # first 20 min, matching the paper
N_TRIALS = ANALYSIS_FRAMES // (FRAMES_PER_BIN * TRIAL_BINS)  # = 20

neural_trials = [
    x.copy()
    for x in np.split(neural_binned, N_TRIALS, axis=1)
]
```

iii. CONVERSION_NOTES Step 4 documents the discrepancy between 20-min and 30-min recordings and resolves it by restricting to 20 minutes, citing the paper's statement that "each session lasted 20 minutes" and that all figures show 0-20 min analysis windows.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial quality filtering is applied. All 20 trials per session are retained. The only filtering is the restriction to the first 20 minutes of data.

ii. N/A

iii. CONVERSION_NOTES documents that the paper treated recordings as continuous with no trial-level quality control.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (processing parameters including neucoeff, baseline method, sigma, window size, and frame rate), all from `suite2p/plane0/`.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy", mmap_mode="r")
neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. CONVERSION_NOTES Step 5 specifies using baseline-corrected fluorescence rather than `spks` or raw `F`, matching the paper's decoding pipeline.

## 2-b. How is the `neural` data processed?

i. The AI reimplements Suite2p's default baseline correction pipeline using scipy functions rather than importing `dcnv.preprocess`:
1. Neuropil subtraction: `Fc = F - neucoeff * Fneu` (neucoeff=0.7 from ops.npy)
2. Gaussian smoothing: `gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)`
3. Minimum filter: `minimum_filter1d(flow, size=win_frames, axis=1)`
4. Maximum filter: `maximum_filter1d(flow, size=win_frames, axis=1)`
5. Baseline subtraction: `Fc - flow`
6. Crop to first 36,000 frames
7. Average into non-overlapping 10-frame bins

Processing is done in chunks of 64 neurons to manage memory.

ii.
```python
neucoeff = float(ops.get("neucoeff", 0.7))
corrected_neuropil = raw_f - np.float32(neucoeff) * raw_fneu
flow = gaussian_filter1d(corrected_neuropil, sigma=sig_baseline, axis=1, mode="reflect")
flow = minimum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
flow = maximum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
baseline_corrected = corrected_neuropil - flow
retained = baseline_corrected[:, :ANALYSIS_FRAMES]
binned[start:stop] = retained.reshape(stop - start, n_bins, FRAMES_PER_BIN).mean(axis=2)
```

iii. CONVERSION_NOTES Step 4 resolves the processing parameters by following the Suite2p ops values stored per session (neucoeff=0.7, maximin baseline, sig_baseline=10, win_baseline=60s) rather than Track2p GUI defaults (neucoeff=0). The AI processes the full trace before cropping to preserve baseline context near the 20-minute boundary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the released F.npy arrays are retained. The AI documents that the released data already contains Track2p-matched, iscell-filtered neurons.

ii. N/A (no filtering code)

iii. CONVERSION_NOTES Steps 1-2 document that Suite2p cell detection and Track2p longitudinal matching have already been applied in the released data. All saved iscell rows have class 1 and probability >0.5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording starting from frame 0, no event-based alignment is needed. The metadata records `temporal_alignment_event` as "Session start; trials are consecutive non-overlapping 60-second windows."

ii.
```python
"temporal_alignment_event": (
    "Session start; trials are consecutive non-overlapping 60-second windows."
),
"off_start": None,
"off_end": None,
```

iii. CONVERSION_NOTES Step 4 documents that there is no stimulus event to align to, as recordings capture spontaneous behavior.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and behavioral data are averaged into non-overlapping bins of 10 consecutive frames, converting 30 Hz to 3 Hz (333.33 ms time bins). This is applied after baseline correction and motion alignment.

ii.
```python
FRAMES_PER_BIN = 10
BINNED_FS_HZ = RAW_FS_HZ / FRAMES_PER_BIN  # 3.0

# Neural binning:
binned[start:stop] = retained.reshape(stop - start, n_bins, FRAMES_PER_BIN).mean(axis=2)

# Motion binning:
binned = aligned_motion.reshape(-1, FRAMES_PER_BIN).mean(axis=1)
```

iii. CONVERSION_NOTES Step 3 cites the paper: "averaging in bins of 10 consecutive timestamps" for all decoding analyses.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from a raw data variable. It is computed from the imaging frame indices and the known frame rate (30 Hz). Specifically, it represents the mean time of the 10 frames within each bin.

ii.
```python
elapsed_time = (
    np.arange(ANALYSIS_FRAMES, dtype=np.float64)
    .reshape(-1, FRAMES_PER_BIN)
    .mean(axis=1)
    / RAW_FS_HZ
).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 documents the input as absolute session time, computed from frame indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices 0 through 35,999 are grouped into bins of 10, the mean frame index within each bin is computed, and divided by 30 Hz to get seconds. This produces bin-center times (e.g., first bin = mean of frames 0-9 = 4.5, divided by 30 = 0.15 s). The same elapsed_time array is shared across all sessions.

ii.
```python
elapsed_time = (
    np.arange(ANALYSIS_FRAMES, dtype=np.float64)
    .reshape(-1, FRAMES_PER_BIN)
    .mean(axis=1)
    / RAW_FS_HZ
).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Each averaged neural/output sample represents the mean of ten frame times, so the corresponding input is the mean time, not the left edge."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The elapsed_time array has the same number of bins as the neural data (3,600 bins), computed from the same frame indices. It is split into trials identically to the neural data. Time continues across trials within a session (not reset per trial).

ii.
```python
input_trials = [
    x[np.newaxis, :].copy()
    for x in np.split(elapsed_time, N_TRIALS)
]
```

iii. Since both are derived from the same frame indexing, alignment is inherent.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy` in the `move_deve` subdirectory. The timestamps and interframe intervals are used to reconstruct camera trigger positions for alignment with neural data.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals_stored = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. CONVERSION_NOTES Step 2 documents the motion energy file as pre-computed global motion energy from behavioral video, with timestamps needed for frame alignment.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three main processing steps:
1. **Alignment**: Camera trigger positions are reconstructed by dividing each inter-frame interval by the median interval and rounding to integer steps. This identifies dropped frames. Motion energy is then linearly interpolated onto the expected 36,000 neural frame positions using `np.interp`.
2. **Binning**: Aligned motion is averaged into non-overlapping 10-frame bins (same as neural data).
3. **Discretization**: Binned motion is discretized into 5 quintiles using session-specific percentile edges at 20/40/60/80th percentiles, with `np.searchsorted(..., side='right')` producing labels 0-4.

ii.
```python
# Alignment
median_interval = float(np.median(intervals_stored))
trigger_steps = np.rint(intervals_stored / median_interval).astype(np.int64)
trigger_positions = np.concatenate([np.array([0], ...), np.cumsum(trigger_steps, ...)])
aligned = np.interp(target_positions, trigger_positions, motion)

# Binning
binned = aligned_motion.reshape(-1, FRAMES_PER_BIN).mean(axis=1)

# Discretization
edges = np.quantile(binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, binned, side="right").astype(np.int64)
```

iii. CONVERSION_NOTES Step 4 documents that camera trigger reconstruction from timestamp ratios is needed because some sessions have dropped video frames. The paper states microscope triggers camera frames, so integer trigger steps are the natural alignment unit.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Session-specific percentile edges at 20th, 40th, 60th, and 80th percentiles are computed on the binned motion energy. Values are assigned to 5 bins (labels 0-4) using `np.searchsorted` with `side='right'`, which means values equal to an edge go into the lower bin except at the maximum.

ii.
```python
edges = np.quantile(binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, binned, side="right").astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 documents using session-specific quintiles as required by the decoder task specification. The AI verified that each class contains exactly 20% of values per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera trigger positions are reconstructed from timestamp interval ratios, and motion energy is linearly interpolated onto the expected imaging frame indices (0 to 35,999). After interpolation, the aligned motion has exactly 36,000 samples matching the neural data. Both are then binned identically with 10-frame averaging.

ii.
```python
target_positions = np.arange(ANALYSIS_FRAMES, dtype=np.float64)
aligned = np.interp(target_positions, trigger_positions, motion)
```

iii. CONVERSION_NOTES Steps 4-5 document that the microscope-triggered camera acquisition means integer timestamp gaps represent skipped triggers, making trigger ordinal reconstruction the physically correct alignment method.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are detected via timestamp interval analysis (intervals significantly larger than the median indicate dropped frames). These are handled by linear interpolation of motion energy onto the expected neural frame grid. The AI validates: (1) stored intervals match timestamp differences, (2) trigger steps are positive, (3) behavior coverage extends to the analysis window end, (4) gap-free sessions are unchanged by alignment. Sessions shorter than 36,000 neural frames raise an error. The 20-minute restriction avoids partial-trial remainders.

ii.
```python
if not np.allclose(intervals_stored, np.diff(timestamps), rtol=1e-8, atol=1e-12):
    raise ValueError(...)
if trigger_positions[-1] < ANALYSIS_FRAMES - 1:
    raise ValueError(...)
identity_alignment = bool(
    len(motion) >= ANALYSIS_FRAMES
    and np.array_equal(trigger_positions[:ANALYSIS_FRAMES], np.arange(ANALYSIS_FRAMES))
)
```

iii. CONVERSION_NOTES Step 10 documents that all 282 missing camera frames within retained windows across all sessions are correctly interpolated, verified by independent reconstruction.

## 6-a. What are the most time-consuming steps of the code?

i. The baseline correction (Gaussian smoothing, minimum filter, maximum filter over full traces) is the most computationally expensive step. The AI processes neurons in chunks of 64 and uses memory-mapped loading to manage memory.

ii.
```python
CHUNK_NEURONS = 64
for start in range(0, n_neurons, CHUNK_NEURONS):
    stop = min(start + CHUNK_NEURONS, n_neurons)
    # ... baseline correction on chunk ...
```

iii. CONVERSION_NOTES Step 7 reports full conversion completed in 33.68 seconds for 41 sessions, with ~2.15 s/session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron processing loop (chunks of 64) could theoretically be done as a single vectorized operation on all neurons at once, though this would increase memory usage. The motion alignment uses `np.interp` which is already vectorized.

ii.
```python
for start in range(0, n_neurons, CHUNK_NEURONS):
    # Process chunk of neurons
```

iii. The chunking is a deliberate memory-efficiency tradeoff. The AI avoided the inefficient `np.insert` loop that would arise from one-at-a-time frame interpolation by using `np.interp` for the entire alignment.

## 6-c. What processing does the code repeat multiple times?

i. The elapsed_time array is computed once and reused for all sessions. No significant repeated processing was identified.

ii.
```python
elapsed_time = (
    np.arange(ANALYSIS_FRAMES, dtype=np.float64)
    .reshape(-1, FRAMES_PER_BIN)
    .mean(axis=1)
    / RAW_FS_HZ
).astype(np.float32)
```

iii. The AI designed the code to avoid redundant computation. The same elapsed_time is shared across all sessions since all use the same 20-minute window.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI processes the full fluorescence trace (up to 54,000 frames for 30-minute sessions) before cropping to the first 36,000 frames. This is intentional: baseline estimation near the 20-minute boundary benefits from having additional data context. However, the extra 18,000 frames of baseline-corrected fluorescence are discarded after cropping.

ii.
```python
# Full trace processing
flow = gaussian_filter1d(corrected_neuropil, sigma=sig_baseline, axis=1, mode="reflect")
flow = minimum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
flow = maximum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
baseline_corrected = corrected_neuropil - flow
# Crop to 20 min
retained = baseline_corrected[:, :ANALYSIS_FRAMES]
```

iii. CONVERSION_NOTES Step 5: "Processing the full available trace before cropping preserves baseline context for 30-min source arrays."
