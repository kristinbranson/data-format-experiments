# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by iterating all directories in the data root (no name-prefix filter). Sessions are sorted subdirectories within each subject. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy` from `move_deve/`. It also loads `spks.npy`, `iscell.npy`, `stat.npy`, and `ops.npy` for validation purposes during `validate_source_session`.

ii.
```python
def discover_sessions() -> tuple[list[str], list[SessionRef]]:
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    sessions: list[SessionRef] = []
    for subject in subjects:
        subject_dir = DATA_ROOT / subject
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            required = (
                session_dir / "suite2p" / "plane0" / "F.npy",
                session_dir / "suite2p" / "plane0" / "Fneu.npy",
                session_dir / "move_deve" / "motion_energy_glob.npy",
            )
            if not all(p.exists() for p in required):
                raise FileNotFoundError(...)
            sessions.append(SessionRef(subject, session_dir.name, session_dir))
    return subjects, sessions
```

Loading neural data:
```python
f = np.load(plane_dir / "F.npy", mmap_mode="r")
fneu = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
```

Loading motion data:
```python
raw = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. The AI documented that the data directory contains six subject folders with session subdirectories, each containing suite2p outputs and behavioral data. Additional files (ops, iscell, stat, spks) are loaded for validation to confirm the data has already been curated by Track2p.

## 1-b. How are the data split into subjects?

i. Subjects are all directories in the data root, sorted alphabetically. Unlike the reference, the AI does not filter by a `jm*` prefix, relying instead on the assumption that only subject directories exist.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
```

iii. The AI noted that the data directory contains six subject folders (jm031-jm046) and a README.md file. Since README.md is not a directory, the lack of prefix filtering produces the same result in practice.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject folder. Each subdirectory contains one daily recording. The AI validates that required files exist in each session directory.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
    required = (
        session_dir / "suite2p" / "plane0" / "F.npy",
        session_dir / "suite2p" / "plane0" / "Fneu.npy",
        session_dir / "move_deve" / "motion_energy_glob.npy",
    )
    if not all(p.exists() for p in required):
        raise FileNotFoundError(...)
    sessions.append(SessionRef(subject, session_dir.name, session_dir))
```

iii. The AI documented that each subject has 6-7 sessions, yielding 41 sessions total (7,7,7,7,6,7 per subject). Sessions are sorted chronologically by directory name.

## 1-d. How are the data split into trials?

i. Trials are 60-second non-overlapping segments of the continuous recording. At 3 Hz (after 10-frame binning), each trial is 180 bins. The AI enforces that sessions divide exactly into complete trials (raises an error if there is a remainder), unlike the reference which silently discards remainders.

ii.
```python
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * OUTPUT_FS_HZ)  # 180

# In validate_source_session:
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(f"Session does not divide into full 60-s trials: {ref.path}")

# In convert_session:
n_trials = n_binned // TRIAL_BINS
for trial in range(n_trials):
    sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
```

iii. The AI noted that all sessions have exactly 36,000 or 54,000 frames which divide evenly into 20 or 30 trials. The exact-division check is a stricter validation than the reference's approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are included. The AI validates that sessions divide exactly into full trials (no remainder to discard).

ii. N/A (no filtering code)

iii. The AI documented that there are no experimental trials or trial-level quality criteria in this dataset. The recording is continuous with no invalid-period markers.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
f = np.load(plane_dir / "F.npy", mmap_mode="r")
fneu = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
```

iii. The AI identified these as standard suite2p output files. The `ops.npy` metadata was used to verify parameters (30 Hz, neucoeff=0.7, etc.) but not as a data source.

## 2-b. How is the `neural` data processed?

i. The AI manually reimplements suite2p's maximin baseline correction rather than calling `dcnv.preprocess()`. The pipeline is: (1) neuropil subtraction `F - 0.7*Fneu`, (2) Gaussian smoothing with sigma=10 frames, (3) minimum filter with window=1800 frames, (4) maximum filter with window=1800 frames, (5) subtract baseline, (6) average into non-overlapping 10-frame bins. Processing is done in 64-neuron chunks.

ii.
```python
corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu  # 0.7
baseline = gaussian_filter(corrected_neuropil, sigma=(0.0, BASELINE_SIGMA_FRAMES))  # sigma=10
baseline = minimum_filter1d(baseline, size=baseline_window, axis=1)  # 1800 frames
baseline = maximum_filter1d(baseline, size=baseline_window, axis=1)
activity = corrected_neuropil - baseline
result[start:stop] = activity.reshape(stop - start, n_binned, DOWNSAMPLE).mean(axis=2)
```

iii. The AI documented in CONVERSION_NOTES Step 4 that the processing follows paper/default Suite2p settings: `Fc=F-0.7*Fneu`, Gaussian sigma 10 frames, min then max filters over 1800 frames, and `Fc-Flow`. They chose to reimplement rather than use `dcnv.preprocess` directly, using scipy filters. They verified results with `np.allclose(rtol=1e-6, atol=1e-5)` against independently computed values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI verified that all neurons in the packaged data already pass Track2p curation criteria (iscell probability > 0.5, tracked across all days).

ii.
```python
# In validate_source_session:
if not np.all(iscell[:, 0] == 1) or not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"Packaged neurons do not all pass Track2p curation: {ref.path}")
```

iii. The AI explicitly validated that the data is pre-curated rather than just assuming it. CONVERSION_NOTES document that every `iscell` label is 1 and every probability > 0.5, so no further filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The AI sets `off_start=0.0` and `off_end=60.0`.

ii.
```python
'temporal_alignment_event': 'start of each non-overlapping 60-second segment',
'off_start': 0.0,
'off_end': 60.0,
```

iii. There is no stimulus event to align to. The AI documented that the recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
DOWNSAMPLE = 10
OUTPUT_FS_HZ = NATIVE_FS_HZ / DOWNSAMPLE  # 3.0
TIME_BIN_MS = 1000.0 / OUTPUT_FS_HZ       # 333.33

# Neural binning (vectorized reshape):
result[start:stop] = activity.reshape(stop - start, n_binned, DOWNSAMPLE).mean(axis=2)

# Motion binning:
motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
```

iii. The AI cited the Methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from bin indices, the downsampling factor, and the sampling rate.

ii.
```python
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
```

iii. The AI noted that since the frame rate is constant at 30 Hz, time can be computed analytically from bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Each time value represents the **center** of the 10-frame bin: `(bin_index * 10 + 4.5) / 30` seconds. The first bin has time 0.15 s (center of frames 0-9). Time continues across trial boundaries within a session.

ii.
```python
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
# For bin 0: (0*10 + 4.5)/30 = 0.15 s
# For bin 1: (1*10 + 4.5)/30 = 0.483 s
```

iii. The AI described the 4.5-frame offset as "the mean acquisition time of the ten samples represented by that bin," arguing this is a more precise temporal representation. This differs from the reference which uses the left edge of each bin (starting at 0.0 s).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices used for neural data, so alignment is inherent. Both share the same 10-frame binning grid.

ii.
```python
# Both use the same indexing:
input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
```

iii. No explicit alignment step is needed since time is derived from the same bin structure.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps (`tstamps.npy`) and interframe intervals (`interframe_int.npy`) are used to detect and repair dropped camera frames.

ii.
```python
raw = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. The AI documented that the motion energy file contains pre-computed global motion energy from the behavioral video, and the timestamp files are needed to identify dropped frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped frames are detected via median interval ratios and filled by linear interpolation using `np.interp`, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 percentile-based bins per session using `np.quantile` and `np.searchsorted`.

ii.
```python
# 1. Alignment/interpolation:
median_interval = float(np.median(positive))
interval_steps = np.maximum(1, np.rint(intervals / median_interval).astype(int))
observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
aligned = np.interp(np.arange(expected_frames, dtype=np.float64), observed_positions, raw)

# 2. Binning:
motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)

# 3. Discretization:
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(quantile_edges, motion_binned, side="right").astype(np.int64)
```

iii. The AI documented that linear interpolation fills gaps so both streams can be indexed identically, and that per-session quintiles produce exactly balanced classes.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session. The AI computes edges at the 20th, 40th, 60th, and 80th percentiles using `np.quantile`, then assigns labels 0-4 using `np.searchsorted(side='right')`. An assertion verifies that each class has exactly 20% of the data.

ii.
```python
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(quantile_edges, motion_binned, side="right").astype(np.int64)
counts = np.bincount(labels, minlength=5)
expected = len(labels) // 5
if not np.array_equal(counts, np.full(5, expected)):
    raise ValueError(f"Motion quintiles are not equal: counts={counts.tolist()}")
```

iii. The AI documented that the five-class discretization is task-required (not from the paper, which uses continuous regression). The exact-balance check ensures no ties distort the quintiles.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. Dropped camera frames make the motion energy array shorter than neural data in some sessions. The AI detects gaps using median interval ratios from `interframe_int.npy`, reconstructs integer frame positions, and uses `np.interp` to fill missing values. After interpolation, the length is verified to match.

ii.
```python
positive = intervals[intervals > 0]
median_interval = float(np.median(positive))
interval_steps = np.maximum(1, np.rint(intervals / median_interval).astype(int))
observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
aligned = np.interp(np.arange(expected_frames, dtype=np.float64), observed_positions, raw)
```

iii. The AI documented 276 missing frames across 10 sessions, and that full-length sessions with timestamp anomalies (jm046 days 3/5/6) are left untouched since no actual camera samples are absent.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected and interpolated (see 4-d). Full-length sessions with timestamp anomalies are left unchanged. An assertion verifies motion energy length matches neural data. The AI requires exact trial division (no remainder frames).

ii.
```python
if len(aligned) != expected_frames or not np.all(np.isfinite(aligned)):
    raise ValueError(f"Failed motion alignment in {move_dir}")

# In validate_source_session:
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(f"Session does not divide into full 60-s trials: {ref.path}")
```

iii. The AI documented that all 276 missing frames are explained by timestamp gaps and filled by interpolation. Suite2p bad frames (only 1 in the entire dataset) are retained since they are finite and processed.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction (Gaussian smoothing + min/max filtering) applied to all neurons in each session. The AI processes neurons in 64-neuron chunks with scipy filters. Total conversion for all 41 sessions took ~48 seconds.

ii. N/A

iii. The AI documented timing per session and estimated total conversion time before running the full dataset.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the key operations: scipy filters operate on neuron chunks, binning uses reshape+mean, and motion interpolation uses `np.interp`. The neuron chunking loop (64 neurons at a time) is necessary for memory management rather than being an inefficiency.

ii.
```python
# Vectorized binning:
result[start:stop] = activity.reshape(stop - start, n_binned, DOWNSAMPLE).mean(axis=2)
# Vectorized interpolation:
aligned = np.interp(np.arange(expected_frames, dtype=np.float64), observed_positions, raw)
```

iii. The AI documented that memory-mapped arrays and chunked processing were chosen to bound memory usage.

## 6-c. What processing does the code repeat multiple times?

i. The AI loads `ops.npy` in `validate_source_session` and then the conversion proceeds without re-loading it. No processing is repeated unnecessarily.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads additional files for validation (ops.npy, iscell.npy, stat.npy, spks.npy) that are not used in the actual conversion. It also collects detailed `trace_example` dictionaries for processing plots and `session_info` metadata dictionaries for every session. The exact-balance assertion on quintiles adds computation that isn't strictly needed.

ii.
```python
# Validation loads in validate_source_session:
spks = np.load(plane / "spks.npy", mmap_mode="r")
iscell = np.load(plane / "iscell.npy", mmap_mode="r")
stat = np.load(plane / "stat.npy", allow_pickle=True)
ops = np.load(plane / "ops.npy", allow_pickle=True).item()
```

iii. These extra loads serve validation purposes but are not needed for the conversion itself.
