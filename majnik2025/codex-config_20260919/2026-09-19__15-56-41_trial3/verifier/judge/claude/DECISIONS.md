# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by iterating over sorted directories under `/app/data`, checking each for the existence of required files (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`). For each session, it loads fluorescence data (`F.npy`, `Fneu.npy`) via memory-mapped numpy loads, acquisition parameters from `ops.npy`, and behavioral data (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`).

ii.
```python
def discover_sessions() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            plane = session_dir / "suite2p" / "plane0"
            behavior = session_dir / "move_deve"
            required = (
                plane / "F.npy", plane / "Fneu.npy", plane / "iscell.npy",
                plane / "ops.npy", behavior / "motion_energy_glob.npy",
                behavior / "tstamps.npy", behavior / "interframe_int.npy",
            )
            if all(path.exists() for path in required):
                found.append((subject_dir.name, session_dir))
    return found
```

Loading neural data:
```python
fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
```

Loading behavioral data:
```python
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
```

iii. The AI validates that required files exist before including a session, which is a robust discovery approach. Loading uses memory mapping for large neural arrays to control memory usage. The CONVERSION_NOTES document each data source and its purpose.

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted directory names under the data directory. Each directory containing valid sessions becomes a subject.

ii.
```python
subjects = sorted({subject for subject, _ in selected})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Subject identification follows the data directory structure. The AI does not filter by `jm*` prefix (unlike the reference) but this makes no difference since all directories in the data folder are `jm*` directories.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder that contains the required suite2p and behavioral files is treated as one session. Sessions are sorted alphabetically (by date).

ii.
```python
for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        # ... check required files exist ...
        found.append((subject_dir.name, session_dir))
```

iii. Each dated subdirectory corresponds to one daily recording session. The sorted order ensures deterministic session ordering.

## 1-d. How are the data split into trials?

i. Trials are defined as 60-second non-overlapping segments of the continuous recording. At 30 Hz with 10-frame binning, each trial has 180 time bins. The AI requires sessions to be exactly divisible into complete trials (raises ValueError otherwise).

ii.
```python
bins_per_trial = frames_per_trial // RAW_FRAMES_PER_BIN  # 1800 // 10 = 180

def split_trials(array: np.ndarray, bins_per_trial: int) -> list[np.ndarray]:
    if array.ndim != 2 or array.shape[1] % bins_per_trial:
        raise ValueError(f"Cannot split array of shape {array.shape}")
    return [
        array[:, start : start + bins_per_trial].copy()
        for start in range(0, array.shape[1], bins_per_trial)
    ]
```

And earlier:
```python
if n_frames % frames_per_trial:
    raise ValueError(f"Session does not divide into complete 60-s trials: {session_dir}")
```

iii. The AI verified that all sessions divide evenly into 60-s trials (no remainder), so raising an error rather than discarding remainders has no practical effect. The CONVERSION_NOTES confirm 1,090 total trials with no truncation.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second trials are retained. The AI notes there are no trial rejection rules in the reference paper.

ii. N/A (no filtering code)

iii. The CONVERSION_NOTES state: "Keep all trials; there are no trial rejection rules, all arrays are finite, missing behavior is recoverable, and the lone Suite2p bad frame is not prescribed for downstream deletion."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`, plus acquisition parameters from `ops.npy`.

ii.
```python
fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
```

iii. These are the standard suite2p output files. The AI also loads `ops.npy` to extract processing parameters (neucoeff, baseline method, window size, etc.) rather than hardcoding them.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction (`Fc = F - 0.7 * Fneu`), then manual reimplementation of suite2p's maximin baseline correction using scipy's `gaussian_filter`, `minimum_filter1d`, and `maximum_filter1d`, then subtraction of the baseline, then 10-frame non-overlapping averaging.

ii.
```python
corrected = raw_f - neucoeff * raw_fneu

if baseline_mode == "maximin":
    flow = gaussian_filter(corrected, sigma=(0.0, sigma))
    flow = minimum_filter1d(flow, size=baseline_window, axis=1)
    flow = maximum_filter1d(flow, size=baseline_window, axis=1)

processed = corrected - flow
binned[start:stop] = processed.reshape(
    stop - start, n_bins, RAW_FRAMES_PER_BIN
).mean(axis=2)
```

iii. The CONVERSION_NOTES state: "Copy the reference maximin logic, but pass each session's Suite2p ops parameters." The AI explicitly chose to reimplement the algorithm rather than call `dcnv.preprocess` from suite2p. Processing is done in 64-neuron chunks to control memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI recognizes that the distributed data is already Track2p-curated (iscell > 0.5, complete longitudinal tracks).

ii. N/A (no filtering code; verification only):
```python
# From CONVERSION_NOTES: "all first iscell values are 1 and probabilities >0.5"
```

iii. The AI verified that all neurons in the distributed data already pass the iscell > 0.5 threshold and are complete longitudinal tracks. Re-filtering would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second blocks of a continuous recording with no stimulus events, no event-based alignment is needed.

ii.
```python
"temporal_alignment_event": "start of each non-overlapping 60-second block",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The AI sets `off_start=0.0` and `off_end=60.0` to describe that each trial starts at the beginning of its 60-second block and ends 60 seconds later.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms per bin). This binning is applied before motion energy discretization.

ii.
```python
RAW_FRAMES_PER_BIN = 10
# ...
binned[start:stop] = processed.reshape(
    stop - start, n_bins, RAW_FRAMES_PER_BIN
).mean(axis=2)
# ...
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
# ...
"time_bin_size": 1000.0 * RAW_FRAMES_PER_BIN / 30.0,  # 333.33 ms
```

iii. This matches the paper's Methods: "averaging in bins of 10 consecutive timestamps." Both streams are binned identically before discretization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the bin index and the acquisition parameters (frame rate from `ops.npy`), not from any raw data variable. It represents absolute elapsed time from session start at the **center** of each 10-frame bin.

ii.
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
```

iii. The AI uses bin centers (middle of each 10-frame window) rather than bin left edges. The CONVERSION_NOTES describe this as "absolute elapsed time at the center of each 10-frame bin."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time is computed as `(bin_index * 10 + 4.5) / 30.0` seconds, giving the center time of each bin. For a 20-minute session, this ranges from 0.15 s to 1199.8167 s; for 30 minutes, up to 1799.8167 s. Time runs continuously across trial boundaries within a session.

ii.
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
```

iii. The AI chose bin centers as a more precise representation of when the binned measurement was taken. The reference uses left edges: `(s + np.arange(trial_frames)) * BIN_FRAMES / FS`, which gives 0.0, 0.333..., etc. The AI's approach gives 0.15, 0.483..., etc.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time input is derived from the same bin indices as the neural data, so it is inherently aligned. It does not reset at trial boundaries; instead, it reflects absolute session time. The same `split_trials` function is applied to both neural and input arrays.

ii.
```python
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
# ...
input_trials = split_trials(input_binned, bins_per_trial)
```

iii. Alignment is guaranteed because time and neural data share the same bin indexing scheme and are split at the same trial boundaries.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Additionally, `tstamps.npy` and `interframe_int.npy` are used to detect and repair dropped camera frames.

ii.
```python
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
```

iii. The motion energy file contains pre-computed global motion energy. The timestamp and interval files are used to identify positions of missing video frames for interpolation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) Missing camera frames are detected via timestamp-based median interval rounding and linearly interpolated using `np.interp`. (2) The repaired trace is averaged into 10-frame bins. (3) The binned signal is discretized into 5 quintile bins using session-specific quantile thresholds (20th, 40th, 60th, 80th percentiles) with `np.searchsorted(..., side='right')`.

ii.
```python
# Missing frame detection and repair:
median_interval = float(np.median(intervals))
frame_steps = np.maximum(np.rint(intervals / median_interval).astype(np.int64), 1)
observed_idx = np.concatenate(
    (np.array([0], dtype=np.int64), np.cumsum(frame_steps, dtype=np.int64))
)
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)

# Binning:
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)

# Discretization:
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
motion_classes = np.searchsorted(quantile_edges, motion_binned, side="right").astype(np.int64)
```

iii. The AI's approach to missing frame detection is more sophisticated than the reference (which uses a simple threshold `dt * 1000 > 0.04`). Both produce the same result since they identify the same missing frames. For single missing frames, `np.interp` (linear interpolation) produces the same result as averaging neighbors.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session. The AI uses `np.quantile` at [0.2, 0.4, 0.6, 0.8] to compute 4 thresholds, then `np.searchsorted(..., side='right')` to assign bins 0-4. An assertion verifies each class has exactly 20% of bins.

ii.
```python
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
if np.unique(quantile_edges).size != 4:
    raise ValueError(f"Non-distinct motion quintile thresholds in {session_dir}")
motion_classes = np.searchsorted(
    quantile_edges, motion_binned, side="right"
).astype(np.int64)
fractions = np.bincount(motion_classes, minlength=5) / motion_classes.size
if not np.allclose(fractions, 0.2, atol=1.0 / motion_classes.size):
    raise AssertionError(f"Unexpected quintile fractions {fractions}")
```

iii. This is functionally equivalent to the reference's `np.percentile` + `np.digitize` approach. Both produce 5 equal-percentile bins with labels 0-4.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Neural and behavioral data are acquired synchronously at 30 Hz (microscope-triggered camera). When behavioral frames are missing, the AI detects gaps via timestamp analysis and linearly interpolates to restore frame-for-frame alignment. After repair, both streams have the same length and are binned identically.

ii.
```python
repaired_motion, missing_mask, raw_motion, observed_idx = repair_motion(
    session_dir, n_frames
)
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
```

iii. The assertion `inferred != deficit or observed_idx[-1] != n_frames - 1` ensures the repair correctly accounts for all missing frames. The AI also verifies observed values are unchanged after interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames (276 total across 9 sessions) are detected via timestamp gap analysis and linearly interpolated. The AI validates that the number of detected gaps exactly equals the length deficit. Sessions are required to divide evenly into complete 60-s trials (all do). The single Suite2p bad frame is retained since the paper doesn't prescribe its exclusion and it gets averaged into a 10-frame bin.

ii.
```python
if raw_motion.size == n_frames:
    return raw_float, missing_mask, raw_float.copy(), np.arange(n_frames)

median_interval = float(np.median(intervals))
frame_steps = np.maximum(np.rint(intervals / median_interval).astype(np.int64), 1)
# ... interpolation ...
if inferred != deficit or observed_idx[-1] != n_frames - 1:
    raise ValueError(...)
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
```

iii. The AI's approach is more rigorous than the reference in validating the repair (checking that inferred gaps match the deficit, verifying observed values are unchanged). The reference simply uses a threshold-based approach and asserts final length equality.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction (scipy gaussian_filter + minimum_filter1d + maximum_filter1d applied to all neurons). The AI processes neurons in 64-neuron chunks. The full conversion takes ~44 seconds for 41 sessions.

ii. N/A

iii. The CONVERSION_NOTES report 43.99 s total conversion time with timing per session. The AI's manual scipy-based implementation runs on CPU only, unlike the reference's `dcnv.preprocess` which can use GPU.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is well-vectorized. The neural processing uses chunked array operations. The motion repair uses `np.interp` (vectorized). The main loop is over sessions (unavoidable). No significant unvectorized loops remain.

ii. N/A

iii. The AI specifically avoided the per-frame `np.insert` loop that the reference has for dropped frame interpolation, using `np.interp` instead.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session is processed once. The validation step (`validate_converted`) re-checks array shapes and label distributions after assembly, but this is intentional verification rather than redundant processing.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and validates `iscell.npy` and `ops.npy` parameters that are used for validation but not all are needed for the core conversion (e.g., `nchannels`, `nplanes`, `badframes` count). The `tstamps.npy` file is loaded in addition to `interframe_int.npy` when only the intervals are needed for gap detection (the reference uses only `interframe_int.npy`). The `validate_converted` function at the end re-scans all arrays. These are minor overhead for validation purposes.

ii.
```python
# Loading extra ops fields:
"nchannels": int(ops.get("nchannels", 1)),
"nplanes": int(ops.get("nplanes", 1)),
"badframes": int(np.count_nonzero(ops.get("badframes", []))),
```

iii. The additional validation is intentional and improves robustness. The overhead is negligible compared to the core processing.
