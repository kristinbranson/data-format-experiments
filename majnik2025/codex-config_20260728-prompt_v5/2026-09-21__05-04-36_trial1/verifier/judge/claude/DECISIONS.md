# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` under the data root, and sessions as subdirectories whose names start with 4 digits (date-formatted). For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. It also reads `ops['nframes']` during discovery. All sessions are processed sequentially in sorted order.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            suite2p_dir = session_dir / "suite2p" / "plane0"
            move_dir = session_dir / "move_deve"
            ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
            f = np.load(suite2p_dir / "F.npy", mmap_mode="r")
            motion = np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r")
            ...
```

```python
def process_session(session: SessionInfo, ...):
    f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
    fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
    motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
    tstamps = np.load(session.move_dir / "tstamps.npy")
```

iii. The AI notes that the data follows standard suite2p output organization with Track2p-tracked neurons. Discovery uses memory-mapped reads to get metadata without fully loading arrays. The `p.name[:4].isdigit()` filter ensures only date-formatted session directories are included.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data root, sorted alphabetically. Each unique subject name is assigned an index as it is encountered.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...

if session.subject not in subject_to_idx:
    subject_to_idx[session.subject] = len(dataset["subjects"])
    dataset["subjects"].append(session.subject)
subject_idx.append(subject_to_idx[session.subject])
```

iii. The `jm*` prefix convention is consistent across the dataset. Alphabetical sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder whose names start with 4 digits (date-formatted), sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
```

iii. The AI applies a `p.name[:4].isdigit()` filter to distinguish session directories from any other subdirectories. Each session directory contains suite2p output and motion energy files for one recording.

## 1-d. How are the data split into trials?

i. Trials are defined as fixed 60-second non-overlapping segments of the continuous recording. After 10-frame binning, each trial contains 180 time bins (60s * 30Hz / 10). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_SECONDS = 60
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES  # 1800 // 10 = 180

def split_time_series_into_trials(arr: np.ndarray, bins_per_trial: int) -> list[np.ndarray]:
    usable = arr.shape[-1] - (arr.shape[-1] % bins_per_trial)
    arr = arr[..., :usable]
    ntrials = usable // bins_per_trial
    return [arr[:, i * bins_per_trial:(i + 1) * bins_per_trial] for i in range(ntrials)]
```

iii. Per instruction, trials are defined as 60-second segments. The recording has no natural trial structure, so fixed-length segmentation is the appropriate approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are retained; only the remainder at the end of each session is discarded.

ii. N/A (no filtering code)

iii. There is no trial-level quality issue in the continuous recordings. The AI notes that since there is no stimulus-driven trial structure, there is no basis for quality-based trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) in `suite2p/plane0/`.

ii.
```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces, consistent with the paper's description of using baseline-corrected fluorescence.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`F - 0.7 * Fneu`), followed by suite2p's `preprocess` function with `baseline='maximin'`, `win_baseline=60.0`, `sig_baseline=10`, `fs=30`, `prctile_baseline=8.0`. The corrected traces are then averaged into non-overlapping 10-frame bins.

ii.
```python
corrected = f - NEUROPIL_COEFF * fneu  # NEUROPIL_COEFF = 0.7
corrected = suite2p_preprocess(
    corrected.copy(),
    baseline=BASELINE_MODE,       # "maximin"
    win_baseline=WIN_BASELINE_SECONDS,  # 60.0
    sig_baseline=SIG_BASELINE_FRAMES,   # 10.0
    fs=FRAME_RATE_HZ,                    # 30.0
    prctile_baseline=PRCTILE_BASELINE,   # 8.0
    batch_size=SUITE2P_BATCH_SIZE,       # 128
    device=torch.device("cpu"),
).astype(np.float32)

binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
```

iii. The AI notes that the paper states "we used baseline corrected fluorescence traces as our dF/F" and that 10-frame binning matches the paper's "averaging in bins of 10 consecutive timestamps". The neuropil coefficient of 0.7 is the suite2p default.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied. All neurons present in the released `F.npy` are used, since the data is already Track2p-filtered to contain only tracked all-day cells.

ii. N/A (no filtering code)

iii. The AI notes that the released suite2p outputs already contain only tracked neurons present across all days, with all `iscell` values already passing the 0.5 threshold. No further filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The temporal alignment event is described as "trial start of fixed 60-second windows tiled across each session".

ii.
```python
"temporal_alignment_event": "trial start of fixed 60-second windows tiled across each session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),  # 60.0
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied to both streams before motion energy is discretized.

ii.
```python
BIN_FRAMES = 10
binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES).astype(np.float32)
...
"time_bin_size": float(1000.0 * BIN_FRAMES / FRAME_RATE_HZ),  # 333.33 ms
```

iii. The AI cites the paper's methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the frame indices and frame rate. Specifically, `frame_times = np.arange(nframes) / FRAME_RATE_HZ` generates per-frame times, and these are then averaged in 10-frame bins, giving bin-center times in seconds from session start.

ii.
```python
frame_times = np.arange(session.nframes, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
binned_time = non_overlapping_mean_last_axis(frame_times, BIN_FRAMES).astype(np.float32)
```

iii. Since the frame rate is constant at 30 Hz, computing time from frame indices is equivalent to having stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices are converted to seconds by dividing by the frame rate (30 Hz), then averaged in 10-frame bins using the same binning function applied to neural and motion data. This yields bin-center times.

ii.
```python
frame_times = np.arange(session.nframes, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
binned_time = non_overlapping_mean_last_axis(frame_times, BIN_FRAMES).astype(np.float32)
```

iii. The bin-center approach gives values like [0.15, 0.483, ...] seconds rather than left-edge values [0, 0.333, ...].

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same frame indices as the neural data and binned with the same function, so alignment is automatic. The time input continues across trial boundaries (absolute session time, not trial-relative time).

ii.
```python
input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
```

iii. By using the same frame grid and binning, the time input is inherently aligned with neural data at every time point.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used for aligning motion samples to imaging frames.

ii.
```python
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
tstamps = np.load(session.move_dir / "tstamps.npy")
```

iii. The motion energy file contains a pre-computed global motion energy signal from behavioral video. Timestamps are needed to handle dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) Motion samples are aligned to the imaging frame timeline using timestamps, with linear interpolation for missing camera frames. (2) The aligned trace is averaged into 10-frame bins. (3) The binned signal is discretized into 5 equal-percentile bins (quintiles) computed within each session, with a rank-based fallback for tied values.

ii.
```python
# Alignment
motion_aligned, missing_mask, raw_frame_idx, timestamp_scale = align_motion_to_imaging(
    motion=motion_raw, tstamps=tstamps, nframes=session.nframes, fs=FRAME_RATE_HZ)

# Binning
binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES).astype(np.float32)

# Discretization
def compute_motion_quintiles(values, nclasses=5):
    quantiles = np.linspace(0.0, 1.0, nclasses + 1)[1:-1]
    thresholds = np.quantile(values, quantiles)
    categories = np.digitize(values, thresholds, right=False).astype(np.int64)
    if np.unique(categories).size < nclasses:
        # fallback: rank-based assignment
        order = np.argsort(values, kind="mergesort")
        categories = np.empty(values.shape[0], dtype=np.int64)
        boundaries = np.linspace(0, values.shape[0], nclasses + 1, dtype=int)
        for cls in range(nclasses):
            categories[order[boundaries[cls]:boundaries[cls + 1]]] = cls
    return categories, thresholds.astype(np.float32)
```

iii. The AI notes that interpolation at dropped frames ensures the motion signal matches imaging frame count. Binning before discretization is necessary because averaging class labels would be meaningless. Per-session quintiles normalize across session-to-session scale changes.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 quintile bins per session using `np.quantile` to compute the 20th, 40th, 60th, and 80th percentile thresholds, then `np.digitize` to assign categories 0-4. A rank-based fallback handles degenerate cases where tied values would produce fewer than 5 unique categories.

ii.
```python
quantiles = np.linspace(0.0, 1.0, nclasses + 1)[1:-1]  # [0.2, 0.4, 0.6, 0.8]
thresholds = np.quantile(values, quantiles)
categories = np.digitize(values, thresholds, right=False).astype(np.int64)
```

iii. Per the instructions, motion energy should be "discretized into five equal-percentile bins, selected per session."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses timestamp-based alignment via `align_motion_to_imaging()`. When motion length already equals imaging frame count, no interpolation is needed. When lengths differ, motion timestamps are mapped to imaging frame indices using the full timestamp span, and missing frames are filled by linear interpolation. Both streams are then binned identically.

ii.
```python
def align_motion_to_imaging(motion, tstamps, nframes, fs):
    ...
    if motion.size == nframes:
        return (motion.astype(np.float32, copy=False), np.zeros(nframes, dtype=bool), ...)

    nominal_step = span / float(nframes - 1)
    frame_idx = np.rint((tstamps - tstamps[0]) / nominal_step).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)

    sums = np.zeros(nframes, dtype=np.float64)
    counts = np.zeros(nframes, dtype=np.int64)
    np.add.at(sums, frame_idx, motion)
    np.add.at(counts, frame_idx, 1)

    aligned = np.full(nframes, np.nan, dtype=np.float64)
    valid = counts > 0
    aligned[valid] = sums[valid] / counts[valid]
    missing = ~valid
    if missing.any():
        aligned[missing] = np.interp(missing_idx, valid_idx, aligned[valid])
    ...
```

iii. The AI uses timestamps for precise mapping rather than inter-frame interval thresholding. This is a more general approach that automatically handles the scale of the timestamps.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames (where motion length < imaging frame count) are handled by timestamp-based alignment and linear interpolation. The AI also detects the timestamp scale automatically. Remainder frames at the end of sessions that don't fill a complete trial are discarded.

ii.
```python
# In align_motion_to_imaging:
if missing.any():
    valid_idx = np.flatnonzero(valid)
    missing_idx = np.flatnonzero(missing)
    aligned[missing] = np.interp(missing_idx, valid_idx, aligned[valid])

# In split_time_series_into_trials:
usable = arr.shape[-1] - (arr.shape[-1] % bins_per_trial)
```

iii. The conversion notes document specific missing frame counts: [0, 1, 2, 3, 116, 148]. All sessions process successfully. Assertions verify alignment correctness.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` (baseline correction), which dominates processing time per session. The conversion logs show `neural_preprocess_s` takes 0.15-0.8s per session while all other steps take <0.02s each. Total conversion for 41 sessions takes ~27 seconds.

ii. N/A

iii. The baseline correction involves sliding window operations over the full session length for every neuron.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting uses a Python loop over trial indices. However, each iteration is a simple array slice, so the loop overhead is negligible. The rank-based fallback in `compute_motion_quintiles` uses a loop over classes, but this only runs in degenerate cases.

ii.
```python
return [arr[:, i * bins_per_trial:(i + 1) * bins_per_trial] for i in range(ntrials)]
```

iii. The loops are minimal and involve only indexing operations. The main computational work (binning, preprocessing) is already vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The AI loads `F.npy` and `motion_energy_glob.npy` twice: once during `discover_sessions()` (memory-mapped for metadata) and once during `process_session()` (full load for processing).

ii.
```python
# In discover_sessions:
f = np.load(suite2p_dir / "F.npy", mmap_mode="r")
motion = np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r")

# In process_session:
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
```

iii. The discovery phase uses memory-mapped reads only for shape metadata, so the overhead is minimal.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `zscore_rows` for visualization in `plot_processing_summary`, but this z-scored neural data is not saved. It also loads `tstamps.npy` even for sessions where motion length already matches imaging frame count (though the alignment function shortcuts in that case). The `detect_timestamp_scale_to_seconds` function computes a scale factor that isn't functionally used when lengths match.

ii.
```python
def zscore_rows(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std
```

iii. The z-scoring is only for visualization purposes. The timestamp loading overhead is negligible.
