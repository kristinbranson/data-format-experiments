# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all sessions by scanning the `data/` directory for subject folders (starting with "jm") and within each, session folders (starting with a 4-digit year). For each session, it loads neural data from `suite2p/plane0/` (F.npy, Fneu.npy, ops.npy) and behavioral data from `move_deve/` (motion_energy_glob.npy, tstamps.npy, interframe_int.npy). Each session is processed individually via `process_session()`, then all processed sessions are assembled into the final dataset via `build_dataset()`.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    ...

def process_session(session: SessionInfo) -> dict:
    s2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"
    neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
    n_frames = int(ops["nframes"])
    motion_raw = np.load(move_dir / "motion_energy_glob.npy")
    timestamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
    ...
```

iii. The AI documented in CONVERSION_NOTES.md Step 2 that the data is organized as subject folders containing session folders, each with `suite2p/plane0/` and `move_deve/` subdirectories. The AI confirmed 6 subjects and 41 total sessions, consistent with the paper's report of 6 mice with at least 6 daily sessions each.

## 1-b. How are the data split into subjects?

i. Subjects are identified by directory names starting with "jm" (e.g., jm031, jm032, ..., jm046). A sorted unique set of subject names is created, and each session is mapped to its subject via a lookup dictionary. The `subjects` list and `subject_idx` array are constructed accordingly.

ii.
```python
subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subject_to_idx[session["info"].subject] for session in processed_sessions], dtype=np.int64),
    ...
}
```

iii. The AI documented that 6 subjects are present (jm031, jm032, jm038, jm039, jm040, jm046), matching the paper's report of 6 mice. The data README confirms jm031=mouse A through jm046=mouse F.

## 1-c. How are the data split into sessions?

i. Each date-named subdirectory within a subject folder constitutes one session. Sessions are discovered via directory iteration and sorted by name. The session count per subject ranges from 6 to 7 (jm040 has 6, all others have 7), totaling 41 sessions.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that session folders are named as "YYYY-MM-DD_a" and that the paper states a minimum of 6 consecutive daily sessions per mouse. All 41 sessions are included.

## 1-d. How are the data split into trials?

i. There are no native trials in the source data -- the recordings are continuous spontaneous-behavior sessions. The AI creates pseudo-trials by segmenting each continuous session into consecutive 2-minute blocks after 10-frame temporal binning. This yields 360 time bins per trial (120 seconds * 30 Hz / 10 frames per bin). Sessions of 20 minutes (36,000 frames) produce 10 trials; sessions of 30 minutes (54,000 frames) produce 15 trials.

ii.
```python
TRIAL_DURATION_SEC = 120.0
BIN_FRAMES = 10

def segment_trials(neural_binned, time_binned, motion_norm_binned, motion_edges, trial_bins):
    n_total_bins = neural_binned.shape[1]
    n_trials = n_total_bins // trial_bins
    for trial_idx in range(n_trials):
        start = trial_idx * trial_bins
        stop = start + trial_bins
        neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
        input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
        output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
        ...
```

iii. The AI justified this in CONVERSION_NOTES.md Step 5: "Convert each continuous session into consecutive 2-minute pseudo-trials after 10-frame averaging. This exactly matches the paper's decoder split granularity." The paper states "splits were done on consecutive 2 minute blocks."

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All pseudo-trials created from the 2-minute segmentation are included. The only check is that each session must produce at least 2 trials (which all do -- minimum is 10).

ii.
```python
if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
data["neural"].append(neural_trials)
```

iii. The AI noted in CONVERSION_NOTES.md Step 3 that "No trial-based curation is described because the recordings are continuous spontaneous-behavior sessions rather than discrete trials." Since trials are synthetically created from continuous data, there is no quality-based filtering criterion from the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from three Suite2p output files per session: `F.npy` (raw fluorescence traces), `Fneu.npy` (neuropil fluorescence traces), and `ops.npy` (Suite2p processing parameters including neucoeff, baseline parameters, and sampling rate).

ii.
```python
def compute_baseline_corrected_fluorescence(s2p_dir: Path) -> tuple[np.ndarray, dict, dict]:
    ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(s2p_dir / "F.npy", mmap_mode="r")
    Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
    ...
```

iii. The AI documented in CONVERSION_NOTES.md Steps 1 and 5 that the paper states analyses used baseline-corrected fluorescence traces with default Suite2p parameters, and the bundled data contains F.npy, Fneu.npy, and ops.npy in each session's suite2p/plane0/ directory.

## 2-b. How is the `neural` data processed?

i. The AI applies Suite2p-style baseline-corrected fluorescence processing: (1) neuropil subtraction using `F - neucoeff * Fneu` with the session-specific `neucoeff` from ops.npy, then (2) Suite2p's `dcnv.preprocess()` function for baseline correction using the session's saved parameters (baseline method, window, sigma, percentile, sampling rate). The result is then temporally binned by averaging non-overlapping windows of 10 frames.

ii.
```python
corrected = np.array(F, dtype=np.float32, copy=True)
corrected -= np.float32(ops["neucoeff"]) * np.asarray(Fneu, dtype=np.float32)
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 2000)),
    device=torch.device("cpu"),
).astype(np.float32, copy=False)

# Then binned:
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)

def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32)
```

iii. The AI justified using Suite2p's `dcnv.preprocess` with session-specific ops parameters because "the paper explicitly states downstream analyses used baseline-corrected fluorescence traces with default Suite2p parameters." The 10-frame binning matches the paper's description of "averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI treats the released data as already containing only the Track2p-matched neurons that passed Suite2p's cell classification (iscell > 0.5) and were present across all recording days for a given subject.

ii.
```python
# No filtering code - all neurons from F.npy are used directly
neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
# ...
"brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
```

iii. The AI documented in CONVERSION_NOTES.md Steps 1, 4, and 5 that the Track2p code uses `iscell_thr = 0.50` for ROI filtering, and the released data already contains only Track2p-matched cells ("Within each subject all sessions have identical neuron counts and row identities"). The paper states functional analyses used neurons successfully tracked across all days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the start of each consecutive 2-minute block within the continuous recording session. The temporal alignment event is defined as "Start of each consecutive 2-minute block cut from a continuous recording session." Each trial starts at the beginning of a 2-minute segment with `off_start=0.0` and `off_end=120.0` seconds.

ii.
```python
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)  # = 360 bins per trial

for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop]
    ...

# metadata:
"temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SEC,  # 120.0
```

iii. The AI noted there is no natural trial structure in the data. The alignment event is the start of each synthetic 2-minute block, which matches the paper's cross-validation splits ("splits were done on consecutive 2 minute blocks").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw data is sampled at 30 Hz (33.33 ms per frame). The AI applies temporal rebinning by averaging non-overlapping windows of 10 frames, producing a time bin size of 333.33 ms (1000 * 10 / 30). This yields 360 bins per 2-minute trial.

ii.
```python
RAW_FS_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ  # = 333.33 ms

neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The AI documented in CONVERSION_NOTES.md Step 3 that the paper specifies "averaging in bins of 10 consecutive timestamps (~333 ms)" for the decoder.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is derived from the session frame count (`nframes` from ops.npy) and the sampling rate (`fs` from ops.npy). A time array is constructed from the frame indices.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. The AI stated in CONVERSION_NOTES.md Step 5 that elapsed session time is encoded as one continuous, time-varying input in seconds, with each pseudo-trial keeping its absolute position within the original recording. This is a task-specific requirement from the benchmark instructions ("Time elapsed from the beginning of the experiment").

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The processing involves: (1) creating a time array where each frame's time is computed as `(frame_index + 0.5) / fs`, placing the time at the center of each frame, (2) temporally binning by averaging 10-frame windows (same as neural data), (3) segmenting into 2-minute trial blocks. The time values are absolute within the session (not reset per trial).

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]

# In segment_trials:
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
```

iii. The AI justified using bin centers (+0.5 offset) to accurately represent the temporal midpoint of each frame. The same 10-frame binning is applied to the time signal as to the neural and behavioral signals.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because both are derived from the same imaging frame indices. The time array has one value per imaging frame, is binned with the same 10-frame windows, and is segmented into the same trial boundaries. They share the same temporal grid.

ii.
```python
# Same binning applied to both:
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]

# Same trial segmentation:
neural_trial = neural_binned[:, start:stop]
input_trial = time_binned[np.newaxis, start:stop]
```

iii. The AI noted the imaging frames serve as the master clock. Since the time input is computed directly from the frame indices and undergoes identical binning and segmentation, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` (global motion energy computed from squared pixel-wise frame differences), `tstamps.npy` (camera timestamps), and `interframe_int.npy` (inter-frame intervals for detecting missing frames).

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. The AI documented in CONVERSION_NOTES.md Step 2 that `motion_energy_glob.npy` contains the processed behavioral signal, and that `tstamps.npy` and `interframe_int.npy` are needed to identify and handle missing behavior frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The processing involves: (1) reconstructing missing behavior frames by analyzing inter-frame intervals to identify gaps and inserting NaN values at those positions, then linearly interpolating, (2) temporally binning by averaging 10-frame windows, (3) global min-max normalization across all sessions, (4) discretization into 5 equal-percentile bins (quintiles) using global quantile edges.

ii.
```python
# Reconstruct missing frames:
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)

# Bin:
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]

# Global normalization:
motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions])
motion_min = float(np.min(motion_all))
motion_max = float(np.max(motion_all))
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)

# Global quintile edges:
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Discretize per session:
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
```

iii. The AI documented that the paper describes motion energy from "pixel-wise difference of consecutive frames" and that the decoder instructions require normalization and discretization into five equal-percentile bins.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The motion energy is first globally min-max normalized to [0, 1], then global quintile edges are computed at the 0th, 20th, 40th, 60th, 80th, and 100th percentiles across all binned motion energy values from all sessions. `np.digitize` is used to assign each binned value to one of 5 classes (Q1-Q5, coded as 0-4).

ii.
```python
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)

# Per trial:
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. The AI noted in CONVERSION_NOTES.md Step 5: "Because the target decoder requires categorical outputs, convert the denoised motion-energy trace into 5 global equal-percentile classes." The resulting global distribution is exactly balanced at 20% per class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The behavior data is aligned to the neural data by using the imaging frames as the master clock. For sessions with missing behavior frames (9 out of 41), the AI reconstructs the full-length behavior trace by analyzing inter-frame timing intervals to identify where frames were dropped, inserting NaN values at those positions, and linearly interpolating. After reconstruction, the behavior trace matches the neural trace length. Both are then binned with the same 10-frame windows and segmented at the same trial boundaries.

ii.
```python
def reconstruct_motion_trace(motion, interframe_int, target_len):
    ...
    median_interval = float(np.median(interframe_int))
    steps = np.rint(interframe_int / median_interval).astype(np.int64)
    steps[steps < 1] = 1
    observed_idx = np.empty(motion.shape[0], dtype=np.int64)
    observed_idx[0] = 0
    observed_idx[1:] = np.cumsum(steps)
    ...
    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    missing_idx = np.flatnonzero(np.isnan(full))
    full = interpolate_nans(full)
    return full, missing_idx
```

iii. The AI documented in CONVERSION_NOTES.md Step 4: "Camera triggered by microscope at 30 Hz... Treat imaging frames as the master clock and reconstruct missing behavior frames from doubled timing gaps before alignment."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data issue is missing behavior frames in 9 out of 41 sessions (ranging from 1 to 148 missing frames). The AI handles this by: (1) detecting missing frames from `interframe_int.npy` by computing the ratio of each interval to the median interval, (2) rounding to determine the number of imaging frames between consecutive behavior frames, (3) placing observed behavior values at their correct positions in a full-length NaN array, (4) linearly interpolating NaN values. If the reconstructed trace length doesn't match `nframes`, an error is raised.

ii.
```python
def interpolate_nans(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    idx = np.arange(x.shape[0])
    valid = ~np.isnan(x)
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid]).astype(np.float32)
    return out

def reconstruct_motion_trace(motion, interframe_int, target_len):
    ...
    if motion.shape[0] == target_len:
        return motion, np.empty(0, dtype=np.int64)
    ...
```

iii. The AI documented that the data README says "The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over." The AI chose interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p baseline-corrected fluorescence computation via `dcnv.preprocess()`, which is called once per session. Total processing for all 41 sessions takes ~86 seconds (~2.1 seconds per session on average), with sessions having more neurons (e.g., jm039 with 746 neurons) taking ~3.5 seconds and smaller sessions (e.g., jm031 with 221 neurons) taking ~0.5-0.7 seconds.

ii.
```python
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    ...
).astype(np.float32, copy=False)
```

iii. The AI documented in CONVERSION_NOTES.md Step 6 that "the main cost is Suite2p-style fluorescence preprocessing" and in Step 7 estimated the full conversion at ~85.6 seconds, which matched the actual 86.08 seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could potentially be vectorized is the per-trial segmentation loop in `segment_trials()`, which iterates over trials to slice arrays. However, this loop is already efficient since it only performs array slicing and `np.digitize` per trial. The per-session loop in `main()` cannot be easily vectorized since each session requires loading different files.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop]
    input_trial = time_binned[np.newaxis, start:stop]
    output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
    ...
```

iii. The AI documented that binning uses "reshape-and-mean rather than Python loops" and behavior reconstruction is "vectorized via timing-step accumulation." The trial segmentation loop is simple slicing, which is already near-optimal.

## 6-c. What processing does the code repeat multiple times?

i. The motion normalization is conceptually computed twice: once globally across all sessions in `build_dataset()` (to determine quantile edges), and then again per session when creating trials. The quantile edge computation requires a global pass, so the per-session re-normalization in the trial-building loop is a necessary second pass.

ii.
```python
# Global pass in build_dataset():
motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions])
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Per-session pass:
for session in processed_sessions:
    motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
```

iii. The AI documented this is by design since global statistics must be computed before per-session discretization. The two-pass approach is standard for global normalization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several preview/diagnostic data structures that are not included in the final pickle output: (1) `neural_preview` dict containing sample traces of F, Fneu, corrected, and processed for a single neuron, (2) `motion_preview` dict with raw/reconstructed motion traces, missing frame indices, and timestamps. These are used only for `--show-processing` plots. Additionally, the `select_sample_sessions()` function includes logic to find sessions with missing behavior frames, which is only used in `--sample` mode.

ii.
```python
# Preview data computed but not saved to output:
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    "corrected": corrected[sample_neuron, :preview_len].copy(),
    "processed": processed[sample_neuron, :preview_len].copy(),
}

motion_preview = {
    "raw": np.asarray(motion_raw[:preview_len], dtype=np.float32),
    "reconstructed": motion_full[:preview_len].copy(),
    ...
}
```

iii. The AI designed these preview structures to support the `--show-processing` visualization mode required by the instructions. While they consume memory during processing, they are discarded when not plotting. The final pickle output contains only the decoder-format data.
