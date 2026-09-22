# Decisions

**Note**: The AI's CONVERSION_NOTES.md describes an earlier/simpler version of the code (e.g., hardcoded parameters, `np.insert` for dropped frames, `np.percentile`+`np.digitize` for discretization). However, the actual deployed code at `/app/convert_data.py` is significantly more sophisticated and closely matches the human reference. This analysis is based on the **actual deployed code**.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data root, and sessions as subdirectories whose name starts with 4 digits. For each session, it loads `ops.npy` (suite2p metadata), `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`. This matches the reference approach.

ii.
```python
def list_sessions() -> list[SessionPath]:
    sessions: list[SessionPath] = []
    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("jm"):
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name[:4].isdigit():
                continue
            sessions.append(SessionPath(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    return sessions

# Loading per session:
ops = load_ops(session.path)  # ops.npy
raw_F, raw_Fneu = load_neural_arrays(session.path)  # F.npy, Fneu.npy
raw_motion, tstamps, interframe = load_behavior_arrays(session.path)  # motion_energy_glob.npy, tstamps.npy, interframe_int.npy
```

iii. The AI documents in CONVERSION_NOTES that released data are post-Track2p matched-cell outputs, so no match matrix loading is needed.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm`, sorted alphabetically. This matches the reference.

ii.
```python
subjects = sorted({session.subject for session, _ in processed_sessions})
```

iii. Each `jm*` directory represents one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder whose name starts with 4 digits (date-based names like `2023-10-18_a`), sorted alphabetically. This matches the reference.

ii.
```python
for session_dir in sorted(subject_dir.iterdir()):
    if not session_dir.is_dir() or not session_dir.name[:4].isdigit():
        continue
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as 60-second non-overlapping segments after 10-frame binning, yielding 180 time bins per trial. The code requires the number of bins to be exactly divisible by the trial length (raises an error if not). In practice, all sessions divide exactly. This matches the reference.

ii.
```python
bins_per_trial = int(round(trial_seconds * fs / bin_frames))  # 180
neural_trials = split_trials_2d(neural_binned, bins_per_trial)

def split_trials_2d(arr, trial_bins):
    n_rows, n_time = arr.shape
    if n_time % trial_bins != 0:
        raise ValueError(f"Time dimension {n_time} is not divisible by trial_bins {trial_bins}.")
    n_trials = n_time // trial_bins
    reshaped = arr.reshape(n_rows, n_trials, trial_bins)
    return [reshaped[:, i, :] for i in range(n_trials)]
```

iii. Per instruction, trials are 60-second non-overlapping segments.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second segments are kept. This matches the reference.

ii. N/A

iii. There is no natural trial structure or quality criteria to filter on.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (preprocessing parameters) from `suite2p/plane0/`. This matches the reference.

ii.
```python
f = np.load(session_path / "suite2p" / "plane0" / "F.npy", allow_pickle=True).astype(np.float32)
fneu = np.load(session_path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True).astype(np.float32)
ops = np.load(session_path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()
```

iii. Standard suite2p output files.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction using `ops["neucoeff"]` (0.7), followed by suite2p's `dcnv.preprocess` using parameters from `ops.npy` (baseline, win_baseline, sig_baseline, prctile_baseline). The AI reads all parameters from `ops.npy` rather than hardcoding, which is slightly more robust than the reference (which hardcodes them).

ii.
```python
def preprocess_neural(F, Fneu, ops):
    fs = float(ops["fs"])
    fc = F - np.float32(ops["neucoeff"]) * Fneu
    corrected = dcnv.preprocess(
        fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=fs,
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 2000)),
        device=torch.device("cpu"),
    )
    return corrected.astype(np.float32), info
```

iii. The CONVERSION_NOTES state: "For decoder conversion, recompute baseline-corrected fluorescence from F and Fneu using Suite2p-style defaults from ops.npy."

**Note**: The batch_size differs: AI uses `ops.get("batch_size", 2000)` (default 2000), reference hardcodes 128. This could cause minor numerical differences in the `dcnv.preprocess` computation due to batching effects.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons are included. This matches the reference, as the released data already contain only tracked cells that passed iscell filtering.

ii. N/A

iii. CONVERSION_NOTES document that the released data already embody the paper's iscell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. The AI sets `temporal_alignment_event` to "start of each derived 60-second trial window" with `off_start=0.0` and `off_end=60.0`. The reference sets `temporal_alignment_event` to "session_start" with `off_start=None` and `off_end=None`. Both approaches are valid descriptions of the same alignment.

ii.
```python
"temporal_alignment_event": "start of each derived 60-second trial window",
"off_start": 0.0,
"off_end": float(DEFAULT_TRIAL_SECONDS),
```

iii. There is no stimulus event to align to; the recording is continuous.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms). This matches the reference exactly.

ii.
```python
DEFAULT_BIN_FRAMES = 10
DEFAULT_FS = 30.0

neural_binned = mean_bin_2d(corrected_F, bin_frames).astype(np.float32)
motion_binned = mean_bin_1d(aligned_motion.astype(np.float32), bin_frames).astype(np.float32)

"time_bin_size": 1000.0 * DEFAULT_BIN_FRAMES / DEFAULT_FS,  # 333.33 ms
```

iii. The paper states neural and behavior traces were "slightly denoised by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from frame indices and the frame rate (from `ops.npy`). This matches the reference approach.

ii.
```python
def build_time_input(n_frames, fs, bin_frames):
    frame_times = full_frame_times_seconds(n_frames, fs)  # np.arange(n_frames) / fs
    binned = mean_bin_1d(frame_times.astype(np.float32), bin_frames)
    return binned.astype(np.float32)
```

iii. Frame rate is constant at 30 Hz, so computing time from frame indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame times are computed as `np.arange(n_frames) / fs`, then averaged within 10-frame bins using `mean_bin_1d`, giving bin CENTERS. For the first bin this gives 0.15s (average of frames 0-9 at 30Hz). Time runs continuously across trials within a session. This matches the reference exactly.

ii.
```python
def full_frame_times_seconds(n_frames, fs):
    return np.arange(n_frames, dtype=np.float64) / fs

def build_time_input(n_frames, fs, bin_frames):
    frame_times = full_frame_times_seconds(n_frames, fs)
    binned = mean_bin_1d(frame_times.astype(np.float32), bin_frames)
    return binned.astype(np.float32)
```

iii. The CONVERSION_NOTES claim time values are "left edge of each bin" starting at 0.0s, but the actual code computes bin centers starting at 0.15s, which matches the reference. The CONVERSION_NOTES are inconsistent with the deployed code.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is constructed from the same frame count as neural data and binned identically, so alignment is inherent. Time is then split into trials with the same `split_trials_1d` function. This matches the reference.

ii.
```python
time_binned = build_time_input(n_frames, fs, bin_frames)
input_trials = split_trials_1d(time_binned, bins_per_trial)
```

iii. Since time is derived from the same indexing as neural data, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy` in the `move_deve` subdirectory. This matches the reference.

ii.
```python
motion = np.load(session_path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session_path / "move_deve" / "tstamps.npy", allow_pickle=True)
interframe = np.load(session_path / "move_deve" / "interframe_int.npy", allow_pickle=True)
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) Motion energy is interpolated onto imaging frame times using timestamps (`np.interp`), (2) averaged in 10-frame bins, (3) discretized per-session into 5 equal-frequency bins using rank-based assignment. This matches the reference exactly.

ii.
```python
# Alignment via interpolation
aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))

# Binning
motion_binned = mean_bin_1d(aligned_motion.astype(np.float32), bin_frames)

# Discretization
def equal_frequency_bins(values, n_bins):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    bins = (ranks * n_bins) // n
    bins = np.minimum(bins, n_bins - 1).astype(np.int64)
    return bins, quantiles
```

iii. The CONVERSION_NOTES describe a simpler approach (`np.percentile`+`np.digitize` and `np.insert`), but the actual deployed code uses the more robust timestamp-based interpolation and rank-based discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses rank-based equal-frequency binning: each value is assigned a bin based on its rank, guaranteeing exactly balanced class counts. This matches the reference's `equal_frequency_bins` function.

ii.
```python
def equal_frequency_bins(values, n_bins):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    bins = (ranks * n_bins) // n
    bins = np.minimum(bins, n_bins - 1).astype(np.int64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    return bins, quantiles
```

iii. The instructions specify "five equal-percentile bins, selected per session." The rank-based approach guarantees this.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI infers the timestamp scale (kiloseconds to seconds conversion), then uses `np.interp` to interpolate motion energy values onto the imaging frame time grid. This matches the reference approach.

ii.
```python
def align_motion_to_imaging(motion, tstamps, interframe, n_frames, fs):
    scale = infer_timestamp_scale_seconds(interframe, fs)
    motion_times = tstamps * scale
    imaging_times = full_frame_times_seconds(n_frames, fs)
    aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
    return aligned.astype(np.float32), info
```

iii. The CONVERSION_NOTES describe an approach using `interframe_int` thresholding and `np.insert`, but the actual deployed code uses the more principled timestamp-based `np.interp` approach, which matches the reference.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are handled by timestamp-based interpolation (`np.interp`), which naturally fills gaps when the motion energy array is shorter than the neural data. The code also detects and logs gaps where inter-frame intervals exceed 1.5x the nominal frame duration. This matches the reference approach.

ii.
```python
aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
gaps = np.where(np.diff(motion_times) > nominal_dt * 1.5)[0]
```

iii. The AI documents 9 sessions with missing motion frames and confirms all are handled correctly.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies suite2p `dcnv.preprocess` baseline correction as the main bottleneck. The code runs it on CPU (`torch.device("cpu")`). This is consistent with the reference approach.

ii. N/A

iii. The baseline correction involves sliding window operations over the full session for every neuron.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The gap detection loop (`for idx in gaps`) computes gap sizes one at a time. However, this is a minor reporting loop, not a performance-critical path. The actual processing uses vectorized `np.interp`. The AI's deployed code has no significant unvectorized loops.

ii.
```python
gap_sizes = []
for idx in gaps:
    missing = int(round((motion_times[idx + 1] - motion_times[idx]) * fs)) - 1
    gap_sizes.append(max(missing, 0))
```

iii. This loop is for logging/metadata only and does not affect performance.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing is present. Each session is processed once in a single pass. This matches the reference.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive metadata per session (gap indices, gap sizes, motion timestamps, imaging timestamps, quantiles, etc.) that is not used by the downstream decoder. It also computes quantiles in `equal_frequency_bins` that are only used for plotting/metadata, not for the actual discretization. This is relatively minor.

ii.
```python
info = {
    "timestamp_scale_to_seconds": scale,
    "motion_len_raw": int(len(motion)),
    "motion_len_aligned": int(len(aligned)),
    "missing_motion_frames": int(n_frames - len(motion)),
    "gap_indices_raw": gaps.astype(np.int64),
    "gap_sizes_frames": np.array(gap_sizes, dtype=np.int64),
    "motion_times_seconds": motion_times,
    "imaging_times_seconds": imaging_times,
}
```

iii. The extra metadata is useful for debugging and validation but adds some overhead.
