# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by iterating over subject directories (`jm031`-`jm046`) within `data/`, and within each subject, iterating over session subdirectories (named `YYYY-MM-DD_a`). For each session, it loads Suite2p neural data (`F.npy`, `Fneu.npy`, `ops.npy`) from `suite2p/plane0/` and behavioral data (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`) from `move_deve/`. All 41 sessions across 6 subjects are loaded.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    ...
    return sessions
```

```python
def process_session(session: SessionInfo) -> dict:
    s2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"
    neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
    motion_raw = np.load(move_dir / "motion_energy_glob.npy")
    timestamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
    ...
```

iii. The AI justified this by noting the data README documents 6 subject folders with session subfolders, and the paper reports 6 mice with >= 6 daily sessions. The data organization was confirmed through explicit directory listing and dimension checks.

## 1-b. How are the data split into subjects?

i. Subjects are split by their top-level directory names (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`). A sorted unique list of subject names is created, and each session is assigned a `subject_idx` pointing into this list.

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

iii. The AI noted that subject folder names correspond directly to unique mouse IDs from the data release. The data README confirms `jm031`=mouse A through `jm046`=mouse F.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder represents one recording session (one day). Sessions are sorted chronologically by their directory name (date-based). Each session is processed independently and becomes one entry in the output lists.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

iii. The AI confirmed that each subject has 6-7 session directories matching the paper's statement of "at least 6 consecutive days." Session counts: jm031:7, jm032:7, jm038:7, jm039:7, jm040:6, jm046:7, totaling 41.

## 1-d. How are the data split into trials?

i. The source data has no native trial structure (continuous recordings). The AI creates pseudo-trials by segmenting each session into consecutive non-overlapping 2-minute blocks after 10-frame temporal binning. Each 2-minute block = 360 time bins (120 sec * 30 Hz / 10 frames per bin). 20-minute sessions yield 10 trials; 30-minute sessions yield 15 trials. Total: 545 pseudo-trials.

ii.
```python
TRIAL_DURATION_SEC = 120.0
BIN_FRAMES = 10
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)  # = 360

def segment_trials(..., trial_bins: int) -> ...:
    n_total_bins = neural_binned.shape[1]
    n_trials = n_total_bins // trial_bins
    for trial_idx in range(n_trials):
        start = trial_idx * trial_bins
        stop = start + trial_bins
        neural_trial = neural_binned[:, start:stop]
        ...
```

iii. The AI justified this by citing the paper's decoding methods: "splits were done on consecutive 2 minute blocks of the recording." This 2-minute block size exactly matches the paper's cross-validation split granularity.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All complete 2-minute blocks from all sessions are included. Any leftover frames at the end of a session that do not fill a complete 2-minute block are discarded (though in practice, both 20-min and 30-min sessions divide evenly into 2-minute blocks with no remainder).

ii.
```python
n_trials = n_total_bins // trial_bins
# Integer division discards any partial trailing block
```

iii. The AI noted there is no trial-based curation described in the paper because recordings are continuous spontaneous-behavior sessions rather than discrete trials. No session or trial exclusion criteria were mentioned in the reference materials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from three Suite2p output files per session: `F.npy` (raw fluorescence traces), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (processing parameters including `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `prctile_baseline`, `fs`).

ii.
```python
def compute_baseline_corrected_fluorescence(s2p_dir: Path) -> ...:
    ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(s2p_dir / "F.npy", mmap_mode="r")
    Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. The AI documented that the paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and identified F.npy and Fneu.npy as the source variables, consistent with Suite2p's standard outputs.

## 2-b. How is the `neural` data processed?

i. Processing follows two steps: (1) Neuropil subtraction: `Fc = F - neucoeff * Fneu` where `neucoeff=0.7` from stored `ops.npy`; (2) Suite2p baseline correction using `suite2p.extraction.dcnv.preprocess()` with parameters from `ops.npy` (baseline='maximin', win_baseline=60, sig_baseline=10, prctile_baseline=8, fs=30). The result is then averaged in non-overlapping 10-frame bins.

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
    ...
).astype(np.float32, copy=False)
```

```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)

def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32)
```

iii. The AI justified this by noting the paper says analyses used "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and the decoding section specifies "slightly denoised the dF/F ... by averaging in bins of 10 consecutive timestamps." The AI chose to use `dcnv.preprocess` (Suite2p's built-in function) rather than the GUI's `F_processing` which uses `neucoeff=0.0`, because the paper specifies "default Suite2p parameters" and the stored ops.npy contains `neucoeff=0.7`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The data release already contains only Track2p-matched neurons (cells present across all days for each subject), and Suite2p cell classification (iscell > 0.5 threshold) was already applied before Track2p matching.

ii.
```python
# No explicit neuron filtering code - all rows from F.npy are used
neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
# All neurons in the processed array are kept
```

iii. The AI noted that "the bundled neural data already contain matched-cell exports" and cited the paper: "We considered all ROIs above the default threshold of 0.5 as true cells." Since the data release already contains only tracked cells, no additional filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each pseudo-trial is aligned to the start of its respective 2-minute block within the continuous recording. The `temporal_alignment_event` is described as "Start of each consecutive 2-minute block cut from a continuous recording session." `off_start = 0.0`, `off_end = 120.0` seconds.

ii.
```python
'temporal_alignment_event': "Start of each consecutive 2-minute block cut from a continuous recording session",
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,  # 120.0
```

iii. The AI noted there are no natural trial events in these spontaneous behavior recordings. The 2-minute block boundaries serve as the alignment events, matching the paper's decoder cross-validation split structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Raw data is acquired at 30 Hz (33.33 ms per frame). The AI applies 10-frame non-overlapping mean binning, resulting in a time bin size of 333.33 ms (1000 * 10 / 30). This rebinning is applied to neural, behavior, and time data before trial segmentation.

ii.
```python
RAW_FS_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ  # = 333.33 ms
```

iii. The AI cited the paper: "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The metadata records `time_bin_size: 333.33` ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The elapsed time input is derived from the frame index (0 to nframes-1) and the sampling rate `fs` from `ops.npy`. No raw timestamp variable is used directly for the input; instead, frame indices are converted to seconds using the known frame rate.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. The AI noted that no reference input variable exists in the paper's decoding (the paper decoded continuous behavior from neural activity alone). The elapsed-time input is a task-specific requirement from the benchmark instructions. The frame index is the most reliable time reference since imaging frames serve as the master clock.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices are offset by +0.5 to center times at mid-frame, then divided by the sampling rate (30 Hz) to get seconds. The resulting time array is then averaged in the same 10-frame bins as the neural data.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The AI documented that the +0.5 offset centers each time value at the midpoint of the frame acquisition interval. After binning, the first bin center is at 0.167 s and values represent the mean time of the 10 frames in each bin.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input uses absolute elapsed time from the session start, not time relative to each trial. After 10-frame binning, the time array is segmented into the same 2-minute blocks as the neural data. Trial N's time input spans from N*120 to (N+1)*120 seconds approximately.

ii.
```python
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
# where start = trial_idx * trial_bins, stop = start + trial_bins
```

iii. The AI justified keeping absolute session time (rather than resetting per trial) because the instructions specify "Time elapsed from the beginning of the experiment."

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (pre-computed global motion energy), with `tstamps.npy` and `interframe_int.npy` used to identify and reconstruct missing camera frames.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. The AI noted that motion energy was pre-computed in the data release as "squared pixel-wise difference of consecutive frames" (matching the paper's methods). The timing files allow reconstruction of missing camera frames to ensure alignment with imaging frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves: (1) Reconstruct full-length motion trace by inserting missing frames at positions identified by timing gaps, then linearly interpolating; (2) Average in 10-frame non-overlapping bins; (3) Global min-max normalization across all sessions; (4) Discretize into 5 equal-percentile (quintile) bins.

ii.
```python
# Step 1: Reconstruct missing frames
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)

# Step 2: 10-frame binning
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]

# Step 3: Global min-max normalization
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)

# Step 4: Quintile discretization
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
```

iii. The AI justified the normalization and discretization by citing the instructions: "Motion energy, normalized and discretized into five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, the normalized motion energy values are discretized into 5 quintile bins using `np.quantile` at [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] computed over all binned samples from all sessions. `np.digitize` assigns each value to bins labeled 0-4 (Q1-Q5). The edges are computed globally and then applied uniformly.

ii.
```python
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)  # ensure monotonicity

output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. The AI verified that global quintile discretization produces exactly balanced class frequencies (0.2 each) across the full dataset, as confirmed in verification output.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data through framewise synchronization (camera triggered by microscope at 30 Hz). Missing camera frames are reconstructed to match imaging frame count. Both streams are then binned identically (10-frame means) and segmented into the same 2-minute blocks.

ii.
```python
# Ensure behavior matches neural length
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
# Same binning
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
# Same trial segmentation
neural_trial = neural_binned[:, start:stop]
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
```

iii. The AI cited the paper: "Videos were recorded at 30 Hz, with the microscope acquisition acting as a trigger for camera frame acquisition, also allowing for simple synchronisation across the two modalities." Missing frames are handled by reconstructing their positions from timing gaps.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The primary data quality issue is missing camera frames in 9 of 41 sessions (ranging from 1 to 148 missing frames). The AI identifies missing frames by analyzing inter-frame intervals from `interframe_int.npy`: intervals approximately double the median indicate a dropped frame. Missing positions are reconstructed via cumulative step counting, then NaN values at missing positions are linearly interpolated.

ii.
```python
def reconstruct_motion_trace(motion, interframe_int, target_len):
    median_interval = float(np.median(interframe_int))
    steps = np.rint(interframe_int / median_interval).astype(np.int64)
    steps[steps < 1] = 1
    observed_idx = np.empty(motion.shape[0], dtype=np.int64)
    observed_idx[0] = 0
    observed_idx[1:] = np.cumsum(steps)
    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    missing_idx = np.flatnonzero(np.isnan(full))
    full = interpolate_nans(full)
    return full, missing_idx

def interpolate_nans(x):
    idx = np.arange(x.shape[0])
    valid = ~np.isnan(x)
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return out
```

iii. The AI cited the data README which says missing frames "can be obtained by looking at tstamps.npy or interframe_int.npy and treated as missing values for motion energy or they can be interpolated over." The AI chose interpolation to maintain temporal continuity before binning.

## 6-a. What are the most time-consuming steps of the code?

i. The Suite2p baseline correction (`dcnv.preprocess`) is by far the most time-consuming step, dominating per-session processing time. The full conversion takes ~86 seconds for 41 sessions (~2.1 s/session mean), with larger sessions (more neurons) taking proportionally longer (up to ~3.6s for 746-neuron sessions vs ~0.5s for 221-neuron sessions).

ii.
```python
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    ...
)
```

iii. The AI documented timing information in conversion output and CONVERSION_NOTES.md, noting that the main computational cost is the Suite2p baseline correction step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over trials one at a time, slicing arrays. This could be replaced with a single reshape operation (reshape the binned arrays into (n_trials, trial_bins) and slice all trials at once). However, since the number of trials per session is small (10-15), this loop is not a significant bottleneck.

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

iii. The AI noted this in CONVERSION_NOTES.md but considered it low priority since the main bottleneck is the dcnv.preprocess call. The binning itself uses efficient reshape-and-mean operations.

## 6-c. What processing does the code repeat multiple times?

i. The motion normalization is computed twice: once globally in `build_dataset` to determine edges, then again per-session when creating trials. The preview data for plotting (neural_preview, motion_preview) is computed for every session regardless of whether `--show-processing` is requested.

ii.
```python
# Global computation in build_dataset:
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Per-session re-normalization:
for session in processed_sessions:
    motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
```

iii. The AI did not explicitly document this redundancy but noted that the design keeps sessions independent during initial processing and only applies global normalization afterward.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `neural_preview` and `motion_preview` data structures for every session, even when `--show-processing` is not used. These store raw/corrected fluorescence samples and motion reconstruction details for visualization only. This adds memory overhead and minor computation time for every session.

ii.
```python
# Always computed in process_session():
sample_neuron = min(2, processed.shape[0] - 1)
preview_len = min(3000, processed.shape[1])
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    ...
}
# motion_preview similarly always computed
```

iii. The AI did not document this as an inefficiency. The preview data is used only when plotting is requested, but is computed unconditionally.
