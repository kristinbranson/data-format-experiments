# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by iterating over sorted subject directories (folders starting with "jm") under `data/`, then sorted session subdirectories within each subject. For each session it loads: `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `suite2p/plane0/ops.npy` for neural data, and `move_deve/motion_energy_glob.npy`, `move_deve/tstamps.npy` for behavioral data. A two-pass approach is used: Pass 1 scans all sessions to collect motion energy for global normalization/quantile computation; Pass 2 loads neural data session-by-session for conversion.

ii.
```python
def discover_sessions(sample: bool) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
        session_dirs = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )
        for session_dir in session_dirs:
            sessions.append(
                SessionInfo(
                    subject=subject_dir.name,
                    session_id=f"{subject_dir.name}_{session_dir.name}",
                    path=session_dir,
                )
            )
    ...

# Pass 2 loading per session:
ops = load_ops(session)
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI noted that the data directory has 6 subject folders, each containing session subdirectories with Suite2p-format neural data and behavioral data. The data README confirms this structure. The two-pass design avoids holding all neural data in memory simultaneously while still computing global motion quantiles.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the top-level directory names under `data/` (e.g., `jm031`, `jm032`, ..., `jm046`). A sorted unique list of subject names is created, and each session is mapped to its subject via `subject_to_idx`.

ii.
```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
converted["subjects"] = subjects
converted["subject_idx"] = np.array([subject_to_idx[session.subject] for session in sessions], dtype=np.int64)
```

iii. The AI documented that there are 6 subjects (`jm031` through `jm046`) matching both the paper and released data, and that subject identity comes from the directory structure.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder (named by date, e.g., `2023-04-30_a`) constitutes one session. Sessions are processed individually and stored as separate entries in the output lists. Each session produces one entry in `neural`, `input`, `output`, etc.

ii.
```python
for session_dir in session_dirs:
    sessions.append(
        SessionInfo(
            subject=subject_dir.name,
            session_id=f"{subject_dir.name}_{session_dir.name}",
            path=session_dir,
        )
    )
```

iii. The AI identified 41 total sessions across 6 subjects (7 sessions each for 5 mice, 6 for jm040), which matches the released data structure.

## 1-d. How are the data split into trials?

i. Trials are constructed as consecutive non-overlapping 2-minute blocks from the continuous recording. After 10-frame temporal binning at 30 Hz, each 2-minute block contains 360 time bins. The neural, input, and output data are all split into blocks of this size. 20-minute sessions yield 10 trials, 30-minute sessions yield 15 trials.

ii.
```python
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)  # = 360

def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The AI noted that the paper's decoder uses "consecutive 2 minute blocks of the recording" for cross-validation splits, so this block structure is paper-consistent. The 10-frame averaging matches the paper's "averaging in bins of 10 consecutive timestamps."

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial filtering or quality control is applied. All complete 2-minute blocks are retained. The code raises an error if a session would produce fewer than 2 trials, but does not filter individual trials based on data quality. Any partial block at the end of a session (not filling a full 360 bins) is silently discarded.

ii.
```python
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The AI documented that there are no native trials in this dataset and no trial-quality filtering criteria described in the paper. The minimum-2-trials check ensures decoder cross-validation can work.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from three raw files per session: `F.npy` (fluorescence traces), `Fneu.npy` (neuropil traces), and `ops.npy` (Suite2p processing parameters including baseline correction settings).

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)  # loads ops.npy
```

iii. The AI chose F and Fneu over spks.npy because the paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes three processing steps: (1) Suite2p baseline-corrected fluorescence is computed from F and Fneu using `Fc = F - neucoeff * Fneu` followed by Suite2p's `dcnv.preprocess()` with session-specific ops parameters; (2) the result is temporally binned by averaging in non-overlapping windows of 10 frames; (3) the binned data is split into 2-minute block trials.

ii.
```python
def compute_suite2p_baseline_corrected(F, Fneu, ops):
    fc = F.astype(np.float32) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(np.float32)
    return dcnv.preprocess(
        fc.copy(),
        baseline=ops.get("baseline", "maximin"),
        win_baseline=float(ops.get("win_baseline", 60.0)),
        sig_baseline=float(ops.get("sig_baseline", 10.0)),
        fs=float(ops.get("fs", IMAGING_FS)),
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 100)),
        device=torch.device("cpu"),
    ).astype(np.float32)

neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
```

iii. The AI explicitly chose Suite2p's `dcnv.preprocess` over the Track2p GUI's `F_processing` because the latter defaults to `neucoeff=0.0`, which contradicts the paper's stated use of default Suite2p parameters (which use `neucoeff=0.7`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied beyond what is already in the released data. The AI trusts that the provided `suite2p/plane0/F.npy` files contain only neurons tracked across all days (as stated in the data README), and that these neurons already passed Suite2p's `iscell` threshold (>0.5).

ii. There is no neuron filtering code in `convert_data.py`. All neurons from F.npy are used directly:
```python
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
# No filtering step between loading and processing
```

iii. The AI verified that `iscell.npy` values are all above 0.5 in sampled sessions, confirming the data is pre-filtered. The data README explicitly states the suite2p folder "only includes traces for the cells present across all days."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The temporal alignment event is "start of each consecutive 2-minute block." Neural data is split into consecutive 2-minute blocks starting from the beginning of the session. Each trial starts at the block boundary, with `off_start=0.0` and `off_end=120.0` seconds relative to the block start.

ii.
```python
converted["metadata"]["temporal_alignment_event"] = "start of each consecutive 2-minute block; input stores absolute elapsed time from session start"
converted["metadata"]["off_start"] = 0.0
converted["metadata"]["off_end"] = BLOCK_DURATION_SEC  # 120.0
```

iii. The AI noted that the paper uses "consecutive 2 minute blocks of the recording" for cross-validation, and that there are no discrete experimental events in this spontaneous behavior paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10 frames at 30 Hz = 333.33 ms per bin. Temporal rebinning is applied: the raw 30 Hz data is averaged in non-overlapping windows of 10 consecutive frames, producing an effective 3 Hz sampling rate.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
# time_bin_size in ms:
"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,  # = 333.33 ms

def bin_average_2d(x, bin_size):
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2, dtype=np.float64).astype(np.float32)
```

iii. The paper states "averaging in bins of 10 consecutive timestamps" for decoding analyses, which the AI directly replicates.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is derived from the `ops['fs']` (sampling frequency, 30 Hz) and the bin size (10 frames). It is computed as a synthetic time vector based on frame indices, not from any raw data file directly.

ii.
```python
def make_time_input(nbins, fs, bin_size_frames):
    step = bin_size_frames / fs  # = 10/30 = 0.333 sec
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]

time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. The AI noted that the task instructions specify "Time elapsed from the beginning of the experiment" as the decoder input, and that this is a synthetic variable derived from the known sampling rate and temporal binning.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `bin_index * (bin_size_frames / fs)`, giving absolute elapsed time from session start in seconds. For the full session, this ranges from 0.0 to ~1199.7s (20-min sessions) or ~1799.7s (30-min sessions). When split into 2-minute block trials, each trial's input retains the absolute session time (not relative block time).

ii.
```python
step = bin_size_frames / fs  # 10/30 = 0.3333 sec
return (np.arange(nbins, dtype=np.float32) * step)[None, :]

# Split preserves absolute time:
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The AI justified using absolute elapsed time from session start (rather than relative time within each block) because the task says "Time elapsed from the beginning of the experiment."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input vector is generated to have exactly the same number of time bins as the binned neural data (both derived from the same `nframes` and `bin_size`), so alignment is inherent by construction. Both are split into identical 2-minute blocks.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# neural_binned.shape[1] == len(time_binned) by construction
```

iii. The AI noted this is trivially aligned since both the neural data and the time vector derive from the same frame count and binning parameters.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy` (global motion energy from videography) and `move_deve/tstamps.npy` (timestamps for behavior frames, used for alignment when frames are missing).

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI identified these from the data structure and the paper's description of motion energy computed from "pixelwise difference... squared... summed across pixels" of consecutive video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves: (1) align motion energy to imaging frame grid using timestamps (interpolating missing frames); (2) average in 10-frame bins; (3) normalize globally to [0,1] using min-max across all sessions; (4) discretize into 5 equal-percentile bins using global quintile thresholds.

ii.
```python
# Step 1: Align
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

# Step 2: Bin average
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)

# Step 3: Global min-max normalization
all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

# Step 4: Discretize
motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12))
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The AI preserved the paper's 10-frame averaging before discretization, and used global normalization and quintile thresholds to satisfy the instruction requirement of "five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is first normalized to [0,1] using global min-max normalization across all sessions. Then global quintile edges at 20th, 40th, 60th, and 80th percentiles are computed from the pooled normalized data. `np.digitize` maps each value to one of 5 bins (0-4).

ii.
```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

def motion_to_bins(x, quantile_edges):
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```

iii. The AI chose global (across all sessions) quintile computation to preserve cross-session comparability, noting the instruction requires "five equal-percentile bins." The global distribution is exactly 20% per bin by construction.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to the imaging frame grid using timestamp-based mapping. When behavior and imaging frame counts match, motion is used directly. When there are missing camera frames, timestamps are used to map behavior samples to the nearest imaging frame indices, averaging duplicates and linearly interpolating gaps. After alignment, both motion and neural data undergo the same 10-frame binning, ensuring temporal alignment.

ii.
```python
def align_motion_to_imaging(motion, tstamps, nframes):
    if len(motion) == nframes:
        aligned = motion.astype(np.float32, copy=False)
        return aligned, stats

    # Map behavior timestamps to imaging frame indices
    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)

    # Average duplicates, interpolate gaps
    summed = np.zeros(nframes, dtype=np.float64)
    counts = np.zeros(nframes, dtype=np.int64)
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
    ...
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
    return aligned.astype(np.float32), stats
```

iii. The AI noted that the paper says "microscope acquisition acting as a trigger... simple synchronisation" and the data README says missing frames can be "interpolated over" using timestamps, which is what the code implements.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (9 sessions have fewer behavior samples than imaging frames) are handled by timestamp-based alignment: behavior samples are mapped to imaging frame indices using normalized timestamps, and missing positions are filled via linear interpolation. No sessions or trials are dropped due to missing data. The code validates that after alignment, neural and motion data have matching lengths.

ii.
```python
# From align_motion_to_imaging:
known_idx = np.flatnonzero(good)
missing = np.flatnonzero(~good)
if missing.size:
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```

iii. The AI documented the specific sessions with missing frames (e.g., `jm031/2023-10-22_a` missing 116 frames) and verified that interpolation produces consistent results via sanity checks against raw data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p baseline correction (`dcnv.preprocess`) during Pass 2, which processes the full F and Fneu arrays for each session on CPU. The conversion log shows Pass 2 taking 57.79s out of 60.97s total. Individual 30-minute sessions with many neurons (e.g., jm039 with 746 neurons) take ~2.2s each.

ii.
```python
# Per the timing output:
# Pass 1 (behavior scan): 2.39s
# Pass 2 (neural conversion): 57.79s
# Total conversion time: 60.97s
```

iii. The AI identified this and designed the two-pass architecture to separate the cheap behavior scanning (Pass 1) from the expensive neural processing (Pass 2), keeping memory bounded to one session at a time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop over sessions (both Pass 1 and Pass 2) cannot easily be vectorized because sessions have different sizes and Suite2p baseline correction is inherently per-session. The `align_motion_to_imaging` function uses `np.add.at` (scatter-add) which is already vectorized. The block splitting uses list comprehensions over slices, which are already efficient NumPy operations.

ii.
```python
# The per-session loop is the main structure:
for idx, session in enumerate(sessions):
    ...
    neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
    neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
    ...
```

iii. The AI noted that the two-pass streaming design was chosen to keep memory bounded, and that Suite2p's baseline correction is the bottleneck which cannot be further vectorized beyond what Suite2p already does internally.

## 6-c. What processing does the code repeat multiple times?

i. The code loads motion energy and timestamps twice for each session: once in Pass 1 (to compute global normalization) and again in Pass 2 (for alignment during the main conversion). It also calls `align_motion_to_imaging` twice per session and loads `ops.npy` twice per session (once in Pass 1 metadata collection, once in Pass 2).

ii.
```python
# Pass 1:
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

# Pass 2 (same session):
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The AI acknowledged this redundancy but justified the two-pass design by noting that it avoids holding all neural data in memory simultaneously. The behavior data is small so the redundant loading is cheap (~2.4s total for Pass 1).

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects detailed per-session metadata (session paths, duration, frame counts, missing frame stats) in `session_meta` stored in `metadata['session_info']`. While useful for documentation, this is not used by the decoder. The `show_processing` plotting code generates visualization plots that are not used downstream. The code also stores `motion_binned_by_session` from Pass 1 but re-derives `motion_binned` in Pass 2 (though the stored values are used for normalization).

ii.
```python
session_meta.append({
    "session_id": session.session_id,
    "subject": session.subject,
    "session_path": str(session.path),
    "nframes_raw": nframes,
    "duration_sec": nframes / float(ops["fs"]),
    "nneurons": int(np.load(session.path / "suite2p" / "plane0" / "F.npy", mmap_mode="r").shape[0]),
    "behavior_missing_frames": align_stats["missing_frames"],
    "behavior_duplicate_timestamp_bins": align_stats["duplicate_timestamp_bins"],
    "binned_timepoints": int(len(motion_binned)),
    "n_trials_expected": int(len(motion_binned) // BLOCK_BINS),
})
```

iii. The AI included the metadata for audit and reproducibility purposes, which is reasonable for a conversion pipeline even if not strictly needed by the decoder.
