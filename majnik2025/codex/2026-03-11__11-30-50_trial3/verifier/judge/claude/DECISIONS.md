# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by discovering subject directories starting with `jm` under the `data/` root, then finding session subdirectories whose names start with 4 digits. For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. A two-pass pipeline is used: pass 1 scans behavior streams for all sessions, pass 2 processes neural data.

ii.
```python
def discover_sessions(sample: bool) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
        session_dirs = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )
        for session_dir in session_dirs:
            sessions.append(SessionInfo(subject=subject_dir.name, session_id=..., path=session_dir))
```

```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI's CONVERSION_NOTES.md documents that subjects are identified as `jm*` directories, sessions are subdirectories, and the data structure follows the Track2p paper dataset organization. The two-pass design was chosen for memory efficiency.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data root directory, sorted alphabetically. The AI identifies 6 subjects: jm031, jm032, jm038, jm039, jm040, jm046.

ii.
```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Each `jm*` directory represents one mouse, consistent with the paper describing 6 mice.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a date-named subdirectory within a subject's folder (filtered to directories whose names start with 4 digits). Sessions are sorted alphabetically.

ii.
```python
session_dirs = sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
)
```

iii. The AI documents that the dataset contains 41 total sessions (7 per mouse except jm040 with 6). This matches the released data.

## 1-d. How are the data split into trials?

i. Since the data has no natural trial structure, the AI creates pseudo-trials as consecutive non-overlapping 2-minute blocks. After 10-frame temporal averaging, each block contains 360 time bins. This yields 10 trials for 20-minute sessions and 15 trials for 30-minute sessions (545 total trials).

ii.
```python
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)  # 360

def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

iii. The AI justifies this choice based on the paper's statement that "splits were done on consecutive 2 minute blocks of the recording" for their decoding cross-validation. The CONVERSION_NOTES.md explicitly documents this decision.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial-level quality filtering. All complete 2-minute blocks are retained. An assertion checks that each session produces at least 2 trials.

ii.
```python
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The CONVERSION_NOTES.md states there are no native trials and no trial curation rules beyond ensuring at least 2 trials per session for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p parameters) from `suite2p/plane0/`.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)  # loads ops.npy
```

iii. The AI identified these as the standard Suite2p output files and documented that the paper uses "baseline corrected fluorescence traces" from Suite2p.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`F - neucoeff * Fneu`) followed by Suite2p's `dcnv.preprocess` for baseline correction, using per-session `ops.npy` parameters. Then the data is temporally binned by averaging in 10-frame bins.

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

neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)  # 10-frame bins
```

iii. The AI documents that the paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and that 10-frame averaging matches the paper's "averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron quality filtering is applied. All neurons present in the provided `F.npy` are used, since the released data already contains only matched/tracked neurons.

ii. N/A (no filtering code)

iii. The AI's CONVERSION_NOTES.md documents that the data README states the `suite2p/` folder already contains only neurons "present across all days," and that all `iscell` values are above 0.5 in the sampled sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since the data is continuous and trials are consecutive 2-minute blocks, there is no event-based alignment. The temporal alignment event is described as "start of each consecutive 2-minute block."

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
"off_start": 0.0,
"off_end": BLOCK_DURATION_SEC,  # 120.0
```

iii. The AI justifies this by noting there is no stimulus event to align to in this spontaneous behavior dataset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal averaging, converting the 30 Hz data to an effective 3 Hz (333.33 ms bins). This is done for both neural and behavioral data.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
# ...
"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,  # 333.33 ms

def bin_average_2d(x, bin_size):
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2)
```

iii. The AI justifies this based on the paper's statement about "averaging in bins of 10 consecutive timestamps" for the decoding analysis.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from frame indices using the imaging frame rate (`ops['fs']`) and the bin size, not from any raw timestamp variable.

ii.
```python
def make_time_input(nbins, fs, bin_size_frames):
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

iii. The AI notes that the frame rate is constant at 30 Hz, so computing time from frame indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input represents absolute elapsed time from session start in seconds. Each bin's time is computed as `bin_index * (bin_size_frames / fs)`. This is NOT relative to trial start — it continues across trial boundaries within a session.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# This creates time for the FULL session, then split into blocks
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The AI chose absolute elapsed time from session start as the decoder input, noting this is mandated by the task specification.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is derived from the same frame indices as the neural data (after 10-frame binning), so they are inherently aligned. The time vector for the full session is computed once, then split into blocks in the same way as neural and output data.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# Same length as neural_binned.shape[1], split into same blocks
```

iii. No special alignment is needed since time is computed from the same frame grid.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used to align behavior frames to imaging frames.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI identified the motion energy file as the pre-computed global motion energy from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Align motion to imaging frame grid using timestamps (interpolation for missing frames), (2) average in 10-frame bins, (3) normalize using global min-max normalization across all sessions, (4) discretize into 5 bins using global quintile edges (0.2, 0.4, 0.6, 0.8 quantiles).

ii.
```python
# Alignment
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

# 10-frame binning
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)

# Global min-max normalization
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)

# Quintile discretization
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The AI documents that 10-frame averaging matches the paper's preprocessing, global normalization removes across-session scale differences, and quintile-based discretization ensures balanced class counts.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is normalized to [0,1] via global min-max, then discretized into 5 bins using global quintile edges at the 20th, 40th, 60th, and 80th percentiles. `np.digitize` maps values to bin indices 0-4.

ii.
```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
def motion_to_bins(x, quantile_edges):
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```

iii. Quintile-based binning ensures approximately 20% of data in each bin globally.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to imaging frames using timestamp mapping. Behavior timestamps are normalized to the range [0, nframes-1], and behavior samples are mapped to imaging frame indices. Missing frames are filled by linear interpolation. After alignment, both streams are averaged in the same 10-frame bins.

ii.
```python
def align_motion_to_imaging(motion, tstamps, nframes):
    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)
    summed = np.zeros(nframes, dtype=np.float64)
    counts = np.zeros(nframes, dtype=np.int64)
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
    # interpolate missing
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
    return aligned.astype(np.float32), stats
```

iii. The CONVERSION_NOTES.md states that the microscope triggered the camera for synchronization, and the conversion treats imaging frames as the master clock.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior frames are handled via timestamp-based alignment and linear interpolation. When the motion energy array is shorter than the imaging frames, the AI maps behavior samples onto imaging frame indices and interpolates missing positions. Sessions where binned timepoints aren't divisible by the block size raise an error.

ii.
```python
# In align_motion_to_imaging:
missing = np.flatnonzero(~good)
if missing.size:
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```

iii. The AI documents that 9 sessions have fewer behavior samples than imaging frames, and the alignment procedure handles these via interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is Suite2p's `dcnv.preprocess` baseline correction, which runs on CPU in the AI's code. The full conversion takes about 61 seconds, with the neural conversion pass taking ~58 seconds.

ii. N/A

iii. The CONVERSION_NOTES.md documents that Suite2p baseline correction is the bottleneck and that the two-pass design avoids loading all neural data simultaneously.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The session-by-session processing loop could potentially be parallelized (though not easily vectorized). The block splitting operations use list comprehensions but are already reasonably efficient.

ii. N/A

iii. The AI's code uses vectorized operations for binning (`reshape + mean`) and alignment (`np.add.at`), so there are few obvious vectorization opportunities remaining.

## 6-c. What processing does the code repeat multiple times?

i. The code loads motion energy and timestamps twice: once in pass 1 (behavior scan for global statistics) and once in pass 2 (neural conversion). `load_ops` is also called twice per session (once in pass 1 for frame count, once in pass 2 for preprocessing parameters). The motion alignment is also computed twice.

ii.
```python
# Pass 1:
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy")
tstamps = np.load(session.path / "move_deve" / "tstamps.npy")
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

# Pass 2:
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy")
tstamps = np.load(session.path / "move_deve" / "tstamps.npy")
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The two-pass design was chosen for memory efficiency (avoiding loading all neural data at once), at the cost of redundant behavior loading.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes detailed alignment statistics (`align_stats`) for each session in both passes, though these are only used for metadata. The `session_meta` list stores detailed per-session information that may not be needed for decoding. The motion alignment in pass 2 is redundant since only the binned version (cached from pass 1) is used.

ii.
```python
session_meta.append({
    "session_id": session.session_id,
    "subject": session.subject,
    "session_path": str(session.path),
    "nframes_raw": nframes,
    "duration_sec": nframes / float(ops["fs"]),
    "nneurons": int(np.load(...).shape[0]),
    "behavior_missing_frames": align_stats["missing_frames"],
    "behavior_duplicate_timestamp_bins": align_stats["duplicate_timestamp_bins"],
    "binned_timepoints": int(len(motion_binned)),
    "n_trials_expected": int(len(motion_binned) // BLOCK_BINS),
})
```

iii. The metadata and statistics are useful for documentation and debugging but are not consumed by the downstream decoder.
