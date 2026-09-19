# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by scanning the `data/` directory for folders starting with `jm`, then within each subject folder finds session subdirectories whose names start with a 4-digit year. For each session, it loads `suite2p/plane0/F.npy`, `Fneu.npy`, and `ops.npy` for neural data, and `move_deve/motion_energy_glob.npy` and `tstamps.npy` for behavior. The conversion is done in two passes: first scanning all behavior streams to compute global normalization/quantile thresholds, then processing neural data session-by-session.

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
```

Loading neural data:
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
```

Loading behavior data:
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md that all `jm*` directories represent subjects and all subdirectories represent sessions. The two-pass design was chosen to compute global motion normalization before processing neural data.

## 1-b. How are the data split into subjects?

i. Subjects are identified as directories starting with `jm` in the data root, sorted alphabetically. A mapping from subject name to index is created for `subject_idx`.

ii.
```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI noted that each `jm*` directory represents one mouse, consistent with the dataset organization described in `data/README.md`.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, filtered to those whose names start with a 4-digit number (year), sorted alphabetically.

ii.
```python
session_dirs = sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
)
```

iii. The AI noted that each subdirectory contains suite2p output and motion energy files for one recording session. The filter `p.name[:4].isdigit()` ensures only date-named session directories are included.

## 1-d. How are the data split into trials?

i. The AI splits continuous sessions into consecutive non-overlapping **2-minute (120-second)** blocks. After 10-frame temporal binning at 30 Hz, each block contains 360 time bins. The AI chose 2-minute blocks based on the paper's statement that "splits were done on consecutive 2 minute blocks of the recording."

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

iii. The AI justified this in CONVERSION_NOTES.md Step 5: "The paper's decoder splits recordings into consecutive 2-minute blocks. After 10-frame averaging, each block contains 120 s x 3 Hz = 360 time bins." The AI also verified the data is evenly divisible (20-min sessions yield 10 trials, 30-min sessions yield 15 trials).

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial-level quality filtering. All trials from all sessions are included. The code raises an error if a session would yield fewer than 2 trials, but this never occurs in practice.

ii.
```python
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The AI noted that the data are continuous spontaneous recordings with no native trial structure, so no trial-level quality criteria apply.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p parameters) from `suite2p/plane0/` in each session directory.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)  # loads ops.npy
```

iii. These are the standard suite2p output files. The AI noted that the paper states "baseline corrected fluorescence traces" were used.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`Fc = F - neucoeff * Fneu`) followed by Suite2p's `dcnv.preprocess` for baseline correction. The processing parameters are read from each session's `ops.npy` file (with defaults matching Suite2p defaults). The result is then averaged into 10-frame non-overlapping bins.

ii.
```python
def compute_suite2p_baseline_corrected(F, Fneu, ops):
    fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(np.float32, copy=False)
    return dcnv.preprocess(
        fc.copy(),
        baseline=ops.get("baseline", "maximin"),
        win_baseline=float(ops.get("win_baseline", 60.0)),
        sig_baseline=float(ops.get("sig_baseline", 10.0)),
        fs=float(ops.get("fs", IMAGING_FS)),
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 100)),
        device=torch.device("cpu"),
    ).astype(np.float32, copy=False)
```

Binning:
```python
def bin_average_2d(x, bin_size):
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2, dtype=np.float64).astype(np.float32)
```

iii. The AI justified this by referencing the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." Reading parameters from `ops.npy` was chosen to be "paper-consistent" by using per-session parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the provided `F.npy` files are included. The AI relied on the fact that the released data already contain only tracked neurons that passed Suite2p's `iscell` filter.

ii. No filtering code is present.

iii. The AI documented in CONVERSION_NOTES.md Step 4: "Provided iscell.npy values are all above 0.5 in sampled sessions, consistent with pre-filtered matched output." The data README states the suite2p folder already contains only successfully tracked neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. Trials are contiguous 2-minute blocks starting from the beginning of each session. The `temporal_alignment_event` is described as "start of each consecutive 2-minute block."

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
"off_start": 0.0,
"off_end": BLOCK_DURATION_SEC,
```

iii. The AI noted there is no stimulus event to align to in this spontaneous recording dataset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting 30 Hz data to 3 Hz (333.33 ms time bins). This binning is applied before motion energy discretization.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0

"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,  # 333.33 ms
```

iii. The AI referenced the paper methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not derived from any raw data variable. It is computed from bin indices, the frame rate (`fs` from `ops.npy`), and the bin size.

ii.
```python
def make_time_input(nbins, fs, bin_size_frames):
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

iii. The AI noted that since the frame rate is constant at 30 Hz and bins are 10 frames, time can be computed directly from indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (bin_size_frames / fs)`, giving elapsed seconds from session start. The time series is continuous across the full session before being split into trial blocks.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# Then split into blocks:
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The AI documented this as providing "absolute elapsed time from session start" as required by the task specification.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is derived from the same bin indices as the neural data, so it is inherently aligned. Both use the same binning and block-splitting operations.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# neural and time are split using the same BLOCK_BINS
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. Alignment is guaranteed by construction since both are indexed by the same bin positions.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` and `tstamps.npy` in the `move_deve/` subdirectory of each session.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI noted that `motion_energy_glob.npy` contains pre-computed global motion energy from behavioral video and `tstamps.npy` is needed to handle missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Motion energy is aligned to imaging frames using timestamps (`tstamps.npy`), with linear interpolation over missing camera frames. (2) The aligned signal is averaged into 10-frame bins. (3) The binned signal is normalized globally using min-max normalization across all sessions. (4) Global quintile edges (20th, 40th, 60th, 80th percentiles) are computed across all normalized values. (5) Values are discretized into 5 bins using `np.digitize`.

ii.
```python
# Alignment
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

# Binning
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)

# Global normalization
all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

# Discretization
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The AI justified global normalization for cross-session comparability. The quintile discretization satisfies the task requirement of "five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is first min-max normalized globally, then discretized using global quintile edges at the 20th, 40th, 60th, and 80th percentiles. `np.digitize` maps values into 5 bins (0-4).

ii.
```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)

def motion_to_bins(x, quantile_edges):
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```

iii. The AI noted this produces exactly balanced quintiles globally. The global (rather than per-session) approach was chosen for cross-session comparability.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to the imaging frame grid using behavioral timestamps (`tstamps.npy`). The timestamps are normalized and mapped to imaging frame indices using `np.round`. Values at missing frames are filled by linear interpolation. After alignment, the same 10-frame binning and block-splitting is applied to both streams.

ii.
```python
def align_motion_to_imaging(motion, tstamps, nframes):
    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)
    summed = np.zeros(nframes, dtype=np.float64)
    counts = np.zeros(nframes, dtype=np.int64)
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
    aligned = np.full(nframes, np.nan, dtype=np.float64)
    good = counts > 0
    aligned[good] = summed[good] / counts[good]
    missing = np.flatnonzero(~good)
    if missing.size:
        aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
    return aligned.astype(np.float32), stats
```

iii. The AI documented that because the microscope triggered the camera, they share the same time base, but occasional dropped video frames require interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where motion energy array is shorter than neural data) are handled by aligning behavior timestamps to imaging frame indices and interpolating missing values. An assertion checks that neural and motion binned lengths match. Remainder bins at the end of sessions that don't fill a complete trial block are discarded.

ii.
```python
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
if neural_binned.shape[1] % BLOCK_BINS != 0:
    raise ValueError(...)
```

iii. The AI documented in CONVERSION_NOTES.md that 9 sessions have missing camera frames and that all were successfully handled by the timestamp-based alignment.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p `dcnv.preprocess` baseline correction, run on CPU. The AI's two-pass design also means behavior data is loaded twice (once for quantile computation, once during neural processing).

ii. N/A

iii. The AI documented per-session timing: ~1.5s for 20-min sessions and ~3.6s for 30-min sessions, with total conversion taking ~61 seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The block-splitting loops (`split_into_blocks_1d`, `split_into_blocks_2d`) use Python list comprehensions with array slicing, which could be replaced with `np.split`. However, these are not significant bottlenecks.

ii.
```python
def split_into_blocks_2d(x, block_bins):
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

iii. The AI noted these are not significant performance bottlenecks compared to the Suite2p preprocessing.

## 6-c. What processing does the code repeat multiple times?

i. The two-pass design means behavior data (`motion_energy_glob.npy`, `tstamps.npy`) is loaded and aligned twice: once in pass 1 for global quantile computation, and again in pass 2 during neural processing. The `ops.npy` file is also loaded multiple times per session (once for shape inspection, once for neural processing).

ii.
```python
# Pass 1:
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

# Pass 2 (same files loaded again):
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The AI justified the two-pass approach as necessary for computing global normalization before processing, and as keeping memory usage bounded.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies global min-max normalization to motion energy before computing quintile edges. This normalization is unnecessary because `np.quantile`/`np.percentile` on unnormalized data would produce the same quintile assignments (percentile-based binning is invariant to monotone transformations). The normalization adds processing and the quantile edges are stored in metadata but not used downstream.

ii.
```python
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```

iii. The AI documented this normalization as providing "cross-session comparability," but since the discretization is based on percentiles, the normalization step does not change the final bin assignments.
