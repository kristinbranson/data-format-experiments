# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` under `data/`. Sessions are subdirectories within each subject folder whose names start with 4 digits. For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. Data is processed in two passes: first a behavior-only pass to compute global motion normalization/quantile thresholds, then a second pass that loads neural data and assembles the final structure.

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

```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md that it used the data directory structure following the standard convention. Session directories are filtered to those starting with 4 digits to exclude non-session directories like `ground_truth.csv`. The two-pass design was chosen to compute global motion normalization before assembling per-trial outputs.

## 1-b. How are the data split into subjects?

i. Subjects are identified as directories starting with `jm` in the data root, sorted alphabetically. A sorted set of unique subject names is created and mapped to integer indices.

ii.
```python
for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI noted in CONVERSION_NOTES.md that there are 6 subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), consistent with the paper's "6 mice" description. Each `jm*` directory represents one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder whose names start with 4 digits (filtering out non-session files like `ground_truth.csv`), sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
session_dirs = sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
)
```

iii. The AI documented that sessions correspond to daily recordings and that session folder names represent dates (e.g., `2023-04-30_a`). Total sessions discovered: 41.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. The AI creates pseudo-trials as consecutive non-overlapping 2-minute blocks after 10-frame temporal binning. Each 2-minute block contains 360 time bins (120s * 30Hz / 10 frames per bin). This differs from the reference, which uses 60-second non-overlapping segments at native 30 Hz (1800 frames per trial).

ii.
```python
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)  # = 360

def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

iii. The AI justified 2-minute blocks by citing the paper's description that "splits were done on consecutive 2 minute blocks of the recording" for the decoding analysis. This is documented extensively in CONVERSION_NOTES.md Steps 3-5.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. However, if the binned timepoints of a session are not evenly divisible by the block size (360 bins), trailing bins are discarded. In practice, all sessions divide evenly (20-min sessions yield 10 trials, 30-min sessions yield 15 trials).

ii.
```python
def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    ...
```

iii. The AI documented that there are no native trials and no trial curation rules in the paper. Sessions with fewer than 2 trials would raise an error, but this does not occur in practice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p parameters) from `suite2p/plane0/`.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)  # loads ops.npy
```

iii. The AI documented that these are standard Suite2p output files and that the paper states "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)."

## 2-b. How is the `neural` data processed?

i. Two processing steps: (1) Neuropil subtraction using `neucoeff` from `ops.npy` (default 0.7): `Fc = F - neucoeff * Fneu`. (2) Suite2p baseline correction via `dcnv.preprocess` using per-session `ops.npy` parameters. Then, (3) 10-frame temporal bin averaging to reduce from 30 Hz to 3 Hz.

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

neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
```

iii. The AI justified reading parameters from `ops.npy` for each session rather than hardcoding them, noting this is more faithful to "using the default Suite2p parameters" as stated in the paper. The 10-frame averaging follows the paper's description of "averaging in bins of 10 consecutive timestamps" for decoding analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. The AI relies on the fact that the provided data already contains only neurons tracked across all days (pre-filtered by Track2p with `iscell > 0.5`).

ii. N/A (no filtering code)

iii. The AI extensively documented in CONVERSION_NOTES.md Step 4 that `iscell.npy` values in the released data are all above 0.5, consistent with pre-filtered matched output from Track2p. The `data/README.md` states the `suite2p/` folder already contains only successfully tracked neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block within a session. There is no stimulus event; the recording is continuous. Time input stores absolute elapsed time from session start.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
"off_start": 0.0,
"off_end": BLOCK_DURATION_SEC,  # 120.0
```

iii. The AI justified this by noting there is no stimulus-driven trial structure. The paper's decoding analyses use consecutive 2-minute blocks as the temporal unit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal bin averaging, reducing the native 30 Hz to 3 Hz (333.33 ms per bin). This differs from the reference, which keeps the native 30 Hz (33.33 ms per bin).

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,  # 333.33 ms

neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
```

iii. The AI cited the paper's description: "averaging in bins of 10 consecutive timestamps" for the decoding analysis. This is documented in CONVERSION_NOTES.md Step 3 under "Neural data time bin."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from a raw data variable. It is computed from the session's frame rate (`ops['fs']`) and the bin size. For each binned time point, time = bin_index * (bin_size_frames / fs).

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

iii. The AI justified this as a direct computation from known constants: the frame rate and bin size together determine the time of each bin.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as absolute elapsed time from session start: `time = bin_index * (10 / 30)` seconds. This gives the start time of each 10-frame bin. For each trial, the time vector spans from the trial's position within the session (not reset to zero per trial).

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# Then split into blocks:
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The AI documented that elapsed time from session start is mandated by the task specification. The time vector is not reset per trial, so later trials have larger time values.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is constructed from the same binned grid as the neural data (same number of bins, same bin size), then split into the same trial blocks. This ensures perfect alignment by construction.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
# neural_binned.shape[1] ensures same length
```

iii. No explicit alignment is needed because both are derived from the same frame grid and bin size.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` and `tstamps.npy` in the `move_deve` subdirectory of each session.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented that `motion_energy_glob.npy` contains pre-computed global motion energy from behavioral video, and `tstamps.npy` contains timestamps for behavior frames, used to handle missing video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) Align motion energy to imaging frame grid using `tstamps.npy` (mapping behavior timestamps to imaging frame indices, interpolating missing frames). (2) Average in 10-frame bins. (3) Global min-max normalization across all sessions. (4) Discretize into 5 bins using global quintile edges computed from all sessions.

ii.
```python
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
# Global normalization:
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
# Per-session discretization:
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The AI justified global min-max normalization to remove scale differences across sessions before quintile computation. The 10-frame averaging matches the paper's preprocessing. The quintile-based discretization satisfies the task requirement for 5 equal-percentile bins.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, quintile edges at [0.2, 0.4, 0.6, 0.8] quantiles are computed from all sessions' binned motion energy. `np.digitize` is used to assign each value to one of 5 bins (0-4). This differs from the reference, which normalizes by standard deviation and uses `np.percentile` with `np.linspace(0, 100, 6)` edges.

ii.
```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
def motion_to_bins(x, quantile_edges):
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```

iii. The AI documented that global thresholds preserve across-session comparability and that the approach produces exactly balanced quintile bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first aligned to the imaging frame grid using timestamps (mapping behavior frame timestamps to imaging frame indices, with linear interpolation for missing frames). Then both neural and motion data are averaged in the same 10-frame bins, ensuring frame-for-frame alignment. Both are then split into the same 2-minute blocks.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
# After alignment, both are binned identically:
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
# Verification:
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
```

iii. The AI documented that the microscope triggered the camera, enabling simple synchronization. Missing camera frames are handled by timestamp-based alignment with interpolation, using `tstamps.npy`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames (where behavior array is shorter than neural array) are handled by the `align_motion_to_imaging` function, which maps behavior timestamps to imaging frame indices and linearly interpolates over gaps. The code also checks for motion/timestamp length consistency, verifies aligned length matches neural length, and raises errors on critical issues. Trailing frames that don't fill a complete block are discarded during block splitting.

ii.
```python
def align_motion_to_imaging(motion, tstamps, nframes):
    if len(motion) == nframes:
        return motion.astype(np.float32, copy=False), stats
    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    # ... interpolation for missing frames ...
    missing = np.flatnonzero(~good)
    if missing.size:
        aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```

iii. The AI documented 9 sessions with missing behavior frames and verified that all were handled correctly via the timestamp-based alignment approach.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is Suite2p baseline correction (`dcnv.preprocess`), which runs on CPU in the AI's code. The AI's CONVERSION_NOTES.md reports total conversion time of ~61 seconds for all 41 sessions.

ii. N/A

iii. The AI documented per-session timing: ~1.5s for 20-minute sessions, ~3.6s for 30-minute sessions. The two-pass design and session-by-session processing keep memory bounded.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code is already well-vectorized. Bin averaging uses reshape+mean operations. The motion alignment uses `np.add.at` for scatter operations. No significant loop-based inefficiencies are present.

ii. N/A

iii. The AI's code avoids the per-frame `np.insert` loop used in the reference solution for dropped frame handling.

## 6-c. What processing does the code repeat multiple times?

i. The motion energy alignment (`align_motion_to_imaging`) and motion energy loading are performed twice: once in Pass 1 (behavior scan for global normalization) and again in Pass 2 (full conversion). The `ops.npy` file is also loaded twice per session (once for nframes in Pass 1, once in Pass 2).

ii.
```python
# Pass 1:
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
# Pass 2:
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The AI documented the two-pass design as a deliberate tradeoff: it avoids holding all neural data in memory simultaneously by computing global motion statistics first, then processing neural data session-by-session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `align_motion_to_imaging` function returns detailed alignment statistics (`missing_frames`, `duplicate_timestamp_bins`) that are stored in metadata but not used by the decoder. The `motion_aligned` array from Pass 2 is recomputed but only used for plotting. The code also loads `motion_energy_glob.npy` and `tstamps.npy` a second time in Pass 2 even though the binned result is cached.

ii.
```python
# Pass 2 reloads and realigns motion, but only uses cached motion_binned:
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]  # uses cached value
```

iii. The redundant processing is minor in terms of runtime but could be eliminated by caching the aligned motion array from Pass 1 or skipping the realignment when not plotting.
