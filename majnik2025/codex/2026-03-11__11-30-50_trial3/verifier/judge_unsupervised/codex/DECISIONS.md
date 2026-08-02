# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers all data by walking `data/`, taking each top-level `jm*` directory as a subject and each dated subdirectory under it as a session. It does not load pre-defined trials from disk because none exist; instead it loads continuous session-level files and later constructs pseudo-trials from them. It runs in two passes: pass 1 loads per-session metadata plus motion files to compute global motion normalization/quantile thresholds, and pass 2 reloads each session’s neural and motion arrays to build the final converted dataset.

ii.
```python
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
for session in sessions:
    ops = load_ops(session)
    nframes = int(ops["nframes"])
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the agent says the release is organized as subject folders containing daily session folders, with continuous `suite2p` and `move_deve` streams per day. It justified the two-pass structure in the trajectory as necessary to compute global motion-energy quintiles without holding the entire neural dataset in memory at once.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level folder name, e.g. `jm031`, `jm032`, etc. The output `subjects` field is the sorted unique list of those names, and `subject_idx` maps each session to its subject.

ii.
```python
subject=subject_dir.name
```

```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.array([subject_to_idx[session.subject] for session in sessions], dtype=np.int64),
```

iii. The notes explicitly list 6 subject folders and state that the dataset uses one folder per mouse, so the agent treated folder names as authoritative subject IDs.

## 1-c. How are the data split into sessions?

i. Each dated directory under a subject is one session. Sessions remain separate in the output; there is one top-level session entry in `neural`, `input`, and `output` for each dated folder.

ii.
```python
session_dirs = sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
)
```

```python
SessionInfo(
    subject=subject_dir.name,
    session_id=f"{subject_dir.name}_{session_dir.name}",
    path=session_dir,
)
```

```python
converted["neural"].append([x.astype(np.float32, copy=False) for x in neural_trials])
converted["input"].append(input_trials)
converted["output"].append(output_trials)
```

iii. The agent’s notes say the native data are longitudinal daily recordings, so sessions should stay separate rather than being concatenated across days or subjects.

## 1-d. How are the data split into trials?

i. The raw dataset has no native trial table. The agent creates pseudo-trials by averaging each continuous session into 10-frame bins and then splitting the binned time series into consecutive non-overlapping 2-minute blocks. At 30 Hz with 10-frame bins, each trial has 360 time bins.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)
```

```python
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. In the notes, the agent says the paper’s decoder used consecutive 2-minute blocks of continuous recordings, so this was the closest trial structure that preserved the reference temporal handling.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial rejection. Trials are only filtered implicitly by structural checks: neural and motion must have equal binned lengths, session length must be divisible by the 2-minute block size, neural/input/output trial counts must match, and a session must yield at least two trials.

ii.
```python
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(
        f"{session.session_id}: neural/motion binned length mismatch "
        f"{neural_binned.shape[1]} vs {len(motion_disc)}"
    )
if neural_binned.shape[1] % BLOCK_BINS != 0:
    raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")
```

```python
if not (len(neural_trials) == len(input_trials) == len(output_trials)):
    raise ValueError(f"{session.session_id}: trial count mismatch after block splitting")
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The notes describe the recordings as continuous sessions with no native trial QC rules from the paper/code, so the agent limited trial QC to consistency checks required by the decoder format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and the preprocessing parameters stored in `suite2p/plane0/ops.npy`.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
```

iii. The notes say the paper reported using baseline-corrected fluorescence traces for downstream analyses, so the agent chose to reconstruct that from `F`, `Fneu`, and Suite2p ops parameters instead of using raw `F.npy` or `spks.npy` directly.

## 2-b. How is the `neural` data processed?

i. The agent subtracts neuropil using `neucoeff` from `ops`, then runs `suite2p.extraction.dcnv.preprocess(...)` with the baseline parameters from `ops` to obtain a baseline-corrected trace. After that it averages the trace in non-overlapping 10-frame bins and splits it into 2-minute pseudo-trials.

ii.
```python
def compute_suite2p_baseline_corrected(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(
        np.float32, copy=False
    )
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

```python
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
```

iii. The notes justify this as the paper-consistent reconstruction of “baseline corrected fluorescence traces as our dF/F,” followed by the paper’s stated 10-timestamp averaging and 2-minute decoding blocks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies no additional neuron filter during conversion. It trusts the provided `suite2p` exports as already containing cells matched across all days and already filtered upstream by the original Track2p/Suite2p pipeline.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
...
converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))
```

iii. In `CONVERSION_NOTES.md`, the agent cites `data/README.md` saying these `suite2p` directories already contain only tracked neurons present across all days, and it says sampled `iscell.npy` values were already above the default 0.5 threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no native external event in the raw data. The agent defines the alignment event as the start of each consecutive 2-minute block and stores each trial relative to that block boundary, while the input time series itself remains absolute elapsed time from session start.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
"off_start": 0.0,
"off_end": BLOCK_DURATION_SEC,
```

```python
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
```

iii. The notes say the recordings are continuous rather than event-locked, so the agent used the paper’s decoder block boundaries as the practical alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are sampled at 10 imaging frames per bin. With `fs=30 Hz`, that yields `10 / 30 = 0.333... s`, i.e. `333.33 ms` per bin. Yes, temporal rebinning is applied to both neural and motion signals.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
...
"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
```

```python
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
```

iii. The notes cite the paper’s “averaging in bins of 10 consecutive timestamps” and 30 Hz acquisition for both imaging and behavior.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not loaded from a dedicated raw variable. The agent derives it from the imaging frame clock using `ops['fs']` and the number of binned time points in the neural data.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. The agent’s notes say the task specifically required elapsed time as decoder input, and the imaging frame clock is the cleanest source because neural and behavior are synchronized to that grid.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The input is computed as an absolute elapsed-time vector in seconds: `0, step, 2*step, ...`, where `step = 10/fs`. The agent then splits that vector into the same 2-minute blocks used for neural and output data, adding a singleton input dimension.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

```python
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes explicitly state that the input should be absolute time from session start, not time re-zeroed within each block.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated from the same binned neural length and is split using the same block boundaries. Each trial therefore has the same number of time bins as the neural trial extracted at the same positions.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
...
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

```python
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
```

iii. The notes say imaging frames are treated as the master clock, so deriving time from the neural bin count guarantees temporal alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy` as the continuous behavior measure, with `move_deve/tstamps.npy` used to align that measure to the imaging frame grid.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

iii. The notes identify `motion_energy_glob.npy` as the released processed motion-energy stream and `tstamps.npy` as the file needed to handle missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent aligns motion samples to imaging frames, fills missing frame positions by linear interpolation when needed, averages the aligned signal in 10-frame bins, normalizes it using a global min-max computed across all included sessions, and only then discretizes it.

ii.
```python
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
...
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
```

```python
motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12)).astype(
    np.float32
)
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The notes say this preserves the paper’s temporal handling first, then adapts the target to the decoder task’s requirement for normalized and discretized motion categories.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, the agent computes global quintile thresholds across all included binned motion values using the 20th, 40th, 60th, and 80th percentiles. It then uses `np.digitize` to assign each time bin to category `0` through `4`.

ii.
```python
all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```

```python
def motion_to_bins(x: np.ndarray, quantile_edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```

iii. The notes say the task required five equal-percentile bins and that global thresholds preserve comparability across sessions, so the agent deferred discretization until after all paper-like preprocessing.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent uses imaging as the reference clock. If the motion vector already has one sample per imaging frame, it is used directly; otherwise the agent maps behavior timestamps onto frame indices spanning `0..nframes-1`, averages duplicate mappings, and linearly interpolates missing frame positions. The aligned framewise motion is then binned by the same 10-frame rule used for neural data and split with the same 2-minute block boundaries.

ii.
```python
if len(motion) == nframes:
    aligned = motion.astype(np.float32, copy=False)
    ...
if len(motion) != len(tstamps):
    raise ValueError(f"motion/tstamps length mismatch: {len(motion)} vs {len(tstamps)}")
...
frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
```

```python
np.add.at(summed, frame_idx, motion.astype(np.float64))
np.add.at(counts, frame_idx, 1)
...
aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```

iii. The notes cite the dataset README saying missing camera frames should be treated as missing values or interpolated over. The agent chose imaging-as-master with interpolation, and in the trajectory it described this as the paper-consistent synchronization strategy.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main minor-data-error handling is for missing behavior frames. The code accepts exact frame matching when available, otherwise reconstructs a frame-aligned motion series from timestamps, averages duplicate timestamp bins, and linearly interpolates missing frame locations. It also guards against malformed data by raising errors for mismatched motion/timestamp lengths, non-increasing timestamps, no valid mapped samples, or irreconcilable binned-length mismatches.

ii.
```python
if len(motion) != len(tstamps):
    raise ValueError(f"motion/tstamps length mismatch: {len(motion)} vs {len(tstamps)}")

denom = tstamps[-1] - tstamps[0]
if denom <= 0:
    raise ValueError("Non-increasing behavior timestamps")
```

```python
known_idx = np.flatnonzero(good)
if known_idx.size == 0:
    raise ValueError("No valid behavior samples after timestamp mapping")
if known_idx.size == 1:
    aligned[:] = aligned[known_idx[0]]
else:
    missing = np.flatnonzero(~good)
    if missing.size:
        aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```

iii. The notes explicitly mention the dataset README’s instruction that missing video frames can be treated as missing or interpolated over; the agent chose interpolation and logged per-session missing-frame statistics in metadata.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work is the session-by-session neural preprocessing in pass 2: loading large `F.npy`/`Fneu.npy` matrices, running `dcnv.preprocess(...)`, binning neural traces, and then storing split trials. Pass 1 is cheaper because it only scans motion/ops metadata.

ii.
```python
pass1_start = time.perf_counter()
...
summarize_timing(pass1_start, "Pass 1 (behavior scan)")
...
pass2_start = time.perf_counter()
...
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
...
summarize_timing(pass2_start, "Pass 2 (neural conversion)")
```

iii. The trajectory says the script was designed as a two-pass conversion specifically to avoid loading the full neural dataset into memory. That implies the neural conversion pass dominates runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The top-level `for session in sessions` loops are purely sequential and could be parallelized across sessions. The Python-level trial construction also repeatedly slices lists of blocks instead of keeping batched arrays as long as possible. The pass-1 metadata scan and pass-2 per-session alignment work also repeat similar per-session operations.

ii.
```python
for session in sessions:
    ops = load_ops(session)
    ...
```

```python
for idx, session in enumerate(sessions):
    ...
    neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
    input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
    output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The agent’s notes focus on correctness rather than speed. Nothing in the notes suggests it attempted aggressive vectorization beyond NumPy reshaping for the 10-frame averages.

## 6-c. What processing does the code repeat multiple times?

i. The script repeats session I/O and some motion processing across its two passes. It loads `ops.npy` in both passes, loads motion files in both passes, and calls the motion-alignment routine twice per session: once during the behavior scan and again during the full conversion. It also recomputes session-level metadata that partly overlap with work already done earlier.

ii.
```python
for session in sessions:
    ops = load_ops(session)
    ...
    motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

```python
for idx, session in enumerate(sessions):
    ops = load_ops(session)
    ...
    motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. In the trajectory, the agent explicitly says it is doing a two-pass conversion to compute global motion-energy quintiles first. That explains, but does not eliminate, the repeated loading/alignment work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some information that is only diagnostic, not used by downstream decoding: per-session alignment stats stored in metadata, optional processing plots, repeated full-resolution motion alignment in pass 2 even though pass 1 already cached the binned motion needed for thresholds, and motion arrays loaded in pass 2 for plotting despite only the already-binned/discretized output being retained. The sample-mode session-selection logic based on unique `nframes` is also just for convenience.

ii.
```python
session_meta.append(
    {
        "behavior_missing_frames": align_stats["missing_frames"],
        "behavior_duplicate_timestamp_bins": align_stats["duplicate_timestamp_bins"],
        ...
    }
)
```

```python
show_processing_ids = {session.session_id for session in sessions[:2]} if args.show_processing else set()
...
if session.session_id in show_processing_ids:
    plot_processing(...)
```

```python
motion_binned_by_session[session.session_id] = motion_binned
...
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The notes frame these as sanity-check and audit features. They are useful for validation, but they are not required for the final decoder inputs/outputs themselves.
