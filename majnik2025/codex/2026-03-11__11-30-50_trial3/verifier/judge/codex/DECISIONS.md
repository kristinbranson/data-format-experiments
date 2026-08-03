# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all sessions by scanning `data/` for subject directories whose names start with `jm`, then scanning each subject for date-like session subdirectories. It uses a two-pass pipeline. In pass 1 it loads `ops.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and a memmapped `F.npy` shape to collect metadata and compute global motion-energy bin edges. In pass 2 it reloads `ops.npy`, `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy` to build the converted dataset. It does not load `interframe_int.npy`.

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

iii. In `CONVERSION_NOTES.md`, the AI says the provided `suite2p/plane0` exports already contain matched neurons, so it should “read these directly,” keep all six mice and all sessions, and use a two-pass conversion “so it can compute global motion-energy quintiles without ever loading the full neural dataset into memory at once.”

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the top-level directories in `data/` whose names start with `jm`. They are sorted lexicographically, and `subject_idx` is built from those sorted names.

ii.
```python
for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[session.subject] for session in sessions], dtype=np.int64),
```

iii. The notes say “Keep all six mice and all provided sessions” and map “Subject folder name (e.g. `jm038`)” directly to `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each session is one date-like subdirectory under a subject directory. Sessions are sorted within subject, then flattened into one global session list in subject-major order.

ii.
```python
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

iii. The notes describe “daily session folders named as dates” and say sessions should remain separate rather than merged across mice or days.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions with no native trial structure. It first averages data in 10-frame bins, then splits each session into consecutive non-overlapping 2-minute blocks. Each block is treated as one trial with 360 time bins.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)
```

```python
def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

```python
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The notes justify this as matching the paper’s decoding setup: “Construct pseudo-trials as consecutive 2-minute blocks” after 10-frame averaging because the raw data have no native trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control filtering. Instead, the code enforces structural validity at the session level: neural and motion lengths must match after binning, the binned session length must divide evenly into 2-minute blocks, and each session must yield at least two blocks.

ii.
```python
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(
        f"{session.session_id}: neural/motion binned length mismatch "
        f"{neural_binned.shape[1]} vs {len(motion_disc)}"
    )
if neural_binned.shape[1] % BLOCK_BINS != 0:
    raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")
...
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The notes emphasize preserving all released data and only rejecting structurally invalid sessions. No explicit justification for per-trial QC was given because the dataset is continuous rather than trial-based.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and per-session Suite2p parameters in `suite2p/plane0/ops.npy`.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)
```

```python
def compute_suite2p_baseline_corrected(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(
        np.float32, copy=False
    )
```

iii. The notes say the paper used “baseline-corrected fluorescence traces,” so the AI chose to reconstruct that signal from `F`, `Fneu`, and `ops.npy` rather than use `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil using the session’s `neucoeff` from `ops.npy`, applies `suite2p.extraction.dcnv.preprocess(...)` with per-session Suite2p parameters, and then averages the resulting trace in 10-frame bins.

ii.
```python
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
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
```

iii. The notes explicitly justify baseline-corrected fluorescence and 10-frame averaging as the paper-consistent neural representation: use “Suite2p’s own preprocessing logic and the per-session `ops.npy` parameters,” then “10-frame temporal averaging before segmentation.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply a second neuron-level quality-control filter inside `convert_data.py`. It trusts the provided `suite2p/plane0` export as already Track2p-matched and already filtered to accepted cells, then uses all rows in `F.npy`.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
...
converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))
```

iii. The notes say the released data already store neurons “present across all days” and that “No additional ROI filtering beyond trusting provided matched Suite2p export” is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not align neural data to a stimulus or behavioral event. It bins continuous session activity, then defines each trial as starting at the start of a consecutive 2-minute block. The metadata names that block start as the alignment event, while the input still stores absolute elapsed session time.

ii.
```python
"metadata": {
    "time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
    "temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
    "off_start": 0.0,
    "off_end": BLOCK_DURATION_SEC,
```

```python
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
```

iii. The notes say the data are continuous and “the paper’s decoder splits recordings into consecutive 2-minute blocks,” so pseudo-trials should be defined by block boundaries rather than an external event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 imaging frames per bin at 30 Hz, so the final temporal resolution is `333.33 ms` per bin. Yes, temporal rebinning is applied to both neural and behavioral streams.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
```

```python
"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
```

```python
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
```

iii. The notes justify this with the paper’s stated decoder preprocessing: “Use 10-frame temporal averaging before segmentation.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is derived from the imaging clock, specifically the session sampling rate from `ops['fs']` and the implicit binned frame index. It is not taken from behavioral timestamps.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. The notes map “Imaging frame clock derived from `ops['fs']`” to the decoder input and say the input should be “absolute within-session elapsed time.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a constant step size of `10 / fs` seconds, uses `np.arange(nbins)` to create a full-session elapsed-time vector after 10-frame averaging, and then splits that vector into the same 2-minute blocks as the neural data.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

```python
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes explicitly say “Use absolute elapsed time from session start as the sole decoder input,” not relative block time.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated to have the same number of 10-frame bins as the binned neural trace, then split with the same block boundaries into per-trial arrays.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
...
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
...
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes’ stated plan was to derive input from the imaging frame clock and split it into the same 2-minute block structure as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion-energy signal is derived from `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`. The script does not use `interframe_int.npy`.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

iii. The notes say to “align behavior to imaging using `tstamps.npy` and interpolate only missing camera frames,” treating the imaging frames as the master clock.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI maps motion samples onto the imaging frame grid using normalized behavior timestamps, averages duplicate assignments, linearly interpolates missing imaging-frame bins, averages the aligned motion signal in 10-frame bins, normalizes all binned values globally with min-max scaling, and only then discretizes to categories.

ii.
```python
frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
frame_idx = np.clip(frame_idx, 0, nframes - 1)
...
np.add.at(summed, frame_idx, motion.astype(np.float64))
np.add.at(counts, frame_idx, 1)
...
aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```

```python
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
...
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
```

iii. The notes justify this as preserving paper-style synchronization and denoising while handling missing frames from the released data: “Treat imaging frames as the reference clock” and “10-frame averaging” before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global quintile thresholds from all min-max-normalized, 10-frame-averaged motion values across the included sessions. It uses the 20th, 40th, 60th, and 80th percentiles as bin edges and assigns integer labels with `np.digitize`.

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

iii. The notes say the task requires categorical outputs, so motion should be “discretize[d] ... into 5 equal-percentile bins” after the continuous preprocessing steps.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code aligns behavior to the imaging frame grid first, then bins both behavior and neural activity by the same 10-frame bin size, checks that the binned lengths match, and finally splits both into the same 2-minute blocks.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]
...
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
...
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
```

```python
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The notes state that “Imaging frames are treated as master clock,” with behavior aligned to that grid using `tstamps.npy` and interpolation where needed.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. If motion already matches the imaging length, it is used directly. Otherwise the script maps behavior timestamps to imaging-frame indices, averages duplicates, and linearly interpolates missing positions. It raises errors for impossible cases such as non-increasing timestamps, no valid mapped samples, or mismatched binned lengths. It also validates the final output with `verify_data_format`.

ii.
```python
if len(motion) == nframes:
    aligned = motion.astype(np.float32, copy=False)
```

```python
if denom <= 0:
    raise ValueError("Non-increasing behavior timestamps")
...
if known_idx.size == 0:
    raise ValueError("No valid behavior samples after timestamp mapping")
...
aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```

```python
valid, errors, warnings = verify_data_format(converted)
if not valid:
    raise RuntimeError("Converted data failed validation:\n" + "\n".join(errors))
```

iii. The notes explicitly describe missing behavior frames as a release issue and justify handling them by timestamp-based interpolation onto the imaging grid.

## 6-a. What are the most time-consuming steps of the code?

i. The heaviest work is the second pass over sessions, especially `compute_suite2p_baseline_corrected(...)` via `dcnv.preprocess(...)` on full-session neural matrices. The first behavior-only pass is comparatively cheap.

ii.
```python
pass1_start = time.perf_counter()
...
summarize_timing(pass1_start, "Pass 1 (behavior scan)")
```

```python
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
...
summarize_timing(pass2_start, "Pass 2 (neural conversion)")
```

iii. The trajectory says the script uses a two-pass pipeline, and runtime logs show pass 1 is short while pass 2 dominates. The notes also describe pass 2 as the session-by-session neural conversion stage.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The motion-alignment core is already vectorized with `np.add.at` and `np.interp`, so the script avoids the obvious one-by-one dropped-frame insertion loop. The remaining Python-level loops are mostly session loops and the list-based block splitting, which could be represented more directly with reshaped arrays or views.

ii.
```python
np.add.at(summed, frame_idx, motion.astype(np.float64))
np.add.at(counts, frame_idx, 1)
```

```python
def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

iii. There is no explicit justification in the notes for these remaining loops. The main stated performance rationale is the two-pass design for global quantiles without holding all neural data in memory.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several operations because of its two-pass structure. It loads `ops.npy` twice per session, loads `motion_energy_glob.npy` and `tstamps.npy` twice per session, runs `align_motion_to_imaging(...)` twice per session, and reads `F.npy` once in pass 1 for metadata (`nneurons`) and again in pass 2 for the actual conversion.

ii.
```python
for session in sessions:
    ops = load_ops(session)
    ...
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

```python
for idx, session in enumerate(sessions):
    ops = load_ops(session)
    F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
    ...
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The trajectory explicitly justifies the two-pass design as a way to compute global motion-energy quintiles “without ever loading the full neural dataset into memory at once.”

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is in pass 2, where `motion_aligned` is recomputed even though the actual conversion uses the already stored `motion_binned_by_session` result from pass 1. That recomputed aligned trace is only needed for optional plotting. The script also spends time on diagnostic plotting and validation that are useful for auditing but not consumed by downstream decoder analyses.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]
```

```python
if session.session_id in show_processing_ids:
    plot_processing(
        ...
        motion_aligned=motion_aligned,
```

```python
valid, errors, warnings = verify_data_format(converted)
```

iii. The notes and trajectory frame these steps as audit/debug support: processing plots, validation, and detailed session metadata were added to check correctness rather than because the downstream decoder consumes them.
