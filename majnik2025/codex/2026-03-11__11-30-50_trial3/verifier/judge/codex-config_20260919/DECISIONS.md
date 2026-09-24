# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every `jm*` subject directory under `data/`, then date-named session directories. In a first pass it loads each session's `ops.npy`, motion energy, and behavior timestamps. In a second pass it loads `F.npy`, `Fneu.npy`, motion energy, and timestamps. Full mode includes all discovered sessions; sample mode selects two sessions.

ii.
```python
for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
    session_dirs = sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )
...
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. The notes say the release has six `jm*` mice and 41 daily sessions, and explicitly choose to retain all provided mice and sessions rather than force agreement with paper summary counts. The two-pass design is justified as a way to compute dataset-wide behavior thresholds without retaining all unbinned neural data in memory.

## 1-b. How are the data split into subjects?

i. A subject is a sorted directory whose name starts with `jm`. The output subject list is the sorted unique set of session subject names, and each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.array(
    [subject_to_idx[session.subject] for session in sessions], dtype=np.int64
),
```

iii. The notes identify each `jm*` directory as a mouse and state that sessions remain separate rather than being merged across mice.

## 1-c. How are the data split into sessions?

i. Each date-named child directory of a subject is one session. Sessions are sorted by subject and path, and each becomes one entry in `neural`, `input`, and `output`.

ii.
```python
session_dirs = sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
)
...
converted["neural"].append([x.astype(np.float32, copy=False) for x in neural_trials])
converted["input"].append(input_trials)
converted["output"].append(output_trials)
```

iii. The agent states that session directories are daily recordings and preserves them as distinct sessions for longitudinal analysis.

## 1-d. How are the data split into trials?

i. The agent creates consecutive, non-overlapping 120-second pseudo-trials after 10-frame averaging. Each trial has 360 bins at 3 Hz. This does not follow the task's explicit request for 60-second trials.

ii.
```python
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)
...
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :] for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :] for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The notes justify 2-minute blocks because the paper used consecutive 2-minute blocks as cross-validation units. The agent prioritized that paper detail over the task's direct instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filtering. The code requires at least two trials per session and requires the binned session length to be exactly divisible by 360; otherwise it raises an error.

ii.
```python
if neural_binned.shape[1] % BLOCK_BINS != 0:
    raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")
...
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The notes report that the source has no native trials and do not identify any principled trial-rejection criterion. They expect all released 20- and 30-minute sessions to divide evenly into the chosen blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is reconstructed from per-session Suite2p `F.npy` and `Fneu.npy`, with processing parameters taken from `ops.npy`.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
ops = load_ops(session)
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
```

iii. The agent cites the paper's use of Suite2p baseline-corrected fluorescence and rejects raw fluorescence or `spks.npy` as less faithful to the stated paper analysis.

## 2-b. How is the `neural` data processed?

i. The code subtracts neuropil fluorescence using the session's `neucoeff` (default 0.7), applies Suite2p `dcnv.preprocess` with session parameters and a maximin default baseline, casts to float32, and averages each neuron over non-overlapping groups of 10 frames.

ii.
```python
fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(
    np.float32, copy=False
)
return dcnv.preprocess(
    fc.copy(), baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=float(ops.get("fs", IMAGING_FS)),
    prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
    batch_size=int(ops.get("batch_size", 100)), device=torch.device("cpu"),
)
...
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
```

iii. The notes explain that this reconstructs the paper's Suite2p-style baseline-corrected fluorescence using recorded per-session settings, followed by the paper's stated 10-timestamp averaging for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed during conversion; every row of the provided `F.npy` is retained.

ii.
```python
converted["neural"].append([x.astype(np.float32, copy=False) for x in neural_trials])
converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))
```

iii. The notes establish that the supplied Suite2p exports already contain cells tracked across all days and that sampled `iscell.npy` values already exceed the 0.5 threshold. Thus another Track2p or `iscell` filtering pass would duplicate upstream curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Continuous neural data starts at session time zero and is cut into consecutive blocks with no temporal shift. Metadata calls the start of each 2-minute block the alignment event and reports offsets 0 to 120 seconds.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
"off_start": 0.0,
"off_end": BLOCK_DURATION_SEC,
...
return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

iii. The agent notes that these are pseudo-trials from a continuous spontaneous recording with no experimental event. It uses block starts as the practical event, but its 2-minute alignment window conflicts with the required 60-second trials and the reference metadata's session-start description.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are averaged per output bin, producing 3-Hz data and a bin size of approximately 333.33 ms. Neural and motion streams are rebinned identically before trial splitting.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
...
"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
```

iii. The notes quote the paper's use of averages over 10 consecutive timestamps to denoise both fluorescence and behavior while preserving alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the binned sample index, the per-session imaging frequency from `ops.npy`, and the fixed 10-frame bin width; it is not read from a raw timestamp array.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```

iii. The notes state that the imaging frame clock is the reference clock and that absolute elapsed time from session start is mandated by the decoder task.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code multiplies zero-based binned-frame indices by `10 / fs`, yielding bin-left-edge times in seconds. It then splits that continuous vector into trial-shaped `(1, 360)` arrays without resetting time at block boundaries.

ii.
```python
time_binned = make_time_input(
    neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES
)[0]
input_trials = [x[None, :].astype(np.float32)
                for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The agent explicitly chooses absolute within-session time, rather than relative trial time, so later blocks retain their position in the recording.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One time value is generated for every 10-frame neural bin and both arrays are sliced with the same block boundaries. The code relies on their common length and construction rather than an explicit per-trial alignment assertion.

ii.
```python
time_binned = make_time_input(neural_binned.shape[1], ...)[0]
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :] for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The planned sanity check was to compare trial input values with frame indices and the 30-Hz sampling rate. The shared index grid makes the input and neural columns correspond directly.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from `move_deve/motion_energy_glob.npy`; `move_deve/tstamps.npy` and the imaging frame count from `ops.npy` are used to align it to imaging.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

iii. The notes identify motion energy as the paper's global video-motion measure and use timestamps because the data README warns that a subset of sessions has missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Raw motion samples are mapped to an imaging-frame grid; duplicate mappings are averaged and empty imaging frames are linearly interpolated. The aligned signal is averaged in 10-frame bins, globally min-max normalized across all included sessions, and digitized using dataset-wide quintile edges.

ii.
```python
np.add.at(summed, frame_idx, motion.astype(np.float64))
np.add.at(counts, frame_idx, 1)
...
aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
...
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
all_motion = np.concatenate(all_motion_binned)
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8])
```

iii. The agent justifies alignment/interpolation and 10-frame averaging from the README and paper. It says global normalization and thresholds preserve across-session comparability, but this conflicts with the task's explicit requirement that five equal-percentile bins be selected per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20/40/60/80-percentile edges are computed once from all normalized binned motion samples across all sessions. `np.digitize` assigns integer labels 0 through 4. Therefore the bins are globally, not per-session, balanced.

ii.
```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
def motion_to_bins(x: np.ndarray, quantile_edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```

iii. The notes explicitly choose global quintiles for cross-session comparability and report exactly 20% per category globally. This is contrary to “five equal-percentile bins, selected per session” and the reference implementation's session-local edges.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If lengths already match, motion is treated as frame-aligned. Otherwise, behavior timestamps are normalized over their observed span, rounded onto `0..nframes-1`, duplicate bins are averaged, and missing bins are interpolated. Neural and motion are then averaged in matching 10-frame bins and split using identical block boundaries; an equality check enforces their binned lengths.

ii.
```python
frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
...
aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
...
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
```

iii. The agent treats imaging as the master clock because acquisition triggered the camera. It considers timestamp-based interpolation the README-supported way to repair missing behavior frames before shared binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion frames are filled by linear interpolation after timestamp mapping; multiple motion samples assigned to one imaging bin are averaged. Invalid timestamp conditions, length mismatches, nondivisible trial lengths, and too few trials cause explicit errors rather than silent repair. No neural missing-value imputation is performed.

ii.
```python
if known_idx.size == 1:
    aligned[:] = aligned[known_idx[0]]
else:
    missing = np.flatnonzero(~good)
    if missing.size:
        aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
...
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
```

iii. The notes report nine sessions with slight behavior shortfalls and choose interpolation rather than dropping sessions. They emphasize validation checks and preservation of the released data.

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p baseline correction over every neuron and frame is the dominant computation. Loading full `F.npy` and `Fneu.npy` is the other substantial I/O and memory cost. The full run performs this work session by session on CPU.

ii.
```python
return dcnv.preprocess(
    fc.copy(), ..., batch_size=int(ops.get("batch_size", 100)),
    device=torch.device("cpu"),
)
```

iii. The notes explicitly identify full-session Suite2p baseline correction as moderately expensive, estimate about two minutes for the complete conversion, and explain that session-wise processing bounds peak memory.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The session loops cannot readily be combined because sessions have different neuron counts and lengths. Trial splitting uses Python list comprehensions that could be replaced by reshape/view operations, though the target format ultimately requires lists. Session metadata and output assembly also use small Python loops. Missing-frame accumulation and interpolation are already vectorized with `np.add.at`, `flatnonzero`, and `np.interp`.

ii.
```python
return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
...
for idx, session in enumerate(sessions):
    ...
```

iii. The notes do not claim a specific loop-vectorization opportunity. Their optimization discussion instead emphasizes streaming one session at a time and memory mapping files used only for shape inspection.

## 6-c. What processing does the code repeat multiple times?

i. `ops.npy`, raw motion, and timestamps are loaded in both passes. Motion-to-imaging alignment is also computed in both passes even though the first pass caches the binned aligned motion used for conversion; the second result is needed only for an optional processing plot. Neural baseline processing itself occurs only once per session.

ii.
```python
# pass 1
ops = load_ops(session)
motion = np.load(...)
tstamps = np.load(...)
motion_aligned, align_stats = align_motion_to_imaging(...)
...
# pass 2
ops = load_ops(session)
motion = np.load(...)
tstamps = np.load(...)
motion_aligned, _ = align_motion_to_imaging(...)
```

iii. The agent describes the two-pass organization as necessary for global thresholds and low neural-memory use, but does not acknowledge that raw behavior loading and alignment are redundantly repeated after `motion_binned_by_session` has already been cached.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Global min-max normalization is mathematically unnecessary before global quantiles because it is monotonic and does not change category assignments. The second-pass aligned continuous motion is unused in conversion after recomputation, except when optional plots are requested. `usable` local variables in block splitters are calculated but do not affect slicing. Optional plotting produces diagnostics that the decoder does not use.

ii.
```python
motion_binned_norm = ((motion_binned - motion_min) /
                      max(motion_max - motion_min, 1e-12)).astype(np.float32)
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
...
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The notes justify normalization as enabling global comparison and plotting, but categorical quantile labels would be identical without it. They justify diagnostic plots as sanity checks; those are optional and not part of the saved decoder data.
