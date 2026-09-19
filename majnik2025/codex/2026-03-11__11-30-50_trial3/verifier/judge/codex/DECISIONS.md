# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers data by scanning the fixed `data/` directory for subject folders whose names start with `jm`, then scans each subject for date-like session folders. It uses a two-pass pipeline: pass 1 loads per-session behavior-side files (`ops.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and `F.npy` shape via `mmap`) to compute global motion thresholds and metadata; pass 2 reloads each session and loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy` to build the final session/trial arrays.

ii.
```python
DATA_ROOT = Path("data")

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

for session in sessions:
    ops = load_ops(session)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    ...

for idx, session in enumerate(sessions):
    ops = load_ops(session)
    F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
    Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, the AI justifies a two-pass design so it can compute motion thresholds before neural conversion while keeping memory bounded to one session at a time.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the top-level directories under `data/` whose names start with `jm`. They are sorted lexicographically, and the final `subjects` list is the sorted unique set of subject IDs seen in the discovered sessions.

ii.
```python
for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...

subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI states that there are 6 subject folders named `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and treats each folder as one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories inside each subject directory whose names begin with 4 digits, i.e. date-like names such as `2023-10-18_a`. Each session becomes one entry in the `sessions` list and later one entry in the converted top-level `neural`/`input`/`output` lists.

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

iii. In `CONVERSION_NOTES.md` Step 2, the AI describes each daily recording folder as one session and preserves that organization directly.

## 1-d. How are the data split into trials?

i. The AI does not use 60-second trials. It defines pseudo-trials as consecutive non-overlapping 2-minute blocks after 10-frame binning. At 30 Hz with 10-frame bins, each block is `120 * 30 / 10 = 360` time bins.

ii.
```python
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)

def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The AI explicitly justifies this in `CONVERSION_NOTES.md` Step 5: “Construct pseudo-trials as consecutive 2-minute blocks” because the paper’s decoder used consecutive 2-minute blocks, even though the task instructions asked for 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-level quality-control filtering. Instead it enforces structural checks: neural and motion must have matching binned lengths, the number of binned timepoints must be exactly divisible by the 2-minute block size, and each session must yield at least 2 trials.

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

iii. There is no explicit QC rationale for trials in the notes beyond format validity. The AI’s notes frame these as sanity checks needed by the decoder format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, using `ops.npy` to obtain preprocessing parameters.

ii.
```python
ops = load_ops(session)
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
...
def compute_suite2p_baseline_corrected(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it chose “Suite2p baseline-corrected fluorescence reconstructed from `F.npy`, `Fneu.npy`, and `ops.npy`” because the paper said analyses used baseline-corrected fluorescence rather than `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil using `ops['neucoeff']` (defaulting to `0.7` if absent), then runs `suite2p.extraction.dcnv.preprocess` with parameters pulled from `ops.npy` and forces CPU execution. After that, it averages the processed traces in non-overlapping 10-frame bins.

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

neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI argues that per-session `ops.npy` parameters are the most paper-faithful way to reconstruct Suite2p-style baseline-corrected fluorescence and explicitly rejects using the Track2p GUI helper or raw `spks.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron filtering. It trusts the provided `suite2p` exports as already matched and pre-filtered across days, and uses every row in `F.npy` / `Fneu.npy`.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
...
converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))
```

iii. `CONVERSION_NOTES.md` Steps 2, 4, and 10 state that the release already contains neurons “present across all days” and already reflects the Track2p / Suite2p curation, so the AI intentionally avoided reapplying `iscell` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of its own 2-minute block, not to session start. Its metadata says the temporal alignment event is “start of each consecutive 2-minute block,” with `off_start = 0.0` and `off_end = 120.0`.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
    "off_start": 0.0,
    "off_end": BLOCK_DURATION_SEC,
    ...
}
...
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this through the paper’s use of consecutive 2-minute decoding blocks rather than any external behavioral or stimulus event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavior traces into non-overlapping bins of 10 raw frames at 30 Hz, giving `1000 * 10 / 30 = 333.33 ms` per bin.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0

motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
...
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
...
"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says this matches the paper’s “10 consecutive timestamps” denoising rule and keeps neural and behavior streams synchronized.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not read time from any raw timestamp file. It derives time from the session frame rate in `ops.npy`, the chosen 10-frame bin size, and the bin index within the session.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]

time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI states that absolute elapsed time from session start is the required decoder input and can be computed from the constant imaging frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a uniformly spaced time vector with step size `bin_size_frames / fs` after temporal binning, then splits that vector into the same 2-minute blocks used for neural and output data.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]

time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes do not add much beyond Step 5’s statement that the task requires absolute elapsed time from session start. There is no more elaborate justification.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time with neural activity by constructing the time vector at the same binned resolution as the neural traces and then applying the same block-splitting indices to both arrays.

ii.
```python
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
...
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The AI’s implicit rationale is that both arrays live on the same post-binning frame grid. The notes do not provide a separate explicit justification beyond the shared block definition.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion output from `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`. It does not use `interframe_int.npy`.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says it treats imaging frames as the master clock and uses timestamps to map behavior samples to that grid when camera frames are missing.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI remaps behavior timestamps to imaging-frame indices, averages duplicate assignments, linearly interpolates missing imaging-frame positions, averages the aligned trace into 10-frame bins, then globally min-max normalizes the binned motion across all included sessions and discretizes it with global quintile thresholds.

ii.
```python
frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
...
np.add.at(summed, frame_idx, motion.astype(np.float64))
np.add.at(counts, frame_idx, 1)
...
if missing.size:
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
...
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
...
all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12)).astype(
    np.float32
)
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly justifies timestamp-based interpolation, 10-frame averaging, and global normalization / global quintiles as a way to preserve cross-session comparability.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses 5 global categories derived from the pooled binned motion values of all included sessions. It first min-max normalizes pooled motion to `[0, 1]`, computes the 20th/40th/60th/80th percentiles once globally, and then uses `np.digitize` to assign each binned timepoint to classes `0` through `4`.

ii.
```python
all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
def motion_to_bins(x: np.ndarray, quantile_edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 states the rationale directly: “Global thresholds preserve across-session comparability.”

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data by projecting each behavior timestamp onto the imaging frame grid for that session, averaging any duplicated frame assignments, interpolating missing imaging-frame positions, and then checking that the binned behavior and neural traces have the same length before trial/block splitting.

ii.
```python
def align_motion_to_imaging(
    motion: np.ndarray, tstamps: np.ndarray, nframes: int
) -> tuple[np.ndarray, dict]:
    ...
    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    ...
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
    ...
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
    ...

if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(
        f"{session.session_id}: neural/motion binned length mismatch "
        f"{neural_binned.shape[1]} vs {len(motion_disc)}"
    )
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI justifies this as using imaging as the reference clock because the behavior stream occasionally has missing camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing behavior samples by timestamp-based interpolation onto the imaging grid. It raises hard errors for malformed behavior timestamps, mismatched motion/timestamp lengths, neural-motion binned length mismatches, and sessions that do not divide evenly into the chosen 2-minute trial blocks. It does not implement a “discard the remainder” path.

ii.
```python
if len(motion) != len(tstamps):
    raise ValueError(f"motion/tstamps length mismatch: {len(motion)} vs {len(tstamps)}")
...
if denom <= 0:
    raise ValueError("Non-increasing behavior timestamps")
...
if known_idx.size == 0:
    raise ValueError("No valid behavior samples after timestamp mapping")
...
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
if neural_binned.shape[1] % BLOCK_BINS != 0:
    raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")
```

iii. The AI’s notes justify interpolation of missing camera frames and otherwise prefer fail-fast checks. There is no justification for not keeping partial trailing data beyond the decision to use exact 2-minute blocks.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify Suite2p baseline correction as the expensive step. The code structure also shows a full pass over all sessions to align/bin behavior and a second pass to load and preprocess all neural data, so the dominant cost is per-session `dcnv.preprocess` over full `F` and `Fneu` matrices.

ii.
```python
def compute_suite2p_baseline_corrected(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    ...
    return dcnv.preprocess(
        fc.copy(),
        ...
        device=torch.device("cpu"),
    ).astype(np.float32, copy=False)

for session in sessions:
    ...

for idx, session in enumerate(sessions):
    ...
    neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
```

iii. `CONVERSION_NOTES.md` Step 6 says baseline correction is the main cost and that the full conversion remains “moderately expensive” because it is run session-by-session on CPU.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the motion-alignment step with `np.add.at` and `np.interp`, so the remaining Python-level loops are mostly bookkeeping loops over sessions and list-based trial splitting. The most obvious remaining loop-like work is creating Python lists of trial slices in `split_into_blocks_1d/2d` and constructing `input_trials` / `output_trials`.

ii.
```python
def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    ...
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

def split_into_blocks_1d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    ...
    return [x[i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. There is no explicit justification in the notes for these remaining loops. The AI’s efficiency discussion instead emphasizes the two-pass design and avoiding full-dataset neural loading.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats several operations by design: it loads `ops.npy` in both passes; it loads `motion_energy_glob.npy` and `tstamps.npy` in both passes; it runs `align_motion_to_imaging` twice per session; and it touches `F.npy` in pass 1 for metadata and again in pass 2 for the actual neural conversion.

ii.
```python
for session in sessions:
    ops = load_ops(session)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
    ...
    "nneurons": int(np.load(session.path / "suite2p" / "plane0" / "F.npy", mmap_mode="r").shape[0]),

for idx, session in enumerate(sessions):
    ops = load_ops(session)
    F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
    Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly presents the two-pass design as an intentional tradeoff: repeated lightweight behavior processing is accepted so the script can compute global motion thresholds before session-by-session neural conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is in pass 2: the AI recomputes `motion_aligned` for every session even though the saved dataset uses the cached `motion_binned_by_session` result, and `motion_aligned` is only used for optional plotting. It also computes and stores detailed `session_info` metadata that the decoder does not consume.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]
...
if session.session_id in show_processing_ids:
    plot_processing(
        ...
        motion_aligned=motion_aligned,
        ...
    )

"metadata": {
    ...
    "session_info": session_meta,
}
```

iii. The notes do not justify this as necessary for the decoder. The likely rationale is debugging and auditability, since the same notes emphasize verification plots and detailed session metadata.
