# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subject/session folders under `data/`, then does a two-pass load. In pass 1 it loads `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy` for every session to build global motion-energy statistics. In pass 2 it reloads each session and loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy` to build the converted dataset.

ii.
```python
for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
    session_dirs = sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )

for session in sessions:
    ops = load_ops(session)
    nframes = int(ops["nframes"])
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)

for idx, session in enumerate(sessions):
    ops = load_ops(session)
    F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
    Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as a paper-driven two-pass pipeline: first scan behavior to compute global motion quantiles cheaply, then do session-by-session neural preprocessing to keep memory bounded.

## 1-b. How are the data split into subjects?

i. Subjects are the top-level directories in `data/` whose names start with `jm`, sorted lexicographically. The converted file stores sorted unique subject IDs and maps each session to a subject index.

ii.
```python
for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...

subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes say each `jm*` directory is one mouse and that all six provided mice should be kept rather than dropped to force agreement with paper summary counts.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject directory whose names begin with four digits, again sorted lexicographically. Each subject-day folder becomes one converted session.

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

iii. The notes describe each date-like subdirectory as one daily recording session and keep sessions separate rather than merging across days or mice.

## 1-d. How are the data split into trials?

i. The AI decided there are no native trials, so it constructs pseudo-trials as consecutive non-overlapping 2-minute blocks after 10-frame temporal averaging. At 30 Hz with 10-frame bins, each trial has `360` time bins.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)

neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The AI explicitly justified this in the notes as matching the paper’s “consecutive 2 minute blocks” and “10 consecutive timestamps” decoder preprocessing.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial behavioral or neural quality filter. The AI only enforces structural constraints: the binned session length must be divisible by block length, and each session must yield at least two trials.

ii.
```python
if neural_binned.shape[1] % BLOCK_BINS != 0:
    raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")

if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The notes frame the data as continuous sessions with no native trial table, so the AI did not add trial-level QC beyond ensuring the converted structure is valid for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, with `ops.npy` used to supply preprocessing parameters such as `neucoeff`, `fs`, and baseline settings.

ii.
```python
ops = load_ops(session)
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
...
fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(
    np.float32, copy=False
)
```

iii. The notes say the paper’s main analyses used baseline-corrected fluorescence rather than `spks.npy`, so the AI chose to reconstruct that representation from `F`, `Fneu`, and Suite2p parameters.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil using `ops["neucoeff"]` (default `0.7`), then runs `suite2p.extraction.dcnv.preprocess` using parameters read from `ops.npy`, and finally averages the resulting traces in 10-frame bins.

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

iii. The notes say this was chosen to mimic the paper’s “baseline corrected fluorescence traces as our dF/F” while avoiding the Track2p GUI helper, which the AI judged less faithful.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron filtering during conversion. It trusts the released `suite2p` exports as already matched across days and already filtered as cells.

ii.
```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
...
converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))
```

iii. In the notes, the AI states the release data already reflect Track2p matching and `iscell > 0.5` curation, so it deliberately avoids a second filtering pass.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the start of each consecutive 2-minute block as the trial alignment event. It stores this explicitly in metadata while also saying the time input remains absolute from session start.

ii.
```python
"metadata": {
    "temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
    "off_start": 0.0,
    "off_end": BLOCK_DURATION_SEC,
}
...
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
```

iii. The justification in the notes is that the recordings are continuous and the paper’s decoder used consecutive 2-minute blocks rather than event-locked stimulus trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so the time bin size is `333.333... ms`. Yes: the AI explicitly rebins both neural and behavior data by averaging every 10 consecutive frames.

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
...
def bin_average_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2, dtype=np.float64).astype(np.float32)

"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
```

iii. The notes justify this directly from the paper’s decoding description: average in bins of 10 timestamps before forming 2-minute blocks.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a dedicated raw time array. It is synthesized from the imaging sampling rate in `ops.npy` plus bin indices.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]

time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. The AI’s notes say the task requires elapsed time from session start and that the continuous recordings have no event table, so frame/bin index plus `fs` is the natural source.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one time value per 10-frame bin using `step = bin_size_frames / fs`, generating a monotonically increasing absolute time vector from session start, then slices that vector into 2-minute trials.

ii.
```python
step = bin_size_frames / fs
return (np.arange(nbins, dtype=np.float32) * step)[None, :]
...
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes explicitly say the input should be “absolute elapsed time from session start,” not time relative to each block.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated at the same binned length as `neural_binned`, then split into the same consecutive 2-minute blocks as `neural` and `output`.

ii.
```python
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
...
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes mention a sanity check that reconstructed the elapsed-time vector from raw frame timing and matched it exactly to the converted `input`.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/tstamps.npy` used to map behavior samples onto the imaging frame grid. The AI does not use `interframe_int.npy`.

ii.
```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

iii. The notes justify this by saying missing video frames should be handled with timestamps and that imaging frames should be treated as the master clock.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI maps motion samples to imaging frames using normalized timestamps, averages duplicate mappings, linearly interpolates missing frame positions, averages the aligned signal in 10-frame bins, applies a global min-max normalization across all included sessions, and only then discretizes.

ii.
```python
frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
...
np.add.at(summed, frame_idx, motion.astype(np.float64))
np.add.at(counts, frame_idx, 1)
...
aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])

motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
...
all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
```

iii. The notes justify the interpolation and 10-frame averaging as paper-consistent temporal handling, and the global normalization as a prelude to cross-session quintile discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, the AI computes global quintile thresholds at the 20th, 40th, 60th, and 80th percentiles of the pooled binned motion signal, then assigns category labels `0..4` with `np.digitize`.

ii.
```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
def motion_to_bins(x: np.ndarray, quantile_edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)

motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The notes say the task requires categorical outputs, so the AI preserved continuous processing until the end and then used global quintiles to produce balanced classes.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data in two stages: first map behavior timestamps onto the imaging-frame grid and interpolate missing imaging-frame positions, then average both streams in 10-frame bins and split them into the same 2-minute blocks.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]
...
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
...
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(
        f"{session.session_id}: neural/motion binned length mismatch "
        f"{neural_binned.shape[1]} vs {len(motion_disc)}"
    )
```

iii. The notes explicitly justify this as “imaging frames are the master clock” and cite missing-camera-frame handling as the main reason to align by `tstamps.npy`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or irregular behavior samples by mapping timestamps to imaging frames, averaging duplicates, and linearly interpolating missing frame positions. It raises errors for impossible cases such as non-increasing timestamps or zero valid mapped samples. Implicitly, any leftover frames not fitting the bin or block structure are dropped by truncation in the binning/splitting helpers.

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
if missing.size:
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])

usable = (len(x) // bin_size) * bin_size
x = x[:usable]
```

iii. The notes present missing-camera-frame interpolation as a deliberate paper/data consistency choice and report raw-data sanity checks on a session with missing frames.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies Suite2p baseline correction as the expensive step and explicitly says full-session `F.npy`/`Fneu.npy` loading plus per-session `dcnv.preprocess` dominate runtime. It treats the first-pass behavior scan as cheap.

ii.
```python
neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
...
summarize_timing(pass1_start, "Pass 1 (behavior scan)")
summarize_timing(pass2_start, "Pass 2 (neural conversion)")
```

iii. In the notes: “Suite2p baseline correction requires loading full `F.npy` and `Fneu.npy` arrays for each session” and “Full conversion may still be moderately expensive because baseline correction is performed per session on CPU.”

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious remaining non-vectorized work is at the session/trial list-construction level rather than within signal alignment: the code loops over sessions twice, does Python list construction for every block, and uses per-session file loads during sample selection. The dropped-frame interpolation itself is already vectorized compared with an insertion loop.

ii.
```python
for session in sessions:
    ...

for idx, session in enumerate(sessions):
    ...

return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
return [x[i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```

iii. The notes emphasize memory-bounded two-pass processing rather than maximal vectorization, so the AI appears to have accepted some Python-level looping as a tradeoff.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats session discovery-related work and motion processing work across two passes. It loads `ops.npy` once in the first pass and again in the second pass; it also loads `motion_energy_glob.npy` and `tstamps.npy` twice, and it reruns motion alignment in the second pass even though pass 1 already aligned and binned motion for each session.

ii.
```python
for session in sessions:
    ops = load_ops(session)
    ...
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

for idx, session in enumerate(sessions):
    ops = load_ops(session)
    ...
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The notes justify the two-pass design as a memory/runtime tradeoff: global motion thresholds first, then neural conversion session by session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are not needed for the final decoder arrays: `motion_aligned` is recomputed in pass 2 even though only the cached binned motion is used downstream, `align_stats` and `session_meta` are stored only as metadata, `plot_processing` support computes extra plotting inputs, and `verify_data_format` runs as validation rather than conversion. In sample mode, `discover_sessions` also loads `ops.npy` just to diversify frame counts.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]
...
session_meta.append(
    {
        "behavior_missing_frames": align_stats["missing_frames"],
        "behavior_duplicate_timestamp_bins": align_stats["duplicate_timestamp_bins"],
        ...
    }
)
...
valid, errors, warnings = verify_data_format(converted)
```

iii. The notes present these as sanity-checking and documentation aids rather than required downstream features, so the AI intentionally kept some extra processing for auditability.
