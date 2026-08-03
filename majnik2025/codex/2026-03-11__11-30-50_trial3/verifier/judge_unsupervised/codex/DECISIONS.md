# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers sessions by walking `data/`, taking subject folders whose names start with `jm` and session folders whose names begin with a date. It then uses a two-pass load. Pass 1 loads session-level metadata plus behavior (`motion_energy_glob.npy`, `tstamps.npy`) for every session to compute global motion normalization and quantile cutoffs. Pass 2 reloads each session and loads neural fluorescence (`F.npy`, `Fneu.npy`) plus behavior again, converts them, and appends session-level trial lists to the output dictionary. Trials are not loaded from disk because the raw data are continuous sessions; they are created later by block splitting.

ii. ```python
def discover_sessions(sample: bool) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
        session_dirs = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )
        for session_dir in session_dirs:
            sessions.append(SessionInfo(...))

for session in sessions:
    ops = load_ops(session)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)

for idx, session in enumerate(sessions):
    ops = load_ops(session)
    F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
    Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the agent says the raw data are continuous session recordings with one `suite2p` recording and one motion-energy stream per day, so it planned a two-pass pipeline: first scan behavior across all sessions, then process neural data session-by-session. The trajectory also shows it relied on `data/README.md` to justify using the provided `suite2p` and `move_deve` folder structure directly.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level folder. Every directory in `data/` whose name starts with `jm` is treated as one mouse. The final dataset stores the sorted unique subject IDs in `subjects` and maps each session to its subject with `subject_idx`.

ii. ```python
for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...
    sessions.append(
        SessionInfo(
            subject=subject_dir.name,
            session_id=f"{subject_dir.name}_{session_dir.name}",
            path=session_dir,
        )
    )

subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.array([subject_to_idx[session.subject] for session in sessions], dtype=np.int64),
```

iii. The agent’s notes say the data package contains 6 subject folders named `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and `data/README.md` describes each subject folder as one mouse. The trajectory shows it explicitly documented that subject folder names are the canonical subject IDs.

## 1-c. How are the data split into sessions?

i. Sessions are split by one folder level below each subject. Any subdirectory whose first four characters are digits is treated as a recording day. The session ID is the concatenation of subject ID and folder name, e.g. `jm038_2023-04-30_a`.

ii. ```python
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

iii. In `CONVERSION_NOTES.md`, the agent states that each subject contains daily session folders named as dates and that this matches the organization described in `data/README.md`. It justified using these folders directly because the dataset has one imaging session per day and no higher-level trial table.

## 1-d. How are the data split into trials?

i. The script creates pseudo-trials from continuous sessions by binning time series into 10-frame bins and then cutting each session into consecutive non-overlapping 2-minute blocks. At 30 Hz and 10-frame averaging, each derived trial contains 360 time bins. Neural, input, and output streams are all split with the same block boundaries.

ii. ```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)

def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    ...
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

def split_into_blocks_1d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = len(x) // block_bins
    ...
    return [x[i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The notes say there are no native trials in the data and cite the paper/methods statement that decoding used “consecutive 2 minute blocks of the recording” after averaging in bins of 10 consecutive timestamps. The agent therefore used those blocks as decoder-format trials.

## 1-e. How are trials filtered based on quality controls?

i. There is effectively no per-trial quality-control filtering beyond structural checks. Trials are kept if the session can be evenly divided into 2-minute blocks after 10-frame binning, if neural and output streams have matching binned lengths, and if the session yields at least two blocks. Missing behavior frames are interpolated before trial construction rather than causing trial rejection.

ii. ```python
if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)
if neural_binned.shape[1] % BLOCK_BINS != 0:
    raise ValueError(...)

if not (len(neural_trials) == len(input_trials) == len(output_trials)):
    raise ValueError(...)
if len(neural_trials) < 2:
    raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The agent’s notes say the dataset has no native trials and that the main behavioral defect is occasional missing camera frames, which `data/README.md` says can be interpolated over. The agent therefore justified retaining all contiguous 2-minute blocks once the continuous streams had been aligned and repaired.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the Suite2p fluorescence and neuropil traces plus session-specific Suite2p processing parameters. Concretely, the script uses `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and `suite2p/plane0/ops.npy`.

ii. ```python
ops = load_ops(session)
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)

def compute_suite2p_baseline_corrected(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(
        np.float32, copy=False
    )
    return dcnv.preprocess(...)
```

iii. In the notes, the agent rejected `spks.npy` as the main neural signal because the methods excerpt says “We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses.” The trajectory also shows it inspected Track2p loader code and the data notebook before choosing `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The script first subtracts neuropil using `Fc = F - neucoeff * Fneu`, with `neucoeff` read from `ops.npy`. It then applies `suite2p.extraction.dcnv.preprocess` using baseline-related parameters from `ops.npy`, producing baseline-corrected fluorescence. After that it averages every 10 frames and splits the result into 2-minute blocks.

ii. ```python
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

iii. The notes explicitly justify this as an attempt to match the paper’s “baseline corrected fluorescence traces” rather than reusing Track2p GUI `F_processing`, which the agent found uses `neucoeff=0.0` by default. The trajectory shows it inspected Suite2p’s preprocessing signature before implementing this path.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The conversion script applies no new neuron filtering. It assumes the provided `suite2p` exports are already curated to include only cells above the Suite2p `iscell` threshold and only cells tracked across all days for each mouse. As a result, every row of `F.npy` is retained.

ii. ```python
F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
...
converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))
```

iii. The notes justify this by citing Track2p reference code that filters by `iscell > 0.5` and then keeps only rows present across all days, together with `data/README.md`, which says the released `suite2p` folders already contain “the successfully tracked neurons” with matched rows across days. The agent therefore decided a second filtering pass would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no natural task event in the raw dataset, so the script uses the start of each derived 2-minute block as the trial-alignment event. Neural data are not shifted relative to behavior; they are simply binned on the imaging clock and then cut into common block boundaries shared with `input` and `output`.

ii. ```python
"temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
"off_start": 0.0,
"off_end": BLOCK_DURATION_SEC,

neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
...
neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
```

iii. The notes say the recordings are continuous spontaneous sessions with no trial table or event list, and that the closest paper-consistent unit for decoder-format trials is the consecutive 2-minute block used in the paper’s decoding procedure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at a 30 Hz imaging rate, so each bin is `10 / 30 = 0.333...` seconds or `333.33 ms`. Yes, temporal rebinning is applied by averaging groups of 10 consecutive samples for both neural and behavioral streams.

ii. ```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0

def bin_average_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2, dtype=np.float64).astype(np.float32)

"time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,
```

iii. The agent’s notes cite the methods text saying that imaging and videography were both recorded at 30 Hz and that decoding used traces “averaging in bins of 10 consecutive timestamps.” That is the direct basis for the 333.33 ms bin size.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is derived from the imaging sampling rate in `ops.npy` and the number of binned imaging samples in the session. It is not read from `tstamps.npy`; instead it is synthesized from evenly spaced imaging bins on the neural clock.

ii. ```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]

time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. In the notes, the agent says the decoder input was mandated to be elapsed time from the beginning of the experiment and that imaging frames should be treated as the master clock. That led it to derive time from `ops['fs']` rather than from the behavior timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script constructs a monotonic absolute time vector in seconds after binning, with one value per 10-frame bin. The spacing is `bin_size_frames / fs`, so with 10-frame bins at 30 Hz the step is `0.333...` seconds. This vector is then split into 2-minute blocks, but each block retains absolute session time rather than resetting to zero.

ii. ```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]

time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes say this was a deliberate choice: use “absolute elapsed time from session start” because the recordings are continuous sessions and the task specifically asks for time from the beginning of the experiment. The trajectory repeats that justification during the mapping-planning step.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction to the neural data because it is created after neural binning using the neural session length, then split with the exact same block boundaries as the neural matrices.

ii. ```python
neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]

neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```

iii. The notes describe imaging frames as the reference clock for the whole conversion. Because both `neural` and `input` are derived on that same binned imaging timeline, the agent viewed their alignment as exact.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from the precomputed behavioral motion-energy stream and its timestamps: `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`. The script does not recompute motion energy from raw video frames.

ii. ```python
motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

iii. The notes say Track2p reference code has no behavior loader and that the released data package already exposes processed motion energy in `move_deve/`. The trajectory shows the agent using `data/README.md` to justify consuming that processed variable directly because raw videography was not part of the released files.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script maps motion-energy samples onto the imaging-frame grid, averages duplicate timestamp bins, linearly interpolates missing bins, averages every 10 frames, then applies one global min-max normalization across all included sessions. The normalized continuous motion signal is what gets discretized afterward.

ii. ```python
motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)

all_motion = np.concatenate(all_motion_binned)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)

motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12)).astype(
    np.float32
)
```

iii. The notes justify the timestamp mapping and interpolation from `data/README.md`, which explicitly says missing motion frames can be treated as missing or interpolated. They justify the 10-frame averaging from the paper’s decoding description. The extra global min-max normalization is justified only as an adaptation to the decoder task requirement that the output be normalized before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The script computes global quintile thresholds from the concatenated normalized motion-energy values from all included sessions, using the 20th, 40th, 60th, and 80th percentiles. It then converts each normalized value to an integer bin `0` through `4` with `np.digitize`.

ii. ```python
quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)

def motion_to_bins(x: np.ndarray, quantile_edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)

motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```

iii. The notes say the paper’s target is continuous motion, so discretization is a task-driven adaptation. The agent chose to “discretize motion energy only at the final step” and to use “global quintiles” so the bins would be comparable across sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by treating the imaging frames as the master clock. Behavior timestamps are mapped onto imaging frame indices, missing frames are interpolated on that imaging grid, the result is binned in the same 10-frame windows as `neural`, and the final categorical output is split into the same 2-minute blocks as `neural`.

ii. ```python
frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
...
aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])

motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)

if neural_binned.shape[1] != len(motion_disc):
    raise ValueError(...)

output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. The notes repeatedly justify this with the methods statement that microscope acquisition triggered the camera and the README note that missing frames can be interpolated from timestamps. The agent explicitly describes imaging as the reference clock for alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main repair mechanism is for missing behavior frames. If the motion array already matches the imaging length, it is used directly. Otherwise the script uses `tstamps.npy` to map behavior samples to imaging-frame indices, averages duplicate assignments, and linearly interpolates missing frame positions. More severe inconsistencies cause hard errors: motion/timestamp length mismatches, non-increasing timestamps, no valid behavior samples, no sessions, stream-length mismatches after binning, and sessions that do not yield at least two trials.

ii. ```python
if len(motion) == nframes:
    return aligned, stats

if len(motion) != len(tstamps):
    raise ValueError(...)
if denom <= 0:
    raise ValueError(...)
...
if missing.size:
    aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
...
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. The notes tie the interpolation decision to `data/README.md`, which explicitly mentions missing camera frames and says they may be treated as missing values or interpolated over. The trajectory shows the agent checking this before implementing interpolation and leaving other errors as fail-fast conditions.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant expensive step is per-session neural preprocessing, especially loading the full `F.npy` and `Fneu.npy` arrays and running Suite2p baseline correction via `dcnv.preprocess` on CPU. A secondary cost is the full-dataset behavior scan used to compute global normalization and quantiles.

ii. ```python
for idx, session in enumerate(sessions):
    F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
    Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
    ...
    neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
    neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)

for session in sessions:
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    ...
```

iii. The notes explicitly identify Suite2p baseline correction as the main cost and say full conversion is “moderately expensive because baseline correction is performed per session on CPU.” They also describe the two-pass behavior scan as a deliberate tradeoff to compute global motion thresholds without holding all neural data in memory.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several Python-level loops in place that could have been reduced: iterating over sessions in both passes, the list-comprehension block splitting for 1D and 2D arrays, and the per-session appends into `converted`. These are structurally simple loops over contiguous blocks and sessions.

ii. ```python
for session in sessions:
    ...

for idx, session in enumerate(sessions):
    ...

return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
return [x[i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]

converted["neural"].append([x.astype(np.float32, copy=False) for x in neural_trials])
converted["input"].append(input_trials)
converted["output"].append(output_trials)
```

iii. The agent did not give an explicit optimization justification for these specific loops. Its notes only say that baseline correction is the main bottleneck and that session-by-session processing was chosen to keep memory bounded, so the lack of vectorization here is an implied implementation choice rather than a stated rationale.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several operations: `load_ops(session)` is called during sample selection, again in the behavior pass, and again in the neural pass; motion files are loaded in both passes; motion alignment is performed in both passes; and the imaging length is read first from `ops["nframes"]` and later again from `F.shape[1]`.

ii. ```python
if sample:
    for session in sessions:
        nframes = int(load_ops(session)["nframes"])

for session in sessions:
    ops = load_ops(session)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)

for idx, session in enumerate(sessions):
    ops = load_ops(session)
    motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
    motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
```

iii. The notes explicitly describe the implementation as a two-pass pipeline and justify the repetition as a memory-saving design: compute global motion thresholds first, then process neural data one session at a time. The repeated motion alignment in pass 2 is not separately justified beyond that overall structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is that pass 2 recomputes `motion_aligned` for every session even though the actual output uses the already cached `motion_binned_by_session`; when `--show-processing` is false, that recomputed aligned trace is never used downstream. The code also builds `session_meta`, timing summaries, and optional plotting products that are useful for audit or visualization but not for decoder training itself.

ii. ```python
motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])
motion_binned = motion_binned_by_session[session.session_id]
...
if session.session_id in show_processing_ids:
    plot_processing(..., motion_aligned=motion_aligned, ...)

session_meta.append({...})
...
summarize_timing(pass1_start, "Pass 1 (behavior scan)")
summarize_timing(pass2_start, "Pass 2 (neural conversion)")
```

iii. The notes justify optional plotting and metadata as documentation and sanity-check support, but they do not justify recomputing `motion_aligned` in pass 2 when plots are disabled. That repeated alignment appears to be an incidental inefficiency rather than an intentional downstream requirement.
