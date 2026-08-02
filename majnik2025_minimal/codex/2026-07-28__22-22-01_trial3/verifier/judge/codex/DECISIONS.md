# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code first enumerates subject directories and session subdirectories, then loads each session as one continuous recording. For every session it reads `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. Trials are not loaded from disk; they are created later by splitting each already-loaded session.

ii. 
```python
def session_paths(data_dir: Path, selected_subjects: list[str] | None) -> list[tuple[str, Path]]:
    available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    ...
    for subject in subjects:
        subject_dir = data_dir / subject
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            out.append((subject, session_dir))
    return out

def load_session(subject: str, session_dir: Path) -> dict:
    plane_dir = session_dir / "suite2p" / "plane0"
    move_dir = session_dir / "move_deve"

    fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
    motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)

def build_dataset(data_dir: Path, selected_subjects: list[str] | None) -> dict:
    loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
```

iii. No `CONVERSION_NOTES.md` was present. The rationale is inferred from the trajectory: the agent said it would inspect the data layout first, then concluded that the dataset was arranged per mouse and day with Suite2p outputs plus motion-energy traces, and later said it would preprocess full-session traces before writing any splits.

## 1-b. How are the data split into subjects?

i. Subjects are every directory directly under `data_dir`, sorted alphabetically. The code also supports an optional explicit `--subjects` list. It does not restrict subjects to names matching `jm*`.

ii. 
```python
def session_paths(data_dir: Path, selected_subjects: list[str] | None) -> list[tuple[str, Path]]:
    available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if selected_subjects is None:
        subjects = available_subjects
    else:
        missing = sorted(set(selected_subjects) - set(available_subjects))
        if missing:
            raise FileNotFoundError(f"Unknown subjects: {missing}")
        subjects = sorted(selected_subjects)
```

iii. There is no explicit written justification beyond the trajectory note that the data are arranged “per mouse and day.” The subject-splitting rule is therefore inferred from the code.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory within a subject directory, sorted alphabetically. Each such directory is treated as one recording session.

ii. 
```python
for subject in subjects:
    subject_dir = data_dir / subject
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        out.append((subject, session_dir))
```

iii. The trajectory says the dataset is organized per mouse and day, which matches this one-subdirectory-per-session interpretation.

## 1-d. How are the data split into trials?

i. Trials are artificial consecutive 2-minute blocks cut from each session after both neural and motion traces have been averaged into non-overlapping 10-frame bins. The code requires the binned session length to divide exactly into 2-minute trials.

ii. 
```python
FRAME_BIN_SIZE = 10
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)

if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(
        f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
        f"{TRIAL_BINS}-bin trials"
    )

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
    output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. The trajectory explicitly says the agent had “settled” on averaging both streams in 10-frame windows and then cutting recordings into “the same consecutive 2-minute blocks described in the paper.” The conversion summary printed by the script repeats those choices.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control filter. The only trial-related structural check is that the binned session length must be divisible by the chosen trial length; otherwise the code raises an error instead of dropping or repairing trials.

ii. 
```python
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(
        f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
        f"{TRIAL_BINS}-bin trials"
    )
```

iii. No separate justification was documented. This answer is inferred from the absence of any trial-quality mask plus the explicit divisibility check in `load_session`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`. `ops.npy` is also loaded to supply preprocessing parameters such as `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, and `fs`.

ii. 
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()

neural = suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops)
```

iii. The trajectory says the agent traced whether the files exposed raw traces or processed traces and then concluded it should use Track2p-exported `F/Fneu` together with the saved Suite2p ops parameters.

## 2-b. How is the `neural` data processed?

i. The code subtracts neuropil using `F - neucoeff * Fneu`, then manually applies a Suite2p-like baseline subtraction using `gaussian_filter`, `minimum_filter1d`, and `maximum_filter1d` with parameters from `ops.npy`. After that, it averages the neural trace in non-overlapping 10-frame bins and stores the result as `neural_binned`.

ii. 
```python
def suite2p_baseline_corrected_fluorescence(
    fluorescence: np.ndarray, neuropil: np.ndarray, ops: dict
) -> np.ndarray:
    fc = fluorescence.astype(np.float32) - float(ops["neucoeff"]) * neuropil.astype(np.float32)
    baseline = ops.get("baseline", "maximin")

    if baseline == "maximin":
        win = int(round(float(ops["win_baseline"]) * float(ops["fs"])))
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = minimum_filter1d(flow, win)
        flow = maximum_filter1d(flow, win)
    ...
    return (fc - flow).astype(np.float32)

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The trajectory states that the agent believed the paper-level rules were to use baseline-corrected `dF/F` and to average both neural and behavioral signals in bins of 10 imaging frames before decoding. The printed conversion summary repeats “baseline-corrected fluorescence” and “non-overlapping averages over 10 consecutive frames.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not drop any neurons after loading. Instead, it checks that every exported ROI already has `iscell[:, 1] > 0.5` and aborts if any row fails. The final metadata also states that no extra `iscell` filtering was applied.

ii. 
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")

"neural_processing": {
    "neuropil_subtraction": "F - neucoeff * Fneu",
    "baseline": "Suite2p default maximin baseline subtraction from ops.npy",
    "extra_iscell_filtering_applied": False,
},
```

iii. The trajectory says the agent thought the paper-level rule was `iscell > 0.5`, but it also spent time checking whether the saved matrices were already restricted to tracked cells and whether extra filtering would be a mistake. The resulting code chose validation rather than post hoc filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are not aligned to an external experimental event. The code treats each trial as the start of a consecutive 2-minute block within a session and slices neural bins accordingly. Metadata describes the alignment event as the start of each block.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))

"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block within a session",
    "off_start": 0.0,
    "off_end": TRIAL_DURATION_SECONDS,
    ...
}
```

iii. The trajectory and the conversion summary both explicitly describe trials as “consecutive 2-minute blocks within each session.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 imaging frames per time bin at 30 Hz, so the time bin size is 1/3 second, or 333.333... ms. Yes: non-overlapping temporal rebinning is applied to both neural and motion signals.

ii. 
```python
FRAME_RATE_HZ = 30.0
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)

"time_bin_size": BIN_SIZE_MS,
```

iii. The trajectory says the agent believed both neural and behavioral signals should be averaged in 10-frame bins before decoding, and the printed summary repeats that exact choice.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from any timestamp file. It is synthesized from the session-length binned index using `np.arange(...)` and the constant bin size.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. The conversion summary says the decoder input is “elapsed time from session start at bin centers (seconds).” No raw timestamp-based justification was documented.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code computes time at the center of each averaged 10-frame bin, in seconds from session start. It then slices that session-wide time vector into per-trial segments using the same start and end indices as the neural data.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
...
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. The trajectory and summary both describe the input as elapsed time from session start, measured at bin centers after 10-frame averaging.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned to neural data by construction: both are defined on the same binned session grid and are sliced into trials with the same `start:end` indices.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. No separate note was written, but the shared slicing logic makes the alignment rule explicit in code.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`. The code also uses `move_deve/interframe_int.npy` to infer where camera frames are missing before alignment to imaging frames.

ii. 
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The trajectory says the agent identified Suite2p outputs plus motion-energy traces, then focused on mapping missing camera frames so motion would stay aligned to imaging.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If motion and imaging lengths differ, the code infers how many frames were missed after each video interval from `interframe_int / median_interval`, then linearly interpolates missing values onto the imaging-frame grid. After alignment it averages motion in non-overlapping 10-frame bins and applies a per-session z-score.

ii. 
```python
def infer_missing_frames(interframe_int: np.ndarray) -> np.ndarray:
    median_interval = float(np.median(interframe_int))
    frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
    return np.clip(frame_jumps - 1, 0, None)

for gap_missing in missing_after:
    next_src = src + 1
    if gap_missing:
        aligned[dst : dst + gap_missing] = np.linspace(
            motion_energy[src],
            motion_energy[next_src],
            int(gap_missing) + 2,
            dtype=np.float32,
        )[1:-1]

motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
motion_z = (
    (motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)
).astype(np.float32)
```

iii. The trajectory says the agent wanted to repair camera dropouts on the imaging-frame grid before binning, and the printed summary describes the final choice as interpolation over inferred dropped camera frames followed by 10-frame averaging and session-z-scoring.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After pooling all sessions’ z-scored, binned motion signals, the code computes global quintile edges at 20%, 40%, 60%, and 80%, then discretizes each session with `np.digitize` into five categories.

ii. 
```python
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
...
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions], axis=0)
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)
...
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
...
"output_values": [[
    "lowest_20pct",
    "20_40pct",
    "40_60pct",
    "60_80pct",
    "highest_20pct",
]],
```

iii. The summary says the decoder output is “session-z-scored motion energy discretized with global quintile edges,” and it prints the resulting global cut points as a sanity check.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first aligned to the imaging-frame grid, then temporally averaged with the same 10-frame binning as neural data, and finally cut into trials with the same trial boundaries as neural data.

ii. 
```python
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy,
    n_imaging_frames=n_frames,
    interframe_int=interframe_int,
)

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. The trajectory says the agent’s goal was to keep motion alignment frame-accurate before binning, and the summary explicitly describes alignment on the imaging-frame grid followed by shared 10-frame denoising.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are inferred from `interframe_int.npy` and filled by interpolation. If the inferred count of missing frames does not match the neural-motion length difference, or if the reconstructed aligned motion length is still wrong, the code raises an error. It also raises an error if the binned session cannot be split exactly into the chosen trial blocks.

ii. 
```python
expected_missing = n_imaging_frames - motion_energy.shape[0]
if missing_total != expected_missing:
    raise ValueError(
        f"Could not reconcile missing motion frames: inferred {missing_total}, "
        f"expected {expected_missing}"
    )
...
if dst != n_imaging_frames:
    raise ValueError(f"Aligned motion has {dst} frames, expected {n_imaging_frames}")
...
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(
        f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
        f"{TRIAL_BINS}-bin trials"
    )
```

iii. The trajectory says the agent’s main data-quality concern was mapping missing behavior frames accurately before further processing. No additional fallback logic was documented.

## 6-a. What are the most time-consuming steps of the code?

i. The heaviest work is full-session neural preprocessing: neuropil subtraction plus the gaussian/minimum/maximum baseline filters over every neuron and timepoint for every session. The code also loads and preprocesses all sessions before any splitting or saving.

ii. 
```python
if baseline == "maximin":
    win = int(round(float(ops["win_baseline"]) * float(ops["fs"])))
    flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
    flow = minimum_filter1d(flow, win)
    flow = maximum_filter1d(flow, win)
...
loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
```

iii. The trajectory explicitly says the full conversion is heavier because the script preprocesses every session before any splits are written.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious remaining Python loops are the missing-frame insertion loop inside `align_motion_to_imaging_frames` and the per-trial slicing loop inside `split_session_into_trials`. Both could be refactored into more array-oriented reshaping/indexing.

ii. 
```python
for gap_missing in missing_after:
    next_src = src + 1
    if gap_missing:
        aligned[dst : dst + gap_missing] = np.linspace(
            motion_energy[src],
            motion_energy[next_src],
            int(gap_missing) + 2,
            dtype=np.float32,
        )[1:-1]

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
    output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. No explicit efficiency discussion was documented by the agent; this answer is inferred directly from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code uses a two-pass structure over sessions: first it loads and preprocesses every session into cached full-session arrays, then it iterates over those sessions again to compute motion classes, split trials, and assemble metadata. It also repeatedly performs `subjects.index(session["subject"])` inside the assembly loop.

ii. 
```python
loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
subjects = sorted({session["subject"] for session in loaded_sessions})

for session in loaded_sessions:
    motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
    neural_trials, input_trials, output_trials = split_session_into_trials(
        neural_binned=session["neural_binned"],
        motion_classes=motion_classes,
    )
    ...
    subject_idx.append(subjects.index(session["subject"]))
```

iii. The trajectory notes that the agent intentionally preprocesses sessions before writing splits. The second pass is needed for global motion quantiles, so this repetition appears intentional.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps intermediate quantities such as `motion_binned`, `missing_motion_frame_indices`, `ops`, and rich `session_info` metadata even though the downstream decoder only consumes `neural`, `input`, `output`, and a small amount of structural metadata. It also loads `iscell.npy` only for a validation check.

ii. 
```python
return {
    "subject": subject,
    "session_id": f"{subject}/{session_dir.name}",
    ...
    "neural_binned": neural_binned,
    "motion_binned": motion_binned,
    "motion_z": motion_z,
    "ops": {
        "fs": float(ops["fs"]),
        "neucoeff": float(ops["neucoeff"]),
        "baseline": ops.get("baseline", "maximin"),
        "sig_baseline": float(ops["sig_baseline"]),
        "win_baseline": float(ops["win_baseline"]),
        "prctile_baseline": float(ops["prctile_baseline"]),
    },
    **motion_info,
}
...
"session_info": session_info,
```

iii. No explicit justification was written. The extra retained information seems aimed at validation and documentation rather than the decoder itself.
