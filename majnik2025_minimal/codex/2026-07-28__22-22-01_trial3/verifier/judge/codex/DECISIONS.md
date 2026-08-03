# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the data root, treats every subject-level directory as a mouse, treats every subdirectory inside each subject as a session, and then loads each session's `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. Trials are not loaded directly; they are created later by splitting each fully loaded session after preprocessing.

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
```

iii. There is no `CONVERSION_NOTES.md`, so the justification has to be reconstructed from the trajectory. In the trajectory the agent says the dataset is arranged "per mouse and day with Suite2p outputs plus motion-energy traces" and later describes each recording day as one matched-cell session, which explains why it loads whole sessions first and only then creates trials.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the sorted directory names under the data root. If `--subjects` is supplied, only those named directories are used.

ii. 
```python
available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
if selected_subjects is None:
    subjects = available_subjects
else:
    missing = sorted(set(selected_subjects) - set(available_subjects))
    if missing:
        raise FileNotFoundError(f"Unknown subjects: {missing}")
    subjects = sorted(selected_subjects)
```

iii. The trajectory shows that the agent read the data README and concluded the dataset is organized "per mouse and day," so it used subject directories as mouse identities.

## 1-c. How are the data split into sessions?

i. Sessions are defined as the sorted subdirectories within each subject directory. Each such directory is treated as one imaging day / one session.

ii. 
```python
for subject in subjects:
    subject_dir = data_dir / subject
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        out.append((subject, session_dir))
```

iii. The agent explicitly described the layout as one mouse folder containing day-level recording folders, so the session split follows the on-disk organization directly.

## 1-d. How are the data split into trials?

i. The AI does not use the reference 60-second raw-frame segmentation. Instead, it first preprocesses whole sessions, averages both neural and motion traces into non-overlapping 10-frame bins, and then splits the binned sessions into consecutive 2-minute blocks. With 30 Hz raw data and 10-frame bins, each trial is 360 bins long.

ii. 
```python
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)

def split_session_into_trials(
    neural_binned: np.ndarray,
    motion_classes: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ...
    n_trials = neural_binned.shape[1] // TRIAL_BINS

    for trial_idx in range(n_trials):
        start = trial_idx * TRIAL_BINS
        end = start + TRIAL_BINS
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
        output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. The trajectory states that the agent believed the paper required "both neural plus behavior averaged in bins of 10 imaging frames" and "consecutive 2-minute blocks," then says it had "settled the main structural choices" and would "cut the recording into the same consecutive 2-minute blocks described in the paper."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering step. Instead, the code requires the binned session length to divide evenly into 2-minute trials and raises an error if it does not. It therefore enforces a structural validity check rather than filtering low-quality trials.

ii. 
```python
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(
        f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
        f"{TRIAL_BINS}-bin trials"
    )
```

iii. The agent's trajectory does not mention any trial QC policy. The implemented behavior suggests a fail-fast choice: keep every trial if the session structure matches the assumed 2-minute grid, otherwise reject the session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, using parameters read from `suite2p/plane0/ops.npy`. `iscell.npy` is loaded only as a consistency check.

ii. 
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
neural = suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops)
```

iii. In the trajectory the agent says it was tracing whether the files expose processed traces or only raw `F.npy`, and later says it had concluded the data should use "Suite2p baseline-corrected fluorescence from Track2p-exported F/Fneu using the saved Suite2p ops parameters."

## 2-b. How is the `neural` data processed?

i. The AI manually reimplements Suite2p-like baseline correction rather than calling `suite2p.dcnv.preprocess` as in the reference. It computes `F - neucoeff * Fneu`, then subtracts a baseline estimated from `ops.npy` settings. For the default `maximin` case it uses a Gaussian filter followed by minimum and maximum filters. After that, elsewhere in the pipeline, the corrected trace is averaged in 10-frame windows.

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
```

iii. The trajectory shows the agent concluded the packaged files only expose raw `F.npy`, that the GUI's "`dF/F0` helper applies baseline subtraction rather than a ratio," and that `ops.npy` carried the default parameters (`neucoeff=0.7`, `baseline=maximin`, `win_baseline=60 s`). That reasoning led it to reproduce baseline subtraction from `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out. It assumes Track2p has already exported the matched-cell set and merely checks that all loaded ROIs have `iscell[:, 1] > 0.5`; if any do not, it raises an error.

ii. 
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The trajectory states that the agent thought "the shipped session matrices are already Track2p-matched cell sets" and that extra `iscell` filtering could be a mistake. The check is therefore treated as validation, not curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of a consecutive 2-minute block within a session, not to overall session start. It records the alignment event in metadata with `off_start = 0.0` and `off_end = 120.0`.

ii. 
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block within a session",
    "off_start": 0.0,
    "off_end": TRIAL_DURATION_SECONDS,
    ...
}
```

iii. The trajectory repeatedly describes trials as "consecutive 2-minute blocks" and the conversion summary printed by the script says exactly that.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame bins at 30 Hz, so the final time step is 1/3 s = 333.33 ms. The AI applies non-overlapping mean rebinning to both neural and motion data.

ii. 
```python
FRAME_RATE_HZ = 30.0
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0

def mean_bin_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    nbins = x.shape[1] // bin_size
    return x[:, : nbins * bin_size].reshape(x.shape[0], nbins, bin_size).mean(axis=2)
```

iii. The agent said in the trajectory that the paper-level rules required both streams to be "averaged in bins of 10 imaging frames before decoding," and its printed summary describes this as "Denoising: non-overlapping averages over 10 consecutive frames."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is not read from a raw file. It is synthesized from the index of the rebinned session timeline: the code constructs evenly spaced bin-center times from the beginning of the session.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. The conversion summary printed by the AI script describes the decoder input as "elapsed time from session start at bin centers (seconds)." No separate timestamp file is used for this input.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code computes times on the rebinned grid, places them at bin centers by adding `0.5`, multiplies by the 10-frame bin duration, and then slices the resulting session-long vector into trial segments.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
...
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. The likely rationale is alignment convenience: once neural and motion are both rebinned, the time input is generated on the same bin grid.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time series is aligned by construction to the rebinned neural data. The code creates one elapsed-time vector for the full rebinned session and uses the same `start:end` trial slices for both `neural_binned` and `elapsed_time_seconds`.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. This alignment logic follows directly from the AI's earlier choice to preprocess whole sessions first and only then split all streams into identical 2-minute blocks.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`. `move_deve/interframe_int.npy` is used as auxiliary timing information to infer dropped camera frames and align the motion trace to imaging frames.

ii. 
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
...
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy,
    n_imaging_frames=n_frames,
    interframe_int=interframe_int,
)
```

iii. The trajectory shows the agent explicitly focused on "how missing behavior frames are handled" and later summarized the behavior source as motion energy aligned with dropped-frame repair from `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI uses a multi-step pipeline: infer missing video frames from `interframe_int.npy`, linearly interpolate onto the imaging-frame grid, average the aligned motion signal in non-overlapping 10-frame bins, z-score each session's binned motion trace with both mean subtraction and standard-deviation scaling, and later discretize pooled values into quintiles.

ii. 
```python
def infer_missing_frames(interframe_int: np.ndarray) -> np.ndarray:
    median_interval = float(np.median(interframe_int))
    frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
    return np.clip(frame_jumps - 1, 0, None)

...
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

iii. The trajectory states that the agent wanted to "repair camera dropouts on the frame grid," "average both streams in 10-frame windows," and use "session-z-scored motion energy discretized with global quintile edges."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI pools all per-session z-scored motion values, computes the 20th, 40th, 60th, and 80th percentiles, and then assigns each time bin to one of five quintile classes with `np.digitize`.

ii. 
```python
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)

pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions], axis=0)
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)

for session in loaded_sessions:
    motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
```

iii. The script's own summary calls the output "session-z-scored motion energy discretized with global quintile edges," so this categorization policy is explicit in the AI's final implementation.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data in two stages. First it repairs motion length mismatches on the raw imaging-frame grid using inferred dropped frames from `interframe_int.npy`. Then it mean-bins both neural and motion streams with the same 10-frame window size and finally slices them into the same 2-minute trial boundaries.

ii. 
```python
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy,
    n_imaging_frames=n_frames,
    interframe_int=interframe_int,
)

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)

...
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. The trajectory repeatedly frames the problem as keeping alignment "frame-accurate before binning" and then carrying both streams forward on the same 10-frame grid.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI only actively repairs one class of issue: missing camera frames in motion energy. It infers how many frames are missing from `interframe_int.npy` and linearly interpolates them. Other irregularities are handled by exceptions instead of recovery: mismatched `F`/`Fneu` shapes raise an error, any `iscell <= 0.5` raises an error, unreconcilable motion lengths raise an error, and sessions that cannot be divided into the assumed 2-minute binned trials raise an error.

ii. 
```python
if fluorescence.shape != neuropil.shape:
    raise ValueError(f"{session_dir}: F and Fneu shapes do not match")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
...
if missing_total != expected_missing:
    raise ValueError(
        f"Could not reconcile missing motion frames: inferred {missing_total}, "
        f"expected {expected_missing}"
    )
...
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)
```

iii. The trajectory emphasizes "alignment stays frame-accurate" and then the code reflects a conservative fail-fast style: patch missing motion frames, but otherwise stop rather than silently proceeding.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work is full-session preprocessing across every session: loading the large `.npy` arrays, running the Gaussian/minimum/maximum-filter baseline correction over all neurons and time points, and aligning/binning motion before trial splitting.

ii. 
```python
flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
flow = minimum_filter1d(flow, win)
flow = maximum_filter1d(flow, win)

loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
```

iii. The trajectory says the full conversion is "heavier because I'm preprocessing every session before any splits are written," which matches the structure of the code.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious loops could have been reduced or vectorized. The missing-frame alignment loop writes one gap at a time into the aligned motion array, and the trial-splitting loop slices each trial in Python. There is also a repeated `subjects.index(...)` lookup inside the session loop.

ii. 
```python
for gap_missing in missing_after:
    next_src = src + 1
    if gap_missing:
        inserted_indices.extend(range(dst, dst + int(gap_missing)))
        aligned[dst : dst + gap_missing] = np.linspace(...)[1:-1]
        dst += int(gap_missing)
    aligned[dst] = motion_energy[next_src]
    dst += 1
    src = next_src

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    ...

subject_idx.append(subjects.index(session["subject"]))
```

iii. There is no explicit efficiency justification in notes. These loops appear to be simple implementation choices rather than intentional optimization.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several operations structurally. It makes one full pass over all sessions to preprocess and store session dictionaries, then makes another full pass to discretize, split, and package them. Within the packaging loop it repeatedly casts per-trial slices with `astype(..., copy=False)` and repeatedly performs `subjects.index(...)`.

ii. 
```python
loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
...
for session in loaded_sessions:
    motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
    neural_trials, input_trials, output_trials = split_session_into_trials(...)
    ...
    subject_idx.append(subjects.index(session["subject"]))
```

iii. This repetition follows from the AI's two-stage design: preprocess full sessions first, then do dataset assembly in a second pass.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and carries some information that is not used by the downstream decoder dataset itself. `motion_binned` is stored in each intermediate session dictionary but only `motion_z` is used after that. `missing_motion_frame_indices` is computed and returned from alignment but only the total count is preserved in final metadata. The final dataset also includes several metadata summaries that are not used for decoder training.

ii. 
```python
return aligned, {
    "missing_motion_frames": missing_total,
    "missing_motion_frame_indices": inserted_indices,
}

return {
    ...
    "neural_binned": neural_binned,
    "motion_binned": motion_binned,
    "motion_z": motion_z,
    ...
}

session_info.append(
    {
        "session_id": session["session_id"],
        "subject": session["subject"],
        ...
        "suite2p_ops": session["ops"],
    }
)
```

iii. The trajectory does not justify these extras. They appear to have been kept for debugging and reporting rather than because the final standardized dataset required them.
