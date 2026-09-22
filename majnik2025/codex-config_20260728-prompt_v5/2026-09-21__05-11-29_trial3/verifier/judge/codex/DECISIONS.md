# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories whose names start with `jm`, then scans each subject directory for session subdirectories. It only keeps sessions that contain both `suite2p/plane0/F.npy` and `move_deve/motion_energy_glob.npy`. During session discovery it memory-maps `F.npy` and `motion_energy_glob.npy` to record shapes. During conversion it fully loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`, and `ops.npy`.

ii. ```python
def sorted_subjects(data_root: Path) -> list[str]:
    return sorted(
        path.name for path in data_root.iterdir()
        if path.is_dir() and path.name.startswith("jm")
    )

def discover_sessions(data_root: Path) -> tuple[list[str], list[SessionInfo]]:
    subjects = sorted_subjects(data_root)
    session_infos: list[SessionInfo] = []
    for subject in subjects:
        subject_dir = data_root / subject
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            f_path = session_dir / "suite2p" / "plane0" / "F.npy"
            motion_path = session_dir / "move_deve" / "motion_energy_glob.npy"
            if not f_path.exists() or not motion_path.exists():
                continue
            f_mmap = np.load(f_path, mmap_mode="r")
            motion_mmap = np.load(motion_path, mmap_mode="r")
```

```python
raw_f = np.load(session_info.session_dir / "suite2p" / "plane0" / "F.npy").astype(np.float32, copy=False)
raw_fneu = np.load(session_info.session_dir / "suite2p" / "plane0" / "Fneu.npy").astype(np.float32, copy=False)
raw_motion = np.load(session_info.session_dir / "move_deve" / "motion_energy_glob.npy")
tstamps = np.load(session_info.session_dir / "move_deve" / "tstamps.npy")
interframe_int = np.load(session_info.session_dir / "move_deve" / "interframe_int.npy")
```

iii. The justification in `CONVERSION_NOTES.md` is that the release already contains per-session matched Suite2p exports plus motion-energy/timing arrays, so the converter should use the provided session structure directly. The notes also say memory-mapped discovery was added so cataloging session metadata would not require fully loading every array.

## 1-b. How are the data split into subjects?

i. Subjects are identified by directory name: any top-level directory under `/app/data` whose name starts with `jm`. They are sorted alphabetically and used as the canonical subject list.

ii. ```python
def sorted_subjects(data_root: Path) -> list[str]:
    return sorted(
        path.name for path in data_root.iterdir()
        if path.is_dir() and path.name.startswith("jm")
    )
```

iii. The notes justify this by pointing out that the dataset organization already uses `jm031`, `jm032`, etc. as mouse IDs, and the AI repeatedly describes these as the six mice in the release.

## 1-c. How are the data split into sessions?

i. Each session is a subdirectory inside a subject directory. Sessions are sorted by directory name. The AI represents each session with a `SessionInfo` record containing subject ID, session name, path, neuron count, imaging-frame count, and motion-signal length.

ii. ```python
for subject in subjects:
    subject_dir = data_root / subject
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        f_path = session_dir / "suite2p" / "plane0" / "F.npy"
        motion_path = session_dir / "move_deve" / "motion_energy_glob.npy"
        if not f_path.exists() or not motion_path.exists():
            continue
        f_mmap = np.load(f_path, mmap_mode="r")
        motion_mmap = np.load(motion_path, mmap_mode="r")
        session_infos.append(
            SessionInfo(
                subject=subject,
                session=session_dir.name,
                session_dir=session_dir,
                n_neurons=int(f_mmap.shape[0]),
                n_frames=int(f_mmap.shape[1]),
                motion_len=int(motion_mmap.shape[0]),
            )
        )
```

iii. The justification in the notes is that each daily recording already lives in its own folder and contains the necessary Suite2p and behavior files. Sorting is used for deterministic ordering.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions with no native trials, then creates artificial 60-second non-overlapping trials after 10-frame binning. It trims each session to an integer number of 60-second binned windows and slices the binned neural, input, and output streams with the same trial boundaries.

ii. ```python
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FRAME_RATE_HZ))
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES
```

```python
def split_session_into_trials(
    neural_binned: np.ndarray,
    time_binned: np.ndarray,
    motion_bins: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_bins = neural_binned.shape[1]
    usable = (n_bins // TRIAL_BINS) * TRIAL_BINS
    if usable < TRIAL_BINS * 2:
        raise ValueError("Session does not contain at least two 60-second trials after binning.")

    neural_binned = neural_binned[:, :usable]
    time_binned = time_binned[:usable]
    motion_bins = motion_bins[:usable]
    n_trials = usable // TRIAL_BINS

    neural_trials = []
    input_trials = []
    output_trials = []
    for trial_idx in range(n_trials):
        sl = slice(trial_idx * TRIAL_BINS, (trial_idx + 1) * TRIAL_BINS)
        neural_trials.append(neural_binned[:, sl].astype(np.float32, copy=False))
        input_trials.append(time_binned[np.newaxis, sl].astype(np.float32, copy=False))
        output_trials.append(motion_bins[np.newaxis, sl].astype(np.int64, copy=False))
```

iii. The notes explicitly say there are no native trials in the paper or data and that 60-second trialization is an adaptation required by the decoder task. The trajectory also says the “60 s trials will have to be constructed deterministically from the continuous frame axis.”

## 1-e. How are trials filtered based on quality controls?

i. There is no biological or behavioral trial-quality filter. The only trial-level curation is structural: sessions must contain at least two full 60-second trials after binning, and any leftover partial trial at the end of a session is discarded.

ii. ```python
usable = (n_bins // TRIAL_BINS) * TRIAL_BINS
if usable < TRIAL_BINS * 2:
    raise ValueError("Session does not contain at least two 60-second trials after binning.")

neural_binned = neural_binned[:, :usable]
time_binned = time_binned[:usable]
motion_bins = motion_bins[:usable]
```

iii. The justification in the notes is task-driven rather than paper-driven: the decoder format requires at least two trials per session, and the AI chose to keep all full windows without introducing any extra filtering beyond that requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, with `suite2p/plane0/ops.npy` used to supply the Suite2p preprocessing parameters.

ii. ```python
f_path = session_dir / "suite2p" / "plane0" / "F.npy"
fneu_path = session_dir / "suite2p" / "plane0" / "Fneu.npy"
ops = load_ops(session_dir)

f = np.load(f_path).astype(np.float32, copy=False)
fneu = np.load(fneu_path).astype(np.float32, copy=False)
```

iii. The notes justify this by saying the paper’s downstream analyses use Suite2p-style baseline-corrected fluorescence rather than raw fluorescence or deconvolved spikes, and that the provided release already preserves the needed Suite2p arrays and metadata.

## 2-b. How is the `neural` data processed?

i. The AI computes neuropil-subtracted fluorescence `F - neucoeff * Fneu`, then runs `suite2p.extraction.dcnv.preprocess()` using parameters read from `ops.npy`. After that it averages in non-overlapping bins of 10 imaging frames.

ii. ```python
dff = f.copy()
dff -= np.float32(ops["neucoeff"]) * fneu

dff = dcnv.preprocess(
    dff,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 100)),
    device=torch.device("cpu"),
).astype(np.float32, copy=False)
```

```python
neural_binned = bin_array(dff, BIN_FRAMES)
```

iii. In the trajectory the AI says it wanted to “preserv[e] the paper’s Suite2p-style neural preprocessing and 10-frame binning.” In the notes it argues that using the saved Suite2p parameters is a better match to the paper than switching to `spks.npy` or inventing a different `dF/F` transform.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied inside `convert_data.py`. Every row in the provided `F.npy`/`Fneu.npy` arrays is kept. The AI’s rationale is that the release already contains Track2p-matched cells that have passed the Suite2p `iscell` threshold in the upstream pipeline.

ii. ```python
session_payload = {
    "neural_trials": neural_trials,
    "input_trials": input_trials,
    "output_trials": output_trials,
    "brain_region_idx": np.zeros(session_info.n_neurons, dtype=np.int64),
    "motion_edges": motion_edges.astype(np.float32),
    "n_trials": len(neural_trials),
}
```

iii. `CONVERSION_NOTES.md` says the reference code’s main curation rule is Suite2p `iscell > 0.5`, but that the released matched-session files already satisfy that rule and already contain neurons tracked across all days within each mouse. On that basis, the AI decided no further filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. In actual code, the neural data is not realigned to a task event. It stays on the continuous session frame axis, is binned, and is then cut into contiguous 60-second windows. However, the AI’s metadata describes the alignment event as the “start of each contiguous 60-second session window,” with `off_start = 0.0` and `off_end = 60.0`.

ii. ```python
neural_trials, input_trials, output_trials = split_session_into_trials(
    neural_binned,
    time_binned,
    motion_bins,
)
```

```python
"metadata": {
    "temporal_alignment_event": "start of each contiguous 60-second session window",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
```

iii. The notes and trajectory otherwise frame the recordings as continuous sessions with artificial 60-second segmentation, not event-locked trials. The metadata wording appears to be the AI’s way of describing the imposed windows for the decoder format rather than a paper-derived alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses non-overlapping 10-frame averages at 30 Hz, giving 1/3-second bins or 333.33 ms per bin. The same rebinning is applied to neural data, motion energy, and the time input.

ii. ```python
FRAME_RATE_HZ = 30.0
BIN_FRAMES = 10
BIN_SIZE_SEC = BIN_FRAMES / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SEC * 1000.0
```

```python
neural_binned = bin_array(dff, BIN_FRAMES)
motion_binned = bin_array(full_motion, BIN_FRAMES)
time_binned = bin_array(time_values, BIN_FRAMES)
```

iii. The notes repeatedly justify this choice by citing the paper’s statement that decoding analyses averaged both neural and behavior traces in bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not derive the decoder input from an explicit raw timestamp variable. It constructs time directly from the imaging-frame index and the fixed 30 Hz imaging rate.

ii. ```python
time_values = np.arange(session_info.n_frames, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
```

iii. The notes justify imaging-frame time as the canonical axis because the neural traces are complete on that grid, while behavior sometimes has missing camera frames. The trajectory also says it intended to use “time-as-input” adapted from the session frame axis.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI creates a per-frame time vector in seconds from session start, then applies the same 10-frame averaging used for neural and motion data. Because it bins the time series by averaging frame times, each value is the center of a 10-frame bin rather than the left edge.

ii. ```python
time_values = np.arange(session_info.n_frames, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
time_binned = bin_array(time_values, BIN_FRAMES)
```

iii. The notes say the input should be “elapsed-time array in seconds on the imaging axis, then average in the same 10-frame bins and split into 60 s trials,” with absolute session time preferred over resetting within each trial.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction to the same imaging frame grid as the neural data. It is generated at full imaging-frame resolution, binned with the same `bin_array()` function, and split into trials using exactly the same slices used for the neural matrices.

ii. ```python
time_values = np.arange(session_info.n_frames, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
neural_binned = bin_array(dff, BIN_FRAMES)
time_binned = bin_array(time_values, BIN_FRAMES)
...
input_trials.append(time_binned[np.newaxis, sl].astype(np.float32, copy=False))
```

iii. The notes explicitly say the imaging frame axis should be the canonical time axis and that all modalities should then be binned and trialized together on that grid.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`, using `move_deve/tstamps.npy` to infer which imaging frames have observed camera samples. `interframe_int.npy` is loaded but only used for reporting statistics, not for reconstructing the final output trace.

ii. ```python
raw_motion = np.load(session_info.session_dir / "move_deve" / "motion_energy_glob.npy")
tstamps = np.load(session_info.session_dir / "move_deve" / "tstamps.npy")
interframe_int = np.load(session_info.session_dir / "move_deve" / "interframe_int.npy")
```

```python
full_motion, frame_idx, missing_mask = reconstruct_motion_to_imaging_grid(
    raw_motion,
    tstamps,
    session_info.n_frames,
)
```

iii. The notes justify this by saying the dataset README points to timestamps and inter-frame intervals as ways to recover missing camera frames, and that behavior should be mapped onto the imaging frame grid rather than the other way around.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI reconstructs a dense motion trace on the full imaging-frame grid. It infers the imaging-frame index of each observed camera sample from `tstamps.npy`, inserts the observed motion values at those indices, fills missing positions by linear interpolation, averages the reconstructed trace in non-overlapping 10-frame bins, and then discretizes the binned values.

ii. ```python
def reconstruct_motion_to_imaging_grid(
    motion: np.ndarray,
    tstamps: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_idx = infer_behavior_frame_indices(tstamps, n_frames)
    full_motion = np.full(n_frames, np.nan, dtype=np.float32)
    full_motion[frame_idx] = motion.astype(np.float32, copy=False)
    missing_mask = np.isnan(full_motion)
    valid_idx = np.flatnonzero(~missing_mask)
    full_motion = np.interp(
        np.arange(n_frames, dtype=np.float64),
        valid_idx.astype(np.float64),
        full_motion[valid_idx].astype(np.float64),
    ).astype(np.float32)
    return full_motion, frame_idx, missing_mask
```

```python
motion_binned = bin_array(full_motion, BIN_FRAMES)
motion_bins, motion_edges, discretization_method = discretize_motion_quintiles(motion_binned)
```

iii. The notes justify the approach as “interpolate only missing camera-frame positions” while keeping measured samples unchanged. The trajectory shows this was a deliberate alternative to simpler frame-drop insertion logic, though the AI later had to fix one timestamp-jitter edge case.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes session-specific quintile thresholds from the binned motion-energy trace using the 20th, 40th, 60th, and 80th percentiles, then assigns category labels 0-4 with `np.searchsorted(..., side="right")`. It also includes a rank-based fallback if percentile edges collapse, although that fallback was not used in the final converted dataset.

ii. ```python
def discretize_motion_quintiles(motion_binned: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    discrete = np.searchsorted(edges, motion_binned, side="right").astype(np.int64)
    if np.unique(discrete).size < 5:
        order = np.argsort(motion_binned, kind="stable")
        discrete = np.empty_like(order, dtype=np.int64)
        discrete[order] = np.minimum(4, (5 * np.arange(motion_binned.size)) // motion_binned.size)
        method = "rank_fallback"
    else:
        method = "quantile_edges"
    return discrete, edges, method
```

iii. The justification in the notes is task-driven: the user asked for five equal-percentile bins selected per session, and the AI wanted those labels to be defined on the same 10-frame-averaged motion signal that the decoder actually sees.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to the neural data by treating imaging frames as the canonical grid. Motion samples are placed onto inferred imaging-frame positions, missing positions are interpolated, and then the dense motion trace is binned and trialized in lockstep with the neural data.

ii. ```python
full_motion, frame_idx, missing_mask = reconstruct_motion_to_imaging_grid(
    raw_motion,
    tstamps,
    session_info.n_frames,
)
...
neural_binned = bin_array(dff, BIN_FRAMES)
motion_binned = bin_array(full_motion, BIN_FRAMES)
...
neural_trials, input_trials, output_trials = split_session_into_trials(
    neural_binned,
    time_binned,
    motion_bins,
)
```

```python
if raw_motion.shape[0] == session_info.n_frames:
    if not np.allclose(full_motion[frame_idx], raw_motion.astype(np.float32)):
        raise ValueError(f"Observed motion mismatch in no-drop session {session_info.session_id}")
else:
    if not np.allclose(full_motion[frame_idx], raw_motion.astype(np.float32)):
        raise ValueError(f"Observed motion mismatch at valid timestamps in {session_info.session_id}")
```

iii. The notes justify this by arguing that the paper describes camera triggering from the microscope, so imaging frames are the safest canonical axis, and the release explicitly documents occasional missing camera frames that must be filled to restore framewise alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI focuses on missing or inconsistent behavior samples. It reconstructs missing motion frames by interpolation, checks that the inferred number of missing frames matches the imaging-motion length difference, verifies that observed motion samples survive unchanged at non-missing indices, and throws errors on non-finite outputs or sessions too short to yield two trials.

ii. ```python
inferred_missing = int(missing_mask.sum())
expected_missing = int(session_info.missing_motion_frames)
if inferred_missing != expected_missing:
    raise ValueError(
        f"Missing-frame mismatch for {session_info.session_id}: "
        f"expected {expected_missing}, inferred {inferred_missing}"
    )

if raw_motion.shape[0] == session_info.n_frames:
    if not np.allclose(full_motion[frame_idx], raw_motion.astype(np.float32)):
        raise ValueError(f"Observed motion mismatch in no-drop session {session_info.session_id}")
else:
    if not np.allclose(full_motion[frame_idx], raw_motion.astype(np.float32)):
        raise ValueError(f"Observed motion mismatch at valid timestamps in {session_info.session_id}")
```

```python
if not np.isfinite(neural_binned).all():
    raise ValueError(f"Non-finite neural values after preprocessing for {session_info.session_id}")
if not np.isfinite(time_binned).all():
    raise ValueError(f"Non-finite input values for {session_info.session_id}")
if not np.isfinite(motion_binned).all():
    raise ValueError(f"Non-finite motion values after reconstruction for {session_info.session_id}")
```

iii. The notes present these as sanity checks intended to catch silent alignment failures. They also document one bug discovered during full conversion, where timestamp inference misbehaved in a no-missing-frame session and had to be fixed by special-casing `len(tstamps) == n_frames`.

## 6-a. What are the most time-consuming steps of the code?

i. The AI structured the code as four timed phases per session: loading arrays, neural preprocessing, motion reconstruction, and post-processing. The design and notes imply that Suite2p-style neural preprocessing is expected to dominate runtime, while the rest is comparatively light.

ii. ```python
t0 = time.perf_counter()
...
load_sec = time.perf_counter() - t0

t1 = time.perf_counter()
dff, ops = compute_suite2p_dff(session_info.session_dir)
neural_sec = time.perf_counter() - t1

t2 = time.perf_counter()
full_motion, frame_idx, missing_mask = reconstruct_motion_to_imaging_grid(
    raw_motion,
    tstamps,
    session_info.n_frames,
)
motion_sec = time.perf_counter() - t2
```

iii. `CONVERSION_NOTES.md` emphasizes per-session runtime estimates and says the converter remained fast enough without multiprocessing. The trajectory also highlights “Suite2p-style neural preprocessing” as the heavy processing step worth preserving.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the expensive time-axis averaging with `reshape(...).mean(...)` and replaced per-drop insertion logic with `np.interp`. The remaining obvious Python loops are the per-trial slicing loop in `split_session_into_trials()` and the per-session loop in `build_dataset()`.

ii. ```python
def bin_array(values: np.ndarray, bin_frames: int) -> np.ndarray:
    n_time = values.shape[-1]
    usable = (n_time // bin_frames) * bin_frames
    ...
    if values.ndim == 2:
        trimmed = values[:, :usable]
        return trimmed.reshape(trimmed.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32, copy=False)
```

```python
for trial_idx in range(n_trials):
    sl = slice(trial_idx * TRIAL_BINS, (trial_idx + 1) * TRIAL_BINS)
    neural_trials.append(neural_binned[:, sl].astype(np.float32, copy=False))
    input_trials.append(time_binned[np.newaxis, sl].astype(np.float32, copy=False))
    output_trials.append(motion_bins[np.newaxis, sl].astype(np.int64, copy=False))
```

iii. The notes explicitly list “vectorized 10-frame binning” and “timestamp-derived frame indexing” as speedups already added. They do not claim to have eliminated all Python loops, only the most obvious ones on the time axis.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work. It loads `F.npy` and `Fneu.npy` in `convert_one_session()` for plotting, then loads them again inside `compute_suite2p_dff()`. It also reloads `ops.npy` inside `save_processing_plot()`. In addition, after splitting into trials it reconstructs the full session arrays with concatenation purely to verify that slicing did not alter the data.

ii. ```python
raw_f = np.load(session_info.session_dir / "suite2p" / "plane0" / "F.npy").astype(np.float32, copy=False)
raw_fneu = np.load(session_info.session_dir / "suite2p" / "plane0" / "Fneu.npy").astype(np.float32, copy=False)
...
dff, ops = compute_suite2p_dff(session_info.session_dir)
```

```python
def compute_suite2p_dff(session_dir: Path) -> tuple[np.ndarray, dict]:
    f_path = session_dir / "suite2p" / "plane0" / "F.npy"
    fneu_path = session_dir / "suite2p" / "plane0" / "Fneu.npy"
    ops = load_ops(session_dir)
```

```python
recon_neural = np.concatenate(neural_trials, axis=1)
recon_time = np.concatenate([trial[0] for trial in input_trials])
recon_output = np.concatenate([trial[0] for trial in output_trials])
if not np.allclose(recon_neural, neural_binned):
    raise ValueError("Neural trial splitting failed reconstruction check.")
```

iii. The notes acknowledge one of these redundancies directly: plot generation reloads `ops.npy` inside the plotting helper, which the AI describes as negligible but avoidable.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does several non-essential things that do not affect the saved decoder dataset: optional plotting, per-session timing and ETA bookkeeping, detailed session stats, redundant reconstruction checks after trialization, and loading arrays that are only needed for debugging or plots. It also always loads `interframe_int.npy` even though that array is only summarized in `session_stats`.

ii. ```python
if make_plot:
    save_processing_plot(
        session_info=session_info,
        raw_f=raw_f,
        raw_fneu=raw_fneu,
        dff=dff,
        frame_idx=frame_idx,
        raw_motion=raw_motion,
        full_motion=full_motion,
        missing_mask=missing_mask,
        motion_binned=motion_binned,
        motion_edges=motion_edges,
        motion_bins=motion_bins,
        time_binned=time_binned,
        neural_trials=neural_trials,
        output_trials=output_trials,
        out_dir=out_dir,
        discretization_method=discretization_method,
    )
```

```python
session_stats = {
    ...
    "interframe_median": float(np.median(interframe_int)),
    "load_sec": load_sec,
    "neural_sec": neural_sec,
    "motion_sec": motion_sec,
    "post_sec": post_sec,
    "total_sec": time.perf_counter() - t0,
```

```python
recon_neural = np.concatenate(neural_trials, axis=1)
recon_time = np.concatenate([trial[0] for trial in input_trials])
recon_output = np.concatenate([trial[0] for trial in output_trials])
```

iii. The notes frame these mostly as debugging and validation aids. They were used to support sanity checks and reporting, not the final downstream decoder analyses themselves.
