# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subject directories under `data/` whose names start with `jm`, then discovers session directories within each subject whose names begin with a date-like 4-digit prefix. For each session it loads Suite2p neural files from `suite2p/plane0/` and behavior files from `move_deve/`. It does not load pre-made trials from disk; instead it later constructs pseudo-trials from each continuous session.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    if not sessions:
        raise RuntimeError("No sessions found under data/.")
    return sessions
```

```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. In `CONVERSION_NOTES.md`, the AI says the released dataset is organized as continuous sessions with one matched Suite2p recording and one motion-energy stream per session, so it treats session directories as the atomic loaded unit and derives trials later.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted `jm*` directories under `data/`.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

```python
subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI’s notes explicitly identify the six `jm` directories as the six mice and use the folder naming convention as the subject split.

## 1-c. How are the data split into sessions?

i. Each date-named subdirectory within a subject is treated as one session. The AI stores subject name, session name, and session path in a `SessionInfo` dataclass.

ii.
```python
@dataclass(frozen=True)
class SessionInfo:
    subject: str
    session: str
    path: Path
```

```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

iii. The notes state that each subject contains daily session folders named like `YYYY-MM-DD_a`, so the AI uses those subdirectories as recording sessions.

## 1-d. How are the data split into trials?

i. The AI decides there are no native trials and therefore creates pseudo-trials by first averaging the continuous recording in non-overlapping 10-frame bins and then cutting each session into consecutive 2-minute blocks. Each 2-minute block is one trial.

ii.
```python
BIN_FRAMES = 10
TRIAL_DURATION_SEC = 120.0
```

```python
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
    input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
    output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. In the notes, the AI justifies this as matching the paper’s decoder workflow: “Convert each continuous session into consecutive 2-minute pseudo-trials after 10-frame averaging.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Trials are included if they arise from the 2-minute segmentation, but the AI enforces that each session must yield at least two pseudo-trials.

ii.
```python
if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
```

iii. The notes say all 41 sessions are kept because all yield at least 10 pseudo-trials, and no separate trial curation rule is described.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived primarily from `F.npy` and `Fneu.npy`, with `ops.npy` supplying preprocessing parameters such as `neucoeff`, baseline method, and frame rate.

ii.
```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. The AI’s notes state that the authoritative neural target is Suite2p-style baseline-corrected fluorescence reconstructed from `F`, `Fneu`, and saved `ops.npy` parameters.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil using the per-session `ops["neucoeff"]`, runs `suite2p.extraction.dcnv.preprocess(...)` using the parameters saved in `ops.npy`, and then averages the processed traces in non-overlapping 10-frame bins.

ii.
```python
corrected = np.array(F, dtype=np.float32, copy=True)
corrected -= np.float32(ops["neucoeff"]) * np.asarray(Fneu, dtype=np.float32)
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 2000)),
    device=torch.device("cpu"),
).astype(np.float32, copy=False)
```

```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```

iii. The notes justify this as “paper-consistent Suite2p-style baseline-corrected fluorescence” and also claim the 10-frame averaging matches the decoder denoising described in the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any additional neuron filtering in `convert_data.py`. It assumes the released `suite2p/plane0` arrays already contain Track2p-matched, cell-filtered neurons and keeps all rows in the arrays.

ii.
```python
return {
    "info": session,
    "ops": ops,
    "brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
    "neural_binned": neural_binned,
    ...
}
```

iii. In the notes, the AI argues that the bundled data “already represent the Track2p all-days matched population analyzed in the paper,” so no extra `iscell` or tracking filter is applied at conversion time.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of a consecutive 2-minute block cut from a continuous session, not to session start. It records this as the temporal alignment event in metadata.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
    "off_start": 0.0,
    "off_end": TRIAL_DURATION_SEC,
    ...
}
```

iii. The notes justify this by treating the paper’s 2-minute decoder blocks as the natural trial structure for the benchmark format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are at 10-frame bins, i.e. `333.33 ms` per sample. Yes, temporal rebinning is applied by averaging every 10 imaging/video frames.

ii.
```python
RAW_FS_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ
```

```python
def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32).astype(np.float32, copy=False)
```

iii. The AI’s notes say this is intentional because the paper’s decoder “used averages over 10 consecutive timestamps.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a dedicated time variable. It is derived from frame index and frame rate, specifically `np.arange(n_frames)` and `ops["fs"]`.

ii.
```python
n_frames = int(ops["nframes"])
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. The notes describe this as elapsed session time in seconds from session start, later carried into each pseudo-trial.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes frame-center times in seconds, then averages them in the same non-overlapping 10-frame bins as the neural and motion streams.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
...
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The notes justify this as preserving “absolute position within the original recording” while matching the paper’s 10-frame decoding bins.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by constructing it on the imaging-frame clock, binning it with the same 10-frame averaging as neural activity, and slicing the same 2-minute trial boundaries.

ii.
```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
...
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
```

iii. The AI’s notes explicitly say imaging frames are the master clock and each pseudo-trial keeps its absolute elapsed time from the original recording.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The motion output is derived from `motion_energy_glob.npy`, with `interframe_int.npy` used to reconstruct missing frames. The AI also loads `tstamps.npy`, but only uses it for diagnostic previews, not for the final output values.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. The notes state that the motion-energy source is `motion_energy_glob.npy` and the timing files are used to restore missing camera frames before alignment.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI reconstructs the full motion trace to imaging length by inferring missing frame positions from `interframe_int.npy`, linearly interpolates missing values, averages the result in 10-frame bins, then applies global min-max normalization across all sessions before discretization.

ii.
```python
def reconstruct_motion_trace(motion: np.ndarray, interframe_int: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    ...
    median_interval = float(np.median(interframe_int))
    steps = np.rint(interframe_int / median_interval).astype(np.int64)
    steps[steps < 1] = 1
    ...
    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    missing_idx = np.flatnonzero(np.isnan(full))
    full = interpolate_nans(full)
    return full, missing_idx
```

```python
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
...
motion_min = float(np.min(motion_all))
motion_max = float(np.max(motion_all))
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
```

iii. The notes justify this as using imaging as the master clock, restoring dropped behavior frames from timing gaps, and then satisfying the benchmark requirement for normalized motion energy.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After normalization, the AI thresholds motion energy into five global equal-percentile bins (quintiles) computed across all sessions, then assigns each sample with `np.digitize`.

ii.
```python
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)
```

```python
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. The notes state that categorical outputs are required, so the AI uses “5 global equal-percentile classes” named `Q1` to `Q5`.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by reconstructing missing behavior frames onto the imaging-frame timeline, then binning motion and neural data identically and slicing them with the same 2-minute trial boundaries.

ii.
```python
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
...
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
...
neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. The notes say imaging frames are the master clock and missing behavior frames are reinserted before binning so both modalities remain aligned.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles shorter behavior traces by reconstructing their missing frame positions from timing gaps and filling them by linear interpolation. It raises an error if timing reconstruction does not land on the expected imaging length. It also rejects sessions if they would yield fewer than two pseudo-trials.

ii.
```python
if observed_idx[-1] != target_len - 1:
    raise ValueError(
        f"Timing reconstruction failed: last observed index {observed_idx[-1]} does not match target {target_len - 1}"
    )
```

```python
full = np.full(target_len, np.nan, dtype=np.float32)
full[observed_idx] = motion
missing_idx = np.flatnonzero(np.isnan(full))
full = interpolate_nans(full)
```

iii. In the notes, the AI says the README explicitly mentioned missing camera frames and that they should be treated as missing values or interpolated, so it chose interpolation on the imaging timeline.

## 6-a. What are the most time-consuming steps of the code?

i. The main cost is Suite2p-style neural preprocessing via `dcnv.preprocess(...)` for every session. The AI also does full-session array loads and optional plotting, but the fluorescence preprocessing is the dominant computation.

ii.
```python
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 2000)),
    device=torch.device("cpu"),
).astype(np.float32, copy=False)
```

iii. The notes explicitly identify “Suite2p-style fluorescence preprocessing” as the main cost and report timing per session after conversion.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most heavy array operations are already vectorized. The clearest remaining Python loop is trial segmentation in `segment_trials`, where each trial is sliced and appended one at a time. Session processing is also sequential rather than parallel.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
    input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
    output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
    ...
```

```python
processed_sessions = []
for session in sessions:
    processed_sessions.append(process_session(session))
```

iii. The AI’s notes claim it already vectorized behavior reconstruction and binning, so the remaining opportunities are the smaller Python loops around per-trial and per-session assembly.

## 6-c. What processing does the code repeat multiple times?

i. The code recomputes trial slicing separately for neural, input, and output during `segment_trials`; it also computes motion normalization edges globally and then re-normalizes each session again during final assembly. Preview arrays are also built for every session even though only the first two are ever plotted.

ii.
```python
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
...
for session in processed_sessions:
    motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
    neural_trials, input_trials, output_trials = segment_trials(...)
```

```python
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    "corrected": corrected[sample_neuron, :preview_len].copy(),
    "processed": processed[sample_neuron, :preview_len].copy(),
}
```

iii. This repeated work follows from the AI’s two-pass design: first preprocess/bin all sessions, then compute global motion thresholds, then assemble trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter always prepares diagnostic preview data (`neural_preview`, `motion_preview`) and loads `tstamps.npy`, even though none of that is stored in the final dataset. It also imports and configures Matplotlib unconditionally for optional plotting. These steps are not needed for downstream decoding.

ii.
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

```python
timestamps = np.load(move_dir / "tstamps.npy")
...
motion_preview = {
    "raw": np.asarray(motion_raw[:preview_len], dtype=np.float32),
    "reconstructed": motion_full[:preview_len].copy(),
    "missing_idx": missing_idx[missing_idx < preview_len].copy(),
    "timestamps": np.asarray(timestamps[: min(preview_len, timestamps.shape[0])], dtype=np.float64),
    "interframe_int": np.asarray(interframe_int[: min(preview_len - 1, interframe_int.shape[0])], dtype=np.float64),
}
```

iii. The notes say the preview structures and plots were added for sanity checks and documentation, not because the decoder format required them.
