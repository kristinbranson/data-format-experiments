# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers all subjects and sessions by walking `data/`, then processes each session separately. For each session it loads neural arrays from `suite2p/plane0` and behavior arrays from `move_deve`. Trials are not loaded from disk because the source data are continuous; they are created later by segmentation.

ii. ```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

```python
def process_session(session: SessionInfo) -> dict:
    s2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"
    neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
    motion_raw = np.load(move_dir / "motion_energy_glob.npy")
    timestamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The notes say the dataset contains 6 subject folders, 41 continuous sessions, and no native trial table, so the agent chose full directory traversal and per-session loading.

## 1-b. How are the data split into subjects?

i. Subjects are identified from top-level folder names like `jm031`, sorted lexicographically. The final dataset stores a unique sorted `subjects` list and a `subject_idx` per session.

ii. ```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

```python
subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.array([subject_to_idx[session["info"].subject] for session in processed_sessions], dtype=np.int64),
```

iii. The notes justify this from the data README: each subject folder corresponds to one mouse and folder names map directly to mouse IDs.

## 1-c. How are the data split into sessions?

i. Sessions are identified from date-named subdirectories inside each subject directory, again sorted lexicographically. Each session directory becomes one session in the output.

ii. ```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

```python
for session in sessions:
    processed_sessions.append(process_session(session))
```

iii. The notes say each daily recording day is a separate session folder named `YYYY-MM-DD_a`, and that all 41 sessions are kept.

## 1-d. How are the data split into trials?

i. The source data have no native trials. The agent creates pseudo-trials by first averaging into non-overlapping 10-frame bins and then cutting each continuous session into consecutive 2-minute blocks.

ii. ```python
BIN_FRAMES = 10
TRIAL_DURATION_SEC = 120.0
```

```python
def segment_trials(..., trial_bins: int):
    n_total_bins = neural_binned.shape[1]
    n_trials = n_total_bins // trial_bins
    for trial_idx in range(n_trials):
        start = trial_idx * trial_bins
        stop = start + trial_bins
        neural_trial = neural_binned[:, start:stop]
        input_trial = time_binned[np.newaxis, start:stop]
        output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. The notes say the paper’s decoder used consecutive 2-minute blocks and 10-frame averaging, so the agent used those blocks as the artificial trial dimension required by the target format.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality filter. The agent keeps every full 2-minute block, implicitly drops incomplete trailing data through integer division and trimming, and raises an error if a session would yield fewer than 2 trials.

ii. ```python
n_bins = x.shape[-1] // bin_frames
trimmed = x[..., : n_bins * bin_frames]
```

```python
n_trials = n_total_bins // trial_bins
```

```python
if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
```

iii. The notes explicitly state there is no native trial curation in the source data, so the only QC the agent applied at the trial level was enforcing the benchmark’s minimum of two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence files `F.npy` and `Fneu.npy`, using parameters from `ops.npy`.

ii. ```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. The notes say the agent chose fluorescence-based traces rather than `spks.npy`, because it interpreted the paper as using baseline-corrected fluorescence traces for downstream analyses.

## 2-b. How is the `neural` data processed?

i. The agent subtracts neuropil using Suite2p’s saved `neucoeff`, then runs `suite2p.extraction.dcnv.preprocess` with the per-session baseline parameters from `ops.npy`, and finally averages in 10-frame bins.

ii. ```python
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

iii. The notes say the agent saw a mismatch between the paper’s description and a Track2p GUI helper, and decided the paper was the stronger authority, so it reconstructed a Suite2p-style baseline-corrected fluorescence trace from `F`, `Fneu`, and `ops`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply any new neural QC at conversion time. It assumes the exported Suite2p arrays already contain the Track2p-matched, cell-filtered population and keeps all rows.

ii. ```python
"brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
"neural_binned": neural_binned,
```

There is no code loading `iscell.npy` or dropping neuron rows in `convert_data.py`.

iii. The notes justify this by stating that the release already contains Track2p “save in suite2p format” outputs with cells present across all days, so additional filtering was treated as already baked into the release.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns each trial to the start of the artificial 2-minute block. There is no native stimulus/event alignment in the raw data, so block onset is used as the dataset’s alignment event.

ii. ```python
"temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SEC,
```

```python
start = trial_idx * trial_bins
stop = start + trial_bins
neural_trial = neural_binned[:, start:stop]
```

iii. The notes say the recordings are continuous spontaneous-behavior sessions with no native trial structure, so alignment had to be introduced downstream and the paper’s 2-minute decoder blocks were chosen.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, giving 333.333 ms per time bin. Yes, non-overlapping temporal rebinning is applied to neural, input, and output streams.

ii. ```python
RAW_FS_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ
```

```python
def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32)
```

iii. The notes repeatedly cite the paper’s “averaging in bins of 10 consecutive timestamps” as the basis for this choice.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not loaded from an explicit time file. It is synthesized from the imaging frame index together with the sampling rate in `ops.npy` and the session frame count.

ii. ```python
n_frames = int(ops["nframes"])
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. The notes justify this as the cleanest way to represent “time elapsed from the beginning of the experiment” while keeping the imaging stream as the master clock.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent computes frame-center times in seconds, averages them in non-overlapping 10-frame bins, and then slices the same binned vector into 2-minute pseudo-trials.

ii. ```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

```python
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
```

iii. The notes say the input should remain absolute elapsed time within the original recording, so the per-trial inputs keep their original session-time values rather than resetting to zero.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is aligned by using the same imaging-frame clock, the same 10-frame averaging, and the same trial start/stop indices as the neural data.

ii. ```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

```python
neural_trial = neural_binned[:, start:stop]
input_trial = time_binned[np.newaxis, start:stop]
```

iii. The notes say imaging is the authoritative timeline and all other streams should be aligned to it before segmentation.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion output is derived from `motion_energy_glob.npy`, with `tstamps.npy` and `interframe_int.npy` used to reconstruct missing camera frames when needed.

ii. ```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. The data README explicitly says missing camera frames should be inferred from `tstamps.npy` or `interframe_int.npy`, and the notes cite that as the basis for the reconstruction step.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent reconstructs the full motion trace to imaging length, linearly interpolates inserted missing frames, averages in 10-frame bins, globally min-max normalizes the binned values, and then converts them to classes.

ii. ```python
full = np.full(target_len, np.nan, dtype=np.float32)
full[observed_idx] = motion
missing_idx = np.flatnonzero(np.isnan(full))
full = interpolate_nans(full)
```

```python
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions]).astype(np.float32)
motion_min = float(np.min(motion_all))
motion_max = float(np.max(motion_all))
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
```

iii. The notes say this matches the paper’s motion-energy definition and the task’s requirement to normalize and discretize motion into five categories.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Categories are defined by global equal-percentile quintiles over the normalized binned motion-energy values from all sessions, then assigned with `np.digitize`.

ii. ```python
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)
```

```python
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
output_trial = output_trial[np.newaxis, :]
```

iii. The notes say the benchmark required categorical outputs and specifically asked for five equal-percentile bins, so the agent used global quintiles instead of per-session thresholds.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to imaging by reconstructing it to the imaging frame count, then applying the same 10-frame binning and identical trial boundaries as the neural data.

ii. ```python
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
```

```python
neural_trials, input_trials, output_trials = segment_trials(
    session["neural_binned"],
    session["time_binned"],
    motion_norm,
    motion_edges,
    trial_bins=trial_bins,
)
```

iii. The notes say the microscope-triggered imaging/video setup implies framewise synchrony, so imaging length is the reference and behavior is reconstructed to that timeline.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior frames are detected from timing gaps, reinserted as `NaN`, and linearly interpolated. The code errors out if the timing reconstruction is inconsistent or if an array is all `NaN`. It also drops incomplete trailing bins/blocks by construction.

ii. ```python
if motion.shape[0] == target_len:
    return motion, np.empty(0, dtype=np.int64)
...
if observed_idx[-1] != target_len - 1:
    raise ValueError(...)
...
full = np.full(target_len, np.nan, dtype=np.float32)
full[observed_idx] = motion
missing_idx = np.flatnonzero(np.isnan(full))
full = interpolate_nans(full)
```

```python
if not valid.any():
    raise ValueError("Cannot interpolate an array containing only NaNs.")
```

iii. The notes explicitly mention 9 sessions with behavior/imaging mismatches and say the README allows either missing values or interpolation; the agent chose interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is neural preprocessing per session, especially loading `F/Fneu` and running `dcnv.preprocess`. The full-dataset per-session loop is the main runtime driver.

ii. ```python
neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
```

```python
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    ...
)
```

iii. The notes say “the main cost is Suite2p-style fluorescence preprocessing,” and the code structure supports that: behavior reconstruction and binning are lightweight compared with `dcnv.preprocess`.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the Python loop in `segment_trials`, which slices and appends one trial at a time. The top-level per-session loop could also be parallelized, though that is a larger structural change rather than simple vectorization.

ii. ```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop]
    input_trial = time_binned[np.newaxis, start:stop]
    output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The notes mention that behavior reconstruction and binning were already vectorized, which implies the remaining obvious Python-level loop is trial segmentation.

## 6-c. What processing does the code repeat multiple times?

i. The code scans some session metadata more than once in sample mode, computes preview arrays that duplicate slices of processed data, and performs global motion normalization in one pass before re-normalizing each session again when building trials.

ii. ```python
def session_nframes(session: SessionInfo) -> int:
    return int(np.load(session.path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()["nframes"])
```

```python
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
...
for session in processed_sessions:
    motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
```

iii. The notes describe the implementation as a two-pass conversion: first collect per-session processed data, then compute global motion statistics and finalize trialization. That design necessarily repeats some normalization-related work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds `neural_preview` and `motion_preview` dictionaries, loads `timestamps` only for diagnostics, and generates optional plots. Those artifacts are not used in the final decoder dataset.

ii. ```python
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    "corrected": corrected[sample_neuron, :preview_len].copy(),
    "processed": processed[sample_neuron, :preview_len].copy(),
}
```

```python
motion_preview = {
    "raw": np.asarray(motion_raw[:preview_len], dtype=np.float32),
    "reconstructed": motion_full[:preview_len].copy(),
    "missing_idx": missing_idx[missing_idx < preview_len].copy(),
    "timestamps": np.asarray(timestamps[: min(preview_len, timestamps.shape[0])], dtype=np.float64),
    "interframe_int": np.asarray(interframe_int[: min(preview_len - 1, interframe_int.shape[0])], dtype=np.float64),
}
```

```python
if args.show_processing:
    for session in processed_sessions[:2]:
        plot_processing(session, motion_min, motion_max, motion_edges, plot_path)
```

iii. The notes explicitly frame these as processing plots and previews for sanity checking, not as data consumed by downstream decoder training.
