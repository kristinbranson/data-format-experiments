# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers all sessions by scanning `data/` for subject directories whose names start with `jm`, then scanning each subject for date-named session directories. In full mode it processes every discovered session. For each session it loads Suite2p calcium files from `suite2p/plane0/` and behavior files from `move_deve/`.

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
def process_session(session: SessionInfo) -> dict:
    s2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"

    neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
    n_frames = int(ops["nframes"])

    motion_raw = np.load(move_dir / "motion_energy_glob.npy")
    timestamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
    motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. In `CONVERSION_NOTES.md`, Step 2 says there are 6 `jm*` subject folders and that each session contains `suite2p/plane0/` plus `move_deve/`. Step 5 says “Keep all 41 sessions.” Trajectory step 82 says the script will “use the saved Suite2p `ops.npy` parameters directly” and “reconstruct missing behavior frames from timing gaps.”

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `jm*` directory names. They are sorted and later re-materialized as `subjects`, with each session mapped back to a subject by `subject_idx`.

ii. 
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

```python
subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[session["info"].subject] for session in processed_sessions], dtype=np.int64),
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly maps “Subject folder name (e.g. `jm031`)” to `subjects` and `subject_idx`, and says to use a unique subject list in sorted order.

## 1-c. How are the data split into sessions?

i. Each date-named subdirectory under a subject is treated as one recording session. The agent flattens them into one ordered session list and preserves subject membership through `subject_idx`.

ii. 
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

```python
all_sessions = discover_sessions(data_root)
sessions = select_sample_sessions(all_sessions) if args.sample else all_sessions
```

iii. `CONVERSION_NOTES.md` Step 2 says subject folders contain daily session folders named like `YYYY-MM-DD_a`, and Step 5 says session inclusion should keep all 41 sessions.

## 1-d. How are the data split into trials?

i. The agent assumes there are no native trials and creates pseudo-trials. It first averages the continuous data into non-overlapping 10-frame bins, then segments each session into consecutive 2-minute blocks. Each such block is emitted as one trial.

ii. 
```python
BIN_FRAMES = 10
TRIAL_DURATION_SEC = 120.0
```

```python
def segment_trials(
    neural_binned: np.ndarray,
    time_binned: np.ndarray,
    motion_norm_binned: np.ndarray,
    motion_edges: np.ndarray,
    trial_bins: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_total_bins = neural_binned.shape[1]
    n_trials = n_total_bins // trial_bins
    ...
    for trial_idx in range(n_trials):
        start = trial_idx * trial_bins
        stop = start + trial_bins
        neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
        input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
        output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

```python
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)
```

iii. In Step 5, `CONVERSION_NOTES.md` says “Convert each continuous session into consecutive 2-minute pseudo-trials after 10-frame averaging,” arguing that this matches the paper’s decoder evaluation blocks and gives 10 trials for 20-minute sessions and 15 for 30-minute sessions. Trajectory step 78 says this avoids “inventing arbitrary windows.”

## 1-e. How are trials filtered based on quality controls?

i. The agent does not implement trial-level quality-control filtering based on signal quality or metadata. The only effective filtering is structural: incomplete final blocks are dropped by integer division, and the code raises an error if a session would produce fewer than two pseudo-trials.

ii. 
```python
n_total_bins = neural_binned.shape[1]
n_trials = n_total_bins // trial_bins
```

```python
if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
```

iii. Step 5 of `CONVERSION_NOTES.md` says “Keep all 41 sessions” and that “all sessions yield at least 10 pseudo-trials after processing,” so the agent did not plan any explicit trial QC beyond satisfying the decoder format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from `F.npy` and `Fneu.npy`, with `ops.npy` used to supply Suite2p preprocessing parameters such as `neucoeff`, `baseline`, and `fs`.

ii. 
```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. `CONVERSION_NOTES.md` Step 5 maps `suite2p/plane0/F.npy`, `Fneu.npy`, and `ops.npy` to `neural` and says to compute “Suite2p-style baseline-corrected fluorescence” using the saved per-session parameters.

## 2-b. How is the `neural` data processed?

i. The agent performs neuropil subtraction and then runs `suite2p.extraction.dcnv.preprocess` with parameters read from `ops.npy`. After that, it averages the processed fluorescence into non-overlapping 10-frame bins.

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

iii. In Step 4, `CONVERSION_NOTES.md` says the paper is the authoritative source for downstream preprocessing and that the bundled GUI `dF/F0` helper should not be used. Step 5 says to use “paper-consistent Suite2p-style baseline-corrected fluorescence” with saved `ops.npy` values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The converter does not apply any additional neuron filtering at runtime. It keeps all rows present in the bundled `F.npy`/`Fneu.npy` arrays and assumes the released data are already Track2p-matched and Suite2p cell-filtered.

ii. 
```python
processed = dcnv.preprocess(...)
...
"brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
```

There is no code that loads or thresholds `iscell.npy` before building `neural`.

iii. `CONVERSION_NOTES.md` Step 4 says the exported Suite2p data already contain Track2p-matched cells across all days and are consistent with prior `iscell > 0.5` filtering. Step 5 says “Keep all neurons present in the released matched Suite2p arrays.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns each trial to the start of a consecutive 2-minute block cut from the session, not to a behavioral event. Metadata describes the alignment event as the start of each pseudo-trial block.

ii. 
```python
"metadata": {
    ...
    "temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
    "off_start": 0.0,
    "off_end": TRIAL_DURATION_SEC,
    ...
},
```

```python
start = trial_idx * trial_bins
stop = start + trial_bins
neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
```

iii. In Step 5, `CONVERSION_NOTES.md` says the data are continuous and that 2-minute blocks provide the “natural trial structure” because the paper used 2-minute decoder blocks. The agent therefore treats block start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final dataset is at 333.33 ms resolution. The agent rebins all streams by averaging every 10 native 30 Hz frames into one time bin.

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

iii. Step 3 of `CONVERSION_NOTES.md` quotes the paper’s decoder smoothing rule as “averaging in bins of 10 consecutive timestamps.” Step 5 then makes “Temporal binning” a key decision and says the converter should match that 10-frame averaging.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a dedicated raw variable. It is derived from imaging frame indices and the frame rate stored in `ops["fs"]`.

ii. 
```python
n_frames = int(ops["nframes"])
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. Step 5 of `CONVERSION_NOTES.md` maps “Session frame index / bin centers” to `input[0]` and says to convert these to elapsed seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent creates per-frame timestamps at frame centers, then averages them in the same 10-frame bins used for neural and behavior data. The resulting binned time vector is the decoder input.

ii. 
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
...
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. Step 5 of `CONVERSION_NOTES.md` says to use “bin-center time for each sample inside each 2-minute block,” because the agent decided to rebin all modalities into 10-frame averages before trialization.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is built on the same imaging-frame clock, averaged with the same `bin_array_mean` function as the neural data, and sliced into the same trial windows.

ii. 
```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

```python
neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
```

iii. Step 5 of `CONVERSION_NOTES.md` calls imaging the “master clock” and says elapsed time should remain time-varying and use the same 10-frame bins and 2-minute pseudo-trials as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The motion output is derived from `motion_energy_glob.npy`, with `interframe_int.npy` used to infer dropped frames. The code also loads `tstamps.npy`, but only for preview/plotting rather than the actual reconstruction.

ii. 
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. Step 2 of `CONVERSION_NOTES.md` says `motion_energy_glob.npy` is the behavior signal to decode and that `tstamps.npy` / `interframe_int.npy` reveal missing behavior frames. Step 5 maps those files to `output[0]`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent reconstructs a full-length motion trace when behavior frames are missing by inferring frame steps from `interframe_int.npy`, placing observed values on the reconstructed timeline, and linearly interpolating missing points. It then averages motion in non-overlapping 10-frame bins, applies a global min-max normalization over all sessions, and later discretizes the binned values.

ii. 
```python
def reconstruct_motion_trace(motion: np.ndarray, interframe_int: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    motion = np.asarray(motion, dtype=np.float32)
    interframe_int = np.asarray(interframe_int, dtype=np.float64)
    if motion.shape[0] == target_len:
        return motion, np.empty(0, dtype=np.int64)

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
motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions]).astype(np.float32)
motion_min = float(np.min(motion_all))
motion_max = float(np.max(motion_all))
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
```

iii. Step 4 of `CONVERSION_NOTES.md` says imaging frames should be the master clock and missing behavior frames should be “reconstruct[ed] ... from doubled timing gaps before alignment.” Step 5 makes global min-max normalization and 10-frame averaging explicit design decisions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent computes five global equal-percentile bins on the normalized, 10-frame-averaged motion-energy values and then uses `np.digitize` to assign each binned sample to one of five quintile classes named `Q1` to `Q5`.

ii. 
```python
OUTPUT_CLASS_NAMES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
```

```python
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the target decoder requires categorical outputs and therefore motion energy should be converted into “5 global equal-percentile classes.” It also states the classes are based on globally normalized values to satisfy the task wording.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent aligns motion to neural data by reconstructing the behavior trace to the imaging frame count, then binning motion and neural signals with the same 10-frame averaging and slicing both into the same pseudo-trial boundaries.

ii. 
```python
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
...
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
```

```python
neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. Step 4 of `CONVERSION_NOTES.md` says “Use imaging frames as the reference timeline” and “Reinsert missing behavior frames from timing gaps so both modalities remain frame-aligned before binning.”

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The only explicit data-repair path is for behavior-frame mismatches: the agent reconstructs missing camera frames and interpolates over them. It also raises hard errors if timing reconstruction fails or if the motion range becomes invalid after preprocessing. Incomplete trailing bins or pseudo-trials are silently dropped by trimming and integer division.

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

```python
trimmed = x[..., : n_bins * bin_frames]
...
n_trials = n_total_bins // trial_bins
```

iii. Step 2 of `CONVERSION_NOTES.md` documents 9 sessions with behavior/imaging length mismatches. Step 4 resolves this by reconstructing missing behavior frames. The notes do not describe any other data-cleaning policy beyond keeping all sessions and neurons.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is Suite2p fluorescence preprocessing (`dcnv.preprocess`) for each session. Optional plotting is also extra work, but only when `--show-processing` is used.

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

iii. Step 6 of `CONVERSION_NOTES.md` says “the main cost is Suite2p-style fluorescence preprocessing,” and the per-session timing printouts in `process_session` suggest the agent was monitoring this stage specifically.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes the behavior reconstruction and 10-frame averaging, so the remaining obvious Python loops are mostly bookkeeping loops over sessions and trials. The trial-splitting loop in `segment_trials` could be replaced with reshape-based chunking, but it is much smaller than the vectorized core preprocessing.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
    input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
    output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. Step 6 of `CONVERSION_NOTES.md` says “Behavior reconstruction is vectorized via timing-step accumulation” and “Binning uses reshape-and-mean rather than Python loops,” so the agent’s own view was that the expensive parts had already been vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats some derived processing for non-essential outputs: it computes motion normalization and digitization once for dataset assembly and again inside `plot_processing` for optional figures. In sample mode it may also reopen `ops.npy` multiple times while choosing representative sessions.

ii. 
```python
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
...
for session in processed_sessions:
    motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
    neural_trials, input_trials, output_trials = segment_trials(...)
```

```python
motion_norm_binned = (motion_binned - motion_min) / (motion_max - motion_min)
motion_classes = np.digitize(motion_norm_binned, motion_edges[1:-1], right=False)
```

iii. The notes do not call this out explicitly, but the code structure shows one pass to preprocess sessions, a second pass to build the final dataset, and an optional third reuse of the same derived quantities for plots.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter always builds preview dictionaries for neural and motion traces, loads `tstamps.npy`, and stores intermediate arrays used only for optional plotting, even though these previews are not part of the saved dataset. When `--show-processing` is false, that work is discarded after conversion.

ii. 
```python
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    "corrected": corrected[sample_neuron, :preview_len].copy(),
    "processed": processed[sample_neuron, :preview_len].copy(),
}
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

iii. Step 6 of `CONVERSION_NOTES.md` says the script supports “optional processing figures,” and the implementation keeps plotting previews even during normal conversion. Those previews are not serialized into `converted_data.pkl` and therefore are unnecessary for downstream decoding.
