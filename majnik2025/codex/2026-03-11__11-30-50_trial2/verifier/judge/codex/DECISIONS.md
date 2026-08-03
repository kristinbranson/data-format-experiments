# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for subject directories whose names start with `jm`, then scans each subject directory for date-like session directories. It flattens all subject/session pairs into a single ordered `SessionRef` list. For each session it loads neural and behavior files from `suite2p/plane0` and `move_deve`, then later turns each continuous session into trial blocks.

ii. 
```python
def discover_sessions(data_root: Path) -> list[SessionRef]:
    sessions: list[SessionRef] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
        ):
            sessions.append(
                SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
            )
    return sessions

F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says the provided `data/` folder already contains post-Track2p, all-day matched exports and that all 41 sessions should be included. In trajectory step 58 it also states that it will “use the already tracked neurons” and “use motion energy as behavior.”

## 1-b. How are the data split into subjects?

i. Subjects are the top-level `jm*` directories. The exported `subjects` list is the sorted unique set of `SessionRef.subject` values, and each session gets a `subject_idx` pointing into that list.

ii. 
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...
    SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)

subjects = sorted({session.subject for session in session_refs})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.array([subject_to_idx[session.subject] for session in session_refs], dtype=np.int64),
```

iii. The notes explicitly describe each `jm*` directory as one mouse and say the dataset contains 6 subject folders, so the AI followed the directory naming convention directly.

## 1-c. How are the data split into sessions?

i. Each date-named subdirectory under a subject is treated as one session. The session ordering is deterministic because both subjects and session directories are sorted.

ii. 
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

iii. In the notes, the AI records that each subject folder contains 6 or 7 daily session folders named `YYYY-MM-DD_a` and that one session entry should be created per day.

## 1-d. How are the data split into trials?

i. The AI treats the continuous session as a sequence of consecutive 2-minute trial blocks after first averaging the data in non-overlapping 10-frame bins. Each trial therefore contains `120 s * 30 Hz / 10 = 360` bins. Any trailing partial block is discarded, and a session shorter than one full 2-minute block raises an error.

ii. 
```python
BIN_FRAMES = 10
TRIAL_SECONDS = 120.0

def split_trials(
    neural_binned: np.ndarray,
    time_binned_s: np.ndarray,
    output_one_hot: np.ndarray,
    fs: float,
    bin_frames: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
    usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
    if usable_bins < bins_per_trial:
        raise ValueError("Session is too short to form even one 2-minute trial.")
    ...
    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
        input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
        output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes and trajectory are explicit here. In Step 5 the AI says “Define each trial as one consecutive 2-minute block from a continuous session,” and trajectory step 73 says it chose 2-minute blocks because that matches the paper’s decoding split.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a trial-quality filter based on neural or behavioral quality. The only effective filtering is structural: it drops incomplete trailing bins/blocks, rejects sessions too short to make one 2-minute block, and validates that each session has at least 2 trials with matching dimensions and no NaNs.

ii. 
```python
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")

if n_trials < 2:
    raise ValueError("Each converted session must contain at least 2 trials.")
...
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The notes say there is no native trial structure and focus on creating decoder-compatible blocks rather than quality-based trial exclusion. The validator-driven checks were added to satisfy format requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`, using parameters from `ops.npy` to drive preprocessing.

ii. 
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
...
Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
```

iii. In Step 5 of the notes, the AI says it should reconstruct a paper-consistent fluorescence representation from `F` and `Fneu`, using Suite2p defaults from `ops.npy`.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction and then `suite2p.extraction.dcnv.preprocess` with baseline settings taken from `ops.npy`. After that, it performs additional non-overlapping 10-frame averaging before trialization.

ii. 
```python
def compute_fluorescence_signal(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    """Approximate the paper's Suite2p-based dF/F signal."""
    Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
    processed = suite2p_preprocess(
        Fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        batch_size=min(512, max(32, Fc.shape[0])),
        device=torch.device("cpu"),
    )
    return processed.astype(np.float32, copy=False)

neural_processed = compute_fluorescence_signal(F, Fneu, ops)
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The notes say the paper used baseline-corrected fluorescence and 10-frame averaging for decoding, and trajectory step 76 says the script will “mirror the paper’s preprocessing first” and then use “2-minute blocks and 10-frame bins.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not re-filter neurons inside `convert_data.py`. It assumes the provided rows are already curated and already restricted to neurons tracked across all days, then keeps every row in `F.npy`/`Fneu.npy`.

ii. 
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
...
converted = {
    "neural": neural_trials,
    "input": input_trials,
    "output": output_trials,
    "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
    "summary": summary,
}
```

iii. The notes repeatedly justify this by saying the provided `suite2p` exports are already post-Track2p tracked-cell datasets and that constant neuron counts across days indicate all-day matched rows were already baked in.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI effectively aligns neural data to the start of each consecutive 2-minute block by segmenting the binned session into fixed windows. Its metadata explicitly labels the alignment event as the start of each 2-minute recording block.

ii. 
```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    ...

"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute recording block",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
    ...
}
```

iii. The notes justify this choice by mapping the paper’s decoding blocks into the trial-based validator format. The AI explicitly says it chose 2-minute block starts as the trial event because the source recordings have no native stimulus events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins, so the final temporal resolution is `10 / 30 s = 333.3 ms`. Yes, non-overlapping temporal rebinning is applied to neural and behavioral streams.

ii. 
```python
BIN_FRAMES = 10
...
def average_nonoverlapping(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_frames = x.shape[-1]
    usable = (n_frames // bin_frames) * bin_frames
    ...
    return x.reshape(new_shape).mean(axis=-1)

"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
```

iii. The notes cite the paper’s decoder using “10 consecutive timestamps” and record the key decision “Apply non-overlapping 10-frame averaging before trialization.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is derived from the number of binned time steps plus the imaging frame rate in `ops["fs"]`. It is not taken from a dedicated timestamp file.

ii. 
```python
fs = float(ops["fs"])
...
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

iii. In the notes, the AI says it interprets “time elapsed from the beginning of the experiment” as time from the beginning of the recording session, sampled on the same binned grid used for decoding.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI first rebins the session into 10-frame bins, then computes a monotonic time vector in seconds using `np.arange(n_bins) * (bin_frames / fs)`. The resulting values are absolute times within the session, not times reset within each block.

ii. 
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
...
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

iii. The notes say the AI wanted the time input to live on the same 10-frame grid as neural and behavioral data and to mean elapsed time from session start.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is aligned by construction: it is generated at the same binned resolution as the neural data and then sliced with the same `start:stop` trial boundaries used for neural trials.

ii. 
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
neural_trials, input_trials, output_trials, trial_info = split_trials(
    neural_binned=neural_binned,
    time_binned_s=time_binned_s,
    output_one_hot=output_one_hot,
    fs=fs,
    bin_frames=BIN_FRAMES,
)
...
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. The notes frame this as a decoder-format alignment choice: once all signals are on the same 10-frame grid, the same block boundaries are applied to neural, input, and output.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion output from `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`. It does not use `interframe_int.npy`.

ii. 
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
...
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
```

iii. The notes say the critical choice is to “use `tstamps.npy` to map behavior samples onto the imaging-frame grid,” because some sessions have missing camera frames and behavior must be explicitly aligned to imaging.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI reconstructs a full per-imaging-frame motion trace from `motion_energy_glob.npy` and `tstamps.npy`, averaging duplicate timestamp hits and linearly interpolating missing frames. It then 10-frame-averages the aligned trace and min-max normalizes it within session.

ii. 
```python
def reconstruct_motion_trace(
    motion_energy: np.ndarray, tstamps: np.ndarray, n_imaging_frames: int
) -> tuple[np.ndarray, dict]:
    ...
    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    ...
    np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
    np.add.at(counts, frame_idx, 1)
    valid = counts > 0
    full[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    ...
    if len(missing):
        full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)

motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
motion_binned_norm = normalize_motion(motion_binned)
```

iii. The notes justify this as a paper-inspired synchronization step plus a decoder-inspired 10-frame averaging step. They also explicitly say motion is “normalized within session” before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes session-wise quintile thresholds at the 20th, 40th, 60th, and 80th percentiles of the normalized motion trace. It converts each time bin to one of 5 classes with `np.searchsorted`, then expands that into 5 one-hot binary output channels.

ii. 
```python
def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges

"output_names": [f"motion_energy_q{i}" for i in range(N_OUTPUT_BINS)],
"output_values": [[f"not_q{i}", f"q{i}"] for i in range(N_OUTPUT_BINS)],
```

iii. The notes say the task required discretized motion-energy bins, while trajectory steps 80 and 82 say the AI chose one-hot outputs because the provided decoder handled multi-class targets through binary one-hot channels.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data by reconstructing the motion trace directly on the imaging-frame grid of length `F.shape[1]`, then applying the same 10-frame binning and the same trial boundaries used for neural data.

ii. 
```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
...
neural_trials, input_trials, output_trials, trial_info = split_trials(
    neural_binned=neural_binned,
    time_binned_s=time_binned_s,
    output_one_hot=output_one_hot,
    fs=fs,
    bin_frames=BIN_FRAMES,
)
```

iii. The notes treat imaging-frame alignment as the key synchronization target because the source video and calcium recordings are nominally synchronous but can differ due to dropped camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or irregular motion frames by reconstructing onto a full imaging grid and linearly interpolating gaps. It also checks for malformed motion arrays, invalid timestamp scales, sessions too short to form a block, mismatched trial dimensions, and NaNs. Trailing raw frames or binned time steps that do not fit evenly into the chosen bin/block sizes are discarded.

ii. 
```python
if motion_energy.ndim != 1 or tstamps.ndim != 1:
    raise ValueError("Motion-energy inputs must be 1D arrays.")
if len(motion_energy) != len(tstamps):
    raise ValueError("motion_energy and tstamps must have the same length.")
...
if not np.any(valid):
    raise ValueError("No valid motion frames after alignment.")
...
if len(missing):
    full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)

usable = (n_frames // bin_frames) * bin_frames
...
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")
```

iii. The notes say the data README recommended treating missing camera frames as missing or interpolated, and the AI adopted interpolation plus aggressive validation so format errors would fail loudly instead of silently propagating.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s slowest substantive step is per-session Suite2p fluorescence preprocessing (`suite2p_preprocess`). The remaining work is mostly vectorized NumPy operations, plus optional plotting if `--show-processing` is used.

ii. 
```python
processed = suite2p_preprocess(
    Fc.copy(),
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=min(512, max(32, Fc.shape[0])),
    device=torch.device("cpu"),
)
...
elapsed = time.time() - start_time
summary = {
    ...
    "processing_seconds": float(elapsed),
}
```

iii. In the notes, the AI identifies fluorescence preprocessing as the expensive part and reports that the other major operations were already vectorized enough that full conversion time was acceptable.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy motion-reconstruction and binning operations are already vectorized. The main remaining Python loops are the loop that appends trial slices in `split_trials`, the per-session loop in `main`, and the per-trial validation loop in `validate_converted_session`.

ii. 
```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))

for idx, session in enumerate(sessions, start=1):
    ...
    converted, summary = process_session(session, show_processing=do_plot)

for trial_idx in range(n_trials):
    ...
```

iii. The notes explicitly say “Vectorized motion-frame reconstruction” and “Vectorized non-overlapping bin averaging” were deliberate speedups, so the AI treated the remaining loops as acceptable overhead rather than the main bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts arrays to `float32`/`int64`, repeatedly slices trial windows out of already processed session arrays, and then scans those trial arrays again during validation. Optional plotting also reuses several processed arrays to generate diagnostics.

ii. 
```python
Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
...
motion_energy = motion_energy.astype(np.float32, copy=False)
tstamps = tstamps.astype(np.float64, copy=False)
...
neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
...
for trial_idx in range(n_trials):
    ...
```

iii. The notes do not flag any major repeated scientific computation beyond validation/diagnostic work, which suggests the AI believed most meaningful processing was already done once per session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs optional diagnostic plotting and computes some intermediate quantities mainly for summaries or plots, such as `motion_classes`, `motion_edges`, histogram-ready normalized motion, and `session_summaries` used for console reporting. Those are not needed by downstream decoder training itself.

ii. 
```python
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
...
if show_processing:
    plot_processing_summary(
        ...
        motion_binned_norm=motion_binned_norm,
        motion_classes=motion_classes,
        ...
        motion_edges=motion_edges,
        ...
    )
...
session_summaries.append(summary)
...
mean_time = np.mean([summary["processing_seconds"] for summary in session_summaries])
est_full = mean_time * 41
```

iii. The notes say these additions were for sanity checks, plotting, and runtime estimation, not because the decoder format itself required them.
