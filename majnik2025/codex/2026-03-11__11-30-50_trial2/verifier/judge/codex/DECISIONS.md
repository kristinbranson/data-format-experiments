# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for subject directories whose names start with `jm`, then scans each subject for date-like session directories. For each session it loads calcium data from `suite2p/plane0/F.npy`, `Fneu.npy`, and `ops.npy`, and behavioral data from `move_deve/motion_energy_glob.npy` and `tstamps.npy`. Trials are not loaded directly from disk; they are created later by splitting continuous recordings.

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
...
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says the provided `data/` folder is already a post-Track2p export, so it loads the exported suite2p traces directly and reconstructs the remaining decoder inputs/outputs from those files rather than rerunning tracking.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory name. Any directory under `data/` whose name begins with `jm` is treated as one mouse, and subject order is the sorted directory order.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

iii. The notes state that the dataset has 6 subject folders named `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and that each `jm*` directory is one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are split by subdirectory within each subject. The AI only keeps subject subdirectories whose names look like dates, storing each as one session.

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

iii. The notes justify this by saying each subject folder contains daily session folders named like `YYYY-MM-DD_a`, and each such directory is one daily recording.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous, bins them by 10 frames first, then creates artificial trials as consecutive non-overlapping 2-minute blocks. Each trial therefore has 360 bins.

ii.
```python
BIN_FRAMES = 10
TRIAL_SECONDS = 120.0
...
bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
...
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes say this choice was made to match the paper’s decoding analysis, which used consecutive 2-minute blocks from continuous recordings.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-quality filtering in the usual sense. Instead, it drops trailing bins that do not fill a complete 2-minute block and raises an error if a session is too short to form even one trial. It also validates that each converted session has at least two trials and no NaNs.

ii.
```python
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")
...
if n_trials < 2:
    raise ValueError("Each converted session must contain at least 2 trials.")
...
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The notes frame this as satisfying decoder-format constraints rather than scientific trial curation, because the source dataset has no native trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from suite2p fluorescence files `F.npy` and `Fneu.npy`, with processing parameters read from `ops.npy`.

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
...
Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
```

iii. The notes justify this as reconstructing a paper-consistent fluorescence signal from the exported suite2p traces rather than using raw `F` directly.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction using `ops["neucoeff"]`, applies `suite2p.extraction.dcnv.preprocess` with parameters taken from `ops.npy`, then averages the processed neural trace in non-overlapping 10-frame bins before trial splitting.

ii.
```python
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
...
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The notes say this was intended to mirror the paper’s Suite2p-style baseline-corrected fluorescence and the paper’s 10-frame averaging used for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any extra neural filtering in `convert_data.py`. It assumes the exported suite2p rows are already curated tracked cells and keeps all rows in `F.npy`.

ii.
```python
converted = {
    "neural": neural_trials,
    "input": input_trials,
    "output": output_trials,
    "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
    "summary": summary,
}
```

iii. The notes explicitly say reference `iscell > 0.5` filtering and all-day Track2p matching were already applied upstream in the provided exports, so no re-filtering was done.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of the artificial 2-minute block rather than to session start. Operationally, neural trials are just consecutive chunks from the continuous recording after 10-frame binning.

ii.
```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
...
"temporal_alignment_event": "start of each consecutive 2-minute recording block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The notes justify this by treating the paper’s consecutive 2-minute decoding blocks as the effective alignment unit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame bins at 30 Hz, so each time bin is 333.3 ms. The AI explicitly rebins both neural and behavioral data with non-overlapping averaging.

ii.
```python
BIN_FRAMES = 10
...
def average_nonoverlapping(x: np.ndarray, bin_frames: int) -> np.ndarray:
    ...
    return x.reshape(new_shape).mean(axis=-1)
...
"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
```

iii. The notes say this was chosen to match the paper’s decoder preprocessing, which averaged 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw time variable. It is synthesized from bin index, frame rate `fs`, and `BIN_FRAMES`.

ii.
```python
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

iii. The notes say the AI interpreted the required decoder input as elapsed time from the beginning of the recording session.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time as an absolute-within-session binned time vector in seconds, one value per 10-frame bin, then slices that vector into the same 2-minute trials used for neural/output data.

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
...
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. The notes justify this as carrying “time from the beginning of the recording session” through every trial rather than resetting time to zero within each block.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated at the same 10-frame-binned sampling as the neural data and sliced with identical trial boundaries.

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
if neural[trial_idx].shape[1] != input_[trial_idx].shape[1]:
    raise ValueError("Input time dimension does not match neural time dimension.")
```

iii. The notes report an `np.allclose()` sanity check confirming the reconstructed time vector exactly matched the converted input trial.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy from `move_deve/motion_energy_glob.npy` and uses `move_deve/tstamps.npy` to align those values onto the imaging frame grid.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
...
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
```

iii. The notes say the paper’s behavior variable is global motion energy and that `tstamps.npy` should be used to map motion samples to imaging frames in sessions with missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI maps motion samples to imaging-frame indices from timestamps, averages duplicate assignments, interpolates missing frames, averages the aligned trace in 10-frame bins, min-max normalizes within session, then discretizes into quintiles and one-hot encodes the result.

ii.
```python
frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
...
np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
np.add.at(counts, frame_idx, 1)
...
full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
...
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

iii. The notes justify this as a paper-like synchronized 30 Hz motion signal with interpolation for camera drops, plus the task-required conversion to categorical motion labels.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes session-wise 20/40/60/80% quantiles on the normalized motion trace, assigns each bin to one of five quintile classes, then represents those classes as five one-hot binary output channels.

ii.
```python
def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
...
"output_names": [f"motion_energy_q{i}" for i in range(N_OUTPUT_BINS)],
"output_values": [[f"not_q{i}", f"q{i}"] for i in range(N_OUTPUT_BINS)],
```

iii. The notes say the one-hot export was chosen because the provided decoder expects binary outputs for multi-class variables.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first reconstructed onto the full imaging frame grid, then temporally rebinned in the same 10-frame windows as the neural signal, and finally cut into the same 2-minute trial blocks.

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

iii. The notes say this follows synchronized imaging/video acquisition while explicitly interpolating missing camera frames before matching behavior to neural time bins.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing motion frames by detecting unfilled imaging-frame slots after timestamp mapping and filling them by linear interpolation. It also errors if the motion arrays are malformed, drops leftover bins that do not complete a trial, rejects sessions with too little data, and validates that no NaNs remain in converted trials.

ii.
```python
if len(motion_energy) != len(tstamps):
    raise ValueError("motion_energy and tstamps must have the same length.")
...
full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
...
missing = np.flatnonzero(~valid)
if len(missing):
    full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
...
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
...
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The notes highlight interpolation of missing camera frames as the main data-repair step and report explicit `np.allclose()` rechecks on neural, input, and output streams.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s main expensive step is Suite2p fluorescence preprocessing for every session. The session-level loop over all files is also significant, but the notes treat the neural preprocessing as the dominant cost.

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
for idx, session in enumerate(sessions, start=1):
    converted, summary = process_session(session, show_processing=do_plot)
```

iii. The notes explicitly say “Need to benchmark Suite2p-style fluorescence preprocessing” and later estimate runtime from per-session processing summaries.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the motion-frame reconstruction and 10-frame averaging. The main remaining explicit per-trial loop is the loop that slices binned arrays into trial lists.

ii.
```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes say motion reconstruction and binning were intentionally vectorized with `np.add.at`, `np.interp`, and reshape/mean, implying the remaining list-building loop was left as a simpler non-critical implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats full per-session preprocessing independently for every session, including fluorescence preprocessing, motion reconstruction, normalization, discretization, and validation. It also casts arrays to target dtypes multiple times while creating trial arrays.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    converted, summary = process_session(session, show_processing=do_plot)
    validate_converted_session(converted)
    converted_sessions.append(converted)
...
neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes describe the workflow as “preprocess all sessions” and then validate each converted session, with repeated dtype conversion and repeated session-local normalization/quintiling as part of that design.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extra diagnostic information not needed for downstream decoding, such as per-session summaries, motion-quintile edges, processing times, and optional plotting artifacts. It also computes `motion_classes` mainly for summaries/plots rather than for the exported dataset itself.

ii.
```python
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
...
summary = {
    "subject": session.subject,
    "session_id": session.session_id,
    ...
    "motion_quintile_edges": motion_edges.tolist(),
    "processing_seconds": float(elapsed),
}
converted = {
    "neural": neural_trials,
    "input": input_trials,
    "output": output_trials,
    "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
    "summary": summary,
}
```

iii. The notes emphasize validation and review, so these extra computations were part of the agent’s debugging/documentation workflow rather than the minimal data conversion needed by downstream analyses.
