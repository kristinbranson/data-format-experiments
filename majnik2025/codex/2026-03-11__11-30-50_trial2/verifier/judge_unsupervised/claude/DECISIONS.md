# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all session directories by iterating over subject folders (`jm*`) under `data/`, then iterating over date-stamped session subfolders. For each session, it loads 5 files: `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. All sessions are processed sequentially in a loop.

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

# In process_session:
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md Step 2 that data is organized as subject folders containing session folders, each with `suite2p/plane0/` and `move_deve/` subdirectories. The AI confirmed the data README describes this structure and that the provided suite2p traces already contain only tracked cells present across all days for a given mouse.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the top-level directory names under `data/` (e.g., `jm031`, `jm032`, etc.). A sorted unique list of subject names is created, and each session is mapped to its subject via `subject_to_idx`.

ii.
```python
subjects = sorted({session.subject for session in session_refs})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
dataset = {
    ...
    "subjects": subjects,
    "subject_idx": np.array([subject_to_idx[session.subject] for session in session_refs], dtype=np.int64),
    ...
}
```

iii. The AI noted in Step 2 that there are 6 subject folders (`jm031` through `jm046`) and that each subject's sessions share the same tracked neuron count across days, confirming the subject-level data organization.

## 1-c. How are the data split into sessions?

i. Each date-stamped subdirectory within a subject folder is treated as one session. Sessions are sorted alphabetically (which corresponds to chronological order). All 41 sessions across 6 mice are included.

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

iii. From CONVERSION_NOTES Step 2: subjects have 6 or 7 sessions each (jm040 has 6, others have 7), for a total of 41 sessions. This matches the paper's statement of "minimum of 6 consecutive days."

## 1-d. How are the data split into trials?

i. There are no native trials in the source data (continuous spontaneous-behavior recordings). The AI creates artificial trials by splitting each session into consecutive non-overlapping 2-minute blocks, matching the paper's decoding approach. This yields 360 time bins per trial (120 seconds / 0.333 s per bin). 20-minute sessions produce 10 trials and 30-minute sessions produce 15 trials. Leftover frames that don't fill a complete 2-minute block are discarded.

ii.
```python
TRIAL_SECONDS = 120.0

def split_trials(...):
    bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))  # = 360
    usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
    ...
    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop])
        input_trials.append(time_binned_s[np.newaxis, start:stop])
        output_trials.append(output_one_hot[:, start:stop])
```

iii. From CONVERSION_NOTES Step 5: "Define each trial as one consecutive 2-minute block from a continuous session, matching the paper's decoding split unit exactly." The paper uses "consecutive 2-minute blocks" for its decoding analysis.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied beyond discarding leftover frames that don't complete a full 2-minute block. All complete 2-minute blocks are kept. The AI validates that no NaN values exist in any converted trial and that each session has at least 2 trials.

ii.
```python
def validate_converted_session(converted: dict) -> None:
    ...
    if n_trials < 2:
        raise ValueError("Each converted session must contain at least 2 trials.")
    ...
    if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
        raise ValueError("Converted arrays must not contain NaN values.")
```

iii. From CONVERSION_NOTES Step 3: "No native trials are defined in the source dataset or paper for this analysis." Since the recordings are continuous spontaneous behavior without stimulus-locked events, there is no trial-level quality criterion to apply.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from three raw files per session: `F.npy` (raw fluorescence traces), `Fneu.npy` (neuropil fluorescence traces), and `ops.npy` (Suite2p processing parameters including `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `fs`, and `prctile_baseline`).

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

iii. The AI documented in CONVERSION_NOTES Step 1 that the core source data is suite2p-style per-plane files and that the paper states downstream analyses used "baseline-corrected fluorescence traces treated as dF/F."

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps: (1) neuropil subtraction using `F - neucoeff * Fneu` followed by Suite2p's baseline correction via `suite2p.extraction.dcnv.preprocess`, and (2) non-overlapping averaging in bins of 10 consecutive frames.

ii.
```python
def compute_fluorescence_signal(F, Fneu, ops):
    Fc = F.astype(np.float32) - float(ops["neucoeff"]) * Fneu.astype(np.float32)
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
    return processed.astype(np.float32)

# Then binning:
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES)
```

iii. From CONVERSION_NOTES Step 5: "Use neuropil-subtracted, baseline-corrected fluorescence derived from F and Fneu with Suite2p defaults from ops.npy, because the paper states downstream analyses used baseline-corrected fluorescence as dF/F."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural data filtering is applied at the conversion stage. The AI relies on the fact that the provided data has already been curated through Track2p: Suite2p ROIs with classifier probability > 0.5 have been retained, and only neurons tracked across all days for each subject are included in the exported files.

ii. No explicit filtering code in `convert_data.py`. All rows in `F.npy` are used:
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
# F.shape[0] neurons are all used directly
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
```

iii. From CONVERSION_NOTES Step 4: "Treat provided rows as already curated tracked cells; still confirm all rows pass iscell > 0.5." Step 1 notes: "The default curation rule is to keep ROIs with iscell[:,1] > 0.5."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus event to align to. Trials are defined as consecutive 2-minute blocks starting from the beginning of the session. The temporal alignment event is "start of each consecutive 2-minute recording block." Neural data is split into these blocks after binning.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute recording block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),  # 120.0
```

iii. From CONVERSION_NOTES Step 5: "Interpret 'time elapsed from the beginning of the experiment' as time from the beginning of the recording session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has a time bin size of 333.3 ms (10 frames at 30 Hz). Non-overlapping averaging of 10 consecutive frames is applied to both neural and behavioral data.

ii.
```python
BIN_FRAMES = 10
# In metadata:
"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),  # = 333.33 ms

def average_nonoverlapping(x, bin_frames):
    n_frames = x.shape[-1]
    usable = (n_frames // bin_frames) * bin_frames
    x = x[..., :usable]
    new_shape = x.shape[:-1] + (usable // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
```

iii. From CONVERSION_NOTES Step 3: "Decoding analysis uses 10-frame averages (333.3 ms)" matching the paper's "averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data file directly. It is computed from the frame index, the sampling rate (`ops['fs']` = 30 Hz), and the bin size (10 frames). It represents elapsed time from the start of the session.

ii.
```python
def make_time_input(n_bins, fs, bin_frames):
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)

time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
```

iii. From CONVERSION_NOTES Step 5: "Single input variable named time_from_session_start_s." The paper's decoder does not use time as an explicit input, but the task instructions require it.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `bin_index * (bin_frames / fs)`, where `bin_frames = 10` and `fs = 30`. This gives time in seconds from the session start for each binned time point. The time input increases monotonically across trials within a session (trial 1: 0-119.7s, trial 2: 120-239.7s, etc.).

ii.
```python
def make_time_input(n_bins, fs, bin_frames):
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)

# Then split into trials maintaining absolute time:
input_trials.append(time_binned_s[np.newaxis, start:stop])
```

iii. The instructions specify "Time elapsed from the beginning of the experiment" as the decoder input. The AI interpreted "experiment" as the recording session and maintained absolute session time across trials.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is derived from the same binned time axis as the neural data (same number of bins, same trial boundaries), so alignment is guaranteed by construction. Both neural and time data are split at the same trial boundaries.

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
```

iii. The alignment is trivial since the time input is constructed from the bin indices of the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (global motion energy values) and `tstamps.npy` (timestamps for alignment to imaging frames), both from the `move_deve/` subdirectory.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. From CONVERSION_NOTES Step 2: "move_deve/ with motion_energy_glob.npy, tstamps.npy, interframe_int.npy." The paper describes motion energy as computed from "pixel-wise differences" between consecutive video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Align motion energy samples onto the imaging frame grid using timestamps, averaging multiple samples per frame and interpolating missing frames. (2) Average in non-overlapping 10-frame bins. (3) Normalize within session using min-max scaling. (4) Discretize into 5 equal-percentile (quintile) bins using `np.quantile` at [0.2, 0.4, 0.6, 0.8]. (5) One-hot encode into 5 binary channels.

ii.
```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)

def quintile_one_hot(x):
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. From CONVERSION_NOTES Step 5: "Reconstruct motion on full imaging-frame grid using timestamps, linearly interpolate missing frames, average in 10-frame bins, normalize within session, discretize into 5 equal-percentile bins, then one-hot encode into 5 binary channels."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized using session-wise quintiles. After min-max normalization within each session, the 20th, 40th, 60th, and 80th percentile values are computed. `np.searchsorted` assigns each time bin to one of 5 categories (0-4). The result is then one-hot encoded into 5 binary channels, each of shape `(1, n_timepoints)`, stacked to `(5, n_timepoints)`.

ii.
```python
def quintile_one_hot(x):
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. The instructions say "normalized and discretized into five equal-percentile bins." The AI chose to represent this as 5 one-hot binary channels rather than a single categorical variable with values 0-4.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first aligned to the imaging frame grid (same length as F.npy) using timestamps, then binned with the same 10-frame bins as neural data, and finally split into the same 2-minute trial blocks. This ensures temporal alignment between neural and output data.

ii.
```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
# F.shape[1] = n_imaging_frames, so motion is aligned to the imaging timeline
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
# Same binning as neural data, then same trial splitting in split_trials()
```

iii. From CONVERSION_NOTES Step 5: "Behavior alignment: Use tstamps.npy to map behavior samples onto the imaging-frame grid of length n_imaging_frames; missing camera frames become NaNs that are then interpolated."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data quality issue is missing motion energy frames in 9 of 41 sessions (mismatches between motion energy length and imaging frame count, ranging from 1 to 148 missing frames). The AI handles this by: (1) mapping motion energy samples onto the imaging frame grid using timestamps, (2) averaging multiple samples that map to the same frame, and (3) linearly interpolating any frames with no motion data. Tail frames that don't fill a complete 2-minute trial are silently discarded.

ii.
```python
def reconstruct_motion_trace(motion_energy, tstamps, n_imaging_frames):
    ...
    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)
    full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
    ...
    np.add.at(sums, frame_idx, motion_energy)
    np.add.at(counts, frame_idx, 1)
    valid = counts > 0
    full[valid] = (sums[valid] / counts[valid])
    ...
    missing = np.flatnonzero(~valid)
    if len(missing):
        full[missing] = np.interp(missing, valid_idx, full[valid_idx])
    ...
```

iii. From CONVERSION_NOTES Step 4: "9 sessions have missing behavior frames relative to imaging. Data README says missing camera frames should be treated as missing or interpolated." The data README explicitly recommends interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The Suite2p baseline correction (`suite2p_preprocess`) is the most computationally expensive step per session. The full conversion takes about 57 seconds for 41 sessions (mean 1.38s per session), with sessions containing more neurons taking longer (e.g., jm039 with 746 neurons takes ~2.1s vs jm031 with 221 neurons at ~0.45s).

ii.
```python
processed = suite2p_preprocess(
    Fc.copy(),
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    ...
)
```

iii. From CONVERSION_NOTES Step 7: "Mean processing time per session: 2.26 s" (sample), and full conversion completed in 57.35 seconds. The AI identified Suite2p preprocessing as the potential bottleneck but found it fast enough to not require optimization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main session processing loop iterates sequentially over 41 sessions. Each session is independent, so this loop could be parallelized (e.g., using `multiprocessing`). Within sessions, the trial splitting loop (`for start in range(0, usable_bins, bins_per_trial)`) creates list slices sequentially, though this is minor.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    converted, summary = process_session(session, show_processing=do_plot)
    ...

# Trial splitting loop:
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop])
    ...
```

iii. From CONVERSION_NOTES Step 6: "Need to benchmark Suite2p-style fluorescence preprocessing on sample data before deciding whether further optimization is required." The AI decided no optimization was needed since full conversion took under 1 minute.

## 6-c. What processing does the code repeat multiple times?

i. The code does not perform redundant repeated processing. Each session is processed once. The `average_nonoverlapping` function is called separately for neural data and motion data, but these operate on different arrays. The validation step (`validate_converted_session`) re-iterates over trials to check consistency, but this is a check, not processing.

ii.
```python
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES)
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The AI's code is reasonably non-redundant. No significant repeated computation was identified.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The min-max normalization step (`normalize_motion`) is arguably unnecessary because the quintile discretization is invariant to monotonic transformations - the percentile boundaries would yield the same class assignments with or without normalization. Additionally, the `motion_info` dictionary with detailed alignment statistics is computed but only stored in metadata, not used by downstream decoder training.

ii.
```python
motion_binned_norm = normalize_motion(motion_binned)  # unnecessary before quintile
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)

# motion_info computed but only stored:
info = {
    "n_motion_frames_raw": int(len(motion_energy)),
    "n_imaging_frames": int(n_imaging_frames),
    "n_missing_motion_frames": int((~valid).sum()),
    "motion_frame_dt": float(frame_dt),
}
```

iii. The normalization was a conscious design choice documented in Step 5, though mathematically redundant for percentile-based discretization. The metadata tracking is for documentation purposes.
