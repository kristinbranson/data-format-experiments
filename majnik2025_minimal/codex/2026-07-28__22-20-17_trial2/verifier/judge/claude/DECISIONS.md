# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory (hardcoded list `FULL_SUBJECTS`). For each subject, session subdirectories are found by iterating and filtering for directories whose names start with 4 digits. Within each session, data is loaded from `suite2p/plane0/` (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`) and `move_deve/` (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`).

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]

def select_session_dirs(data_dir: Path, mode: str) -> list[Path]:
    ...
    for subject in subjects:
        subject_dir = data_dir / subject
        sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        ...
        session_dirs.extend(sessions)
    return session_dirs

def load_session(session_dir, ...):
    F = np.load(suite2p_dir / "F.npy")
    Fneu = np.load(suite2p_dir / "Fneu.npy")
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(suite2p_dir / "iscell.npy")
    motion_energy = np.load(move_dir / "motion_energy_glob.npy")
    tstamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The agent inspected the data directory structure and identified the subject/session hierarchy. It also loads `ops.npy` to read per-session processing parameters and `iscell.npy` for validation (though all ROIs already pass the iscell threshold). The agent loads `tstamps.npy` in addition to files used by the reference.

## 1-b. How are the data split into subjects?

i. Subjects are defined via a hardcoded list `FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]`. A sorted set of unique subject names from session records is used for the final subject list.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]
...
subjects = sorted({record.subject for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent identified the six subject directories by inspecting the data. The hardcoded list ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, filtered to those whose names start with 4 digits, and sorted alphabetically. Each subdirectory contains one recording session.

ii.
```python
sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The agent observed that session directories are named with date prefixes (e.g., `2023-10-22_a`), so filtering on leading digits and sorting gives chronological order.

## 1-d. How are the data split into trials?

i. The continuous recording is split into non-overlapping 2-minute (120-second) blocks. After 10-frame temporal binning, each trial has 360 time bins. Any remainder frames that don't fill a complete trial are discarded.

ii.
```python
def convert_dataset(..., trial_duration_s: float = 120.0):
    ...

def split_into_trials(neural_binned, motion_binned, fs, frame_bin, trial_duration_s):
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))
    n_complete_trials = neural_binned.shape[1] // bins_per_trial
    ...
    for trial_idx in range(n_complete_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:end])
        ...
```

iii. From CONVERSION_NOTES.md: "the paper states that ... cross-validation splits were based on consecutive 2-minute blocks. The conversion mirrors that structure directly." The agent chose 120s trials to match the paper's cross-validation block size.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks are included. Incomplete blocks at the end of a session are discarded.

ii. N/A (no filtering code beyond truncating incomplete trials)

iii. The agent did not identify any quality control criteria for trials in the reference paper or code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. Processing parameters are read from `ops.npy`.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

iii. These are standard suite2p output files. The agent also loads `ops.npy` to use per-session parameters for baseline correction rather than hardcoding them.

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) neuropil subtraction with coefficient from `ops.npy` (0.7), (2) suite2p maximin baseline correction via `dcnv.preprocess`, (3) temporal averaging in non-overlapping bins of 10 frames.

ii.
```python
def baseline_correct_fluorescence(F, Fneu, ops, device):
    neucoeff = float(ops.get("neucoeff", 0.7))
    Fc = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    if suite2p_preprocess is not None:
        return suite2p_preprocess(Fc.copy(), baseline=baseline, ...)
    ...

neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32)

def average_nonoverlapping(arr, bin_size):
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., :n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. From CONVERSION_NOTES.md: "the paper states that decoding used slightly denoised dF/F and behaviour traces, averaged in bins of 10 consecutive timestamps." The 10-frame averaging was taken directly from the paper's methods.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All ROIs in the suite2p output are included. The agent validates that all `iscell` probabilities are >= 0.5 but does not filter based on this.

ii.
```python
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. From CONVERSION_NOTES.md: "All inspected iscell.npy files had probabilities above 0.5 for every stored ROI... No extra cell filtering was applied beyond what is already encoded in the Track2p export."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. The continuous recording is segmented into consecutive 2-minute blocks starting from the beginning. The temporal alignment event is described as "start of each consecutive 2-minute block."

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block cut from a continuous session",
"off_start": 0.0,
"off_end": float(trial_duration_s),
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity, so alignment is simply to the session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is temporally rebinned by averaging 10 consecutive frames. At 30 Hz, this gives a time bin size of 333.33 ms (effective rate ~3 Hz).

ii.
```python
def convert_dataset(..., frame_bin: int = 10, ...):
    ...
neural_binned = average_nonoverlapping(neural, frame_bin)
...
"time_bin_size": float((frame_bin / 30.0) * 1000.0),  # 333.33 ms
```

iii. From CONVERSION_NOTES.md: "Both neural and motion traces were averaged in non-overlapping bins of 10 imaging frames. At 30 Hz, that yields a time bin size of 10 / 30 = 0.333... s or 333.333... ms." This was based on the paper's statement about denoising by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and the time bin size, representing the center of each time bin since session start.

ii.
```python
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
...
input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. Since frames are acquired at a constant rate and the bin size is known, time can be computed directly. The `+ 0.5` offset places the time at the center of each bin.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `(bin_index + 0.5) * time_bin_s` where `time_bin_s = frame_bin / fs = 10/30` seconds. This gives the center time of each bin relative to session start.

ii.
```python
time_bin_s = frame_bin / fs  # 10/30 = 0.3333 s
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
```

iii. Using bin centers rather than bin edges provides a more accurate time representation for each averaged data point.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time axis is computed from the same binned indices as the neural data, so alignment is inherent. Both share the same number of time bins per trial.

ii.
```python
# Both use the same start:end indices from the binned data
neural_trials.append(neural_binned[:, start:end])
input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. Since time is computed from the same bin indices, it is automatically aligned with the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. `interframe_int.npy` and `tstamps.npy` are also loaded for dropped frame detection and repair.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The motion energy file contains the pre-computed global motion energy from the behavioral video. The timing files are used to identify and repair dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) dropped frame repair by inserting NaN at detected gaps and interpolating, (2) 10-frame temporal averaging, (3) per-session min-max normalization, (4) global quintile discretization into 5 bins.

ii.
```python
# Step 1: Dropped frame repair
motion_aligned, motion_info = align_motion_to_imaging(motion_energy, tstamps, interframe_int, target_frames)

# Step 2: 10-frame averaging
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0]

# Step 3: Per-session min-max normalization
def normalize_session_outputs_in_place(session_records):
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials])
        normalized = minmax_normalize(concatenated)
        ...

# Step 4: Global quintile discretization
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(trial, global_edges, right=False)
```

iii. From CONVERSION_NOTES.md: "per-session min-max normalization applied after 10-frame averaging; discretization into 5 equal-frequency bins using global quintiles within the exported dataset."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using global quintile edges computed from all sessions. The edges are at the 20th, 40th, 60th, and 80th percentiles of the normalized pooled data. `np.digitize` with `right=False` maps values to bins 0-4.

ii.
```python
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins." Quintile edges ensure approximately equal bin counts. The `right=False` convention means values exactly at an edge go to the higher bin.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to imaging frames by detecting camera gaps (where `interframe_int > 1.5 * median(interframe_int)`) and inserting NaN values at gap positions, then interpolating. This is only done when the motion trace is shorter than the imaging trace. After alignment, both streams are binned identically in 10-frame windows.

ii.
```python
def detect_gap_indices(interframe_int):
    median_dt = float(np.median(interframe_int))
    return np.flatnonzero(interframe_int > 1.5 * median_dt)

def align_motion_to_imaging(motion_energy, tstamps, interframe_int, target_frames):
    ...
    if diff > 0:
        # insert NaN at detected gaps, then interpolate
        for pos in gap_positions:
            insert_at = int(pos + 1 + offset)
            repaired = np.insert(repaired, insert_at, np.nan)
            offset += 1
        repaired = interpolate_nans_1d(repaired)
    ...
```

iii. From CONVERSION_NOTES.md: "I only repaired sessions where len(motion_energy_glob.npy) < n_imaging_frames. This avoids overcorrecting sessions whose timestamps contain irregular intervals but whose motion trace already matches imaging length."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three strategies: (1) dropped camera frames are detected via timing gaps and repaired by NaN insertion + interpolation, (2) sessions where motion energy is longer than neural data are trimmed to match, (3) incomplete trials at the end of sessions are discarded. An assertion validates iscell probabilities.

ii.
```python
if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info
if diff > 0:
    info["length_fix_strategy"] = "insert_nan_at_detected_camera_gaps_then_interpolate"
    ...
# Incomplete trial handling
n_complete_trials = neural_binned.shape[1] // bins_per_trial
neural_binned = neural_binned[:, :n_complete_trials * bins_per_trial]
```

iii. The agent documented 276 total repaired motion positions across the full dataset. The repair strategy only activates when the motion trace is shorter than imaging, avoiding unnecessary modifications.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `dcnv.preprocess` baseline correction, which involves sliding window operations over the full recording for every neuron. When suite2p is not available, the fallback uses scipy's `gaussian_filter`, `minimum_filter1d`, and `maximum_filter1d`. Loading `.npy` files is I/O-bound but relatively fast.

ii.
```python
return suite2p_preprocess(Fc.copy(), baseline=baseline, win_baseline=win_baseline,
    sig_baseline=sig_baseline, fs=fs, prctile_baseline=prctile_baseline,
    batch_size=128, device=device)
```

iii. GPU acceleration mitigates the cost of baseline correction. The 10-frame binning and discretization are computationally light.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation loop uses `np.insert` in a loop, reallocating the array each iteration. The trial-splitting loop iterates over trials one at a time. The per-session normalization loop iterates over records and trials.

ii.
```python
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan)
    offset += 1
```

iii. The number of dropped frames is small (276 total across all sessions), so the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The output continuous trials are iterated over multiple times: once for per-session min-max normalization, once for concatenation for global edges, and once for discretization. The session info loop also re-iterates over output trials to compute summary statistics.

ii.
```python
# First pass: normalization
normalize_session_outputs_in_place(session_records)
# Second pass in build_dataset: concatenate for global edges
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))
# Third pass: discretization
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False)
# Fourth pass: summary stats
for trial in record.output_continuous_trials:
    normalized_trials.append({...})
```

iii. The multiple passes are a code organization choice prioritizing clarity over performance. The data volume is small enough that the overhead is negligible.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` and `iscell.npy` which are not strictly needed for the conversion (iscell is only used for validation, tstamps is loaded but not used in the gap detection logic which uses `interframe_int.npy`). The `summarize_dataset` function computes detailed statistics that are only printed, not used in the output. The per-session min-max normalization is an extra step that the reference does not perform. The code also has a fallback baseline correction implementation that duplicates suite2p's functionality.

ii.
```python
tstamps = np.load(move_dir / "tstamps.npy")  # loaded but only passed to align_motion_to_imaging
iscell = np.load(suite2p_dir / "iscell.npy")  # only used for validation check
```

iii. Loading extra files and computing summaries adds minimal overhead but increases code complexity. The fallback baseline correction ensures the code works without suite2p installed.
