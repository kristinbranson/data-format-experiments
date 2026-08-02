# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from each session's `suite2p/plane0/` directory (F.npy, Fneu.npy, ops.npy, iscell.npy) and `move_deve/` directory (motion_energy_glob.npy, tstamps.npy, interframe_int.npy). Subjects are selected from a hardcoded list of 6 mice. Sessions are found as sorted subdirectories whose names begin with 4 digits.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The AI loads all standard suite2p outputs plus motion energy files. It additionally loads `ops.npy` to read preprocessing parameters and `iscell.npy` to verify cell quality, as well as `tstamps.npy` for timing metadata. The CONVERSION_NOTES.md states: "Input neural data came from F.npy and Fneu.npy" and "Camera timing metadata came from move_deve/tstamps.npy and move_deve/interframe_int.npy."

## 1-b. How are the data split into subjects?

i. Subjects are defined via a hardcoded list: `["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]`, rather than dynamically discovering directories.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]
# ...
subjects = sorted({record.subject for record in session_records})
```

iii. The AI hardcodes subject names based on what it found in the data directory. The CONVERSION_NOTES.md lists all 6 subjects with their neuron counts.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, filtered to those whose names start with 4 digits, and sorted alphabetically.

ii.
```python
sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The digit-prefix filter ensures only date-named session directories are included, excluding any non-session subdirectories.

## 1-d. How are the data split into trials?

i. Trials are consecutive 2-minute (120-second) non-overlapping blocks after 10-frame temporal binning. At 30 Hz with 10-frame bins, each trial has 360 binned timepoints. Remainder frames are discarded.

ii.
```python
def split_into_trials(neural_binned, motion_binned, fs, frame_bin, trial_duration_s):
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))
    n_complete_trials = neural_binned.shape[1] // bins_per_trial
    # ...
    for trial_idx in range(n_complete_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
```
Called with `trial_duration_s=120.0` and `frame_bin=10`.

iii. The AI justifies this based on the paper's statement: "splits were done on consecutive 2 minute blocks of the recording." The CONVERSION_NOTES.md says: "Each continuous session was then segmented into consecutive 2-minute blocks."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks are retained.

ii. N/A

iii. There is no mention of trial filtering in the CONVERSION_NOTES.md or the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. Processing parameters are read from `ops.npy`.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

iii. CONVERSION_NOTES.md: "Input neural data came from F.npy and Fneu.npy in each session's suite2p/plane0 folder."

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) neuropil subtraction with coefficient from ops.npy (0.7), (2) baseline correction via suite2p's `dcnv.preprocess` using maximin method with parameters from ops.npy, (3) temporal averaging in non-overlapping bins of 10 frames.

ii.
```python
Fc = F.astype(np.float32, copy=False) - neucoeff * Fneu.astype(np.float32, copy=False)
# ...
return suite2p_preprocess(
    Fc.copy(), baseline=baseline, win_baseline=win_baseline,
    sig_baseline=sig_baseline, fs=fs, prctile_baseline=prctile_baseline,
    batch_size=128, device=device,
)
# ...
neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md: "Neural traces were processed as Suite2p-style baseline-corrected fluorescence using each session's ops.npy parameters." And: "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" (quoting the paper).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI loads `iscell.npy` and asserts that all stored ROIs have probability >= 0.5, but does not actually filter any neurons out. All neurons in `F.npy` are retained.

ii.
```python
iscell = np.load(suite2p_dir / "iscell.npy")
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. CONVERSION_NOTES.md: "All inspected iscell.npy files had probabilities above 0.5 for every stored ROI, matching the paper's stated iscell > 0.5 threshold." And: "No extra cell filtering was applied beyond what is already encoded in the Track2p export."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block. Since the recording is continuous and trials are contiguous segments, this is equivalent to alignment to session start.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block cut from a continuous session",
"off_start": 0.0,
"off_end": float(trial_duration_s),
```

iii. CONVERSION_NOTES.md: "Each continuous session was then segmented into consecutive 2-minute blocks."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal averaging, reducing the resolution from 30 Hz (33.33 ms bins) to 3 Hz (333.33 ms bins).

ii.
```python
def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
# ...
neural_binned = average_nonoverlapping(neural, frame_bin)  # frame_bin=10
```
Metadata: `"time_bin_size": float((frame_bin / 30.0) * 1000.0)` = 333.33 ms

iii. CONVERSION_NOTES.md: "Both neural and motion traces were averaged in non-overlapping bins of 10 imaging frames. At 30 Hz, that yields a time bin size of 10 / 30 = 0.333... s or 333.333... ms." The AI cites the paper: "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed synthetically from the bin indices and the known frame rate / bin size.

ii.
```python
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
```

iii. Since the frame rate is constant at 30 Hz and the bin size is 10 frames, time is computed as bin centers. No explicit timestamp file is used for the input.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `(bin_index + 0.5) * time_bin_s`, where `time_bin_s = frame_bin / fs = 10/30 s`. The `+0.5` places the time value at the center of each bin. The time axis spans the entire session, so each trial's input slice gives time from session start.

ii.
```python
time_bin_s = frame_bin / fs  # 10/30 = 0.3333s
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
# ...
input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. CONVERSION_NOTES.md: "Decoder input is time_from_session_start_s, represented as a time-varying trace sampled at bin centers."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time axis is computed over the same binned timepoints as the neural data. When trials are extracted, the same start:end slice is used for both neural and input arrays, ensuring perfect alignment.

ii.
```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end])
    input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. Alignment is guaranteed by using the same indices for slicing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy`. Frame timing metadata from `interframe_int.npy` (and `tstamps.npy`) is used to detect dropped camera frames.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. CONVERSION_NOTES.md: "Motion output came from move_deve/motion_energy_glob.npy. Camera timing metadata came from move_deve/tstamps.npy and move_deve/interframe_int.npy."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Dropped frame repair — if motion trace is shorter than imaging frames, NaN values are inserted at detected camera gaps and linearly interpolated. (2) 10-frame temporal averaging. (3) Per-session min-max normalization. (4) Global discretization into 5 quintile bins.

ii.
```python
# Dropped frame repair
gap_idx = detect_gap_indices(interframe_int)  # interframe_int > 1.5 * median
repaired = np.insert(repaired, insert_at, np.nan)
repaired = interpolate_nans_1d(repaired)

# 10-frame averaging
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0]

# Per-session min-max normalization
def minmax_normalize(x):
    return (x - x_min) / (x_max - x_min)

# Global quintile discretization
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(trial, global_edges, right=False)
```

iii. CONVERSION_NOTES.md: "Decoder output is motion energy after: per-session min-max normalization applied after 10-frame averaging; discretization into 5 equal-frequency bins using global quintiles within the exported dataset."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using global quintile edges computed from all sessions. Edges are at the 20th, 40th, 60th, and 80th percentiles of the concatenated min-max-normalized motion energy. `np.digitize` maps values to bins 0-4.

ii.
```python
all_motion_concat = np.concatenate(all_motion, axis=0)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
```

iii. CONVERSION_NOTES.md: "Output class fractions are exactly balanced at 0.2 for each of the five motion bins by construction."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy is first aligned to imaging frame count by detecting camera gaps (interframe intervals > 1.5 * median) and inserting interpolated values at gap positions. After alignment, both neural and motion traces undergo the same 10-frame averaging and trial splitting, using identical indices.

ii.
```python
def detect_gap_indices(interframe_int):
    median_dt = float(np.median(interframe_int))
    return np.flatnonzero(interframe_int > 1.5 * median_dt)

motion_aligned, motion_info = align_motion_to_imaging(
    motion_energy=motion_energy, tstamps=tstamps,
    interframe_int=interframe_int, target_frames=neural.shape[1],
)
# Then same binning and trial splitting as neural
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0]
```

iii. CONVERSION_NOTES.md: "I only repaired sessions where len(motion_energy_glob.npy) < n_imaging_frames... For those short sessions, I detected large timing gaps with interframe_int > 1.5 * median(interframe_int), inserted one missing motion sample at each detected gap, then linearly interpolated over the missing values."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three strategies: (1) Dropped camera frames are detected and interpolated (see 4-d). (2) If motion trace is longer than imaging frames, it is trimmed to match. (3) Remainder frames that don't fill a complete trial are discarded. (4) The iscell assertion catches any unexpected data quality issues.

ii.
```python
# Trim excess motion frames
if diff < 0:
    return motion_energy[:target_frames], info

# Discard remainder
n_complete_trials = neural_binned.shape[1] // bins_per_trial
neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]

# iscell assertion
if np.any(iscell_prob < 0.5):
    raise ValueError(...)
```

iii. CONVERSION_NOTES.md: "Total repaired motion positions in the full dataset: 276." The AI documents the discrepancy between stated 20-minute sessions and actual 30-minute sessions for some mice.

## 6-a. What are the most time-consuming steps of the code?

i. The suite2p `dcnv.preprocess` baseline correction is the most computationally intensive step, involving sliding window operations over all neurons for the full session length. The code also includes a fallback pure-NumPy/SciPy implementation that would be slower without GPU acceleration.

ii.
```python
if suite2p_preprocess is not None:
    return suite2p_preprocess(Fc.copy(), baseline=baseline, ...)
# Fallback:
Flow = gaussian_filter(Fc, [0.0, sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
```

iii. No explicit justification provided. The use of GPU acceleration via torch device mitigates this.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame insertion loop iterates one gap at a time, calling `np.insert` which reallocates the array each iteration. This could be vectorized by pre-computing all insertion positions and building the output array in one pass.

ii.
```python
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan)
    offset += 1
```

iii. No explicit justification. The number of gaps is typically small (276 total across 41 sessions), so the impact is minimal.

## 6-c. What processing does the code repeat multiple times?

i. The output processing iterates over trials multiple times: once for concatenating continuous motion values, once for discretization, and once more for computing per-trial summary statistics. The session records are also iterated multiple times in `build_dataset`.

ii.
```python
# First pass: collect all motion
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))

# Second pass: discretize
for record in session_records:
    for trial in record.output_continuous_trials:
        bins = np.digitize(trial, global_edges, right=False)

# Third pass: summary stats
for trial in record.output_continuous_trials:
    normalized_trials.append({"min": ..., "max": ..., "mean": ...})
```

iii. No justification provided. The repeated iteration is for clarity rather than efficiency.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` but does not use it for alignment (only `interframe_int.npy` is used for gap detection). The per-session min-max normalization is applied before global quintile discretization; since quintile binning is rank-based, normalization within each session before global pooling could distort the global distribution compared to normalizing globally or not normalizing at all. The detailed `session_info` metadata and motion trial summaries are computed but not used by the decoder.

ii.
```python
tstamps = np.load(move_dir / "tstamps.npy")  # loaded but not used in gap detection
# ...
normalize_session_outputs_in_place(session_records)  # min-max before discretization
# ...
info["motion_trial_summary_before_discretization"] = normalized_trials[:2]  # metadata only
```

iii. No explicit justification for loading tstamps.npy. The CONVERSION_NOTES.md describes min-max normalization as part of the pipeline but doesn't justify why it's done before global quintile binning.
