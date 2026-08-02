# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` in `data/`, then discovers sessions as subdirectories with date-formatted names. For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. All sessions are processed individually via `process_session()`.

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

iii. The AI documented in CONVERSION_NOTES.md that the directory structure follows a standard convention with subject folders containing session subfolders. It loads the same suite2p files as the reference but additionally loads `ops.npy` for preprocessing parameters and `tstamps.npy` (instead of `interframe_int.npy`) for motion alignment.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data root, sorted alphabetically. A unique sorted set of subject names is built and used for `subject_idx` mapping.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...
# In build_dataset:
subjects = sorted({session.subject for session in session_refs})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI identified 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046) matching the paper's "full dataset of 6 mice."

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, filtered to those with date-formatted names (first 4 chars are digits, length >= 10), sorted alphabetically.

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
```

iii. Each subdirectory corresponds to one daily recording session. The date-format filter is an additional safeguard compared to the reference (which accepts all subdirectories).

## 1-d. How are the data split into trials?

i. Trials are consecutive non-overlapping 2-minute (120-second) blocks of the continuous recording, after 10-frame temporal binning. Each trial contains 360 time bins (120s * 30Hz / 10 frames = 360 bins). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_SECONDS = 120.0
BIN_FRAMES = 10

def split_trials(...):
    bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))  # 360
    usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
    ...
    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop])
```

iii. The AI chose 2-minute trials based on the paper's decoding methodology: "the paper uses consecutive 2-minute blocks." The 10-frame binning follows the paper's "averaging in bins of 10 consecutive timestamps."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied beyond discarding incomplete trailing blocks. The AI validates that each session produces at least 2 trials and checks for NaN values.

ii.
```python
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")
# In validate_converted_session:
if n_trials < 2:
    raise ValueError("Each converted session must contain at least 2 trials.")
if np.isnan(neural[trial_idx]).any() or ...:
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The AI noted there is no native trial structure and no trial-level quality criteria described in the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (suite2p parameters) from `suite2p/plane0/`.

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

iii. The AI noted these are standard suite2p output files. It loads `ops.npy` to extract preprocessing parameters rather than hardcoding them.

## 2-b. How is the `neural` data processed?

i. Three-step processing: (1) neuropil subtraction (`Fc = F - neucoeff * Fneu`) using `neucoeff` from `ops.npy`, (2) suite2p baseline correction via `suite2p.extraction.dcnv.preprocess` with parameters from `ops.npy`, (3) non-overlapping 10-frame temporal averaging.

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
        ...
    )
    return processed.astype(np.float32)

# Then binned:
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES)
```

iii. The AI justified reading parameters from `ops.npy` as being more robust than hardcoding. The 10-frame averaging matches the paper's decoding preprocessing ("averaging in bins of 10 consecutive timestamps").

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. All neurons present in the suite2p `F.npy` are included.

ii. N/A (no filtering code)

iii. The AI documented that the provided data files are already post-Track2p exports with all-day matched neurons, so the curation (iscell > 0.5 + cross-day tracking) has already been applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block within the session. There is no stimulus event to align to since recordings are of spontaneous behavior.

ii.
```python
'temporal_alignment_event': 'start of each consecutive 2-minute recording block',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),  # 120.0
```

iii. The AI noted there is no stimulus-driven trial structure, so trials are artificial segments of continuous recordings.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal averaging, converting from the native 30 Hz to an effective 3 Hz (333.3 ms time bins). The `time_bin_size` metadata is set to 333.3 ms.

ii.
```python
BIN_FRAMES = 10

def average_nonoverlapping(x, bin_frames):
    usable = (n_frames // bin_frames) * bin_frames
    x = x[..., :usable]
    new_shape = x.shape[:-1] + (usable // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)

# metadata:
'time_bin_size': float(1000.0 * BIN_FRAMES / 30.0),  # 333.3 ms
```

iii. The paper explicitly states "averaging in bins of 10 consecutive timestamps" for their decoding analysis. The AI followed this methodology.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index, frame rate (from `ops.npy`), and bin size.

ii.
```python
def make_time_input(n_bins, fs, bin_frames):
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

iii. Since the frame rate is constant (30 Hz), computing time from bin indices is equivalent to loading timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (bin_frames / fs)`, giving absolute elapsed time from session start in seconds. Each trial preserves its absolute session time (not reset to zero per trial).

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
# In split_trials:
input_trials.append(time_binned_s[np.newaxis, start:stop])  # absolute time preserved
```

iii. The instruction says "Time elapsed from the beginning of the experiment," which the AI interprets as absolute session time.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural data are inherently aligned because both are derived from the same binned frame indices. The time vector is sliced with the same start:stop indices as the neural data when splitting into trials.

ii.
```python
# In split_trials - same indices for all:
neural_trials.append(neural_binned[:, start:stop])
input_trials.append(time_binned_s[np.newaxis, start:stop])
output_trials.append(output_one_hot[:, start:stop])
```

iii. No explicit alignment step is needed since all data streams share the same temporal grid after 10-frame binning.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (global motion energy signal) and `tstamps.npy` (timestamps for alignment) in the `move_deve` subdirectory.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video. Timestamps are used to map motion samples onto the imaging frame grid.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) motion energy is mapped onto the imaging frame grid using timestamps and interpolated for missing frames, (2) 10-frame temporal averaging, (3) min-max normalization to [0, 1], (4) per-session quintile discretization into 5 bins with one-hot encoding.

ii.
```python
# Step 1: Align to imaging frames
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
# Step 2: Bin
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
# Step 3: Normalize
def normalize_motion(x):
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)
# Step 4: Quintile + one-hot
def quintile_one_hot(x):
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    classes = np.searchsorted(edges, x, side="right")
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. The AI justified 10-frame binning from the paper. Min-max normalization removes scale differences. Per-session quintiles ensure balanced bins within each session. One-hot encoding was chosen for compatibility with the decoder.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using per-session quintiles (20th, 40th, 60th, 80th percentile thresholds computed within each session after normalization). The result is one-hot encoded into 5 binary channels.

ii.
```python
def quintile_one_hot(x):
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. Per-session quintiles guarantee exactly 20% of data in each bin within every session. The AI chose this over global pooling.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy timestamps are used to compute which imaging frame each motion sample corresponds to. Samples are mapped to the nearest imaging frame, values at the same frame are averaged, and missing frames are linearly interpolated. After alignment, the same 10-frame binning and trial splitting are applied identically to both streams.

ii.
```python
def reconstruct_motion_trace(motion_energy, tstamps, n_imaging_frames):
    frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)
    full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
    ...
    np.add.at(sums, frame_idx, motion_energy)
    np.add.at(counts, frame_idx, 1)
    valid = counts > 0
    full[valid] = (sums[valid] / counts[valid])
    # Interpolate missing
    missing = np.flatnonzero(~valid)
    full[missing] = np.interp(missing, valid_idx, full[valid_idx])
    return full, info
```

iii. The AI noted that imaging and video are synchronized at 30 Hz but occasional camera frames are dropped. Using timestamps to reconstruct the trace and interpolating missing frames ensures frame-for-frame alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are detected via timestamps and interpolated. The AI uses `np.interp` for linear interpolation of gaps. NaN checks are performed on all converted arrays. Remainder frames/bins that don't fill a complete trial are discarded.

ii.
```python
# Missing frame detection and interpolation in reconstruct_motion_trace
missing = np.flatnonzero(~valid)
if len(missing):
    full[missing] = np.interp(missing, valid_idx, full[valid_idx])

# NaN validation
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The AI documented that 9/41 sessions have missing motion frames (ranging from 1 to 148 missing frames). All are handled by interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is suite2p's `preprocess` function for baseline correction, which the AI runs on CPU. Mean processing time per session is ~1.4 seconds.

ii. N/A (timing information from output logs)

iii. The AI benchmarked processing at ~1.4s per session, with total conversion time of ~57s for 41 sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is already well-vectorized. The motion reconstruction uses `np.add.at` and `np.interp` instead of loops. Temporal binning uses reshape/mean. The trial splitting loop is simple slicing.

ii.
```python
# Vectorized motion reconstruction
np.add.at(sums, frame_idx, motion_energy)
np.add.at(counts, frame_idx, 1)
# Vectorized binning
def average_nonoverlapping(x, bin_frames):
    new_shape = x.shape[:-1] + (usable // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
```

iii. The AI explicitly documented vectorization as a speedup in CONVERSION_NOTES.md.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once in a single pass.

ii. N/A

iii. The AI processes all sessions in a single sequential loop with no redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The min-max normalization before quintile binning is unnecessary since quintile computation is rank-based and normalization does not change ranks. The normalization step's output is immediately overwritten by the discretization.

ii.
```python
motion_binned_norm = normalize_motion(motion_binned)  # min-max normalize
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)  # discretize
```

iii. The AI did not explicitly acknowledge this redundancy. The normalization is harmless but adds no value before rank-based binning.
