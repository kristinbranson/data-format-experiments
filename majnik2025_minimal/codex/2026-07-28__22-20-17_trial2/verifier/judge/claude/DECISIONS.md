# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating through a hardcoded list of 6 subject names (`FULL_SUBJECTS`). For each subject, it finds session subdirectories whose names start with a 4-digit date prefix. For each session, it loads suite2p files (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`) from `suite2p/plane0/` and behavioral files (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`) from `move_deve/`.

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

# In load_session:
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The AI identified the 6 subject directories from the data folder and hardcoded them. It also loads `ops.npy` to read per-session suite2p processing parameters and `iscell.npy` to validate cell classification. It additionally loads `tstamps.npy` beyond what the reference uses.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of 6 subject names matching the data directory structure. Subjects are sorted alphabetically when building the dataset.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]
...
subjects = sorted({record.subject for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI identified the 6 mice from the data directory and the paper's description of 6 mice imaged daily.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, filtered to only include directories whose names start with a 4-digit string (date prefix), and sorted alphabetically. Each session directory corresponds to one daily recording.

ii.
```python
sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The AI noted that session folders follow the `YYYY-MM-DD_a` naming convention from the data README, hence the date-prefix filter.

## 1-d. How are the data split into trials?

i. The AI splits sessions into 120-second (2-minute) non-overlapping trials, based on the paper's description that "splits were done on consecutive 2 minute blocks of the recording" for cross-validation. After binning (10 frames at 30 Hz = 333.33 ms bins), each trial has 360 time bins.

ii.
```python
def convert_dataset(..., trial_duration_s: float = 120.0, ...):
    ...

def split_into_trials(..., trial_duration_s: float, ...):
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))
    n_complete_trials = neural_binned.shape[1] // bins_per_trial
    ...
    for trial_idx in range(n_complete_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:end])
```

iii. The AI chose 2-minute trials because the paper states "splits were done on consecutive 2 minute blocks" for decoding cross-validation. The instructions say "Split sessions into 60-second trials," but the AI followed the paper's methodology instead.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Incomplete trials at the end of a session (that don't fill a full 2-minute block) are discarded.

ii.
```python
n_complete_trials = neural_binned.shape[1] // bins_per_trial
neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
```

iii. No justification given for the absence of trial filtering beyond standard truncation of incomplete trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (processing parameters) from `suite2p/plane0/`. The AI also loads `iscell.npy` for validation.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")
```

iii. The AI loads ops.npy to use per-session Suite2p parameters rather than hardcoding defaults, and iscell.npy to validate that all cells pass the 0.5 threshold (as expected for Track2p output).

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - neucoeff * Fneu`), where `neucoeff` is read from `ops.npy` (default 0.7). Then suite2p's `dcnv.preprocess` is called with parameters from `ops.npy` (baseline='maximin', win_baseline=60.0, sig_baseline=10, fs=30, prctile_baseline=8.0). A fallback reimplementation is provided if suite2p is unavailable.

ii.
```python
def baseline_correct_fluorescence(F, Fneu, ops, device):
    neucoeff = float(ops.get("neucoeff", 0.7))
    ...
    Fc = F.astype(np.float32, copy=False) - neucoeff * Fneu.astype(np.float32, copy=False)
    if suite2p_preprocess is not None:
        return suite2p_preprocess(
            Fc.copy(), baseline=baseline, win_baseline=win_baseline,
            sig_baseline=sig_baseline, fs=fs, prctile_baseline=prctile_baseline,
            batch_size=128, device=device,
        ).astype(np.float32, copy=False)
    # fallback reimplementation...
```

iii. The AI confirmed from the methods that the paper uses "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and verified that ops.npy contains the actual Suite2p defaults.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI loads `iscell.npy` and raises an error if any cell has probability < 0.5, but does not actually filter neurons out. All neurons in `F.npy` are included since the Track2p output already contains only tracked cells.

ii.
```python
iscell = np.load(suite2p_dir / "iscell.npy")
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. The AI noted that the Track2p suite2p-format output already contains only the successfully tracked neurons, so no further filtering is needed. The iscell check is a sanity validation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. The recording is continuous with no stimulus events. Trials are contiguous 2-minute segments from the beginning of the session.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block cut from a continuous session",
"off_start": 0.0,
"off_end": float(trial_duration_s),
```

iii. Since there is no stimulus-driven trial structure and the recording is continuous spontaneous activity, alignment to session start is the natural choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting from 30 Hz to 3 Hz (333.33 ms time bins).

ii.
```python
def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

neural_binned = average_nonoverlapping(neural, frame_bin)
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0]
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from bin indices and the bin duration, not from any raw data variable. It represents seconds from session start at the center of each time bin.

ii.
```python
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
```

iii. Since the frame rate is constant at 30 Hz and time bins are 10 frames, time can be computed directly from indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `(bin_index + 0.5) * time_bin_s`, where `time_bin_s = frame_bin / fs = 10/30 s`. The `+0.5` places the time at the center of each bin rather than the left edge. The time runs continuously across trials within a session.

ii.
```python
time_bin_s = frame_bin / fs
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. No explicit justification provided for the bin-center convention.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time axis is computed from the same bin indices used for the neural data, so alignment is inherent. Each time point corresponds to the center of the same bin as the neural data.

ii. Same code as 3-b above.

iii. N/A - alignment is automatic since both are indexed by the same bin structure.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. `interframe_int.npy` and `tstamps.npy` are used to detect dropped camera frames.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The AI identified these files from the data directory structure and the README's description of missing frame handling.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) Dropped frames detected via median-based threshold on interframe intervals (>1.5x median) and repaired by inserting NaN then interpolating. (2) The trace is averaged into 10-frame bins. (3) Min-max normalization is applied per session. (4) The normalized signal is discretized into 5 quintile bins using global edges computed across all sessions.

ii.
```python
# Gap detection
def detect_gap_indices(interframe_int):
    median_dt = float(np.median(interframe_int))
    return np.flatnonzero(interframe_int > 1.5 * median_dt)

# Normalization
def minmax_normalize(x):
    x_min = float(np.min(x)); x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min)

# Discretization (global edges)
all_motion_concat = np.concatenate(all_motion, axis=0)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(trial, global_edges, right=False)
```

iii. The AI chose min-max normalization because the instructions mention "normalized and discretized." Global quintile edges were chosen to ensure consistent bin meanings across sessions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After min-max normalization per session, global quintile edges are computed from the concatenated normalized motion energy across all sessions. Values are discretized using `np.digitize` with edges at [0.2, 0.4, 0.6, 0.8] quantiles, producing 5 categories (0-4).

ii.
```python
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
```

iii. The AI chose global edges to maintain consistent bin meanings across sessions, reasoning that per-session edges would make class labels incomparable across sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When the motion energy trace is shorter than the neural trace, gap indices are detected via the median-based threshold on interframe intervals. NaN values are inserted at gap positions, then the trace is padded if still short, and finally NaNs are interpolated. When longer, the motion trace is trimmed. After alignment, both traces are binned together.

ii.
```python
def align_motion_to_imaging(motion_energy, tstamps, interframe_int, target_frames):
    ...
    if diff > 0:
        gap_positions = gap_idx[:diff]
        for pos in gap_positions:
            insert_at = int(pos + 1 + offset)
            repaired = np.insert(repaired, insert_at, np.nan)
            offset += 1
        if repaired.shape[0] < target_frames:
            pad = np.full(target_frames - repaired.shape[0], np.nan)
            repaired = np.concatenate([repaired, pad])
        repaired = repaired[:target_frames]
        repaired = interpolate_nans_1d(repaired)
```

iii. The AI followed the data README's guidance that missing frames can be identified from timestamps/interframe intervals and interpolated.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected and interpolated (see 4-d). If the motion trace is longer than neural, it is trimmed. Incomplete trials at the end of sessions are discarded. An iscell validation check raises an error if unexpected cell classifications are found.

ii.
```python
if diff < 0:
    return motion_energy[:target_frames], info
...
if np.any(iscell_prob < 0.5):
    raise ValueError(...)
```

iii. The AI took a defensive approach, raising errors for unexpected conditions rather than silently handling them.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `dcnv.preprocess` baseline correction, which runs per-session across all neurons. The AI noted this in the trajectory: "Most of the time here is the per-session Suite2p-style baseline correction over the 41 continuous recordings."

ii. N/A

iii. The agent trajectory explicitly mentions this as the bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation loop uses `np.insert` in a for-loop, which reallocates the array each iteration. This could be vectorized by pre-allocating and filling in all interpolated values at once.

ii.
```python
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan)
    offset += 1
```

iii. The number of dropped frames is typically small, so this has minimal performance impact.

## 6-c. What processing does the code repeat multiple times?

i. The code iterates over output trials twice in `build_dataset`: once to discretize them and once to compute per-trial summary statistics. The `normalize_session_outputs_in_place` function also concatenates and re-splits trials, which is repeated logic.

ii.
```python
# In build_dataset:
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False)
    discretized_trials.append(bins)
# And again:
for trial in record.output_continuous_trials:
    normalized_trials.append({...})
```

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code applies min-max normalization per session before global discretization. This normalization step is unnecessary because `np.digitize` with global edges would produce the same relative ordering regardless of per-session scaling. The code also computes and stores per-trial motion summary statistics in metadata that are not used downstream. Loading `tstamps.npy` is also unnecessary as only `interframe_int.npy` is needed for gap detection.

ii.
```python
def normalize_session_outputs_in_place(session_records):
    ...
    normalized = minmax_normalize(concatenated)
    ...
```

iii. N/A
