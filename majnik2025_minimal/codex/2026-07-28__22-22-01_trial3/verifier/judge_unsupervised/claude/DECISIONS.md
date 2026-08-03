# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all subject directories under the data root, then iterates over all session subdirectories within each subject. For each session, it loads `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. All sessions are collected into a list of dicts by `session_paths()` + `load_session()`, then assembled into the final dataset by `build_dataset()`.

ii.
```python
def session_paths(data_dir: Path, selected_subjects: list[str] | None) -> list[tuple[str, Path]]:
    available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    ...
    for subject in subjects:
        subject_dir = data_dir / subject
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            out.append((subject, session_dir))
    return out

def load_session(subject: str, session_dir: Path) -> dict:
    plane_dir = session_dir / "suite2p" / "plane0"
    move_dir = session_dir / "move_deve"
    fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
    motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The agent recognized the data organization from the `data/README.md` and `load_data.ipynb`: 6 subject folders each containing 6-7 session subdirectories with Suite2p format neural data and behavioral data. The agent verified shapes and file existence across all subjects.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the top-level directory names under the data root (e.g., `jm031`, `jm032`, ..., `jm046`). A sorted unique list of subject names is maintained, and each session is mapped to its subject via `subjects.index(session["subject"])`.

ii.
```python
subjects = sorted({session["subject"] for session in loaded_sessions})
...
subject_idx.append(subjects.index(session["subject"]))
```

iii. The agent followed the data README which states each subject has its own folder. The subject IDs are the folder names.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder (e.g., `2023-10-18_a`) corresponds to one recording session/day. Sessions are sorted alphabetically (which corresponds to chronological order given the YYYY-MM-DD naming convention).

ii.
```python
for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
    out.append((subject, session_dir))
```

iii. The agent followed the data README which states each session folder corresponds to one recording day.

## 1-d. How are the data split into trials?

i. After binning, each session's continuous recording is split into consecutive non-overlapping 2-minute (120-second) blocks. With 10-frame bins at 30 Hz, each bin is 1/3 second, so each trial has 360 bins (`TRIAL_BINS = 360`). The code validates that the total number of bins is evenly divisible by the trial length.

ii.
```python
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)  # 360

def split_session_into_trials(...):
    n_trials = neural_binned.shape[1] // TRIAL_BINS
    for trial_idx in range(n_trials):
        start = trial_idx * TRIAL_BINS
        end = start + TRIAL_BINS
        neural_trials.append(neural_binned[:, start:end])
```

iii. The agent followed the paper's methods which state "splits were done on consecutive 2 minute blocks of the recording" for decoding cross-validation.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All consecutive 2-minute blocks from every session are included. The only validation is that the total binned frames are evenly divisible by the trial length (360 bins), ensuring no partial trials.

ii.
```python
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)
```

iii. The paper and data README do not describe any trial-level quality filtering. Since the data comes from Track2p-exported sessions (already curated), no additional trial filtering was deemed necessary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence), along with parameters from `ops.npy` (neucoeff, baseline method, sig_baseline, win_baseline).

ii.
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
```

iii. The agent identified from the paper ("We used baseline corrected fluorescence traces as our dF/F") and the Track2p GUI code that the proper neural signal is computed from F and Fneu using Suite2p baseline correction, not just raw F.

## 2-b. How is the `neural` data processed?

i. Processing follows Suite2p's baseline-corrected fluorescence: (1) neuropil subtraction: `Fc = F - neucoeff * Fneu`, (2) baseline estimation using the maximin method (Gaussian smoothing, then min filter, then max filter), (3) baseline subtraction: `F_corrected = Fc - baseline`. After correction, the data is temporally binned by averaging every 10 consecutive frames.

ii.
```python
def suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops):
    fc = fluorescence.astype(np.float32) - float(ops["neucoeff"]) * neuropil.astype(np.float32)
    if baseline == "maximin":
        win = int(round(float(ops["win_baseline"]) * float(ops["fs"])))
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = minimum_filter1d(flow, win)
        flow = maximum_filter1d(flow, win)
    ...
    return (fc - flow).astype(np.float32)

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The agent traced the dF/F computation through the Track2p GUI code (`data_management.py:F_processing`) and verified it matches Suite2p's default baseline correction using parameters from `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural cell filtering is applied beyond what Track2p already provides. The code loads `iscell.npy` and asserts all cells have `iscell > 0.5`, but does not remove any cells since Track2p already exports only tracked cells passing this threshold.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The agent verified that all cells in the Track2p-exported data already pass the iscell > 0.5 threshold, so additional filtering would be redundant. The assertion serves as a sanity check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each consecutive 2-minute block within the session. Since the continuous recording is simply sliced into equal-length blocks starting from frame 0, the alignment event is the beginning of each block. `off_start = 0.0` and `off_end = 120.0` seconds.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block within a session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SECONDS,  # 120.0
```

iii. The paper states "splits were done on consecutive 2 minute blocks." There is no specific stimulus or event to align to since the experiment involves spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw imaging rate is 30 Hz. The data is rebinned by averaging 10 consecutive frames, yielding a bin size of 10/30 = 0.3333 seconds (333.33 ms). This rebinning is applied to both neural and motion energy data.

ii.
```python
FRAME_RATE_HZ = 30.0
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ  # 0.3333...
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0  # 333.33...

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE)
```

iii. The paper states "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent implemented this exactly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data variable. It is computed synthetically as the elapsed time at each bin center, based on the bin index and bin size.

ii.
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. The instructions specify "Time elapsed from the beginning of the experiment" as the decoder input. Since this is simply a time axis, it is generated from the bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the elapsed time is computed as `(bin_index + 0.5) * bin_size_seconds`, where bin_index runs from the start of the session (not from the start of each trial). The `+0.5` centers the time at the middle of each bin.

ii.
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
# Then for each trial:
input_trials.append(elapsed_time_seconds[start:end][None, :])
```

iii. The agent chose to use session-level elapsed time (not trial-relative time), which means the input reflects actual time from the session start, not from the trial start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The elapsed time array has the same number of time bins as the neural data (one time value per bin), so they are inherently aligned. The same slicing indices are used for both neural and input data when splitting into trials.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end])
    input_trials.append(elapsed_time_seconds[start:end][None, :])
```

iii. Both neural and input share the same temporal grid derived from the binned frames.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` (global motion energy) and `interframe_int.npy` (inter-frame intervals for detecting missing camera frames).

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The data README describes `motion_energy_glob.npy` as "processed behavioural data (motion energy extracted from videography of spontaneous behaviour)."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Align motion energy to imaging frame grid by linearly interpolating missing camera frames (detected via interframe_int.npy). (2) Temporally bin by averaging 10 consecutive frames. (3) Z-score normalize per session. (4) Compute global quintile edges across all sessions. (5) Digitize into 5 categories using `np.digitize`.

ii.
```python
motion_aligned, motion_info = align_motion_to_imaging_frames(motion_energy, n_frames, interframe_int)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE)
motion_z = (motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)
...
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES)
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False)
```

iii. The paper describes using motion energy as a behavioral measure, and the instructions require "normalized and discretized into five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy (after z-scoring) is discretized into 5 categories using global quintile edges computed across all sessions. The quantile breakpoints are at 20%, 40%, 60%, and 80%. `np.digitize` maps each value to a bin index 0-4.

ii.
```python
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES)
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
```

iii. The instructions state "discretized into five equal-percentile bins." The quintile edges divide the data into 5 equal-frequency bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy is first aligned to the imaging frame grid (handling missing camera frames via interpolation), then binned with the same 10-frame windows as the neural data. This ensures both share the same temporal grid. The code explicitly validates that binned neural and motion lengths match.

ii.
```python
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy, n_imaging_frames=n_frames, interframe_int=interframe_int)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE)
if neural_binned.shape[1] != motion_binned.shape[0]:
    raise ValueError(...)
```

iii. The data README notes that "In some recordings there might be some missing frames from the camera" and that interframe_int can be used to identify and interpolate over them.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where motion_energy length < imaging frames) are detected by analyzing inter-frame intervals from `interframe_int.npy`. The median interval is used to infer how many frames are missing at each gap, and linear interpolation fills those gaps. An assertion verifies the total inferred missing frames equals the expected difference. Sessions where motion and imaging perfectly match require no correction.

ii.
```python
def infer_missing_frames(interframe_int):
    median_interval = float(np.median(interframe_int))
    frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
    return np.clip(frame_jumps - 1, 0, None)

def align_motion_to_imaging_frames(motion_energy, n_imaging_frames, interframe_int):
    ...
    missing_after = infer_missing_frames(interframe_int)
    ...
    aligned[dst : dst + gap_missing] = np.linspace(
        motion_energy[src], motion_energy[next_src], int(gap_missing) + 2)[1:-1]
```

iii. The agent noted from the data README that missing camera frames can occur and chose interpolation, consistent with the README's suggestion that they "can be interpolated over."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading all `.npy` files for every session across all subjects (I/O bound), (2) Computing Suite2p baseline-corrected fluorescence for each session, which involves Gaussian filtering, min filtering, and max filtering over the full neuron x time matrix, (3) The motion energy alignment with interpolation for sessions with missing frames.

ii.
```python
# Baseline correction involves filtering over full (n_neurons, n_frames) matrices
flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
flow = minimum_filter1d(flow, win)
flow = maximum_filter1d(flow, win)
```

iii. The agent noted these are inherently computationally expensive operations but necessary for matching the reference processing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `align_motion_to_imaging_frames` function uses an explicit Python loop to insert interpolated frames one gap at a time. This could potentially be vectorized using cumulative sum indexing and array assignment. The `split_session_into_trials` function uses a Python loop over trials, which could use array reshape operations instead.

ii.
```python
# Loop in align_motion_to_imaging_frames:
for gap_missing in missing_after:
    next_src = src + 1
    if gap_missing:
        aligned[dst : dst + gap_missing] = np.linspace(...)
        dst += int(gap_missing)
    aligned[dst] = motion_energy[next_src]
    ...

# Loop in split_session_into_trials:
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end])
```

iii. These loops are not performance bottlenecks in practice (the trial loop runs ~10 times, the alignment loop runs only for sessions with missing frames), but could theoretically be vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The code processes all sessions sequentially in `load_session()`, but then iterates over the loaded sessions again in `build_dataset()` to split into trials. The motion z-scoring is done per-session in `load_session()`, and then the z-scored values are concatenated again in `build_dataset()` to compute global quantiles. This is not redundant per se, but involves two passes over the data.

ii.
```python
# First pass in load_session():
motion_z = (motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)

# Second pass in build_dataset():
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES)
for session in loaded_sessions:
    motion_classes = np.digitize(session["motion_z"], motion_bin_edges, ...)
```

iii. This two-pass design is reasonable because global quantile edges require all sessions to be loaded first.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `iscell.npy` and validates all values > 0.5, but since Track2p-exported data already contains only tracked cells, this check is always true and no filtering is applied. The code also stores detailed per-session metadata (ops parameters, missing frame counts, session dates) in the output, some of which may not be used by the downstream decoder.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(...)
# iscell is loaded but never used to filter - all cells already pass
```

iii. The iscell loading serves as a sanity check rather than functional filtering. The extra metadata, while not strictly necessary for the decoder, provides useful documentation.
