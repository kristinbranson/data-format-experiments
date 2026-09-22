# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every top-level directory under `/app/data` as a subject, then loads every subdirectory whose name starts with a date as a session. For each session it loads Suite2p calcium files (`ops.npy`, `iscell.npy`, `F.npy`, `Fneu.npy`) and behavior files (`motion_energy_glob.npy`, `interframe_int.npy`). Trials are not loaded from disk directly; they are created later by splitting each processed session into consecutive 60-second windows.

ii. 
```python
def sorted_subjects(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir())


def sorted_sessions(subject_dir: Path) -> list[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )
```

```python
for subject_dir in sorted_subjects(data_root):
    for session_dir in sorted_sessions(subject_dir):
        neural_trials, input_trials, output_trials, _ = session_to_trials(session_dir)
```

```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
f = np.load(plane_dir / "F.npy").astype(np.float32)
fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)
motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy").astype(np.float32)
interframe = np.load(session_dir / "move_deve" / "interframe_int.npy")
```

iii. In the trajectory, the AI said it would "load all 6 subjects and all longitudinal sessions in date order" and then split each session into trials later (Step 48). Earlier steps show it inspected the data layout and concluded the dataset was organized by subject folders with date-named session subfolders (Steps 11-12, 48).

## 1-b. How are the data split into subjects?

i. Subjects are defined as the sorted top-level directories in `/app/data`. The subject names are the directory names themselves.

ii.
```python
def sorted_subjects(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir())

subjects = [subject_dir.name for subject_dir in sorted_subjects(data_root)]
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI inspected the dataset directory structure and stated that it would "load all 6 subjects" (Step 48). Its earlier filesystem inspection showed six subject folders in `/app/data` (Steps 11-12).

## 1-c. How are the data split into sessions?

i. Sessions are defined as the sorted subdirectories within each subject whose names begin with four digits, i.e. date-prefixed folders such as `2023-10-18_a`.

ii.
```python
def sorted_sessions(subject_dir: Path) -> list[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )
```

```python
for subject_dir in sorted_subjects(data_root):
    for session_dir in sorted_sessions(subject_dir):
        print(f"Converting {subject_dir.name}/{session_dir.name}...")
```

iii. The AI examined the README and folder names, then said it would use all longitudinal sessions "in date order" (Step 48). The trajectory shows it verified the date-based naming pattern across all mice (Steps 11-12).

## 1-d. How are the data split into trials?

i. The AI treats each session as continuous data and splits the processed, binned session into consecutive non-overlapping 60-second trials. Because data are binned into 10-frame bins at 30 Hz first, each trial contains 180 bins. Any remainder after the last full trial is dropped.

ii.
```python
trial_bins = int(TRIAL_SECONDS * FRAME_RATE_HZ / DENOISE_BIN_FRAMES)
n_complete_trials = neural_binned.shape[1] // trial_bins
usable = n_complete_trials * trial_bins
neural_binned = neural_binned[:, :usable]
motion_bins = motion_bins[:usable]
time_binned = time_binned[:usable]
```

```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_binned[np.newaxis, start:end].astype(np.float32, copy=False))
    output_trials.append(motion_bins[np.newaxis, start:end].astype(np.int64, copy=False))
```

iii. The AI explicitly planned to "split every session into consecutive 60-second trials" (Step 48). In later status updates it described the result as the "expected 20- or 30-trial splits" (Step 63), confirming this segmentation scheme.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial-by-trial quality-control filter. It only enforces that each session must yield at least two complete 60-second trials; otherwise it raises an error instead of exporting that session.

ii.
```python
n_complete_trials = neural_binned.shape[1] // trial_bins
if n_complete_trials < 2:
    raise ValueError(f"{session_dir}: fewer than two complete 60-second trials")
```

iii. The trajectory does not record a separate trial-quality rationale. The only explicit rationale is the Step 48 plan to split sessions into 60-second trials and the requirement from the task that sessions need at least two trials for decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported `neural` data is derived from Suite2p fluorescence traces `F.npy` and `Fneu.npy`. The AI also loads `ops.npy` to retrieve Suite2p preprocessing parameters and `iscell.npy` to validate the ROIs.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
f = np.load(plane_dir / "F.npy").astype(np.float32)
fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)
```

iii. The AI said the notebook confirmed `F.npy` was raw fluorescence and that it needed to compute the proper processed trace "the way described in the paper" (Step 24). It then converged on using Track2p-tracked Suite2p traces as the neural source (Steps 40, 48, 73).

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil from raw fluorescence using `ops['neucoeff']` (defaulting to 0.7), then runs Suite2p's `preprocess` baseline-correction function with parameters pulled from `ops.npy` (default baseline `maximin`, 60 s window, sigma 10, percentile 8). After that, it averages the trace in non-overlapping 10-frame bins.

ii.
```python
fc = f - float(ops.get("neucoeff", 0.7)) * fneu

dff = preprocess(
    fc.copy(),
    baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=float(ops.get("fs", FRAME_RATE_HZ)),
    prctile_baseline=float(ops.get("prctile_baseline", 8)),
    batch_size=100,
    device=torch.device("cpu"),
)
```

```python
neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
```

iii. The trajectory shows this was one of the AI's most deliberate choices. It reasoned that `F.npy` should not be used directly, rejected the Track2p GUI helper because it hard-coded `neucoeff=0.0`, verified Suite2p was installed, and decided to reproduce "Suite2p-style neuropil subtraction plus `maximin` baseline correction, then the paper's 10-frame averaging before trializing" (Steps 24, 32, 36, 40, 48).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not remove neurons, but it asserts that every ROI in `iscell.npy` has confidence greater than 0.5. If any tracked ROI fails this check, the session conversion aborts.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found tracked ROIs below the 0.5 iscell threshold")
```

iii. In the trajectory the AI stated that the "core constraints from the methods" included "Suite2p `iscell > 0.5`" (Step 9) and later summarized the intended pipeline as using "Track2p-tracked neurons" with `iscell > 0.5` (Steps 40, 73). It treated this as validation rather than a separate filtering pass.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align neural data to an experimental event from the paper. Instead, it defines artificial 60-second trials and stores each trial as the consecutive bins within that window. In the saved metadata it describes the alignment event as the "start of each consecutive 60-second trial."

ii.
```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
```

```python
"metadata": {
    "temporal_alignment_event": "start of each consecutive 60-second trial",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
```

iii. The AI's justification was implicit: the recording is continuous, so it planned to "split every session into consecutive 60-second trials" rather than align to any stimulus event (Step 48). The trajectory does not show a separate discussion of using session start versus trial start for the metadata field.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral data by averaging non-overlapping groups of 10 frames. At 30 Hz, this yields 3 Hz data with 333.33 ms bins.

ii.
```python
FRAME_RATE_HZ = 30.0
DENOISE_BIN_FRAMES = 10
```

```python
neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES).astype(np.float32)
```

```python
"time_bin_size": 1000.0 * DENOISE_BIN_FRAMES / FRAME_RATE_HZ,
```

iii. The AI repeatedly cited the paper's 10-frame denoising step and said it had confirmed from the methods that it should "denoise neural plus motion traces by averaging 10 frames before decoding" (Step 9), then repeated that plan in Steps 40, 48, and 73.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not load timestamps from file for this input. It derives time from the frame index and the assumed imaging frame rate of 30 Hz.

ii.
```python
frame_times_s = np.arange(raw_frames, dtype=np.float32) / FRAME_RATE_HZ
```

iii. The AI explicitly framed this as a choice between "raw seconds, bin centers, or another aligned representation" (Step 44), and then decided to "build decoder input as binned elapsed time from session start in seconds" (Step 48).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI first creates a per-frame time vector from frame indices, then averages those timestamps in the same 10-frame bins used for neural and motion data. This means the exported time is effectively the mean time within each 10-frame bin, i.e. bin centers rather than bin left edges.

ii.
```python
frame_times_s = np.arange(raw_frames, dtype=np.float32) / FRAME_RATE_HZ
time_binned = mean_bin_1d(frame_times_s, DENOISE_BIN_FRAMES).astype(np.float32)
```

iii. The trajectory directly documents this decision process: after checking how `input` is used by the decoder, the AI said it needed to choose between "raw seconds, bin centers, or another aligned representation after the 10-frame averaging" (Step 44), and then selected "binned elapsed time from session start in seconds" (Step 48).

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time signal is aligned by constructing it at frame resolution for the full session, applying the same 10-frame averaging as the neural data, truncating to the same number of complete trials, and slicing trials with the same `[start:end]` bin indices.

ii.
```python
time_binned = mean_bin_1d(frame_times_s, DENOISE_BIN_FRAMES).astype(np.float32)
usable = n_complete_trials * trial_bins
time_binned = time_binned[:usable]
```

```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
    input_trials.append(time_binned[np.newaxis, start:end].astype(np.float32, copy=False))
```

iii. The AI's Step 44 note makes this explicit: it inspected how `input` is used during training so it could decide on a representation "after the 10-frame averaging." Step 48 then states the result: time is binned and aligned to the processed session timeline.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The `output` variable is derived from `move_deve/motion_energy_glob.npy`. When the motion array is shorter than the neural recording, the AI also uses `move_deve/interframe_int.npy` to infer dropped camera frames and reconstruct a full-length motion trace.

ii.
```python
motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy").astype(
    np.float32
)
interframe = np.load(session_dir / "move_deve" / "interframe_int.npy")
```

iii. The AI identified "genuine video-frame drops" in some sessions and said it needed to use the interframe-interval file to align motion energy without inventing ad hoc preprocessing (Step 19). Its final plan also explicitly called for reconstructing the trace from `interframe_int.npy` (Step 48).

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI reconstructs missing camera samples by turning interframe intervals into integer frame steps, placing known motion samples at those inferred frame positions, and linearly interpolating missing positions. It then averages motion in 10-frame bins and converts the result to 5 discrete categories.

ii.
```python
median_dt = float(np.median(interframe))
frame_steps = np.rint(interframe / median_dt).astype(np.int64)
frame_steps = np.maximum(frame_steps, 1)
known_positions = np.concatenate([[0], np.cumsum(frame_steps)])
```

```python
full_motion = np.full(target_len, np.nan, dtype=np.float32)
full_motion[known_positions] = motion
valid = np.flatnonzero(~np.isnan(full_motion))
return np.interp(
    np.arange(target_len, dtype=np.float64), valid, full_motion[valid]
).astype(np.float32)
```

```python
motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES).astype(np.float32)
motion_bins = discretize_equal_percentile(motion_binned, MOTION_NBINS)
```

iii. The trajectory shows several explicit justifications: the AI found real dropped frames (Step 19), concluded NaNs were not allowed by the decoder (Step 44), and then planned to "reconstruct a full-length motion-energy trace aligned to neural frames by inserting missing camera frames from `interframe_int.npy` and linearly interpolating them" before 10-frame averaging (Step 48).

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI bins each session's motion-energy trace into 5 equal-percentile bins computed within that session. It uses interior quantiles as thresholds and `np.digitize` to assign labels 0-4. If tied quantiles would collapse edges, it falls back to rank-based equal-frequency assignment so that exactly five classes remain.

ii.
```python
def discretize_equal_percentile(values: np.ndarray, nbins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
    if np.unique(edges).shape[0] == edges.shape[0]:
        return np.digitize(values, edges, right=False).astype(np.int64)

    order = np.argsort(values, kind="mergesort")
    bins = np.empty(values.shape[0], dtype=np.int64)
    for bin_idx, idx in enumerate(np.array_split(order, nbins)):
        bins[idx] = bin_idx
    return bins
```

iii. The AI's Step 48 plan says the decoder output would be "per-session 5-bin motion-energy categories." The equal-percentile interpretation comes from the instructions and the function it implemented; the trajectory does not record a separate argument about the tie-handling fallback.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by reconstructing a motion vector with the same number of raw frames as the neural recording, then applying the same 10-frame binning and same trial slices as for the neural data.

ii.
```python
neural = suite2p_dff(session_dir)
raw_frames = neural.shape[1]
motion = reconstruct_motion_energy(session_dir, raw_frames)
```

```python
neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES).astype(np.float32)
```

```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
    output_trials.append(motion_bins[np.newaxis, start:end].astype(np.int64, copy=False))
```

iii. The trajectory gives a direct rationale: the AI saw real dropped frames (Step 19), decided missing values could not remain in the export because the decoder rejects NaNs (Step 44), and then planned to reconstruct a "full-length motion-energy trace aligned to neural frames" before binning and trialization (Step 48).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing camera frames by reconstructing the motion trace from interframe intervals and interpolating missing samples. It raises errors if interframe lengths or reconstructed frame positions are inconsistent with the neural recording, and it drops any leftover partial trial at the end of a session by truncating to the number of complete 60-second trials.

ii.
```python
if interframe.shape[0] != motion.shape[0] - 1:
    raise ValueError(
        f"{session_dir}: unexpected interframe length {interframe.shape[0]} "
        f"for motion length {motion.shape[0]}"
    )
...
if int(known_positions[-1]) != target_len - 1:
    raise ValueError(
        f"{session_dir}: reconstructed motion spans {int(known_positions[-1]) + 1} "
        f"frames but neural data has {target_len}"
    )
```

```python
full_motion = np.full(target_len, np.nan, dtype=np.float32)
full_motion[known_positions] = motion
valid = np.flatnonzero(~np.isnan(full_motion))
return np.interp(
    np.arange(target_len, dtype=np.float64), valid, full_motion[valid]
).astype(np.float32)
```

```python
usable = n_complete_trials * trial_bins
neural_binned = neural_binned[:, :usable]
motion_bins = motion_bins[:usable]
time_binned = time_binned[:usable]
```

iii. The AI's explicit rationale was that there were real dropped frames in the dataset (Step 19) and that NaNs were unacceptable because the decoder rejects them (Step 44). Later updates show it actively watched for neural/behavior length mismatches while the conversion ran (Steps 53, 55, 58).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is Suite2p preprocessing of full-session neural traces in `suite2p_dff`, especially `preprocess(...)` over every neuron and frame. Motion reconstruction and disk I/O are secondary; the rest of the work is mostly simple array slicing and averaging.

ii.
```python
dff = preprocess(
    fc.copy(),
    baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=float(ops.get("fs", FRAME_RATE_HZ)),
    prctile_baseline=float(ops.get("prctile_baseline", 8)),
    batch_size=100,
    device=torch.device("cpu"),
)
```

iii. The trajectory does not contain a formal performance analysis, but the AI repeatedly monitored the long end-to-end run and specifically noted waiting for "the longer 30-minute sessions" to finish (Step 58). Given the implementation, the full-trace Suite2p preprocessing is the clear dominant computation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious loop is the per-trial assembly loop, which repeatedly slices and appends one trial at a time for neural, input, and output arrays. The tie-handling fallback in `discretize_equal_percentile` also loops over five bins, although that path is rarely used.

ii.
```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_binned[np.newaxis, start:end].astype(np.float32, copy=False))
    output_trials.append(motion_bins[np.newaxis, start:end].astype(np.int64, copy=False))
```

```python
for bin_idx, idx in enumerate(np.array_split(order, nbins)):
    bins[idx] = bin_idx
```

iii. The trajectory does not include an explicit vectorization discussion. This assessment is inferred from the final code: the AI already avoided the most obvious frame-insertion loop by using a vectorized `np.interp` reconstruction, leaving trial assembly as the main remaining Python loop.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated numerical processing, but the code does rescan the subject directories twice in `build_dataset`: once to create `subjects` and again to iterate through sessions. It also repeats dtype conversions inside the per-trial loop.

ii.
```python
subjects = [subject_dir.name for subject_dir in sorted_subjects(data_root)]
...
for subject_dir in sorted_subjects(data_root):
    for session_dir in sorted_sessions(subject_dir):
```

```python
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
input_trials.append(time_binned[np.newaxis, start:end].astype(np.float32, copy=False))
output_trials.append(motion_bins[np.newaxis, start:end].astype(np.int64, copy=False))
```

iii. The trajectory does not contain a separate justification about repeated work. This is inferred from inspection of the code structure rather than from an explicit statement by the AI.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is that `session_to_trials` returns `motion_bins` as a fourth output even though `build_dataset` ignores it. The code also constructs extra metadata fields (`session_ids`, `trial_counts`, descriptive strings) that are not needed by the downstream decoder.

ii.
```python
def session_to_trials(session_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    ...
    return neural_trials, input_trials, output_trials, motion_bins
```

```python
neural_trials, input_trials, output_trials, _ = session_to_trials(session_dir)
```

```python
session_ids = []
trial_counts = []
...
"metadata": {
    ...
    "session_ids": session_ids,
    "trial_counts": trial_counts,
```

iii. The trajectory does not call out any unnecessary work explicitly. This conclusion comes from the final implementation rather than from a recorded justification.
