# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats every top-level directory in `/app/data` as a subject, every subdirectory inside each subject as a session, and then loads each session's Suite2p fluorescence files plus the motion-energy stream. `ops.npy`, `F.npy`, and `Fneu.npy` are always loaded; `motion_energy_glob.npy` is always loaded; `tstamps.npy` is loaded only if motion repair is needed because the motion array is shorter than the neural recording.

ii. <Code snippets>
```python
def list_subjects(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def list_sessions(subject_dir: Path) -> list[Path]:
    return sorted(path for path in subject_dir.iterdir() if path.is_dir())
```

```python
for subject_dir in list_subjects(data_root):
    for session_dir in list_sessions(subject_dir):
        neural_trace, fs = preprocess_neural(session_dir)
        ...
        motion_trace, n_motion_repaired = repair_motion_energy(session_dir, n_frames)
```

```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
...
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
```

iii. In the trajectory, the agent said it would "load each subject/session in chronological order and use the tracked-cell `suite2p/plane0` outputs directly," then later summarized that it used the "tracked Suite2p outputs already provided per session" and repaired motion only when camera frames were actually missing.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted top-level directories under `/app/data`. The subject names stored in the output are the directory names.

ii. <Code snippets>
```python
def list_subjects(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())
```

```python
subjects = [subject_dir.name for subject_dir in list_subjects(data_root)]
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The trajectory repeatedly refers to "subject/session" traversal and, in the final summary, says sessions are "ordered by subject then date." It does not mention any extra subject-name filter beyond using the directory structure.

## 1-c. How are the data split into sessions?

i. Each session is one sorted subdirectory inside a subject directory. The AI processes sessions in that sorted order and produces one output session per directory.

ii. <Code snippets>
```python
def list_sessions(subject_dir: Path) -> list[Path]:
    return sorted(path for path in subject_dir.iterdir() if path.is_dir())
```

```python
for subject_dir in list_subjects(data_root):
    for session_dir in list_sessions(subject_dir):
        ...
        neural_sessions.append(neural_trials)
        input_sessions.append(input_trials)
        output_sessions.append(output_trials)
```

iii. The trajectory says the plan is to "load each subject/session in chronological order" and the final summary says sessions are ordered by subject then date.

## 1-d. How are the data split into trials?

i. The AI creates artificial trials by splitting each session into contiguous non-overlapping 60-second segments after 10-frame temporal binning. Partial data at the end of a session that do not fill a complete trial are dropped.

ii. <Code snippets>
```python
BIN_SIZE_FRAMES = 10
TRIAL_DURATION_SEC = 60.0
```

```python
bins_per_trial = int(round(TRIAL_DURATION_SEC / (BIN_SIZE_FRAMES / fs)))
neural_trials, input_trials, output_trials = split_into_trials(
    neural_binned,
    time_binned_sec,
    motion_classes,
    bins_per_trial,
)
```

```python
def split_into_trials(
    neural_binned: np.ndarray,
    time_binned_sec: np.ndarray,
    motion_classes: np.ndarray,
    bins_per_trial: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    total_bins = neural_binned.shape[1]
    n_trials = total_bins // bins_per_trial
    usable = n_trials * bins_per_trial

    neural_binned = neural_binned[:, :usable]
    time_binned_sec = time_binned_sec[:usable]
    motion_classes = motion_classes[:usable]
```

iii. The trajectory plan explicitly says the output should "split every session into 60-second trials," and the final summary repeats "splits each session into 60-second trials."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. The only filtering at trialization time is structural: the AI discards any trailing bins that do not make a complete 60-second trial.

ii. <Code snippets>
```python
total_bins = neural_binned.shape[1]
n_trials = total_bins // bins_per_trial
usable = n_trials * bins_per_trial

neural_binned = neural_binned[:, :usable]
time_binned_sec = time_binned_sec[:usable]
motion_classes = motion_classes[:usable]
```

iii. The trajectory does not describe any trial rejection criteria beyond 60-second trialization. The final validation summary reports all sessions kept and only describes structural checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`; `ops.npy` is used to provide preprocessing parameters such as sampling rate and neuropil coefficient.

ii. <Code snippets>
```python
plane_dir = session_dir / "suite2p" / "plane0"
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. In the trajectory, the agent says it "confirmed the notebook guidance is explicit: `F.npy` is raw fluorescence," and later says it would compute "Suite2p-style baseline-corrected fluorescence from `F.npy` and `Fneu.npy` using the saved `ops.npy` parameters."

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil from raw fluorescence and then runs Suite2p's `dcnv.preprocess` for baseline correction. Unlike the human reference, it takes the baseline-related parameters and the sampling rate from `ops.npy` rather than hard-coding them.

ii. <Code snippets>
```python
fs = float(ops["fs"])
neuropil_corrected = F - float(ops.get("neucoeff", 0.7)) * Fneu
baseline_corrected = dcnv.preprocess(
    neuropil_corrected.copy(),
    baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=fs,
    prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
    batch_size=64,
    device=torch.device("cpu"),
)
```

iii. The trajectory says the agent wanted to "mirror Suite2p’s fluorescence preprocessing directly from the installed package," that `preprocess` "returns baseline-corrected fluorescence after a maximin baseline step," and that it was "writing the converter against the session metadata in `ops.npy` ... rather than being hard-coded on assumption."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The implemented code does not apply any explicit neuron-quality filtering such as `iscell.npy` thresholding. All rows in `F.npy`/`Fneu.npy` are carried through.

ii. <Code snippets>
```python
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
...
brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The trajectory is internally inconsistent here. An early message said "keep Suite2p’s `iscell > 0.5` convention," but the final implementation never loads `iscell.npy`, and later summaries instead describe using the tracked Suite2p outputs directly. The effective implemented decision is therefore no extra neuron QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to session start. The AI does not perform event-locked alignment; it keeps continuous session time, bins the session uniformly, and then slices the session into consecutive trials.

ii. <Code snippets>
```python
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
...
neural_trials, input_trials, output_trials = split_into_trials(
    neural_binned,
    time_binned_sec,
    motion_classes,
    bins_per_trial,
)
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "session start",
    "off_start": 0.0,
    "off_end": float(TRIAL_DURATION_SEC),
    ...
},
```

iii. The trajectory plan says the data should be trialized after continuous-session preprocessing rather than aligned to a behavioral event, and the final summary describes 60-second session segments with "time elapsed from the beginning of the session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion data by averaging non-overlapping 10-frame windows. With the 30 Hz sampling rate, this gives 333.33 ms bins.

ii. <Code snippets>
```python
BIN_SIZE_FRAMES = 10
```

```python
def mean_bin_2d(array: np.ndarray, bin_size: int) -> np.ndarray:
    n_rows, n_frames = array.shape
    usable = (n_frames // bin_size) * bin_size
    if usable != n_frames:
        array = array[:, :usable]
    return array.reshape(n_rows, -1, bin_size).mean(axis=2, dtype=np.float32)
```

```python
time_bin_size_ms = 1000.0 * BIN_SIZE_FRAMES / float(common_fs)
```

iii. The trajectory explicitly says it would "average non-overlapping 10-frame bins, matching the paper’s decoding preprocessing," and the final summary repeats "averages neural and behavior traces in non-overlapping 10-frame bins."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not read a recorded timestamp stream for decoder input time. It derives time from the neural frame index and the imaging sampling rate `fs` from `ops.npy`.

ii. <Code snippets>
```python
fs = float(ops["fs"])
...
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
```

iii. The trajectory focuses on `tstamps.npy` only for repairing missing motion frames. For decoder input time, it describes "time elapsed from the beginning of the session," implying computed session time rather than a separate raw variable.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI first creates a full-resolution per-frame time vector in seconds, then averages it in the same 10-frame bins used for neural and motion data. This produces bin-center times rather than the left-edge times used in the human reference.

ii. <Code snippets>
```python
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
...
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
```

```python
def mean_bin_1d(array: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (len(array) // bin_size) * bin_size
    if usable != len(array):
        array = array[:usable]
    return array.reshape(-1, bin_size).mean(axis=1, dtype=np.float32)
```

iii. The trajectory did not separately justify this bin-center choice. Its broader rationale was to process neural and behavioral streams with the same 10-frame averaging, and the time input was constructed to follow that same binning pipeline.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is derived from the same session length, averaged into the same bin structure, truncated to the same usable duration, and sliced into the same trial boundaries as the neural data.

ii. <Code snippets>
```python
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
...
neural_trials, input_trials, output_trials = split_into_trials(
    neural_binned,
    time_binned_sec,
    motion_classes,
    bins_per_trial,
)
```

```python
input_trials.append(time_binned_sec[start:end][None, :].astype(np.float32, copy=False))
```

iii. The trajectory's plan was to process continuous session time, neural activity, and motion with one shared binning and trialization scheme so the streams "align and bin cleanly."

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy from `move_deve/motion_energy_glob.npy`. When repair is needed, it also uses `move_deve/tstamps.npy` to infer where video frames were dropped.

ii. <Code snippets>
```python
move_dir = session_dir / "move_deve"
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
...
tstamps = np.load(move_dir / "tstamps.npy")
```

iii. The trajectory says the key source decision was to "respect the motion-energy stream with missing-camera-frame handling," then later says it found that timestamp gaps "exactly account for the missing frames" when repair is necessary.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI optionally repairs shortened motion traces by inserting missing positions based on timestamp gaps, linearly interpolates over those missing positions, averages the repaired trace into 10-frame bins, and then discretizes the binned signal.

ii. <Code snippets>
```python
diffs = np.diff(tstamps)
frame_dt = float(np.median(diffs))
missing_counts = np.maximum(np.round(diffs / frame_dt).astype(int) - 1, 0)
expected_len = int(len(motion) + missing_counts.sum())
```

```python
repaired = np.full(n_frames, np.nan, dtype=np.float32)
positions = np.arange(len(motion), dtype=np.int64)
positions[1:] += np.cumsum(missing_counts)
repaired[positions] = motion

missing_mask = ~np.isfinite(repaired)
valid_idx = np.flatnonzero(~missing_mask)
repaired[missing_mask] = np.interp(
    np.flatnonzero(missing_mask),
    valid_idx,
    repaired[valid_idx],
).astype(np.float32)
```

```python
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
motion_classes = session_motion_bins(motion_binned)
```

iii. The trajectory says the agent found a "practical rule" that timestamp gaps explain shortened behavior streams, that it would "repair shortened motion-energy streams only when camera frames are actually missing," and that it wanted to avoid improvising behavior preprocessing.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI converts motion energy into five equal-frequency categories separately within each session by taking session-wise 20/40/60/80% quantiles and digitizing the binned motion trace against those thresholds.

ii. <Code snippets>
```python
def session_motion_bins(motion_binned: np.ndarray) -> np.ndarray:
    thresholds = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(motion_binned, thresholds, right=False).astype(np.int64)
```

iii. The trajectory plan explicitly says "discretize motion energy into 5 equal-frequency bins separately within each session," and the final summary describes "5 session-wise quintiles."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI forces motion energy to neural length before any joint binning or trialization. If the motion trace is already the same length as the neural recording, it leaves it untouched even if timestamps look irregular; if it is shorter, it reconstructs the missing frame positions from timestamps and interpolates back to the neural frame count.

ii. <Code snippets>
```python
if len(motion) == n_frames:
    return motion, 0
```

```python
expected_len = int(len(motion) + missing_counts.sum())
if expected_len != n_frames:
    raise ValueError(
        f"{session_dir}: repaired motion length would be {expected_len}, "
        f"but neural data has {n_frames} frames"
    )
```

```python
motion_trace, n_motion_repaired = repair_motion_energy(session_dir, n_frames)
...
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
...
neural_trials, input_trials, output_trials = split_into_trials(
    neural_binned,
    time_binned_sec,
    motion_classes,
    bins_per_trial,
)
```

iii. In the trajectory, the agent says that when lengths already match it "won’t 'repair' those sessions," but when motion is shorter "the timestamp gaps exactly account for the missing frames." The stated goal was to keep neural and motion streams aligned without modifying already aligned sessions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing camera frames in the motion stream by reconstructing missing positions from timestamps and interpolating them. It also validates consistency between motion length and neural length and drops trailing partial data during binning/trialization rather than trying to pad it.

ii. <Code snippets>
```python
if len(tstamps) != len(motion):
    raise ValueError(
        f"{session_dir}: motion timestamps length {len(tstamps)} does not match "
        f"motion length {len(motion)}"
    )
```

```python
expected_len = int(len(motion) + missing_counts.sum())
if expected_len != n_frames:
    raise ValueError(
        f"{session_dir}: repaired motion length would be {expected_len}, "
        f"but neural data has {n_frames} frames"
    )
```

```python
usable = (n_frames // bin_size) * bin_size
...
n_trials = total_bins // bins_per_trial
usable = n_trials * bins_per_trial
```

iii. The trajectory emphasizes missing-camera-frame handling as the main data-cleaning issue and later reports the exact sessions where repair was applied. The agent also says it is using the verifier to catch any remaining structural issues.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is the per-session Suite2p baseline correction (`dcnv.preprocess`) on the full neural recording. Everything else is comparatively lightweight array manipulation and I/O.

ii. <Code snippets>
```python
baseline_corrected = dcnv.preprocess(
    neuropil_corrected.copy(),
    baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=fs,
    prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
    batch_size=64,
    device=torch.device("cpu"),
)
```

iii. The trajectory explicitly says "The heaviest step is the per-session Suite2p baseline correction," and earlier notes that it was testing preprocessing cost on a representative session before writing the full converter.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining pure-Python loop is in `split_into_trials`, which appends one trial at a time to Python lists. By contrast, the AI already vectorized the missing-frame reconstruction step instead of using repeated `np.insert` calls.

ii. <Code snippets>
```python
for trial_idx in range(n_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_binned_sec[start:end][None, :].astype(np.float32, copy=False))
    output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

```python
positions = np.arange(len(motion), dtype=np.int64)
positions[1:] += np.cumsum(missing_counts)
repaired[positions] = motion
```

iii. The trajectory does not explicitly discuss vectorization, but its repair design reflects that concern: it says it was quantifying timestamp encoding of dropped frames and then implemented a timestamp-based reconstruction rather than repeated insertion.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated downstream-discarded preprocessing pass. The code performs one preprocessing pipeline per session, then one trial-splitting pass per session. The only repeated structure is the necessary per-session iteration over subjects and sessions.

ii. <Code snippets>
```python
for subject_dir in list_subjects(data_root):
    for session_dir in list_sessions(subject_dir):
        neural_trace, fs = preprocess_neural(session_dir)
        ...
        neural_trials, input_trials, output_trials = split_into_trials(
            neural_binned,
            time_binned_sec,
            motion_classes,
            bins_per_trial,
        )
```

iii. The trajectory presents the work as a single pass: preprocess each session, bin, discretize, split into trials, and save. It does not describe redundant re-processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is no major unnecessary heavy processing that is later discarded. The closest extra work is bookkeeping: constructing verbose `session_info` metadata and descriptive strings that are not needed by the decoder itself.

ii. <Code snippets>
```python
session_info.append(
    {
        "subject": subject_dir.name,
        "session": session_dir.name,
        "n_neurons": int(n_neurons),
        "n_frames_raw": int(n_frames),
        "n_motion_frames_repaired": int(n_motion_repaired),
        "n_time_bins": int(neural_binned.shape[1]),
        "n_trials": int(len(neural_trials)),
        "duration_sec": float(n_frames / fs),
    }
)
```

```python
"metadata": {
    "task_description": (
        "Decode per-time-bin spontaneous motion energy quintiles from "
        "baseline-corrected barrel-cortex calcium activity."
    ),
    ...
    "session_info": session_info,
},
```

iii. The trajectory says it wanted to "report the concrete processing decisions and the sessions where behavior-frame repair was actually applied," which explains why it retained richer metadata than the minimum needed for training.
