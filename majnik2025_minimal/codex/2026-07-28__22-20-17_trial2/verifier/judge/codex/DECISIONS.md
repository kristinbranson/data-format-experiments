# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subjects, then discovers session directories under each subject by looking for date-named subdirectories. For each session it loads Suite2p fluorescence files (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`) and behavioral timing files (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`). Trials are not loaded directly from disk; they are created later by splitting each continuous session.

ii. 
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]

def select_session_dirs(data_dir: Path, mode: str) -> list[Path]:
    ...
    for subject in subjects:
        subject_dir = data_dir / subject
        sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        ...

def load_session(session_dir: Path, device: torch.device, frame_bin: int, trial_duration_s: float) -> SessionRecord:
    suite2p_dir = session_dir / "suite2p" / "plane0"
    move_dir = session_dir / "move_deve"

    F = np.load(suite2p_dir / "F.npy")
    Fneu = np.load(suite2p_dir / "Fneu.npy")
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(suite2p_dir / "iscell.npy")
    motion_energy = np.load(move_dir / "motion_energy_glob.npy")
    tstamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The trajectory says the agent believed the dataset contained six mice and that it should use the Track2p/Suite2p outputs plus synchronized motion traces. It explicitly said it would "load tracked cells, compute dF/F, align motion-energy to imaging frames" and keep the full recorded durations rather than trimming them.

## 1-b. How are the data split into subjects?

i. Subjects are split using a fixed list of six IDs rather than discovering them dynamically from the filesystem.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]
SAMPLE_SUBJECTS = FULL_SUBJECTS

if mode == "full":
    subjects = FULL_SUBJECTS
elif mode == "sample":
    subjects = SAMPLE_SUBJECTS
```

iii. The trajectory shows the agent inspected the directory tree, confirmed six mice, and then committed to using those exact six subjects.

## 1-c. How are the data split into sessions?

i. Sessions are split as sorted date-like subdirectories within each subject folder. In sample mode the AI keeps only the first session per subject; in full mode it keeps them all.

ii.
```python
for subject in subjects:
    subject_dir = data_dir / subject
    sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    if per_subject_limit is not None:
        sessions = sessions[:per_subject_limit]
    session_dirs.extend(sessions)
```

iii. The trajectory shows the agent enumerated all sessions per mouse and said the sessions lined up with the paper's consecutive daily recordings. No more detailed justification than deterministic sorted traversal was recorded.

## 1-d. How are the data split into trials?

i. The AI treats each session as continuous data and splits it into consecutive non-overlapping 2-minute trials after temporal binning. Only complete trials are kept.

ii.
```python
def split_into_trials(..., trial_duration_s: float) -> ...:
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))
    n_complete_trials = neural_binned.shape[1] // bins_per_trial

    neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
    motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
    ...
    for trial_idx in range(n_complete_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
        output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))

def convert_dataset(..., trial_duration_s: float = 120.0) -> ...:
```

iii. The trajectory explicitly says the agent chose "2-minute trials" to stay faithful to what it interpreted as the paper's block-wise decoding analysis. It repeated this choice in its plan, implementation summary, and documentation.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit trial-quality filter. It only drops the trailing remainder that does not fill a full fixed-length trial.

ii.
```python
n_complete_trials = neural_binned.shape[1] // bins_per_trial

neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
```

iii. The trajectory does not record a separate justification for trial QC filtering. The only recorded rationale is the structural choice to use consecutive fixed-length blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from Suite2p fluorescence traces `F.npy` and `Fneu.npy`. The AI also reads `ops.npy` to get preprocessing parameters and `iscell.npy` to verify the expected cell mask assumptions.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")
...
neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
```

iii. The trajectory says the agent verified the exact Suite2p baseline-correction path and concluded the paper used neuropil-subtracted, maximin-baseline traces rather than a separate additional dF/F normalization step.

## 2-b. How is the `neural` data processed?

i. Neural traces are neuropil-subtracted with `neucoeff` from `ops.npy`, then baseline-corrected with Suite2p's `dcnv.preprocess` if available. If Suite2p is unavailable, the script falls back to a local implementation of the same baseline modes. The resulting traces are then averaged in non-overlapping 10-frame bins.

ii.
```python
neucoeff = float(ops.get("neucoeff", 0.7))
...
Fc = F.astype(np.float32, copy=False) - neucoeff * Fneu.astype(np.float32, copy=False)

if suite2p_preprocess is not None:
    return suite2p_preprocess(
        Fc.copy(),
        baseline=baseline,
        win_baseline=win_baseline,
        sig_baseline=sig_baseline,
        fs=fs,
        prctile_baseline=prctile_baseline,
        batch_size=128,
        device=device,
    ).astype(np.float32, copy=False)
...
neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32, copy=False)
```

iii. The trajectory states the agent found the "exact baseline-correction routine" and decided to mirror Suite2p neuropil subtraction plus maximin baseline subtraction from `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out of the exported dataset. Instead it checks that all loaded ROIs already satisfy the expected `iscell >= 0.5` assumption and raises an error if they do not.

ii.
```python
iscell = np.load(suite2p_dir / "iscell.npy")
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. The trajectory says the agent believed the Track2p export already contained the matched cells and that no extra cell filtering should be applied beyond the existing `iscell` thresholding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI effectively aligns neural data to the start of each artificial 2-minute block. It encodes that choice directly in metadata, with trial start at offset 0 and trial end at `trial_duration_s`.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block cut from a continuous session",
    "off_start": 0.0,
    "off_end": float(trial_duration_s),
    ...
}
```

iii. The trajectory justifies this by saying the paper used a block-wise decoding unit and that the converter should mirror consecutive 2-minute blocks cut from continuous sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion data into non-overlapping bins of 10 imaging frames. With 30 Hz acquisition, this gives a 333.33 ms time bin.

ii.
```python
def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

...
"time_bin_size": float((frame_bin / 30.0) * 1000.0),
```

iii. The trajectory says the agent took the paper's instruction to average over 10 imaging frames as the main denoising step and applied the same binning to neural and behavior traces.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not loaded directly from raw timestamps. It is computed from the sampling rate in `ops.npy`, the chosen 10-frame bin width, and the bin index within the session.

ii.
```python
fs=float(ops["fs"]),
...
time_bin_s = frame_bin / fs
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
```

iii. The trajectory says the input should be "absolute time from session start" and treats the constant imaging frame rate as sufficient to reconstruct that axis.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a continuous time axis after 10-frame binning and uses the center of each bin, not the raw frame indices or bin left edges. It then slices that continuous axis into per-trial segments.

ii.
```python
time_bin_s = frame_bin / fs
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
...
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The trajectory and the AI's later documentation explicitly describe the input as absolute time from session start "sampled at the center of each 10-frame bin."

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time to neural data by generating the time axis after the same temporal binning used for the neural traces and then cutting both into the same trial boundaries.

ii.
```python
neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32, copy=False)
...
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The trajectory does not separately justify this step beyond the broader claim that both streams should be averaged together in 10-frame bins and split into the same 2-minute blocks.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy`. The AI also loads `interframe_int.npy` and `tstamps.npy` to repair motion/imaging length mismatches before exporting motion energy.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
...
motion_aligned, motion_info = align_motion_to_imaging(
    motion_energy=motion_energy,
    tstamps=tstamps,
    interframe_int=interframe_int,
    target_frames=neural.shape[1],
)
```

iii. The trajectory says the agent inspected the synchronization artifacts and decided to use the camera timing metadata to repair dropped motion frames only when the motion trace was shorter than imaging.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI repairs short motion traces by inserting missing values at detected camera gaps and interpolating them, averages motion in 10-frame bins, applies per-session min-max normalization, and then discretizes using global quintile thresholds computed across all sessions in the export.

ii.
```python
gap_idx = detect_gap_indices(interframe_int)
...
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
...
repaired = interpolate_nans_1d(repaired)
...
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)

def normalize_session_outputs_in_place(session_records: list[SessionRecord]) -> None:
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
        normalized = minmax_normalize(concatenated)
        ...

all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```

iii. The trajectory explicitly says the agent would "repair only the motion traces that are actually short" and then apply "task-specific motion normalization and quintile discretization." In its final summary it described the output as per-session normalized motion energy discretized into five global quintile bins.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes four global thresholds at the 20th, 40th, 60th, and 80th percentiles of the concatenated normalized motion values from the whole exported dataset, then uses `np.digitize` to assign bins 0 through 4.

ii.
```python
all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
```

iii. The trajectory and the AI's own notes justify this as "global quintiles within each export" after per-session normalization so that the output classes are balanced across the dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by comparing motion length to imaging length, detecting putative camera gaps from `interframe_int`, inserting missing samples only when motion is shorter, interpolating missing values, and then binning motion and neural traces with the same 10-frame averaging before trial splitting.

ii.
```python
def detect_gap_indices(interframe_int: np.ndarray) -> np.ndarray:
    ...
    median_dt = float(np.median(interframe_int))
    return np.flatnonzero(interframe_int > 1.5 * median_dt).astype(np.int64)

if motion_energy.shape[0] == target_frames:
    return motion_energy, info
...
if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info
...
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
...
repaired = interpolate_nans_1d(repaired)
...
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)
```

iii. The trajectory says the agent "pinned the synchronization rule" as repairing only short motion traces and inserting one missing sample at each detected timestamp gap before interpolation, because it believed this matched the README and avoided overcorrection.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or mismatched motion samples by gap detection plus interpolation when motion is shorter than imaging, trimming if motion is longer than imaging, and padding with `NaN` values before interpolation if too few gap locations are detected. It also drops incomplete trial tails and raises an error if the pre-filtered `iscell` assumption is violated.

ii.
```python
if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info
...
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
...
if repaired.shape[0] < target_frames:
    pad = np.full(target_frames - repaired.shape[0], np.nan, dtype=np.float32)
    repaired = np.concatenate([repaired, pad], axis=0)
...
repaired = interpolate_nans_1d(repaired)
...
if np.any(iscell_prob < 0.5):
    raise ValueError(...)
```

iii. The trajectory justifies the motion repair policy as a conservative way to fix only genuine short traces and avoid overcorrecting sessions whose motion arrays already match imaging length.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the per-session Suite2p-style fluorescence baseline correction. A secondary cost comes from loading all sessions and repairing motion traces.

ii.
```python
if suite2p_preprocess is not None:
    return suite2p_preprocess(
        Fc.copy(),
        baseline=baseline,
        win_baseline=win_baseline,
        sig_baseline=sig_baseline,
        fs=fs,
        prctile_baseline=prctile_baseline,
        batch_size=128,
        device=device,
    ).astype(np.float32, copy=False)

session_records = [
    load_session(...)
    for session_dir in session_dirs
]
```

iii. The trajectory explicitly says that "most of the time here is the per-session Suite2p-style baseline correction over the 41 continuous recordings."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the gap-repair loop that calls `np.insert` once per detected gap, reallocating the array repeatedly. Additional per-trial loops in discretization, normalization, and summary computation could also be replaced by more vectorized operations.

ii.
```python
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
    offset += 1

for trial_idx in range(n_complete_trials):
    ...

for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
```

iii. The trajectory does not discuss vectorization directly. This is inferred from the implementation the AI wrote.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats motion aggregation work across multiple stages: it first concatenates each session's motion trials for min-max normalization, then concatenates all sessions again to compute global quintile edges, then iterates over the same trial lists again to discretize and to compute per-trial summary statistics.

ii.
```python
def normalize_session_outputs_in_place(session_records: list[SessionRecord]) -> None:
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
        normalized = minmax_normalize(concatenated)
        ...

all_motion = []
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))
all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
...
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
...
for trial in record.output_continuous_trials:
    normalized_trials.append({
        "min": float(np.min(trial)),
        "max": float(np.max(trial)),
        "mean": float(np.mean(trial)),
    })
```

iii. The trajectory does not explicitly justify this repeated work. It appears to be a byproduct of adding normalization, global discretization, and summary metadata.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `tstamps.npy` but never uses it, computes and stores motion summary metadata that is not used by the decoder, and builds a summary report that is printed separately from the exported dataset. It also spends effort on normalization metadata even though downstream decoding only uses the final discretized motion bins.

ii.
```python
tstamps = np.load(move_dir / "tstamps.npy")
...
motion_aligned, motion_info = align_motion_to_imaging(
    motion_energy=motion_energy,
    tstamps=tstamps,
    interframe_int=interframe_int,
    target_frames=neural.shape[1],
)

normalized_trials = []
for trial in record.output_continuous_trials:
    normalized_trials.append({
        "min": float(np.min(trial)),
        "max": float(np.max(trial)),
        "mean": float(np.mean(trial)),
    })
info["motion_trial_summary_before_discretization"] = normalized_trials[:2]

summary = summarize_dataset(data)
print(json.dumps(summary, indent=2))
```

iii. The trajectory does not separately justify these extra reporting steps. The justifications it records are about validation and documentation rather than the decoder-facing data conversion itself.
