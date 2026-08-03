# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs, enumerates date-named session directories under each subject, and loads each session by reading Suite2p calcium files plus motion-camera files. It loads more files than the reference solution: `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]

def select_session_dirs(data_dir: Path, mode: str) -> list[Path]:
    if mode == "full":
        subjects = FULL_SUBJECTS
        per_subject_limit = None
    elif mode == "sample":
        subjects = SAMPLE_SUBJECTS
        per_subject_limit = 1
    ...
    for subject in subjects:
        subject_dir = data_dir / subject
        sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        ...
        session_dirs.extend(sessions)
    return session_dirs

def load_session(...):
    ...
    F = np.load(suite2p_dir / "F.npy")
    Fneu = np.load(suite2p_dir / "Fneu.npy")
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(suite2p_dir / "iscell.npy")
    motion_energy = np.load(move_dir / "motion_energy_glob.npy")
    tstamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The justification appears in `CONVERSION_NOTES.md` and the trajectory: the AI believed the provided Track2p export already contained matched cells in Suite2p format and that `tstamps.npy` / `interframe_int.npy` were needed to repair camera dropouts. In trajectory step 54 it said it would "repair only the motion traces that are actually short."

## 1-b. How are the data split into subjects?

i. Subjects are the six hard-coded mouse IDs in `FULL_SUBJECTS`. The final `subjects` field is rebuilt from the loaded records as a sorted set of subject names.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]

subjects = sorted({record.subject for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The justification is implicit rather than algorithmic: `CONVERSION_NOTES.md` states the full dataset has 6 subjects, and trajectory step 20 says the AI "confirmed the sessions line up with the paper’s six mice."

## 1-c. How are the data split into sessions?

i. Each session is a date-named subdirectory inside one of the hard-coded subject folders. Sessions are sorted lexicographically and, in sample mode, truncated to the first session per subject.

ii.
```python
for subject in subjects:
    subject_dir = data_dir / subject
    sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    if per_subject_limit is not None:
        sessions = sessions[:per_subject_limit]
    session_dirs.extend(sessions)
```

iii. The AI justified this by matching the delivered directory layout. Trajectory step 20 says it "confirmed the sessions line up with the paper’s six mice and 6–7 consecutive days per mouse."

## 1-d. How are the data split into trials?

i. Trials are artificial consecutive 2-minute blocks cut from each continuous session after 10-frame temporal averaging. The AI first bins the data, then computes how many full 2-minute blocks fit, and discards leftover binned timepoints.

ii.
```python
def split_into_trials(
    neural_binned: np.ndarray,
    motion_binned: np.ndarray,
    fs: float,
    frame_bin: int,
    trial_duration_s: float,
):
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
```

iii. This is explicitly justified in `CONVERSION_NOTES.md`: "The paper states that decoding used slightly denoised dF/F and behavior traces, averaged in bins of 10 consecutive timestamps, and that cross-validation splits were based on consecutive 2-minute blocks. The conversion mirrors that structure directly."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-level quality-control filter. The only trial curation is structural: incomplete tail data that do not fill a full 2-minute trial are dropped.

ii.
```python
n_complete_trials = neural_binned.shape[1] // bins_per_trial

neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
```

iii. No separate trial-QC justification is given. The notes focus on the assumption that the Track2p export already encoded the relevant curation at the cell level, and the code simply enforces complete fixed-length blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from Suite2p fluorescence traces `F.npy` and `Fneu.npy`. The AI also loads `ops.npy` to obtain preprocessing parameters, but the signal itself comes from `F` and `Fneu`.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
...
neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
```

iii. `CONVERSION_NOTES.md` states: "Input neural data came from `F.npy` and `Fneu.npy` in each session’s `suite2p/plane0` folder." Trajectory step 41 also says the agent identified the intended preprocessing as Suite2p neuropil subtraction plus maximin baseline subtraction.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil using `neucoeff`, applies Suite2p-style baseline correction using the per-session `ops.npy` parameters, and then averages the resulting traces in non-overlapping 10-frame bins. If Suite2p is unavailable, it falls back to a local approximation of the maximin baseline routine.

ii.
```python
Fc = F.astype(np.float32, copy=False) - neucoeff * Fneu.astype(np.float32, copy=False)
...
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

iii. The justification is explicit in the notes and trajectory. `CONVERSION_NOTES.md` says the implementation uses Suite2p preprocessing with `ops.npy` parameters and that the paper’s decoding used "slightly denoised dF/F ... averaged in bins of 10 consecutive timestamps." Trajectory step 54 says it would "mirror Suite2p baseline correction from `ops.npy`" and "bin both streams in 10-frame windows."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out of the loaded arrays. Instead, it checks that every ROI already satisfies `iscell >= 0.5` and raises an error if not, assuming the Track2p export is prefiltered.

ii.
```python
iscell = np.load(suite2p_dir / "iscell.npy")
...
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. `CONVERSION_NOTES.md` says all inspected `iscell.npy` files had probabilities above 0.5 and that "No extra cell filtering was applied beyond what is already encoded in the Track2p export."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of an artificial consecutive 2-minute block drawn from the continuous session, not to session start globally and not to any behavioral event.

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

iii. The notes justify this with the paper’s block-wise decoding description: the AI says it mirrored the paper’s "consecutive 2-minute blocks" directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion traces by averaging every 10 imaging frames. At 30 Hz this yields a time bin size of `10 / 30 = 0.333... s`, i.e. `333.333... ms`.

ii.
```python
def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
...
"time_bin_size": float((frame_bin / 30.0) * 1000.0),
"frame_bin_size": int(frame_bin),
```

iii. `CONVERSION_NOTES.md` explicitly says both streams were averaged in non-overlapping bins of 10 imaging frames to match the AI’s interpretation of the paper’s decoding preprocessing.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a dedicated raw timestamp array. It is derived from the sampled frame index after temporal binning together with the imaging frame rate `fs` from `ops.npy`.

ii.
```python
time_bin_s = frame_bin / fs
...
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
...
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes justify this as "absolute time from session start, sampled at the center of each 10-frame bin." The trajectory summary in step 107 also describes the decoder input as absolute time from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a continuous time axis in seconds at the center of each 10-frame bin, then slices that axis into the same 2-minute trial windows as the neural data.

ii.
```python
time_bin_s = frame_bin / fs
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
...
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The justification is explicit in metadata and notes: the AI wanted "Absolute time from session start, sampled at the center of each 10-frame bin."

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is created on the same binned time base as the neural signal and sliced with the same `start:end` indices for each artificial 2-minute trial.

ii.
```python
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The justification is implicit in the trialization strategy. The AI’s notes state that both neural and motion were binned first and then segmented into common blocks, with the input defined on that same bin grid.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output signal is based on `move_deve/motion_energy_glob.npy`. The AI also loads `interframe_int.npy` and `tstamps.npy` as timing metadata for alignment and repair, although `tstamps.npy` is not actually used in the repair logic.

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

iii. `CONVERSION_NOTES.md` says motion output came from `motion_energy_glob.npy` and camera timing metadata came from `tstamps.npy` and `interframe_int.npy`, motivated by the README’s note that some sessions have missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI repairs motion traces only when they are shorter than the imaging trace, detects putative dropped frames by thresholding `interframe_int` relative to its median, inserts `NaN` placeholders at those gaps, linearly interpolates missing values, averages motion in 10-frame bins, applies per-session min-max normalization, and later discretizes using global quintiles.

ii.
```python
def detect_gap_indices(interframe_int: np.ndarray) -> np.ndarray:
    ...
    median_dt = float(np.median(interframe_int))
    return np.flatnonzero(interframe_int > 1.5 * median_dt).astype(np.int64)

...
if diff > 0:
    ...
    for pos in gap_positions:
        insert_at = int(pos + 1 + offset)
        repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
        offset += 1
    ...
    repaired = interpolate_nans_1d(repaired)

motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)

def normalize_session_outputs_in_place(session_records: list[SessionRecord]) -> None:
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
        normalized = minmax_normalize(concatenated)
        ...
        record.output_continuous_trials = new_trials
```

iii. The notes provide the full rationale: only short traces are repaired "to avoid overcorrecting" equal-length traces, missing frames are handled because the data README allows missing-value treatment or interpolation, and 10-frame averaging was chosen because the AI believed the paper’s decoder used that exact denoising step.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After per-session min-max normalization and 10-frame averaging, the AI pools all trial values across the export, computes the 20th/40th/60th/80th percentiles, and uses `np.digitize` to assign five ordinal categories.

ii.
```python
all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
```

iii. `CONVERSION_NOTES.md` says the decoder output is "per-session min-max normalization applied after 10-frame averaging" followed by "discretization into 5 equal-frequency bins using global quintiles within the exported dataset."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural frame count first, then temporally bins motion and neural together, and finally slices them into the same 2-minute trial windows. When motion is longer than neural, it trims; when shorter, it inserts missing values and interpolates.

ii.
```python
if motion_energy.shape[0] == target_frames:
    return motion_energy, info

diff = int(target_frames - motion_energy.shape[0])
...
if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info
...
motion_aligned, motion_info = align_motion_to_imaging(..., target_frames=neural.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)
...
output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The justification is in both notes and trajectory: the AI wanted to "insert the missing motion frames instead of silently shifting behavior against imaging" (trajectory step 41), and it documented that only shorter traces were repaired to avoid overcorrection.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several mismatches heuristically. If motion is shorter than imaging, it inserts missing samples at detected camera gaps and interpolates. If motion is longer, it trims the extra frames. If too few gaps are found, it pads with `NaN` and interpolates. It also discards incomplete trial tails and raises an error if `iscell` suggests unfiltered ROIs.

ii.
```python
if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info

...
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
    offset += 1
if repaired.shape[0] < target_frames:
    pad = np.full(target_frames - repaired.shape[0], np.nan, dtype=np.float32)
    repaired = np.concatenate([repaired, pad], axis=0)
repaired = repaired[:target_frames]
repaired = interpolate_nans_1d(repaired)

if np.any(iscell_prob < 0.5):
    raise ValueError(...)
```

iii. The justification is explicit for the motion stream: `CONVERSION_NOTES.md` cites the data README’s statement that missing camera frames can be treated as missing values or interpolated. The other guardrails are not separately justified beyond the AI’s general goal of keeping neural and motion lengths synchronized.

## 6-a. What are the most time-consuming steps of the code?

i. The AI does not explicitly discuss runtime hotspots, but from the code the dominant expensive step is full-session neural preprocessing in `baseline_correct_fluorescence`, especially the Suite2p preprocessing or its local maximin fallback over every neuron and timepoint. Secondary cost comes from repeated full-session passes over motion data for normalization, discretization, and summary statistics.

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
...
for record in session_records:
    concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
    normalized = minmax_normalize(concatenated)
```

iii. No explicit performance justification was written in `CONVERSION_NOTES.md` or the trajectory. This answer is inferred from the implementation structure.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized. The clearest is the repeated `np.insert` loop for missing motion frames. The trial-splitting loop also repeatedly appends slices one trial at a time, and motion-output handling traverses the trial lists multiple times for concatenation, normalization, discretization, and metadata summaries.

ii.
```python
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
    offset += 1

for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
    output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))

for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
```

iii. The AI did not explicitly justify these loops. This is inferred from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats motion-output passes. It first stores continuous per-trial motion, then re-concatenates those trials for per-session min-max normalization, then loops again to collect all motion values for global quintile edges, then loops again to discretize each trial, then loops again to build metadata summaries, and later loops again for dataset summaries.

ii.
```python
def normalize_session_outputs_in_place(session_records: list[SessionRecord]) -> None:
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
        ...
        for trial in record.output_continuous_trials:
            ...

all_motion = []
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))

for record in session_records:
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

iii. No explicit justification is given. The repetition appears to be an implementation convenience rather than a documented design choice.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `tstamps.npy` but never uses it for alignment. It also preserves and reprocesses continuous normalized motion trials even though the downstream decoder only consumes the discretized categories. The per-trial motion summary metadata is also computed solely for documentation, not for downstream analysis.

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
```

iii. No explicit justification is given for these extra steps. The only partial rationale is the AI’s general emphasis on documentation and sanity checks in `CONVERSION_NOTES.md`.
