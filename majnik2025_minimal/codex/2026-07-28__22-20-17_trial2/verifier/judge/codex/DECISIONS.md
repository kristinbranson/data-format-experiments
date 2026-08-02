# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over a hardcoded list of six subject IDs, then over subdirectories within each subject whose names begin with four digits. For each session it loads Suite2p fluorescence files (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`) and motion files (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`), then immediately preprocesses and trializes the session rather than storing a raw-session object.

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
```

```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The justification in `CONVERSION_NOTES.md` is that the delivered dataset consists of Track2p-matched Suite2p outputs for six mice and that decoding should mirror the paper’s 10-frame-binned continuous-session representation. The trajectory also shows the agent explicitly deciding to use “the paper’s six mice and 6–7 consecutive days per mouse” and to load full continuous recordings before binning and splitting into trials.

## 1-b. How are the data split into subjects?

i. Subjects are split by membership in the hardcoded subject list `["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]`. The final `subjects` field is rebuilt from the loaded session records by taking the sorted set of `record.subject`.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]
SAMPLE_SUBJECTS = FULL_SUBJECTS
```

```python
subjects = sorted({record.subject for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[record.subject])
```

iii. The notes justify this by stating that the full dataset contains six subjects and the trajectory says the agent “confirmed the sessions line up with the paper’s six mice.” There is no further code-driven discovery rule; the split is based on a fixed list chosen from inspection.

## 1-c. How are the data split into sessions?

i. Each qualifying subdirectory under a hardcoded subject folder is treated as one session, provided the directory name starts with four digits. Sessions are sorted lexicographically and then loaded one by one into `SessionRecord` objects.

ii.
```python
for subject in subjects:
    subject_dir = data_dir / subject
    sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
    if per_subject_limit is not None:
        sessions = sessions[:per_subject_limit]
    session_dirs.extend(sessions)
```

```python
session_date = session_dir.name.split("_")[0]
...
return SessionRecord(
    subject=session_dir.parent.name,
    session_name=session_dir.name,
    session_date=session_date,
    ...
)
```

iii. The justification is mostly implicit: the agent inspected the data layout, concluded that the relevant recordings are per-day subdirectories for each mouse, and then encoded that assumption as “all date-like subdirectories” inside the known six subject folders.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions and artificially splits them into consecutive 2-minute blocks after 10-frame temporal averaging. Trial boundaries are therefore imposed on the binned continuous data, not taken from any raw trial metadata.

ii.
```python
def split_into_trials(
    neural_binned: np.ndarray,
    motion_binned: np.ndarray,
    fs: float,
    frame_bin: int,
    trial_duration_s: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))
    n_complete_trials = neural_binned.shape[1] // bins_per_trial
```

```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
    output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The justification is explicit in both `CONVERSION_NOTES.md` and the trajectory: the agent believed the paper’s decoder used “consecutive 2-minute blocks” and therefore “mirrors that structure directly.” Step 54 in the trajectory says it would “split sessions into 2-minute trials.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a separate trial-quality filter. The only trial-level exclusion is structural: incomplete trailing bins that do not fill a full 2-minute block are discarded by truncation before trial splitting.

ii.
```python
n_complete_trials = neural_binned.shape[1] // bins_per_trial

neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
```

```python
info = {
    "time_bin_s": float(time_bin_s),
    "bins_per_trial": int(bins_per_trial),
    "n_trials": int(n_complete_trials),
    "dropped_binned_timepoints": int(dropped_bins),
}
```

iii. The notes do not claim any trial-quality curation beyond constructing fixed-length consecutive blocks. The justification is therefore implicit: once the agent decided on fixed 2-minute trialization, it kept only complete blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived directly from `F.npy` and `Fneu.npy`, with preprocessing parameters read from `ops.npy`. `iscell.npy` is loaded only as a consistency check, not as a numerical input to the neural traces.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")
```

```python
neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
```

iii. `CONVERSION_NOTES.md` states that neural input came from `F.npy` and `Fneu.npy`, and that the session `ops.npy` parameters define the Suite2p-style baseline correction. The trajectory also shows the agent verifying that the intended neural signal should be neuropil-subtracted and baseline-corrected rather than further normalized by baseline.

## 2-b. How is the `neural` data processed?

i. Neural traces are processed by neuropil subtraction followed by Suite2p-style baseline correction using the per-session `ops.npy` parameters. After that, the AI averages the corrected traces in non-overlapping bins of 10 imaging frames before trialization.

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
```

```python
neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32, copy=False)
```

iii. The notes justify the first stage as mirroring Suite2p preprocessing and claim the paper’s decoder used “slightly denoised dF/F ... averaged in bins of 10 consecutive timestamps.” The trajectory repeats that rationale and says the converter will “mirror Suite2p baseline correction ... [and] bin both streams in 10-frame windows.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied to the stored rows. The AI only checks that all rows in `iscell.npy` already satisfy `iscell >= 0.5`; if not, it aborts.

ii.
```python
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

```python
return SessionRecord(
    ...
    brain_region_idx=np.zeros(neural.shape[0], dtype=np.int64),
    ...
)
```

iii. `CONVERSION_NOTES.md` says the source data already contain Track2p-matched cells saved back into Suite2p format and that no extra cell filtering was applied beyond what is already encoded in the export. It also cites inspection of `iscell.npy` files showing all ROI probabilities above 0.5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of an imposed consecutive 2-minute block cut from a continuous session. Within each block, neural samples are represented at 10-frame-bin resolution.

ii.
```python
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block cut from a continuous session",
    "off_start": 0.0,
    "off_end": float(trial_duration_s),
    ...
}
```

iii. The justification is that the agent interpreted the reference decoding analysis as operating on consecutive 2-minute blocks. That interpretation appears in both the notes and trajectory and is what drove the alignment choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have a temporal resolution of 10 imaging frames per sample, which at 30 Hz is `10/30 = 0.333... s` or `333.333... ms`. Yes: the AI explicitly rebins both neural and motion streams by non-overlapping averaging over 10-frame windows.

ii.
```python
def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

```python
"metadata": {
    "time_bin_size": float((frame_bin / 30.0) * 1000.0),
    ...
    "frame_bin_size": int(frame_bin),
}
```

iii. `CONVERSION_NOTES.md` explicitly justifies the 10-frame averaging by citing “slightly denoised dF/F and behavior traces, averaged in bins of 10 consecutive timestamps.” The trajectory mirrors that explanation.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time trace is generated from frame indices and the imaging frame rate stored in `ops["fs"]`; it is not loaded from a raw timestamp array. Because the AI rebins first, the actual values correspond to centers of 10-frame bins rather than raw frames.

ii.
```python
time_bin_s = frame_bin / fs
...
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
...
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

```python
input_names": ["time_from_session_start_s"],
```

iii. The notes justify this as “absolute time from session start, sampled at bin centers.” The trajectory’s final summary says the decoder input is “absolute time from session start.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time after temporal averaging, using one scalar per 10-frame bin. It converts bin index to seconds using `frame_bin / fs` and uses bin centers by adding `0.5` before scaling.

ii.
```python
time_bin_s = frame_bin / fs
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
```

```python
"input_description": "Absolute time from session start, sampled at the center of each 10-frame bin.",
```

iii. The notes explicitly say the decoder input is “represented as a time-varying trace sampled at bin centers,” which is the stated rationale for the center-of-bin convention.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Input time is aligned by construction to the same 10-frame bins and the same trial slices used for neural data. Each trial’s input array is sliced with the same `start:end` bin indices as the corresponding neural trial.

ii.
```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The justification is implicit in the code structure and explicit in the notes’ statement that both neural and motion traces were averaged in the same non-overlapping 10-frame bins before trialization.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion signal is numerically derived from `move_deve/motion_energy_glob.npy`, with `interframe_int.npy` used to detect gaps when the motion trace is shorter than imaging. `tstamps.npy` is loaded but not used by the code.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

```python
motion_aligned, motion_info = align_motion_to_imaging(
    motion_energy=motion_energy,
    tstamps=tstamps,
    interframe_int=interframe_int,
    target_frames=neural.shape[1],
)
```

iii. `CONVERSION_NOTES.md` says motion output came from `motion_energy_glob.npy` and camera timing metadata came from `tstamps.npy` and `interframe_int.npy`. It further justifies consulting timing metadata because the data README warned that some sessions contain missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first aligns motion to imaging length by inserting missing values at detected camera gaps and interpolating. It then averages motion in 10-frame non-overlapping bins, applies per-session min-max normalization, pools all normalized trial values across the export, and later discretizes them into five global bins.

ii.
```python
if diff > 0:
    info["length_fix_strategy"] = "insert_nan_at_detected_camera_gaps_then_interpolate"
    ...
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
    ...
    repaired = interpolate_nans_1d(repaired)
```

```python
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)
```

```python
def normalize_session_outputs_in_place(session_records: list[SessionRecord]) -> None:
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
        normalized = minmax_normalize(concatenated)
        ...
        record.output_continuous_trials = new_trials
```

iii. The notes justify these steps by citing the README statement that missing camera frames may be interpolated, the paper’s 10-timestamp averaging for decoder analyses, and a task-specific choice to apply “per-session min-max normalization ... after 10-frame averaging” before equal-frequency binning.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global quintile cut points from all normalized motion values in the exported dataset and assigns categories with `np.digitize`. The resulting labels correspond to 0–20, 20–40, 40–60, 60–80, and 80–100 percentiles.

ii.
```python
all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```

```python
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
```

iii. `CONVERSION_NOTES.md` justifies this as “discretization into 5 equal-frequency bins using global quintiles within the exported dataset,” matching the decoder task’s requirement for five equal-percentile bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to the neural data in two stages: first it repairs the continuous motion trace to the same frame count as the continuous neural trace, then it averages both streams with the same 10-frame windows and slices them into the same 2-minute trials.

ii.
```python
motion_aligned, motion_info = align_motion_to_imaging(
    motion_energy=motion_energy,
    tstamps=tstamps,
    interframe_int=interframe_int,
    target_frames=neural.shape[1],
)
```

```python
neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32, copy=False)
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)
...
output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The justification in the notes is that some sessions have missing camera frames, so motion should be repaired only when it is shorter than imaging to avoid “overcorrecting” sessions whose lengths already match. The trajectory repeats that exact rationale in step 48.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or mismatched motion samples heuristically. If motion is shorter than imaging, it detects large camera-interval gaps, inserts `NaN`s, interpolates them, and pads with additional `NaN`s if needed before interpolation; if motion is longer than imaging, it trims the excess. It also aborts if an `iscell` probability below 0.5 is encountered and discards incomplete trailing bins when making full-length trials.

ii.
```python
if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info
```

```python
if diff > 0:
    ...
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
    ...
    if repaired.shape[0] < target_frames:
        pad = np.full(target_frames - repaired.shape[0], np.nan, dtype=np.float32)
        repaired = np.concatenate([repaired, pad], axis=0)
    repaired = repaired[:target_frames]
    repaired = interpolate_nans_1d(repaired)
```

```python
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. The notes justify interpolation by citing the dataset README’s statement that missing camera frames can be treated as missing values or interpolated. The “repair only short sessions” rule is explicitly justified in the notes and step 48 of the trajectory as a way to avoid overcorrecting sessions whose motion length already matches imaging.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant expensive step is per-session neural baseline correction, especially the Suite2p-style `preprocess` call or its local fallback over long `(neurons x time)` arrays. Secondary costs come from repeatedly loading large `.npy` files and from repeated concatenation/insertion operations for motion processing.

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
```

```python
repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
```

iii. Step 67 of the trajectory explicitly says “Most of the time here is the per-session Suite2p-style baseline correction over the 41 continuous recordings.” There is no explicit AI justification for the remaining hotspots; those are evident from the code structure.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized or at least reduced: repeated `np.insert` inside `align_motion_to_imaging`, per-trial slicing/appending in `split_into_trials`, per-trial `np.digitize` in `build_dataset`, and the per-session reconstruction loop in `normalize_session_outputs_in_place`.

ii.
```python
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan).astype(np.float32, copy=False)
    offset += 1
```

```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
    output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

```python
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
```

iii. The AI only explicitly commented on runtime in relation to baseline correction. There is no explicit justification for leaving these loops unvectorized; this is an evaluation of the implementation choices visible in the final code.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly traverses trial arrays for motion handling: once to build trialized continuous outputs, again to min-max normalize them session by session, again to concatenate all trials for global quantile edges, again to discretize each trial, and again to compute summary metadata. It also recomputes and stores some per-trial summaries that are not needed for the final exported tensors.

ii.
```python
for record in session_records:
    concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
    normalized = minmax_normalize(concatenated)
    ...
    record.output_continuous_trials = new_trials
```

```python
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))
...
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
    discretized_trials.append(bins)
```

```python
for trial in record.output_continuous_trials:
    normalized_trials.append({
        "min": float(np.min(trial)),
        "max": float(np.max(trial)),
        "mean": float(np.mean(trial)),
    })
```

iii. There is no explicit justification in the notes or trajectory for these repeated passes. The repeated processing is an implementation artifact of the staged design in `convert_dataset`, `normalize_session_outputs_in_place`, `build_dataset`, and `summarize_dataset`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` but never uses it. It also computes and stores extensive metadata and summaries (`session_date`, `session_info`, `motion_trial_summary_before_discretization`, dataset summaries) that are not used by downstream decoder training, and it constructs continuous-motion trial summaries only to discard the continuous values after discretization in the exported dataset.

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
```

```python
session_date = session_dir.name.split("_")[0]
session_info = {
    "subject": session_dir.parent.name,
    "session_name": session_dir.name,
    "session_date": session_date,
    ...
}
```

```python
normalized_trials = []
for trial in record.output_continuous_trials:
    normalized_trials.append({
        "min": float(np.min(trial)),
        "max": float(np.max(trial)),
        "mean": float(np.mean(trial)),
    })
info["motion_trial_summary_before_discretization"] = normalized_trials[:2]
```

iii. The notes justify extra metadata as documentation and sanity checking, but they do not claim it is required by downstream analysis. The unused `tstamps.npy` load has no explicit justification in the final materials.
