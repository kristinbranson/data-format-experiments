# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six subject IDs, walks each subject directory under `/app/data`, finds date-named session folders, and loads each session from `suite2p/plane0` and `move_deve`. Within each session it loads fluorescence (`F.npy`, `Fneu.npy`), Suite2p metadata (`ops.npy`, `iscell.npy`), and behavior (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`). It then converts each continuous session into trial lists before building the final dataset dictionary.

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

iii. `CONVERSION_NOTES.md` says the source data already contain Track2p-matched cells saved back into Suite2p format, so the agent treated the delivered per-session Suite2p folders plus `move_deve` motion files as the authoritative source. In the trajectory, the agent also explicitly confirmed that the dataset consisted of six mice with 6 to 7 daily sessions and continuous per-day recordings.

## 1-b. How are the data split into subjects?

i. Subjects are defined by top-level directory names (`jm031`, `jm032`, etc.). Later, the final `subjects` list is the sorted unique set of `record.subject`, and `subject_idx` maps each session to its subject.

ii.
```python
for subject in subjects:
    subject_dir = data_dir / subject
    sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])

subjects = sorted({record.subject for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[record.subject])
```

iii. The data README says each subject has its own folder and that the six subject IDs map to the paper’s mice A-F. The trajectory shows the agent enumerated these folders and used that layout directly.

## 1-c. How are the data split into sessions?

i. Sessions are defined as one dated subdirectory per recording day inside each subject folder. The agent sorts them lexicographically, which matches chronological order because the names use `YYYY-MM-DD_a`.

ii.
```python
sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
...
session_date = session_dir.name.split("_")[0]
session_name = session_dir.name
```

iii. The data README says each session subfolder corresponds to one recording day and the `_a` suffix can be ignored. The agent’s trajectory shows it checked the per-subject session lists and relied on those dated directories.

## 1-d. How are the data split into trials?

i. The raw recordings are continuous sessions, so the agent creates artificial trials by first binning the neural and motion traces, then cutting each session into consecutive non-overlapping 2-minute blocks. Each block becomes one “trial.”

ii.
```python
def split_into_trials(neural_binned, motion_binned, fs, frame_bin, trial_duration_s):
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))
    n_complete_trials = neural_binned.shape[1] // bins_per_trial
    ...
    for trial_idx in range(n_complete_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
        output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. `CONVERSION_NOTES.md` says the paper’s decoding analysis used consecutive 2-minute blocks and that the conversion mirrors that structure. The trajectory also shows the agent explicitly chose a 2-minute split because the paper said cross-validation splits were done on consecutive 2-minute blocks.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only trial-level curation is structural: incomplete trailing bins are dropped during binning, incomplete trailing 2-minute blocks are dropped during trialization, and sessions would fail earlier if cell filtering assumptions were violated.

ii.
```python
def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    ...

def split_into_trials(...):
    n_complete_trials = neural_binned.shape[1] // bins_per_trial
    neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
    motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
```

iii. The notes do not describe any additional trial QC. The agent treated the recordings as continuous behavior and only removed incomplete end fragments created by its own binning and block segmentation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence traces `F.npy` and neuropil traces `Fneu.npy`, with preprocessing parameters taken from `ops.npy`. `iscell.npy` is used only as a sanity check on cell inclusion.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")

neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
```

iii. `CONVERSION_NOTES.md` says the input neural data came from `F.npy` and `Fneu.npy`. In the trajectory, the agent quoted the methods text saying the paper used baseline-corrected fluorescence traces as dF/F using default Suite2p parameters.

## 2-b. How is the `neural` data processed?

i. The agent computes Suite2p-style baseline-corrected fluorescence by subtracting neuropil (`F - neucoeff * Fneu`) and then subtracting a baseline estimated with Suite2p’s default maximin-style routine. It uses `suite2p.extraction.dcnv.preprocess` when available and otherwise reproduces the same logic locally. Afterward it averages neural traces in non-overlapping bins of 10 imaging frames.

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
Flow = gaussian_filter(Fc, [0.0, sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
...
neural_binned = average_nonoverlapping(neural, frame_bin).astype(np.float32, copy=False)
```

iii. The notes say the implementation mirrors Suite2p-style baseline correction using each session’s `ops.npy` parameters. In the trajectory, the agent explicitly investigated whether the paper meant full dF/F normalization or only Suite2p baseline correction, then concluded the paper matched neuropil subtraction plus maximin baseline subtraction without an additional division step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not actively re-filter cells. Instead it assumes the Track2p-exported Suite2p folders already contain only matched cells passing the paper’s `iscell > 0.5` rule. It verifies this assumption by checking whether any row has `iscell_prob < 0.5`, and raises an error if so.

ii.
```python
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. `CONVERSION_NOTES.md` says all inspected `iscell.npy` files had probabilities above 0.5 and that no extra cell filtering was applied beyond the Track2p export. The trajectory shows the agent checked the `iscell` ranges and decided the export was already filtered.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions do not define a stimulus event, because the data are continuous spontaneous recordings. The agent therefore aligns each neural trial to the start of the corresponding 2-minute block cut from the session. `metadata["temporal_alignment_event"]` is set to the start of each block.

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

iii. `CONVERSION_NOTES.md` says each session was segmented into consecutive 2-minute blocks and used those as trials. The trajectory shows the agent intentionally used the paper’s 2-minute decoding blocks as the alignment anchor because there was no event-aligned task structure in the source data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so each time bin is `10 / 30 = 0.333...` seconds, or `333.333...` ms. Yes, explicit temporal rebinning is applied to both neural and motion traces.

ii.
```python
def average_nonoverlapping(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n_complete = arr.shape[-1] // bin_size
    ...

time_bin_s = frame_bin / fs
...
"time_bin_size": float((frame_bin / 30.0) * 1000.0),
"frame_bin_size": int(frame_bin),
```

iii. The notes say the paper’s decoding analysis slightly denoised dF/F and behavior by averaging bins of 10 consecutive timestamps, and explicitly state the resulting time bin size is 333.333 ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. This input is not loaded from a dedicated raw array. It is derived from the imaging sampling rate in `ops.npy`, the total number of imaging frames after binning, and the bin indices within each session.

ii.
```python
time_bin_s = frame_bin / fs
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
...
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. `CONVERSION_NOTES.md` says the decoder input is `time_from_session_start_s`, represented as a time-varying trace sampled at bin centers. The trajectory also states that the agent chose absolute time from session start as the decoder input.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent constructs a continuous time axis for the full session after 10-frame binning, uses bin centers rather than edges, and slices that binned time axis into the same 2-minute trial blocks as the neural data.

ii.
```python
time_bin_s = frame_bin / fs
...
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes explicitly say the input is sampled at bin centers. The trajectory shows the agent wanted the input to match the same binned timeline as the paper’s denoised neural and behavioral traces.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is built from the same binned session timeline as the neural data and is sliced with the same `[start:end]` indices for each 2-minute trial, so it is aligned one-to-one with neural time bins.

ii.
```python
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
output_trials.append(motion_binned[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. This alignment is implicit in the shared trialization code. `CONVERSION_NOTES.md` also frames the input as a time-varying trace sampled on the same binned timeline as the decoder variables.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from the precomputed motion-energy time series `move_deve/motion_energy_glob.npy`. The agent also reads `tstamps.npy` and `interframe_int.npy` to repair length mismatches against the imaging trace.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")

motion_aligned, motion_info = align_motion_to_imaging(
    motion_energy=motion_energy,
    tstamps=tstamps,
    interframe_int=interframe_int,
    target_frames=neural.shape[1],
)
```

iii. The data README says `move_deve` contains processed behavioral data in `motion_energy_glob.npy`, and that `tstamps.npy` and `interframe_int.npy` can be used to identify missing camera frames. The trajectory shows the agent relied on that processed motion-energy stream because the raw videos were not provided.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent does not recompute motion energy from video. Instead it repairs missing camera frames when needed, bins the motion trace in 10-frame averages, min-max normalizes it per session, then later discretizes it into categorical bins.

ii.
```python
motion_aligned, motion_info = align_motion_to_imaging(...)
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)

def normalize_session_outputs_in_place(session_records: list[SessionRecord]) -> None:
    for record in session_records:
        concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
        normalized = minmax_normalize(concatenated)
        ...
        record.output_continuous_trials = new_trials
```

iii. `CONVERSION_NOTES.md` says motion output came from `motion_energy_glob.npy`, was aligned to imaging frames, averaged in non-overlapping bins of 10 frames, and then min-max normalized per session before discretization. The trajectory shows the agent viewed normalization and discretization as task-specific additions on top of the paper’s 10-frame denoising.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After per-session min-max normalization, the agent concatenates all motion values from all trials in the export, computes the 20th/40th/60th/80th percentiles, and uses those global edges to assign integer classes 0-4 with `np.digitize`.

ii.
```python
all_motion = []
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))
all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the decoder output is discretized into 5 equal-frequency bins using global quintiles within the exported dataset. The trajectory’s final summary repeats that the output is per-session normalized and then discretized into 5 global quintile bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent aligns motion to the imaging trace at the frame level first. If the motion series is shorter than the neural series, it detects camera gaps from `interframe_int`, inserts NaNs at those gap positions, interpolates across the missing values, and pads or trims to the neural frame count. It then applies the same 10-frame binning and same 2-minute trial slices as the neural data.

ii.
```python
def detect_gap_indices(interframe_int: np.ndarray) -> np.ndarray:
    median_dt = float(np.median(interframe_int))
    return np.flatnonzero(interframe_int > 1.5 * median_dt).astype(np.int64)

if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info

if diff > 0:
    info["length_fix_strategy"] = "insert_nan_at_detected_camera_gaps_then_interpolate"
    ...
    repaired = interpolate_nans_1d(repaired)

motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0].astype(np.float32, copy=False)
```

iii. The data README says missing camera frames can be identified from `tstamps.npy` or `interframe_int.npy` and treated as missing or interpolated. `CONVERSION_NOTES.md` says the agent only repaired sessions where the motion trace was actually shorter than the imaging trace, to avoid overcorrecting sessions that already matched imaging length.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion samples are repaired by inserting NaNs at detected camera gaps and linearly interpolating. If all values were missing, the helper would return zeros. Excess motion frames are trimmed. Incomplete trailing bins and incomplete trailing 2-minute blocks are dropped. Unexpected `iscell < 0.5` entries are treated as a hard error instead of being filtered dynamically.

ii.
```python
def interpolate_nans_1d(x: np.ndarray) -> np.ndarray:
    ...
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    ...

if diff < 0:
    info["length_fix_strategy"] = "trim_excess_motion_frames"
    return motion_energy[:target_frames], info

trimmed = arr[..., : n_complete * bin_size]
...
if np.any(iscell_prob < 0.5):
    raise ValueError(...)
```

iii. The justification in `CONVERSION_NOTES.md` is that the data README explicitly allowed missing motion frames to be treated as missing values or interpolated, and the agent chose interpolation only when the motion trace was shorter than the imaging trace. The notes also explain that the agent kept full delivered session lengths rather than trimming longer recordings to the 20-minute duration stated in the copied methods text.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is the per-session Suite2p-style baseline correction over large `F` and `Fneu` matrices, especially when `suite2p_preprocess` or the local Gaussian/minimum/maximum filtering fallback runs across all neurons and frames. The second major cost is repeatedly concatenating and iterating through motion-output trials to normalize, discretize, and summarize them after trialization.

ii.
```python
neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
...
Flow = gaussian_filter(Fc, [0.0, sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
...
concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials], axis=0)
...
all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
```

iii. The trajectory explicitly says “Most of the time here is the per-session Suite2p-style baseline correction over the 41 continuous recordings.” The rest follows from the code structure: motion processing is revisited several times after trialization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could be vectorized or collapsed: the trial-building loop in `split_into_trials`, the session/trial loops used to collect motion values for global quantiles, the session/trial loops used to discretize motion and compute metadata summaries, and the loop that reconstructs per-trial normalized motion after session-level concatenation.

ii.
```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)

for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))
...
for record in session_records:
    ...
    for trial in record.output_continuous_trials:
        bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
        discretized_trials.append(bins)
```

iii. The agent did not justify these loops as intentional; they are simply how the converter was written. They are not needed for correctness, and much of this work could be done with reshaping or batched array operations.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly walks the same session/trial structure for related motion-output work: once to normalize per session, again to gather all values for global quintiles, again to discretize trials, again to build per-trial summary metadata, and again in `summarize_dataset` to count class frequencies. It also revisits the same binned session timeline separately to create neural, input, and output trial lists.

ii.
```python
normalize_session_outputs_in_place(session_records)
...
all_motion = []
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))
...
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
...
for trial in record.output_continuous_trials:
    normalized_trials.append({
        "min": float(np.min(trial)),
        "max": float(np.max(trial)),
        "mean": float(np.mean(trial)),
    })
...
for session_trials in data["output"]:
    for trial in session_trials:
        values, counts = np.unique(trial, return_counts=True)
```

iii. This is a code-structure observation rather than an explicit stated decision. The trajectory does show the agent focusing on correctness and validation first, not reducing passes over the data.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` into `align_motion_to_imaging` but never uses it. It stores continuous motion trials in `output_continuous_trials`, then discards them after discretization in the final dataset. It also computes per-trial motion summary stats only for metadata, even though the decoder never uses them. More broadly, it builds session metadata such as repair counts and motion summaries that are useful for notes but irrelevant to downstream decoding.

ii.
```python
tstamps = np.load(move_dir / "tstamps.npy")
...
def align_motion_to_imaging(motion_energy, tstamps, interframe_int, target_frames):
    motion_energy = np.asarray(motion_energy, dtype=np.float32)
    tstamps = np.asarray(tstamps)
    interframe_int = np.asarray(interframe_int)
    ...

output_continuous_trials: list[np.ndarray]
...
normalized_trials = []
for trial in record.output_continuous_trials:
    normalized_trials.append({
        "min": float(np.min(trial)),
        "max": float(np.max(trial)),
        "mean": float(np.mean(trial)),
    })
info["motion_trial_summary_before_discretization"] = normalized_trials[:2]
```

iii. The notes justify rich metadata and validation, but these extra computations are not consumed by `train_decoder.py`. They were added for documentation and sanity-checking rather than the decoder itself.
