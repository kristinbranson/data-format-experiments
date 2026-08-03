# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over a hardcoded list of 6 subject IDs (`FULL_SUBJECTS`), finds all session subdirectories within each subject folder under `/app/data/`, and for each session loads the Suite2p neural data files (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`) from `suite2p/plane0/` and behavioral data files (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`) from `move_deve/`. Each session is processed individually by `load_session()`, producing a `SessionRecord`. All session records are collected into a list for downstream assembly.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]

def select_session_dirs(data_dir: Path, mode: str) -> list[Path]:
    # ...
    for subject in subjects:
        subject_dir = data_dir / subject
        sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        # ...
        session_dirs.extend(sessions)
    return session_dirs

def load_session(session_dir, device, frame_bin, trial_duration_s):
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

iii. The AI followed the data organization described in the data README.md and the paper's load_data.ipynb notebook. The subject list was determined from the 6 subject folders present in the data directory, matching the paper's statement of "6 mice imaged daily."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by their folder names under the data directory (e.g., `jm031`, `jm032`, etc.). Each subject's sessions are grouped by iterating over subdirectories within the subject folder. A sorted list of unique subjects is maintained, and each session is mapped to its subject via `subject_to_idx`.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]

subjects = sorted({record.subject for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
# ...
subject_idx.append(subject_to_idx[record.subject])
```

iii. The AI identified 6 subjects from the data directory structure, consistent with the paper's statement of 6 mice used for all subsequent analyses.

## 1-c. How are the data split into sessions?

i. Within each subject directory, all subdirectories whose names start with a 4-digit year (e.g., `2023-10-18_a`) are treated as sessions. They are sorted alphabetically (which equals chronological order given the date naming convention). Each session directory corresponds to one recording day.

ii.
```python
sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

iii. The AI followed the data README which states "Each subject folder contains a number of session folders, each corresponding to one recording day."

## 1-d. How are the data split into trials?

i. Each continuous recording session is split into consecutive non-overlapping 2-minute (120-second) blocks. After 10-frame temporal binning (at 30 Hz), each trial contains exactly 360 binned timepoints. Incomplete blocks at the end of sessions are dropped.

ii.
```python
def split_into_trials(neural_binned, motion_binned, fs, frame_bin, trial_duration_s):
    time_bin_s = frame_bin / fs
    bins_per_trial = int(round(trial_duration_s / time_bin_s))  # 360
    n_complete_trials = neural_binned.shape[1] // bins_per_trial
    neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
    motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
    for trial_idx in range(n_complete_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:end])
        # ...
```

iii. The methods text states "splits were done on consecutive 2 minute blocks of the recording" and "averaging in bins of 10 consecutive timestamps." The AI directly implemented this.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Only incomplete trailing blocks (fewer than 360 bins) are dropped. All complete 2-minute blocks are kept.

ii.
```python
n_complete_trials = neural_binned.shape[1] // bins_per_trial
neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
```

iii. The paper does not describe any trial-level quality filtering. The AI noted this in CONVERSION_NOTES.md and kept all complete trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence traces), with processing parameters from `ops.npy`.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
```

iii. The paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The load_data.ipynb also loads F.npy and notes to "compute dF/F the way as described in the paper."

## 2-b. How is the `neural` data processed?

i. Processing follows Suite2p's baseline correction pipeline: (1) neuropil subtraction: `Fc = F - 0.7 * Fneu`, (2) maximin baseline estimation: Gaussian smoothing then min-max filtering, (3) baseline subtraction: `dF/F = Fc - Flow`. Then 10-frame non-overlapping averaging is applied for denoising.

ii.
```python
def baseline_correct_fluorescence(F, Fneu, ops, device):
    neucoeff = float(ops.get("neucoeff", 0.7))
    Fc = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    # maximin baseline:
    Flow = gaussian_filter(Fc, [0.0, sig_baseline])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    return (Fc - Flow).astype(np.float32)

# Then binning:
neural_binned = average_nonoverlapping(neural, frame_bin)  # frame_bin=10
```

iii. The paper says "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and "averaging in bins of 10 consecutive timestamps." The AI used the suite2p_preprocess function when available, with a fallback implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied beyond what Track2p already provides. The AI verifies that all iscell probabilities are >= 0.5 (they are, since Track2p export already filters) but does not remove any neurons.

ii.
```python
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. The paper states "We considered all ROIs above the default threshold of 0.5 as true cells." The AI's CONVERSION_NOTES.md confirms: "All inspected iscell.npy files had probabilities above 0.5 for every stored ROI."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the start of its corresponding 2-minute block within the session. The first trial starts at frame 0 of the session, the second at frame 3600 (120s * 30Hz), etc. There is no specific stimulus event; the alignment is to the start of each consecutive temporal block.

ii.
```python
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end])
```

iii. The paper describes "splits were done on consecutive 2 minute blocks." There is no stimulus onset or behavioral event to align to in this spontaneous activity paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 333.33 ms (10 frames / 30 Hz). Temporal rebinning is applied: non-overlapping averaging of 10 consecutive imaging frames.

ii.
```python
def average_nonoverlapping(arr, bin_size):
    n_complete = arr.shape[-1] // bin_size
    trimmed = arr[..., : n_complete * bin_size]
    new_shape = (*trimmed.shape[:-1], n_complete, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

# time_bin_size in metadata:
"time_bin_size": float((frame_bin / 30.0) * 1000.0),  # 333.33 ms
```

iii. The paper states "averaging in bins of 10 consecutive timestamps" at 30 Hz imaging rate.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from a raw data variable. It is computed from the frame indices and the known sampling rate (from `ops["fs"]`). Specifically, it is the time coordinate of each binned timepoint, calculated as the center of each 10-frame bin.

ii.
```python
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
# ...
input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. The instructions specify "Time elapsed from the beginning of the experiment" as the decoder input. Since frame timing is regular at 30 Hz, the AI computed this from frame indices rather than raw timestamp data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `(bin_index + 0.5) * time_bin_s` where `time_bin_s = frame_bin / fs = 10/30 = 0.333s`. The `+0.5` places the timestamp at the center of each bin. The time is absolute from session start (not reset per trial).

ii.
```python
time_bin_s = frame_bin / fs  # 10/30 = 0.333s
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
```

iii. The AI chose bin-center timestamps to accurately represent the temporal position of each averaged sample.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time axis is computed from the same array indices as the neural data, so alignment is inherent. Each trial's input slice corresponds exactly to the same time bins as its neural data slice.

ii.
```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end])
    input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. Since both neural and input data use the same binned frame indexing, they are automatically aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in each session's `move_deve/` directory. Camera timing metadata from `tstamps.npy` and `interframe_int.npy` are used for alignment when frame counts differ.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The paper describes motion energy as "pixel-wise difference of consecutive frames" from videography, which is stored in `motion_energy_glob.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Align motion trace to imaging frame count (insert NaN at detected gaps and interpolate, or trim if too long), (2) 10-frame non-overlapping averaging, (3) per-session min-max normalization to [0,1], (4) discretization into 5 bins using global quintile edges.

ii.
```python
motion_aligned, motion_info = align_motion_to_imaging(motion_energy, tstamps, interframe_int, target_frames=neural.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0]
# ...
normalize_session_outputs_in_place(session_records)  # per-session min-max
# ...
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(trial, global_edges, right=False)
```

iii. The AI followed the paper's instruction to average in 10-frame bins and added per-session normalization before discretization. The task instructions say "normalized and discretized into five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After per-session min-max normalization, all motion values across all sessions/trials are pooled. Global quintile edges are computed at the 20th, 40th, 60th, and 80th percentiles. `np.digitize` maps each value to one of 5 bins (0-4), producing equal-frequency categories.

ii.
```python
all_motion_concat = np.concatenate(all_motion, axis=0)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins." The AI computed global quintiles to ensure each bin contains approximately 20% of all data points.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first aligned to imaging frame count via `align_motion_to_imaging()`, then binned with the same 10-frame averaging and split into the same trial boundaries as neural data.

ii.
```python
motion_aligned, _ = align_motion_to_imaging(motion_energy, tstamps, interframe_int, target_frames=neural.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0]
# Then same split_into_trials as neural
```

iii. The paper notes videos were recorded at 30 Hz "with the microscope acquisition acting as a trigger for camera frame acquisition, also allowing for simple synchronisation across the two modalities." The AI aligned frame counts and used identical binning/trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. When motion energy traces are shorter than imaging traces (missing camera frames), the AI detects gaps using `interframe_int > 1.5 * median(interframe_int)`, inserts NaN at detected gap positions, then linearly interpolates over them. When motion traces are longer, they are trimmed. Total repaired frames in full dataset: 276.

ii.
```python
def align_motion_to_imaging(motion_energy, tstamps, interframe_int, target_frames):
    if motion_energy.shape[0] == target_frames:
        return motion_energy, info
    if diff < 0:  # motion longer than imaging
        return motion_energy[:target_frames], info
    if diff > 0:  # motion shorter than imaging
        gap_idx = detect_gap_indices(interframe_int)
        # Insert NaN at gaps, then interpolate
        repaired = np.insert(repaired, insert_at, np.nan)
        repaired = interpolate_nans_1d(repaired)
```

iii. The data README states: "In some recordings there might be some missing frames from the camera... treated as missing values for motion energy or they can be interpolated over." The AI followed this guidance.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) `baseline_correct_fluorescence()` which applies Gaussian filtering and min/max filtering over entire session traces (n_neurons x ~36000-54000 frames), (2) loading large .npy files from disk for each session, and (3) the suite2p_preprocess call (when available) which operates on full fluorescence matrices.

ii.
```python
# Full-session Gaussian + min/max filtering:
Flow = gaussian_filter(Fc, [0.0, sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
```

iii. These are inherently expensive operations on large arrays (hundreds of neurons x tens of thousands of frames).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The trial-splitting loop in `split_into_trials()` could use a single reshape operation instead of per-trial slicing. (2) The per-gap-position insertion loop in `align_motion_to_imaging()` uses repeated `np.insert` calls. (3) The `normalize_session_outputs_in_place()` loop concatenates and re-slices per trial.

ii.
```python
# Trial splitting loop (could be vectorized with reshape):
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end])

# Gap insertion loop (repeated np.insert):
for pos in gap_positions:
    insert_at = int(pos + 1 + offset)
    repaired = np.insert(repaired, insert_at, np.nan)
    offset += 1
```

iii. These loops operate on small numbers of iterations (max ~15 trials per session, max ~116 gap positions) so the performance impact is minimal.

## 6-c. What processing does the code repeat multiple times?

i. Motion data is iterated multiple times: (1) during `normalize_session_outputs_in_place()` which concatenates all trials then splits back, (2) in `build_dataset()` which again concatenates all trials to compute global quintile edges, (3) again in `build_dataset()` for per-trial discretization, and (4) again for computing summary statistics. The motion trial data is also flattened/reshaped redundantly.

ii.
```python
# In normalize_session_outputs_in_place:
concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials])
# In build_dataset:
all_motion.append(trial.reshape(-1))  # iterate all trials again
# In build_dataset again:
for trial in record.output_continuous_trials:  # iterate for discretization
    bins = np.digitize(trial, global_edges)
```

iii. The multiple passes over motion data are logically separate steps (normalization, edge computation, discretization) but could be combined.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Per-session min-max normalization (`normalize_session_outputs_in_place`) is applied before global quintile discretization. Since quintile binning is rank-based, normalization within each session is not strictly necessary (though it changes cross-session relative rankings). (2) Detailed `session_info` metadata including per-trial motion statistics are computed but not used by the decoder. (3) The `summarize_dataset()` function computes comprehensive statistics that are printed but not part of the final pickle.

ii.
```python
# Per-session normalization before discretization:
normalize_session_outputs_in_place(session_records)

# Detailed per-trial stats computed but unused by decoder:
normalized_trials.append({
    "min": float(np.min(trial)),
    "max": float(np.max(trial)),
    "mean": float(np.mean(trial)),
})
```

iii. The metadata and summaries serve documentation purposes. The per-session normalization was the AI's interpretation of the task instruction to produce "normalized" motion energy, though it adds processing not described in the paper.
