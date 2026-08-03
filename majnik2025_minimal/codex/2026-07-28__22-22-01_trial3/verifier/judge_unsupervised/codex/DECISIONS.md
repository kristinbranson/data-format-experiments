# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all data by scanning the top-level subject directories under `data/`, then scanning each subject's session directories, and then loading per-session NumPy files from `suite2p/plane0` and `move_deve`. For each session it loads `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. All sessions are loaded eagerly by `build_dataset()` before any final dataset assembly.

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

iii. The trajectory shows the agent first established that the dataset was "arranged per mouse and day with Suite2p outputs plus motion-energy traces" (step 7), then confirmed the actual tree and file shapes across all subjects and sessions (steps 18 and 22).

## 1-b. How are the data split into subjects?

i. Subjects are defined by the top-level directories in `data/`. The code uses sorted directory names as subject IDs and optionally restricts them with `--subjects`.

ii. 
```python
available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
if selected_subjects is None:
    subjects = available_subjects
else:
    missing = sorted(set(selected_subjects) - set(available_subjects))
    if missing:
        raise FileNotFoundError(f"Unknown subjects: {missing}")
    subjects = sorted(selected_subjects)
```

iii. The data README recovered from the trajectory says that unzipping the release yields "6 folders, one for each subjects" and that each subject folder corresponds to a mouse ID such as `jm031` or `jm046` (step 10).

## 1-c. How are the data split into sessions?

i. Sessions are defined as the immediate subdirectories inside each subject directory. Each session directory corresponds to one recording day and is kept separate as one dataset session.

ii. 
```python
for subject in subjects:
    subject_dir = data_dir / subject
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        out.append((subject, session_dir))
```

iii. The data README says each subject folder contains session folders, "each corresponding to one recording day," and the trajectory confirms the agent relied on that organization (steps 7 and 10).

## 1-d. How are the data split into trials?

i. The source recordings are continuous sessions, so the agent creates pseudo-trials by first preprocessing full-session traces and then cutting them into consecutive 2-minute blocks. The number of trials per session is whatever fits exactly after 10-frame binning: 10 trials for 20-minute sessions and 15 for 30-minute sessions.

ii. 
```python
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)

def split_session_into_trials(
    neural_binned: np.ndarray,
    motion_classes: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ...
    n_trials = neural_binned.shape[1] // TRIAL_BINS

    for trial_idx in range(n_trials):
        start = trial_idx * TRIAL_BINS
        end = start + TRIAL_BINS
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
        output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. The paper methods state that decoding used "consecutive 2 minute blocks of the recording," and the agent explicitly says it would "cut the recording into the same consecutive 2-minute blocks described in the paper" (methods in step 8; trajectory step 34).

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level rejection. Every 2-minute block is kept if the session survives earlier checks. The only effective filtering is upstream: the session must have matching binned neural and motion lengths, and the binned session length must divide evenly into 2-minute blocks. Otherwise the script raises an error instead of dropping bad trials.

ii. 
```python
if neural_binned.shape[1] != motion_binned.shape[0]:
    raise ValueError(f"{session_dir}: binned neural and motion lengths do not match")
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(
        f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
        f"{TRIAL_BINS}-bin trials"
    )
```

iii. The agent's reasoning focused on session-level alignment and format checks rather than any trial curation rule from the paper. In step 34 it says the plan is to preprocess full sessions, repair dropouts, bin, and cut into 2-minute blocks; no trial rejection criterion is mentioned.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw fluorescence traces `F.npy`, the neuropil traces `Fneu.npy`, and Suite2p parameters from `ops.npy`. The code does not use `spks.npy` for the final dataset.

ii. 
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
...
neural = suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops)
```

iii. The paper says they used "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)," and the trajectory shows the agent deliberately compared raw `F`, processed fluorescence, and `spks` before choosing processed fluorescence (steps 8, 21, 23, 33, 47).

## 2-b. How is the `neural` data processed?

i. The agent reconstructs Suite2p-style baseline-corrected fluorescence by subtracting neuropil (`F - neucoeff * Fneu`) and then subtracting the baseline estimated using the Suite2p `baseline`, `sig_baseline`, `win_baseline`, and `prctile_baseline` settings from `ops.npy`. After that, the traces are denoised by non-overlapping 10-frame averaging.

ii. 
```python
def suite2p_baseline_corrected_fluorescence(
    fluorescence: np.ndarray, neuropil: np.ndarray, ops: dict
) -> np.ndarray:
    fc = fluorescence.astype(np.float32) - float(ops["neucoeff"]) * neuropil.astype(np.float32)
    baseline = ops.get("baseline", "maximin")

    if baseline == "maximin":
        win = int(round(float(ops["win_baseline"]) * float(ops["fs"])))
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = minimum_filter1d(flow, win)
        flow = maximum_filter1d(flow, win)
    elif baseline == "constant":
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = np.amin(flow)
    elif baseline == "constant_prctile":
        flow = np.percentile(fc, float(ops["prctile_baseline"]), axis=1, keepdims=True)
    else:
        flow = 0.0

    return (fc - flow).astype(np.float32)
...
neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The justification came from three places in the trajectory: step 12 cites the paper-level rule to use baseline-corrected dF/F and 10-frame averaging; step 17 shows the repo GUI's `F_processing()` implementation; step 25 confirms the saved Suite2p defaults are `neucoeff=0.7`, `baseline='maximin'`, `sig_baseline=10`, and `win_baseline=60`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies only the Suite2p/Track2p cell-quality rule and no extra neuron filtering. It checks that every exported ROI has `iscell[:,1] > 0.5` and errors out if not. It also assumes the Track2p-exported matrices already contain only tracked cells shared across all days.

ii. 
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The paper says they considered "all ROIs above the default threshold of 0.5 as true cells." The data README says the release already contains only cells present across all days. The trajectory shows the agent checked both assumptions: step 12 notes the `iscell > 0.5` rule; steps 21, 27, and 31 show the agent verified the exported matrices were already tracked-cell sets and that all `iscell` probabilities were indeed above 0.5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent defines the alignment event as the start of each artificial 2-minute decoding block. Neural data are first kept in full-session frame order, then binned, then sliced into consecutive windows. Within each trial/block, neural time starts at that block boundary.

ii. 
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block within a session",
    "off_start": 0.0,
    "off_end": TRIAL_DURATION_SECONDS,
    ...
}
...
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
```

iii. The paper does not describe discrete behavioral trials; it describes continuous recordings and decoder splits on consecutive 2-minute blocks. The trajectory shows the agent explicitly used those blocks as the operative alignment structure (steps 34 and 53).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 imaging frames per bin at 30 Hz, so each bin is `10 / 30 = 0.333...` s, or `333.333...` ms. Yes, non-overlapping temporal rebinning is applied once to both neural and motion traces.

ii. 
```python
FRAME_RATE_HZ = 30.0
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0
...
neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The paper methods say that for decoding they "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent called that out in step 12 and implemented it directly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The `input` time variable is not taken from a stored timestamp array. It is synthesized from the number of binned neural frames plus the fixed imaging rate and fixed bin size, so it is implicitly derived from session length in `F.npy` together with the constants `FRAME_RATE_HZ` and `FRAME_BIN_SIZE`.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. The trajectory shows the agent knew `tstamps.npy` existed (steps 18 and 29) but still settled on elapsed time from the binned frame grid, describing the decoder input as "elapsed time from session start at bin centers" (steps 34, 53, and 61).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code creates a regular time axis at the centers of the 10-frame bins. It uses `arange`, adds `0.5` to move from left edges to bin centers, and multiplies by bin size in seconds. No interpolation, no use of recorded timestamps, and no per-session offset correction are applied.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
...
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. The justification in the trajectory is pragmatic rather than paper-driven: step 34 frames the decoder input as session time after binning, and the conversion summaries in steps 53 and 61 report "elapsed time from session start at bin centers (seconds)."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time axis is created on the exact same binned frame grid as `neural`, and the same `start:end` slices are used when splitting into 2-minute trials. Therefore every neural sample has a matching elapsed-time sample.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
...
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. The trajectory repeatedly describes the pipeline as preprocessing full-session traces, binning them together, then cutting aligned 2-minute blocks (steps 28, 34, and 43).

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from the processed motion-energy trace in `move_deve/motion_energy_glob.npy`, with `interframe_int.npy` used to reconcile missing camera frames to the imaging frame grid.

ii. 
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
...
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy,
    n_imaging_frames=n_frames,
    interframe_int=interframe_int,
)
```

iii. The paper defines motion energy from videography frame differences, but the released dataset only ships the processed result. The data README, recovered in step 10, says `move_deve` contains processed behavioral data in `motion_energy_glob.npy` and that dropped frames can be identified from `tstamps.npy` or `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code optionally interpolates missing camera frames onto the imaging frame grid, averages the aligned motion trace in non-overlapping 10-frame bins, and then z-scores the binned motion separately within each session before any discretization.

ii. 
```python
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy,
    n_imaging_frames=n_frames,
    interframe_int=interframe_int,
)

motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
motion_z = (
    (motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)
).astype(np.float32)
```

iii. The interpolation choice comes from the data README, which explicitly says missing motion frames "can be interpolated over" (step 10). The 10-frame averaging comes from the paper methods (step 8). The per-session z-score is the agent's own normalization choice, reflected in its prototype comparisons and final conversion summaries (steps 45, 47, 53, and 61).

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After per-session z-scoring, the agent pools all sessions' z-scored motion values, computes global 20th/40th/60th/80th percentile edges, and uses `np.digitize` to map each time bin to one of five quintile categories.

ii. 
```python
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions], axis=0)
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)
...
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
```

iii. The trajectory shows this was an explicit decision. In step 43 the agent says broken accuracy would usually indicate "alignment or discretization" issues, and in step 47 it compared per-session versus pooled-threshold variants before settling on "global quintile edges" reported in steps 53 and 61.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion trace is aligned to imaging frames first, then binned with the same 10-frame windows as the neural data, and then split into the same `start:end` trial windows. Missing frames are inserted between adjacent observed motion samples by linear interpolation.

ii. 
```python
def align_motion_to_imaging_frames(
    motion_energy: np.ndarray,
    n_imaging_frames: int,
    interframe_int: np.ndarray,
) -> tuple[np.ndarray, dict]:
    ...
    missing_after = infer_missing_frames(interframe_int)
    ...
    for gap_missing in missing_after:
        next_src = src + 1
        if gap_missing:
            aligned[dst : dst + gap_missing] = np.linspace(
                motion_energy[src],
                motion_energy[next_src],
                int(gap_missing) + 2,
                dtype=np.float32,
            )[1:-1]
            dst += int(gap_missing)
        aligned[dst] = motion_energy[next_src]
        dst += 1
        src = next_src
```

iii. The data README says missing motion frames may be treated as missing or interpolated. The trajectory shows the agent inspected the mismatch cases and chose interpolation to keep frame-accurate alignment before binning (steps 28, 29, 32, and 34).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main repair path is for missing motion-camera frames: they are inferred from `interframe_int.npy` and filled by linear interpolation. Most other anomalies are handled by fail-fast validation rather than repair: shape mismatches, unexpected `iscell <= 0.5`, irreconcilable missing-frame counts, or session lengths that do not split cleanly into 2-minute blocks all raise `ValueError`.

ii. 
```python
if fluorescence.shape != neuropil.shape:
    raise ValueError(f"{session_dir}: F and Fneu shapes do not match")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
...
if missing_total != expected_missing:
    raise ValueError(
        f"Could not reconcile missing motion frames: inferred {missing_total}, "
        f"expected {expected_missing}"
    )
...
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(
        f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
        f"{TRIAL_BINS}-bin trials"
    )
```

iii. The trajectory justification is explicit for missing behavior frames. Step 28 says some behavior traces are short because camera frames dropped, and step 34 says those dropouts would be repaired on the frame grid before binning. This matches the options described in the data README from step 10.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading all large `.npy` arrays for every session, applying the Suite2p-style baseline correction filters over full-session fluorescence matrices, interpolating missing motion frames where needed, computing the full pooled motion distribution, and materializing per-trial arrays for every session before writing the pickle.

ii. 
```python
loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
...
flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
flow = minimum_filter1d(flow, win)
flow = maximum_filter1d(flow, win)
...
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions], axis=0)
...
for session in loaded_sessions:
    motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
    neural_trials, input_trials, output_trials = split_session_into_trials(...)
```

iii. The agent itself notes in step 59 that the full conversion is heavier because it is "preprocessing every session before any splits are written." That matches the code structure.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the Python loops that split a session into trials and the repeated per-session appends during dataset assembly. `split_session_into_trials()` could reshape whole binned arrays into `(n_trials, ...)` blocks instead of slicing in Python. `subjects.index(...)` inside the session loop is also repeated lookup work that could be precomputed.

ii. 
```python
for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
    output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
...
for session in loaded_sessions:
    ...
    neural.append(neural_trials)
    input_data.append(input_trials)
    output.append(output_trials)
    subject_idx.append(subjects.index(session["subject"]))
```

iii. The agent did not justify these as intentional choices in prose; they are simply the implementation it wrote in step 49. By inspection, these are straightforward Python loops over already-array-like data.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts arrays with `astype(...)`, repeatedly slices trial windows in Python, repeatedly searches `subjects.index(session["subject"])`, and repeats the same `start:end` boundary logic for `neural`, `input`, and `output`. It also computes and stores per-session summary dictionaries purely for metadata and console reporting.

ii. 
```python
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
...
subject_idx.append(subjects.index(session["subject"]))
...
session_info.append(
    {
        "session_id": session["session_id"],
        "subject": session["subject"],
        ...
    }
)
```

iii. This is mostly visible from direct code inspection rather than an explicit stated rationale. The trajectory instead emphasizes correctness and validator compliance over performance tuning (steps 34, 38, and 48).

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `missing_motion_frame_indices` during motion interpolation but never stores them in the final dataset. It also builds rich metadata and console summaries that are not used by the decoder. More generally, all of `session_info`, tracked-neuron summary stats, and conversion-summary text are extra reporting work rather than inputs to downstream analyses.

ii. 
```python
inserted_indices: list[int] = []
...
if gap_missing:
    inserted_indices.extend(range(dst, dst + int(gap_missing)))
...
return aligned, {
    "missing_motion_frames": missing_total,
    "missing_motion_frame_indices": inserted_indices,
}
...
session_info.append({...})
...
"tracked_neurons_per_subject": tracked_neurons_per_subject,
"tracked_neurons_mean": float(tracked_counts.mean()),
"tracked_neurons_std": float(tracked_counts.std(ddof=0)),
...
print("Sanity checks against reference materials:")
```

iii. The instructions asked for sanity checks and documentation, so some extra reporting was deliberate. The trajectory shows the agent was explicitly trying to capture validation-oriented metadata and logs, not just the minimal decoder inputs and outputs (steps 48, 54, 59, and 62).
