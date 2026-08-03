# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the dataset by scanning `/app/data` for subject folders named `jm*`, then scanning each subject for dated session folders. In full mode it keeps every such session; in sample mode it keeps the first session per subject. For each kept session it loads neural arrays from `suite2p/plane0` and behavior arrays from `move_deve`, converts them into per-trial blocks, and finally assembles the required decoder dictionary.

ii.
```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )

def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )

def collect_session_dirs(root: Path, mode: str) -> list[Path]:
    session_dirs = []
    for subject_dir in sorted_subject_dirs(root):
        sessions = sorted_session_dirs(subject_dir)
        if mode == "sample":
            if sessions:
                session_dirs.append(sessions[0])
        else:
            session_dirs.extend(sessions)
    return session_dirs
```

```python
for session_index, session_dir in enumerate(session_dirs):
    neural_full, ops = load_suite2p_dff(session_dir)
    n_frames = neural_full.shape[1]
    motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
    neural_trials, time_trials, motion_trials = session_to_trials(neural_full, motion_full, n_frames)
```

iii. `CONVERSION_NOTES.md` says the release contains 6 subject folders and 41 full sessions, and the trajectory shows the agent using the data README plus directory listing to conclude that subject folders correspond to mice and dated subfolders correspond to recording sessions.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined purely by folder structure: each top-level `jm*` directory is a mouse. The script creates a sorted unique `subjects` list and a `subject_idx` array that maps each session to its subject.

ii.
```python
subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}

"subject_idx": np.array(
    [subject_to_index[record["subject"]] for record in session_records], dtype=np.int64
),
```

iii. The trajectory includes the data README text stating that each subject folder corresponds to one subject and that the `jm031`-style names are the mouse identifiers.

## 1-c. How are the data split into sessions?

i. Each dated folder inside a subject folder is treated as one session. The agent equates one recording day with one session and preserves the on-disk ordering by sorting those folder names lexicographically.

ii.
```python
def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )
```

```python
session_id = session_dir.name
session_key = f"{subject}/{session_id}"
```

iii. `CONVERSION_NOTES.md` explicitly says “Session unit: one recording day per session.” The trajectory also records the README passage that each such dated subfolder corresponds to one recording day.

## 1-d. How are the data split into trials?

i. Sessions are split into consecutive non-overlapping 2-minute blocks. At 30 Hz this is 3600 frames per block. Each block becomes one trial, so 20-minute sessions yield 10 trials and 30-minute sessions yield 15 trials.

ii.
```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)

usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
n_trials = usable_frames // TRIAL_FRAMES

neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
```

iii. `CONVERSION_NOTES.md` says “Trial unit: consecutive 2-minute blocks extracted from each continuous recording, matching the paper's decoding split strategy.” In the trajectory the agent cites the methods text saying the paper’s cross-validation splits were done on consecutive 2-minute blocks.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no trial-level quality control. The script only enforces structural requirements: it drops any trailing partial block, requires at least one full 2-minute block to make any trial at all, and requires at least two resulting trials per session.

ii.
```python
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
if usable_frames < TRIAL_FRAMES:
    raise ValueError(f"Session has only {session_duration_frames} frames, not enough for one 2-minute block")
if usable_frames != session_duration_frames:
    print(
        f"  Dropping trailing frames: kept {usable_frames} of {session_duration_frames} for consecutive 2-minute blocks"
    )

n_trials = usable_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The notes do not claim any biological or behavioral trial filtering. The agent appears to have inferred that the decoder format’s “at least two trials per session” rule was the only trial-level filter it needed to enforce.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data are derived from the Suite2p fluorescence traces `F.npy` and `Fneu.npy`, using parameters from `ops.npy` to determine the neuropil coefficient and baseline-preprocessing settings.

ii.
```python
plane_dir = session_dir / "suite2p" / "plane0"
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md` and the trajectory both state that the agent intentionally reconstructed a Suite2p-style `dF/F` from `F`, `Fneu`, and `ops` rather than using `spks.npy` or raw `F.npy` directly.

## 2-b. How is the `neural` data processed?

i. The agent neuropil-subtracts fluorescence, runs Suite2p’s `dcnv.preprocess` to estimate a baseline-corrected trace, reconstructs the baseline as `Fcorr - preprocess(Fcorr)`, divides by that baseline to form a `dF/F`-like quantity, and then averages over non-overlapping 10-frame bins.

ii.
```python
Fcorr = F
Fcorr -= float(ops["neucoeff"]) * Fneu

dff_num = Fcorr.copy()
dff_num = dcnv.preprocess(
    F=dff_num,
    baseline=ops["baseline"],
    win_baseline=ops["win_baseline"],
    sig_baseline=ops["sig_baseline"],
    fs=ops["fs"],
    prctile_baseline=ops["prctile_baseline"],
    batch_size=ops.get("batch_size", 100),
    device=torch.device("cpu"),
)

baseline = Fcorr
baseline -= dff_num
np.maximum(baseline, 1e-3, out=baseline)
dff_num /= baseline
```

```python
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. The notes document this exact formula and say it was meant to “stay grounded in the provided Suite2p defaults.” The trajectory shows the agent explicitly debating whether the paper’s “dF/F” meant Suite2p’s baseline-corrected trace or an additional division step, then choosing the latter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies no extra neuron filtering at conversion time. It accepts every row already present in the exported `suite2p/plane0` files, implicitly relying on the release having already restricted the matrices to tracked cells that passed upstream Suite2p/Track2p curation.

ii.
```python
neural_full, ops = load_suite2p_dff(session_dir)
...
data["brain_region_idx"].append(np.zeros(record["n_neurons"], dtype=np.int64))
```

iii. The data README recovered in the trajectory says these per-session Suite2p folders already contain “the neural data for the successfully tracked neurons.” The load-data notebook also says users should compute `dF/F` themselves from those traces, implying the cell set was already fixed upstream.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the start of each synthetic 2-minute block. There is no stimulus or task event; the alignment event is simply block onset within the continuous recording.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute block from a continuous recording",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
```

iii. `CONVERSION_NOTES.md` says the trialization and alignment are based on consecutive 2-minute decoding blocks. The trajectory shows the agent concluded there was no better event in this spontaneous-behavior dataset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins at 30 Hz, so each bin is 333.33 ms. Both neural and behavior traces are temporally rebinned this way.

ii.
```python
FRAME_RATE_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FRAME_RATE_HZ
```

```python
neural_binned = ... .mean(axis=3)
motion_binned = ... .mean(axis=2)
```

iii. The notes and the recovered methods text both say the decoding analysis slightly denoised traces by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. This input is not read from a raw time variable. It is synthesized from the nominal imaging frame rate, the 10-frame binning scheme, and the 2-minute block index within each session.

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. `CONVERSION_NOTES.md` says the decoder input is “absolute time elapsed from session start.” In the trajectory the agent explicitly decided not to use drifted camera timestamps to redefine the experiment clock.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script computes the center time of each 10-frame bin for a single 2-minute block, then adds `trial_index * 120 s` so the value is absolute session time rather than trial-relative time.

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The notes state that “The decoder input is not trial-relative time; it is absolute time-from-session-start in seconds.” The final agent message repeats that choice.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time arrays are created inside the same `session_to_trials` function that bins and chunks neural data. Each trial therefore gets a `1 x T` time array with the same number of bins as its neural matrix and the same block boundaries.

ii.
```python
def session_to_trials(
    neural_full: np.ndarray, motion_full: np.ndarray, session_duration_frames: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ...
    return neural_trials, time_trials, motion_trials
```

```python
data["neural"].append(record["neural_trials"])
data["input"].append(record["input_trials"])
```

iii. The agent’s notes say the same non-overlapping bins and 2-minute block structure are used across neural, input, and output streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output variable comes from the processed behavior file `move_deve/motion_energy_glob.npy`. The auxiliary file `interframe_int.npy` is used only to locate missing camera frames when the motion-energy vector is shorter than the neural recording.

ii.
```python
move_dir = session_dir / "move_deve"
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
...
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The data README recovered in the trajectory says `move_deve` contains “processed behavioural data (motion energy extracted from videography of spontaneous behaviour `motion_energy_glob.npy`).”

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent repairs missing camera frames by inserting `NaN`s at inferred gaps, linearly interpolates over those gaps, averages the repaired trace in non-overlapping 10-frame bins, then applies a global min-max normalization before discretization.

ii.
```python
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
...
for gap in jumps:
    repaired[dst] = motion[src]
    src += 1
    dst += 1
    if gap:
        repaired[dst:dst + gap] = np.nan
        dst += gap
...
if nan_mask.any():
    idx = np.arange(n_frames)
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask]).astype(np.float32)
```

```python
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
...
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_scale = motion_max - motion_min
```

iii. The notes justify interpolation by pointing to the release README, which says missing camera frames may be treated as missing or interpolated. They justify 10-frame averaging by the paper’s “10 consecutive timestamps” statement. The normalization step is documented as an adaptation to the decoder task.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent computes global 20th, 40th, 60th, and 80th percentiles across all binned motion values from all sessions, after global min-max normalization, and assigns each binned time point to one of five categories with `np.digitize`.

ii.
```python
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)
```

```python
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. `CONVERSION_NOTES.md` says the output is “globally normalized, and discretized into 5 equal-percentile bins,” and adds that the quintiles are computed globally rather than separately per session to preserve a single category scale.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first repaired to the same frame count as the neural recording, then chunked into the same consecutive 2-minute blocks and averaged with the same 10-frame binning. Alignment is therefore by shared imaging-frame index.

ii.
```python
n_frames = neural_full.shape[1]
motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
neural_trials, time_trials, motion_trials = session_to_trials(neural_full, motion_full, n_frames)
```

iii. The recovered methods text says the microscope acquisition triggered camera frame acquisition, which the agent used to justify framewise synchronization after repairing dropped camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code has one explicit repair path: if `motion_energy_glob.npy` is shorter than the imaging data, it infers the missing frame positions from `interframe_int.npy`, inserts missing samples as `NaN`, then linearly interpolates them. If the inferred gap count does not match the observed length mismatch, or the repaired array length is wrong, the script raises an error instead of guessing. Other issues are also handled by raising `ValueError`.

ii.
```python
expected_missing = n_frames - len(motion)
if missing_count != expected_missing:
    raise ValueError(
        f"{session_dir}: inferred {missing_count} missing motion frames but expected {expected_missing}"
    )
...
if dst != n_frames or src != len(motion):
    raise ValueError(
        f"{session_dir}: motion repair finished at dst={dst}, src={src}, expected ({n_frames}, {len(motion)})"
    )
```

```python
nan_mask = np.isnan(repaired)
if nan_mask.any():
    idx = np.arange(n_frames)
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask]).astype(np.float32)
```

iii. The notes and trajectory both cite the data README line saying missing camera-frame indices can be obtained from `tstamps.npy` or `interframe_int.npy` and treated as missing or interpolated.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive work is loading large `F.npy` and `Fneu.npy` arrays for every session, running `suite2p.extraction.dcnv.preprocess(...)` over full recordings to reconstruct the neural signal, reshaping/averaging long traces into blocks, and then serializing the large session-by-trial dataset to pickle.

ii.
```python
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
...
dff_num = dcnv.preprocess(
    F=dff_num,
    baseline=ops["baseline"],
    win_baseline=ops["win_baseline"],
    sig_baseline=ops["sig_baseline"],
    fs=ops["fs"],
    prctile_baseline=ops["prctile_baseline"],
    batch_size=ops.get("batch_size", 100),
    device=torch.device("cpu"),
)
```

iii. The trajectory shows the sample conversion spending noticeable time “reconstruct[ing] `dF/F` session by session,” which matches what this code does.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the Python loop that rebuilds motion arrays around missing-frame gaps, the list-comprehension style per-trial extraction in `session_to_trials`, and the per-trial loop that normalizes and digitizes motion one trial at a time.

ii.
```python
for gap in jumps:
    repaired[dst] = motion[src]
    src += 1
    dst += 1
    if gap:
        repaired[dst:dst + gap] = np.nan
        dst += gap
```

```python
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
neural_trials = [
    neural_binned[:, trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
]
motion_trials = [
    motion_binned[trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
]
```

```python
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. This is an evaluation of the code structure itself rather than a claim from the notes. These are the places where the implementation stays in Python even though the underlying operations are array-friendly.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the entire preprocessing pipeline twice when the workflow builds both sample and full datasets. Inside one build it also repeats per-session copies and per-trial normalization/digitization work after already computing global motion statistics.

ii.
```python
def collect_session_dirs(root: Path, mode: str) -> list[Path]:
    ...
    if mode == "sample":
        ...
    else:
        session_dirs.extend(sessions)
```

```python
for session_index, session_dir in enumerate(session_dirs):
    neural_full, ops = load_suite2p_dff(session_dir)
    ...
```

```python
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

iii. The trajectory shows the agent separately running sample conversion, sample verification, sample training, full conversion, full verification, and full training. The code itself also recomputes normalized motion trial by trial after having already computed global min/max and quantile thresholds.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary computation is the global min-max normalization of motion before percentile binning: the quintile assignments would be identical if the quantiles were computed on raw binned motion because min-max scaling is monotonic. The code also keeps floating-point `motion_trials` only as an intermediate so it can later discard them and keep only the categorical output, and it computes summary statistics used only for logs/metadata.

ii.
```python
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)
...
motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

```python
summary = {
    "duration_minutes_unique": ...,
    "total_missing_behavior_frames": ...,
    "paper_check_tracked_neurons_mean": ...,
    "paper_check_tracked_neurons_std": ...,
}
```

iii. This follows directly from the code. The downstream decoder consumes categorical `output` and does not use the temporary normalized motion traces or the logging-only summary values.
