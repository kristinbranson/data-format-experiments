# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data`, keeps subject directories whose names start with `jm`, then keeps per-subject session directories whose first four characters are digits. For each session it loads Suite2p data from `suite2p/plane0` and behavior data from `move_deve`. Neural loading uses `ops.npy`, `F.npy`, and `Fneu.npy`. Motion-energy loading uses `motion_energy_glob.npy`, and `interframe_int.npy` is also loaded when the motion trace is shorter than the neural trace.

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
plane_dir = session_dir / "suite2p" / "plane0"
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
...
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. In the trajectory, the agent first inspected the on-disk layout and said it wanted to mirror the reference processing exactly. Later summaries say it uses one recording day per session and repairs missing motion-energy frames from `interframe_int.npy` (steps 4, 24, 56, 111). The trajectory does not record any separate argument for excluding non-`jm*` subjects or non-date-like session folders beyond following the observed directory convention.

## 1-b. How are the data split into subjects?

i. Subjects are directory names under `/app/data` that start with `jm`, sorted lexicographically.

ii. 
```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )
```

```python
subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}
```

iii. The trajectory shows the agent repeatedly enumerating subjects this way during exploration and in the conversion summaries (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`). It did not give a separate justification beyond treating the folder naming convention as the subject definition (steps 17, 75, 103, 111).

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject directory whose name begins with four digits, sorted lexicographically. The AI treats one recording day as one session.

ii. 
```python
def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )
```

```python
for session_index, session_dir in enumerate(session_dirs):
    subject = session_dir.parent.name
    session_id = session_dir.name
    session_key = f"{subject}/{session_id}"
```

iii. The trajectory explicitly says the agent’s “plausible trial structure” was “each recording day as a session,” and the final summary repeats that decision (steps 24, 111). No more detailed session-splitting justification was recorded.

## 1-d. How are the data split into trials?

i. The AI does not use 60-second trials. It splits each continuous session into consecutive non-overlapping 2-minute blocks (`TRIAL_SECONDS = 120.0`). Any trailing frames that do not fill a full 2-minute block are dropped. Trials are formed before output discretization is attached to the final dataset.

ii. 
```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES
```

```python
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
if usable_frames != session_duration_frames:
    print(
        f"  Dropping trailing frames: kept {usable_frames} of {session_duration_frames} for consecutive 2-minute blocks"
    )

n_trials = usable_frames // TRIAL_FRAMES
```

iii. This choice is explicit in the trajectory. The agent said it had a “plausible trial structure” of “the same consecutive 2-minute blocks used in the paper’s decoding,” and later summarized the converter as splitting each session into “consecutive 2-minute trials” (steps 24, 68, 111). The recorded justification is that this matched its reading of the paper’s decoder setup, not the task instruction’s 60-second trial requirement.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial behavioral or neural quality-control filter. The only structural filtering is: trailing frames that do not fill a full 2-minute block are dropped, sessions with fewer than one 2-minute block are rejected, and sessions with fewer than two 2-minute trials are rejected.

ii. 
```python
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
if usable_frames < TRIAL_FRAMES:
    raise ValueError(f"Session has only {session_duration_frames} frames, not enough for one 2-minute block")
...
n_trials = usable_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The trajectory does not discuss trial QC separately. The only stated rationale is preserving the consecutive 2-minute block structure and producing data that satisfy the decoder’s need for at least two trials per session (steps 24, 77, 103). No other QC rule was justified.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from Suite2p `F.npy` and `Fneu.npy`, using preprocessing parameters read from `ops.npy`.

ii. 
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The trajectory shows the agent probing `F.npy`, `Fneu.npy`, `spks.npy`, and `ops.npy` to determine which representation best matched the paper, then deciding to reconstruct a Suite2p-based fluorescence-derived signal rather than using `spks.npy` directly (steps 18, 23, 29, 64, 111).

## 2-b. How is the `neural` data processed?

i. The AI reconstructs a `dF/F`-like signal. It subtracts neuropil (`F - neucoeff * Fneu`), runs `suite2p.extraction.dcnv.preprocess(...)` with parameters from `ops.npy`, then estimates a baseline as `Fcorr - dff_num`, clips the baseline to at least `1e-3`, and divides the preprocessed trace by that baseline.

ii. 
```python
Fcorr = F
Fcorr -= float(ops["neucoeff"]) * Fneu
del Fneu

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
return dff_num.astype(np.float32, copy=False), ops
```

iii. The trajectory contains the main justification here. The agent said it wanted to “reproduce the paper’s `dF/F` step rather than approximate it,” verified Suite2p’s preprocessing function, checked whether the paper’s “dF/F” implied an additional normalization beyond baseline subtraction, and then ran an empirical benchmark across candidate signals (`F`, `spks`, `dF`, `dff_div`) before locking the conversion (steps 29, 35, 42, 45, 48, 64, 67). The final summary describes the chosen neural signal as reconstructed Suite2p-based `dF/F` (step 111).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no extra neuron filter after loading the Suite2p arrays. It does not use `iscell.npy`, SNR thresholds, or any additional ROI selection in `convert_data.py`.

ii. 
```python
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
...
brain_region_idx.append(np.zeros(record["n_neurons"], dtype=np.int64))
```

iii. The trajectory shows the agent noticed `iscell` in the notebooks during exploration, but the final converter never uses it and contains no later justification for additional filtering (steps 13, 14). The practical justification appears to be that the rows of `F.npy` were taken as the already prepared tracked-cell set.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of each artificial 2-minute block, not to session start. Metadata describe the alignment event as “start of each consecutive 2-minute block from a continuous recording,” with `off_start = 0.0` and `off_end = 120.0`.

ii. 
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block from a continuous recording",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
    ...
}
```

```python
neural_trials = [
    neural_binned[:, trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
]
```

iii. The trajectory is explicit that the AI treated each session as continuous data segmented into consecutive 2-minute blocks and used block start as the alignment anchor (steps 24, 68, 111). It did not justify retaining session-start alignment once it adopted 2-minute trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI averages both neural and motion signals into non-overlapping 10-frame bins at a nominal 30 Hz sampling rate. This yields a time bin size of `1000 * 10 / 30 = 333.33 ms`.

ii. 
```python
FRAME_RATE_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FRAME_RATE_HZ
```

```python
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
```

iii. The trajectory says the timestamps looked drifted but the experiment structure still pointed to 10-frame bins at nominal 30 Hz, so timestamps should only be used for gap placement rather than redefining the binning scheme (step 56). The final summary repeats that neural and behavior traces are averaged in 10-frame bins (step 111).

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not read from a stored time variable. It is derived from the bin index, the 10-frame bin width, the nominal 30 Hz frame rate, and the trial index.

ii. 
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The trajectory frames time as a nominal-frame-rate quantity rather than something derived from raw camera timestamps. The agent said it would keep 10-frame bins at nominal 30 Hz and use timestamps only for motion-gap placement (step 56). The final summary says the decoder input is time from session start in seconds (step 111).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one floating-point time series per trial by taking the center of each 10-frame bin, converting frames to seconds, and adding `trial_index * 120` seconds so time continues across the consecutive 2-minute blocks within a session.

ii. 
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The trajectory does not discuss this formula separately. The closest justification is the agent’s commitment to nominal 30 Hz, 10-frame bins, and continuous time-from-session-start despite using 2-minute block trials (steps 56, 111).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is generated with exactly one sample per neural time bin, using the same binned trial structure as `neural`. In the AI’s format, time values correspond to bin centers within each 2-minute block and are returned as shape `(1, T)` arrays.

ii. 
```python
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
...
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
neural_trials = [
    neural_binned[:, trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
]
```

iii. The trajectory’s rationale is implicit rather than explicit: once the agent chose nominal 30 Hz and 10-frame averaging, it generated time on the same bin grid as the neural data and carried session elapsed time across artificial 2-minute blocks (steps 56, 111).

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`. When the motion array is shorter than the neural recording, `move_deve/interframe_int.npy` is used to infer missing video frames before interpolation.

ii. 
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
...
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The trajectory explicitly says the agent wanted to use timestamps only for gap placement and that the final converter repairs missing motion-energy frames from `interframe_int.npy` (steps 56, 111).

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI uses a multi-step pipeline: load motion energy, repair missing frames using relative jumps in interframe intervals, insert `NaN` placeholders, linearly interpolate missing values, split into 2-minute blocks, average into non-overlapping 10-frame bins, then globally min-max normalize and later quantize the result.

ii. 
```python
nominal = float(np.median(interframe))
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
...
repaired = np.empty(n_frames, dtype=np.float32)
...
for gap in jumps:
    repaired[dst] = motion[src]
    src += 1
    dst += 1
    if gap:
        repaired[dst:dst + gap] = np.nan
        dst += gap
...
nan_mask = np.isnan(repaired)
if nan_mask.any():
    idx = np.arange(n_frames)
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask]).astype(np.float32)
```

```python
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
...
all_motion = np.concatenate(
    [trial for record in session_records for trial in record["motion_trials"]]
).astype(np.float32, copy=False)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_scale = motion_max - motion_min
```

iii. The trajectory gives two justifications. First, it says timestamps should be used only for gap placement, not for redefining the paper’s nominal 30 Hz binning (step 56). Second, the final summary says the converter repairs missing motion-energy frames, averages traces in 10-frame bins, and discretizes globally normalized motion energy into five quintiles (step 111).

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI does not compute thresholds per session. It concatenates motion from all sessions, computes global 20/40/60/80 percentiles after 10-frame averaging, min-max normalizes each trial with the same global min and max, and then applies `np.digitize` to produce five global quintile categories.

ii. 
```python
all_motion = np.concatenate(
    [trial for record in session_records for trial in record["motion_trials"]]
).astype(np.float32, copy=False)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_scale = motion_max - motion_min
...
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)
```

```python
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. The trajectory’s explicit rationale is outcome-oriented: the agent wanted “globally normalized” motion energy discretized into “5 quintiles,” and its sample/full summaries emphasized that this produced exact global 20% fractions for each category (steps 75, 103, 111). The trajectory does not show any argument for using session-local thresholds.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first forces motion energy to match the neural frame count by reconstructing missing behavior frames from `interframe_int.npy` and linearly interpolating them. It then bins motion and neural signals using the same raw-frame windows and splits both into the same artificial 2-minute trials.

ii. 
```python
motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
neural_trials, time_trials, motion_trials = session_to_trials(neural_full, motion_full, n_frames)
```

```python
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
```

iii. The trajectory explicitly says timestamps are for gap placement only, not for altering the nominal neural/video frame grid, and the final summary says missing motion frames are repaired before 10-frame averaging (steps 56, 111). That is the recorded alignment justification.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI treats missing camera frames as the main data-quality issue. It infers missing frames from enlarged interframe intervals, checks that the inferred missing-frame count matches the neural-motion length mismatch, fills the gaps with `NaN`, linearly interpolates them, and raises an error if the inferred repair does not match the expected frame count. It also drops trailing frames that do not fill a full 2-minute trial block.

ii. 
```python
missing_count = int(jumps.sum())
expected_missing = n_frames - len(motion)
if missing_count != expected_missing:
    raise ValueError(
        f"{session_dir}: inferred {missing_count} missing motion frames but expected {expected_missing}"
    )
```

```python
nan_mask = np.isnan(repaired)
if nan_mask.any():
    idx = np.arange(n_frames)
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask]).astype(np.float32)
```

```python
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
```

iii. The trajectory explicitly frames the timestamps as gap-detection aids and says the converter “repairs missing motion-energy frames from `interframe_int.npy`” before downstream processing (steps 56, 111). It does not discuss any other data mistakes as separate design decisions.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant cost is reconstructing neural traces session by session with `dcnv.preprocess`, which operates over every neuron and frame. Motion repair is cheaper; the remaining assembly work is mostly reshaping, averaging, and list construction.

ii. 
```python
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

```python
for session_index, session_dir in enumerate(session_dirs):
    ...
    neural_full, ops = load_suite2p_dff(session_dir)
```

iii. The trajectory says the sample conversion was taking longer because it had to reconstruct `dF/F` “session by session from the Suite2p arrays” (step 73). That is the clearest recorded performance justification.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most of the heavy frame binning is already vectorized with `reshape(...).mean(...)`. The remaining obvious Python loops are the `for gap in jumps` loop in motion repair, the per-trial list construction loop, and the per-trial output discretization loop.

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
...
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. The trajectory does not contain an explicit efficiency discussion for these loops. The code itself shows that the AI already avoided the human reference’s repeated `np.insert` pattern by preallocating a repaired array and using vectorized interpolation once per session.

## 6-c. What processing does the code repeat multiple times?

i. The code makes a first pass over sessions to build `session_records`, then a second pass to concatenate all motion trials and compute global statistics, and then a third pass over sessions/trials to normalize and discretize motion into outputs. It also creates extra summaries and metadata derived from the same processed arrays.

ii. 
```python
for session_index, session_dir in enumerate(session_dirs):
    ...
    session_records.append(
        {
            ...
            "motion_trials": motion_trials,
        }
    )
```

```python
all_motion = np.concatenate(
    [trial for record in session_records for trial in record["motion_trials"]]
).astype(np.float32, copy=False)
...
for record in session_records:
    data["neural"].append(record["neural_trials"])
    data["input"].append(record["input_trials"])
    output_trials = []
    for motion_trial in record["motion_trials"]:
        motion_norm = (motion_trial - motion_min) / motion_scale
        motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
        output_trials.append(motion_bins[np.newaxis, :])
```

iii. The trajectory implies this repetition came from the decision to use global motion-energy thresholds: the agent first needed all processed motion traces to compute global quintiles, then had to revisit every trial to assign categories (steps 64, 75, 111). No further justification was recorded.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs extra global min-max normalization before `np.digitize`, even though digitizing against correspondingly normalized thresholds is equivalent to digitizing the raw binned motion values. It also computes and prints conversion summaries and stores extensive metadata fields that are not required for the downstream decoder. Those statistics are informative but not consumed by the trial-level decoding data.

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
    "mode": mode,
    "nsessions": len(session_records),
    "subjects": subjects,
    "neurons_per_mouse": neurons_per_mouse,
    "duration_minutes_unique": sorted({round(value, 3) for value in duration_minutes}),
    ...
}
```

iii. The trajectory emphasizes achieving globally balanced output bins and printing dataset-level sanity checks, but it does not justify the extra normalization as necessary for downstream decoding (steps 75, 103, 111). That makes the normalization and some of the summary work effectively discarded from the decoder’s perspective.
