# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads subjects by scanning `/app/data` for directories whose names start with `jm`, then loads session directories under each subject whose first four characters are digits. In full mode it includes every such session; in sample mode it keeps only the first session per subject. During dataset construction it loads calcium data from `suite2p/plane0/F.npy` and `Fneu.npy`, Suite2p metadata from `ops.npy`, and behavior from `move_deve/motion_energy_glob.npy` plus `interframe_int.npy`.

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

plane_dir = session_dir / "suite2p" / "plane0"
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
```

iii. In `CONVERSION_NOTES.md`, the AI says the conversion targets the bundled Track2p release, uses 6 listed `jm*` subjects and 41 sessions, and treats one recording day as one session. The trajectory also shows it deliberately inspected the on-disk Suite2p and behavior arrays before finalizing the loader.

## 1-b. How are the data split into subjects?

i. Subjects are defined by top-level directories in `/app/data` whose names start with `jm`, sorted lexicographically. The final `subjects` list is reconstructed from the parent directory names of the collected sessions and mapped to integer indices for `subject_idx`.

ii. 
```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )

subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}

"subject_idx": np.array(
    [subject_to_index[record["subject"]] for record in session_records], dtype=np.int64
),
```

iii. The notes enumerate the six subject IDs explicitly and say the release contains exactly 6 mice, so the AI used the folder naming convention as the subject split.

## 1-c. How are the data split into sessions?

i. Each session is one dated recording-day directory inside a subject directory. Session order is deterministic because the AI sorts those per-subject directories before collecting them. It further assumes valid session names begin with four digits.

ii. 
```python
def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )

for subject_dir in sorted_subject_dirs(root):
    sessions = sorted_session_dirs(subject_dir)
    ...
    session_dirs.extend(sessions)
```

iii. `CONVERSION_NOTES.md` states “Session unit: one recording day per session,” and the trajectory repeatedly describes “one recording day” as the session definition.

## 1-d. How are the data split into trials?

i. The AI does not use the reference 60-second segments. Instead, it cuts each continuous session into consecutive 2-minute blocks (`TRIAL_SECONDS = 120`), drops any trailing frames that do not fill a full block, requires at least 2 such blocks per session, and then represents each block after 10-frame averaging.

ii. 
```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES

usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
if usable_frames != session_duration_frames:
    print(
        f"  Dropping trailing frames: kept {usable_frames} of {session_duration_frames} for consecutive 2-minute blocks"
    )

n_trials = usable_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The notes say the “Trial unit” is consecutive 2-minute blocks “matching the paper’s decoding split strategy,” and the trajectory explicitly says the agent adopted “the same consecutive 2-minute blocks used in the paper’s decoding.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only trial/session exclusions are structural: trailing frames are discarded if they do not fill a complete 2-minute block, and a session is rejected if it yields fewer than 2 artificial trials.

ii. 
```python
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
if usable_frames != session_duration_frames:
    print(
        f"  Dropping trailing frames: kept {usable_frames} of {session_duration_frames} for consecutive 2-minute blocks"
    )

n_trials = usable_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The notes frame this as trialization rather than QC, and the minimum-two-trials check is consistent with the decoder-format requirement from the instructions rather than a biology-specific quality rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p’s `F.npy` and `Fneu.npy`, while also reading `ops.npy` to obtain the preprocessing parameters such as neuropil coefficient, baseline method, and frame rate.

ii. 
```python
plane_dir = session_dir / "suite2p" / "plane0"
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The notes explicitly say the neural activity pipeline starts by loading `F.npy`, `Fneu.npy`, and `ops.npy` for each session.

## 2-b. How is the `neural` data processed?

i. Neural traces are neuropil-subtracted using the coefficient from `ops.npy`, passed through `suite2p.extraction.dcnv.preprocess(...)`, converted into `dF/F` by reconstructing a baseline as `Fcorr - preprocess(Fcorr)`, clipping that baseline at `1e-3`, dividing, and then averaging non-overlapping 10-frame bins inside each 2-minute trial.

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

neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. The notes justify this by saying the paper mentions decoding on “slightly denoised `dF/F`” and 10-frame averaging, so the AI chose to reconstruct `dF/F` using Suite2p defaults stored in `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply neuron-level QC filters such as `iscell` or activity thresholds. All rows in `F.npy` are carried through once loaded. The only checks are that the Suite2p frame rate equals 30 Hz and that each session is long enough to produce at least two 2-minute blocks.

ii. 
```python
neural_full, ops = load_suite2p_dff(session_dir)
if float(ops["fs"]) != FRAME_RATE_HZ:
    raise ValueError(f"{session_key}: expected {FRAME_RATE_HZ} Hz but found {ops['fs']}")

if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. No explicit QC rationale is documented for neurons; the notes instead emphasize that the bundled release already contains tracked barrel-cortex cells and focus on session/trial blocking rules.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of each artificial 2-minute block cut from the continuous session, not to an external behavioral or stimulus event. The metadata records this block-start alignment and sets offsets from 0 to 120 seconds.

ii. 
```python
"metadata": {
    "time_bin_size": TIME_BIN_SIZE_MS,
    "temporal_alignment_event": "start of each consecutive 2-minute block from a continuous recording",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
    ...
}
```

iii. The notes explicitly say “Trials are aligned to block start” because the AI interpreted the paper’s decoding blocks as the relevant temporal anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at a 30 Hz raw frame rate, so each time bin is `1000 * 10 / 30 = 333.33 ms`. Yes, temporal rebinning is applied by averaging every 10 consecutive frames for both neural and motion signals.

ii. 
```python
FRAME_RATE_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FRAME_RATE_HZ

neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. The notes say the paper used behavior traces “obtained by averaging 10 consecutive timestamps,” and the AI extended that choice to the final exported neural and behavior time series.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw file. It is synthesized from the frame-rate constant, the 10-frame bin index within a 2-minute block, and the block’s offset from session start.

ii. 
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The notes describe the decoder input as “absolute time elapsed from session start” rather than a signal read from disk.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time at the center of each 10-frame bin, producing one time series per 2-minute block. It then adds `trial_index * 120` seconds so each trial carries absolute session time rather than restarting at zero.

ii. 
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The notes explicitly justify this as “not trial-relative time” but “absolute time-from-session-start in seconds.”

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction to the same 2-minute trial boundaries and the same 10-frame bins as the neural signal, so each trial’s time vector has exactly the same number of bins as the corresponding neural matrix.

ii. 
```python
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)

base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The notes state that time is carried through as a 1-by-time array for the same blocked trials used for neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/interframe_int.npy` used to infer where camera frames were dropped so the motion trace can be repaired to the neural frame count.

ii. 
```python
move_dir = session_dir / "move_deve"
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
...
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The notes say the decoder output is motion energy with missing camera frames repaired from `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI repairs missing motion samples using gap sizes inferred from `interframe_int.npy`, inserts `NaN` placeholders into a preallocated array, linearly interpolates them, averages the repaired signal in 10-frame bins aligned to the neural bins, computes global min/max and 20/40/60/80 percentiles over all binned motion values, and later digitizes each trial.

ii. 
```python
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
nominal = float(np.median(interframe))
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)

repaired = np.empty(n_frames, dtype=np.float32)
...
if gap:
    repaired[dst:dst + gap] = np.nan
...
nan_mask = np.isnan(repaired)
if nan_mask.any():
    idx = np.arange(n_frames)
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask]).astype(np.float32)

motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)

all_motion = np.concatenate(
    [trial for record in session_records for trial in record["motion_trials"]]
).astype(np.float32, copy=False)
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
```

iii. The notes justify this by citing the paper’s 10-frame averaging statement and the release note that missing camera frames should be identified from timestamps or `interframe_int.npy` and repaired.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI defines five quintile categories globally across all binned motion values. For each trial it first min-max normalizes the binned motion using dataset-wide extrema, then applies `np.digitize` with the globally normalized 20/40/60/80 percentile thresholds to obtain categories `0` through `4`.

ii. 
```python
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)

for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. The notes explicitly say the output is “globally normalized, and discretized into 5 equal-percentile bins,” with a single categorical scale across animals and days.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is enforced in two stages: first the motion trace is repaired until its frame count matches the neural trace, then both streams are cut to the same usable session length, blocked into the same consecutive 2-minute trials, and averaged over the same 10-frame bins so each motion trial matches its neural trial bin-for-bin.

ii. 
```python
motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
neural_trials, time_trials, motion_trials = session_to_trials(neural_full, motion_full, n_frames)

neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
```

iii. The notes justify this as preserving alignment after missing-frame repair while matching the paper’s 2-minute decoding blocks and 10-frame averaging.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI treats missing camera frames as the main recoverable data issue. It infers the number and positions of missing frames from `interframe_int.npy`, checks that the inferred count matches the neural/behavior length mismatch, linearly interpolates the missing motion values, and raises an error if the mismatch cannot be reconciled. It also drops trailing frames that do not fill a complete 2-minute block and raises on unexpected frame rates or sessions that are too short.

ii. 
```python
expected_missing = n_frames - len(motion)
if missing_count != expected_missing:
    raise ValueError(
        f"{session_dir}: inferred {missing_count} missing motion frames but expected {expected_missing}"
    )

nan_mask = np.isnan(repaired)
if nan_mask.any():
    idx = np.arange(n_frames)
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask]).astype(np.float32)

if float(ops["fs"]) != FRAME_RATE_HZ:
    raise ValueError(f"{session_key}: expected {FRAME_RATE_HZ} Hz but found {ops['fs']}")

if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The trajectory says the agent abandoned a naive timestamp mapping, switched to `interframe_int`-based gap detection, and wanted to repair only genuine missing camera frames. The notes also cite release guidance that missing frames should be treated as missing or interpolated.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is the per-session Suite2p preprocessing used to reconstruct `dF/F`, especially the `dcnv.preprocess(...)` call inside `load_suite2p_dff`. Full-dataset conversion loops through 41 sessions and runs that preprocessing on every session before trialization and motion quantization.

ii. 
```python
def load_suite2p_dff(session_dir: Path) -> tuple[np.ndarray, dict]:
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

iii. The trajectory explicitly says the sample and full runs take longer because the converter has to “reconstruct `dF/F` session by session,” and describes the full conversion as the “heaviest step.”

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loops are the gap-repair loop over `jumps`, the list-based assembly of per-trial time and neural/motion arrays, and the per-trial motion normalization and digitization loop. The heavy frame averaging itself is already vectorized with `reshape(...).mean(...)`.

ii. 
```python
for gap in jumps:
    repaired[dst] = motion[src]
    src += 1
    dst += 1
    if gap:
        repaired[dst:dst + gap] = np.nan
        dst += gap

time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]

for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. The AI did not document a specific justification here. From the code and notes, the implicit tradeoff is that it already vectorized the expensive reshaping/binning operations and left the lighter session/trial assembly in simple Python loops.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats output postprocessing trial by trial after already computing global motion statistics: each `motion_trial` is min-max normalized and digitized separately in a nested loop. It also repeatedly copies per-trial neural and motion slices into Python lists after already holding the full binned arrays in memory.

ii. 
```python
neural_trials = [
    neural_binned[:, trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
]
motion_trials = [
    motion_binned[trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
]

for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. No explicit justification for this repetition is documented. The notes focus on matching the AI’s chosen scientific preprocessing rather than on minimizing repeated per-trial bookkeeping.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only used for conversion-time bookkeeping rather than downstream decoding: `neurons_per_mouse`, `duration_minutes`, and `missing_behavior_by_session` are built for summaries/notes; `motion_norm` is computed transiently and immediately discarded after digitization; and `ops.npy` is returned in full even though only a small subset of fields is used beyond preprocessing and a frame-rate check.

ii. 
```python
duration_minutes = []
neurons_per_mouse = {}
missing_behavior_by_session = {}
...
duration_minutes.append(n_frames / FRAME_RATE_HZ / 60.0)
neurons_per_mouse.setdefault(subject, neural_full.shape[0])
missing_behavior_by_session[session_key] = missing_count

for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. The notes justify this extra work as support for “paper-level sanity checks,” detailed metadata, and missing-frame repair summaries, even though those values are not required by the downstream decoder once the categorical outputs are written.
