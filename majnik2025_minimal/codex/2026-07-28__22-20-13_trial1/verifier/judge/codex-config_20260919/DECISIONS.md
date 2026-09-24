# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` for sorted `jm*` subject directories, then sorted date-like session directories whose first four characters are digits. In full mode it loads every such session. For each session it loads `ops.npy`, `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and, only when motion length differs from neural length, `interframe_int.npy`.

ii.
```python
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

ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
```

iii. The trajectory says each recording day is treated as a session and reports that the full pass processed all 41 recording days. The date-name filter was intended to exclude unrelated folders, while full/sample modes supported complete conversion and cheaper testing.

## 1-b. How are the data split into subjects?

i. Each sorted directory beginning with `jm` is a mouse. The unique parent directory names become `subjects`, and a lookup maps every session to `subject_idx`.

ii.
```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir()
                   if path.is_dir() and path.name.startswith("jm")])

subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}
```

iii. The agent inferred the release convention correctly and later sanity-checked that it found six mice.

## 1-c. How are the data split into sessions?

i. One date-like subdirectory of a mouse is one session/recording day; directories are sorted deterministically.

ii.
```python
def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted([path for path in subject_dir.iterdir()
                   if path.is_dir() and path.name[:4].isdigit()])
```

iii. The trajectory explicitly states “each recording day as a session,” consistent with the continuous daily recordings.

## 1-d. How are the data split into trials?

i. The agent divides each continuous session into consecutive, non-overlapping 120-second blocks, drops a trailing partial block, and requires at least two blocks. After 10-frame averaging, each trial has 360 time bins. This conflicts with the task's required 60-second trials.

ii.
```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
n_trials = usable_frames // TRIAL_FRAMES
```

iii. The agent chose 2-minute blocks because the paper used consecutive 2-minute splits for nested cross-validation, calling this a “plausible trial structure.” It overlooked the explicit conversion instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial filtering. Incomplete trailing blocks are dropped; sessions with fewer than two retained trials raise an error.

ii.
```python
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The minimum was imposed to satisfy the decoder format requirement. The trajectory gives no separate behavioral or neural trial-quality criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p `F.npy` and `Fneu.npy`; preprocessing parameters are taken from `ops.npy`.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The agent investigated the notebook and Suite2p metadata and concluded it should reconstruct the signal rather than use a precomputed trace.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil using `ops["neucoeff"]`, runs Suite2p `dcnv.preprocess` with parameters from `ops`, reconstructs the baseline as corrected fluorescence minus the returned baseline-subtracted trace, clips that baseline to `1e-3`, divides by it to create dF/F, and finally averages every 10 frames.

ii.
```python
Fcorr = F
Fcorr -= float(ops["neucoeff"]) * Fneu
dff_num = dcnv.preprocess(F=dff_num, baseline=ops["baseline"],
    win_baseline=ops["win_baseline"], sig_baseline=ops["sig_baseline"],
    fs=ops["fs"], prctile_baseline=ops["prctile_baseline"],
    batch_size=ops.get("batch_size", 100), device=torch.device("cpu"))
baseline = Fcorr
baseline -= dff_num
np.maximum(baseline, 1e-3, out=baseline)
dff_num /= baseline
```

iii. The trajectory says the notebook requested dF/F and the agent wanted to reuse Suite2p's exact preprocessing instead of approximating a baseline. Its final notes describe the extra division as matching the paper's dF/F statement. The human reference, however, stops at the `dcnv.preprocess` result.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROI/neuron filtering is performed; all rows of `F.npy` are retained. The frame rate is checked to equal 30 Hz.

ii.
```python
if float(ops["fs"]) != FRAME_RATE_HZ:
    raise ValueError(...)
```

iii. The trajectory does not record a decision to use `iscell.npy` or another ROI-quality mask; it treated the supplied Suite2p arrays as the tracked neuron set.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are consecutive slices aligned to each artificial 2-minute block start. Metadata sets the event to block start with offsets 0 and 120 seconds.

ii.
```python
neural_trials = [neural_binned[:, trial_index, :].astype(np.float32, copy=True)
                 for trial_index in range(n_trials)]
"temporal_alignment_event": "start of each consecutive 2-minute block from a continuous recording",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

iii. The agent noted there is no natural event in the continuous recording and therefore used the artificial block boundary. The duration follows its erroneous 2-minute trial choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw 30 Hz samples are averaged in non-overlapping groups of 10, yielding 3 Hz or 333.333 ms bins for both neural and motion streams.

ii.
```python
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FRAME_RATE_HZ
neural_binned = neural_full[:, :usable_frames].reshape(..., BIN_FRAMES).mean(axis=3)
```

iii. The agent followed the paper's statement that decoding denoised dF/F and behavior by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthesized from bin indices, the 10-frame bin width, 30 Hz frame rate, trial index, and 120-second block duration; no timestamp file is used.

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES
             + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [(base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :]
               for trial_index in range(n_trials)]
```

iii. The fixed acquisition rate made index-derived time sufficient. The agent interpreted “experiment” as the start of each recording session.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It uses the center of each 10-frame bin, so the first value is 1/6 second, then adds the absolute block offset; values are float32 seconds and remain continuous across trials.

ii.
```python
(np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES
 + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
```

iii. The agent's notes explicitly say the input is absolute session time, not trial-relative time. It chose bin centers, whereas the reference uses bin left edges beginning at zero.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One time value is generated for every averaged neural bin and the same trial index supplies both arrays, giving shape `(1, 360)` against neural `(neurons, 360)`.

ii.
```python
time_trials = [(base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :]
               for trial_index in range(n_trials)]
```

iii. The agent intentionally carried absolute time through each block and validated shape compatibility with the decoder.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from `move_deve/motion_energy_glob.npy`; `interframe_int.npy` is additionally used when missing camera frames must be located.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The agent identified global motion energy as the supplied behavioral signal and used interframe intervals because the release notes warned of missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing frames are inferred from interval/median ratios, inserted as NaNs into a preallocated array, and linearly interpolated. The repaired trace is averaged over the same 10-frame bins as neural data. All retained binned motion values across all sessions are then globally min-max normalized before categorization.

ii.
```python
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask])
motion_binned = motion_full[:usable_frames].reshape(..., BIN_FRAMES).mean(axis=2)
motion_norm = (motion_trial - motion_min) / motion_scale
```

iii. The agent rejected drift-sensitive timestamp mapping and chose gap detection from unusually long interframe intervals. It applied common global normalization to implement the original prompt's “normalized” wording, although the evaluated reference specifies per-session percentile selection.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds (20th, 40th, 60th, 80th percentiles) are computed once from all binned motion samples across every session. `np.digitize` assigns categories 0–4. Thresholding the normalized values is mathematically equivalent to thresholding raw values with the corresponding global raw quantiles.

ii.
```python
all_motion = np.concatenate([trial for record in session_records
                             for trial in record["motion_trials"]])
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80])
motion_quantiles_norm = (motion_quantiles_raw - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False)
```

iii. The agent sought globally balanced quintiles and reported global output fractions. This conflicts with the instruction and reference decision that bins are selected separately per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion is repaired to exactly the neural frame count, both streams are truncated to the same complete-block length, reshaped into identical trial/bin boundaries, and independently averaged within each same 10 raw frames.

ii.
```python
motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
neural_binned = neural_full[:, :usable_frames].reshape(..., BIN_FRAMES).mean(axis=3)
motion_binned = motion_full[:usable_frames].reshape(..., BIN_FRAMES).mean(axis=2)
```

iii. The agent viewed acquisition as synchronous apart from skipped camera frames and added strict inferred/expected missing-count checks before common indexing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Motion-length mismatches trigger gap inference from interframe intervals. The inferred count must exactly equal the neural-motion length difference; otherwise conversion raises. Missing locations are linearly interpolated. Frame-rate mismatches, reconstruction inconsistencies, zero motion range, too-short sessions, and format errors also raise. Partial final trials are dropped.

ii.
```python
if missing_count != expected_missing:
    raise ValueError(...)
if nan_mask.any():
    repaired[nan_mask] = np.interp(...).astype(np.float32)
if motion_scale <= 0:
    raise ValueError("Motion energy has zero range after processing")
```

iii. The trajectory says the gap logic was chosen after naive timestamp mapping proved drift-sensitive. The agent emphasized explicit checks so misalignment would not pass silently.

## 6-a. What are the most time-consuming steps of the code?

i. Full-session Suite2p baseline preprocessing for every neuron/session is the principal compute cost; loading and storing the large arrays and the full decoder validation are also costly.

ii.
```python
dff_num = dcnv.preprocess(..., device=torch.device("cpu"))
```

iii. During execution the agent repeatedly described reconstructing dF/F across 41 sessions as the heaviest conversion step; it first ran a sample build to reduce iteration cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over `jumps` in motion repair, list comprehensions that copy each trial, the per-session conversion loop, and the per-trial output normalization/digitization loop could be reduced or vectorized. The major 10-frame averaging is already vectorized with reshape/mean.

ii.
```python
for gap in jumps:
    repaired[dst] = motion[src]
    ...
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
```

iii. The trajectory contains no explicit efficiency justification for these loops. The repair loop does preallocate its output, which is substantially better than repeated `np.insert` and is reasonable for sparse gaps.

## 6-c. What processing does the code repeat multiple times?

i. Every session separately repeats file loading, Suite2p baseline processing, missing-frame checks, trial reshaping, and copying. Motion is traversed once to concatenate/compute global statistics and again to normalize and digitize each trial. The input time template is also rebuilt as a list per session despite identical trial lengths.

ii.
```python
for session_index, session_dir in enumerate(session_dirs):
    neural_full, ops = load_suite2p_dff(session_dir)
    ...
all_motion = np.concatenate(...)
for record in session_records:
    for motion_trial in record["motion_trials"]:
```

iii. The agent accepted the two-pass structure because global thresholds cannot be known until all sessions have been processed; no further rationale is recorded.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes conversion summaries and global output counts only for logging, retains `ops` only long enough to validate frame rate, creates many copied trial arrays, and performs an extra baseline reconstruction/division absent from the reference pipeline. It also runs min-max normalization even though percentile classification is invariant under this monotonic affine transform.

ii.
```python
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale)
motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False)
```

iii. The agent used summaries for sanity checks and metadata, and normalization to follow its reading of the task. It did not recognize that normalization cancels out when both samples and thresholds receive the same transform.
