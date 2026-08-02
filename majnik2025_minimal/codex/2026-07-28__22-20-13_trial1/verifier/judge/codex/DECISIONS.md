# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by scanning `/app/data` for subject directories whose names start with `jm`, then scanning each subject directory for session subdirectories whose first four characters are digits. For each session it loads Suite2p fluorescence files plus `ops.npy`, and loads motion-energy files from `move_deve`. Trials are not loaded directly from disk; they are created later by splitting each continuous session in code.

ii. <Code snippets>

```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )

def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )
```

```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
...
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
```

iii. The justification in `CONVERSION_NOTES.md` is that the bundled release contains six `jm*` mice and one recording day per session. The trajectory and notes say the agent treated one recording day as one session and then trialized the continuous recordings later.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories beginning with `jm`, sorted lexicographically.

ii. <Code snippets>

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

iii. `CONVERSION_NOTES.md` says the source inventory contains `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and the paper-level sanity check says the release contains exactly six mice.

## 1-c. How are the data split into sessions?

i. Each session is one recording day subdirectory inside a mouse directory. The AI keeps one session per day and uses the folder name as the session id.

ii. <Code snippets>

```python
def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )
```

```python
subject = session_dir.parent.name
session_id = session_dir.name
session_key = f"{subject}/{session_id}"
```

iii. `CONVERSION_NOTES.md` explicitly states: “Session unit: one recording day per session.”

## 1-d. How are the data split into trials?

i. The AI does not use the continuous recording as a single stream. It cuts each session into consecutive non-overlapping 2-minute blocks, then treats each 2-minute block as a trial. This is done after defining `TRIAL_SECONDS = 120.0`; each trial contains `3600` raw frames, which are later averaged into `360` ten-frame bins.

ii. <Code snippets>

```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES
```

```python
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
...
n_trials = usable_frames // TRIAL_FRAMES
...
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. `CONVERSION_NOTES.md` says the AI chose “consecutive 2-minute blocks extracted from each continuous recording, matching the paper's decoding split strategy.” The trajectory repeats that the agent believed the paper’s decoding split strategy should define the trial unit.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit trial-quality filter, but it discards trailing frames that do not fill a complete 2-minute block and raises an error if a session would produce fewer than two such trials.

ii. <Code snippets>

```python
usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
if usable_frames != session_duration_frames:
    print(
        f"  Dropping trailing frames: kept {usable_frames} of {session_duration_frames} for consecutive 2-minute blocks"
    )
...
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The justification is implicit rather than formal: the code enforces complete 2-minute blocks and the decoder requirement that each session have at least two trials. `CONVERSION_NOTES.md` does not describe a separate quality-control rule for trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `F.npy` and `Fneu.npy`, using parameters stored in `ops.npy` to choose the neuropil coefficient and Suite2p preprocessing settings.

ii. <Code snippets>

```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md` says: “Load `F.npy`, `Fneu.npy`, and `ops.npy`,” then use Suite2p defaults from `ops.npy` instead of inventing new baseline parameters.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction, runs `suite2p.extraction.dcnv.preprocess` with parameters taken from `ops.npy`, reconstructs a baseline estimate, converts the signal to `dF/F`, and then averages the result in non-overlapping 10-frame bins.

ii. <Code snippets>

```python
Fcorr = F
Fcorr -= float(ops["neucoeff"]) * Fneu
```

```python
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

iii. The notes justify this by saying the paper mentions “slightly denoised `dF/F`” and 10-timestamp averaging, so the AI chose to recover `dF/F` from Suite2p preprocessing and then bin by 10 frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons by `iscell`, quality score, or any other explicit criterion. All rows present in `F.npy` are kept.

ii. <Code snippets>

```python
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
...
return dff_num.astype(np.float32, copy=False), ops
```

```python
data["brain_region_idx"].append(np.zeros(record["n_neurons"], dtype=np.int64))
```

iii. The AI’s notes do not claim any neuron-level QC beyond using the bundled tracked-cell release. There is no code path reading `iscell.npy` or removing cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of each artificial 2-minute block rather than to the start of the session. Metadata describes the alignment event as the start of each consecutive 2-minute block.

ii. <Code snippets>

```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block from a continuous recording",
    "off_start": 0.0,
    "off_end": TRIAL_SECONDS,
```

```python
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. `CONVERSION_NOTES.md` says: “Trials are aligned to block start for `metadata.off_start = 0` and `metadata.off_end = 120`.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, giving a time bin size of `333.333... ms`. Yes, temporal rebinning is applied by averaging every 10 consecutive frames.

ii. <Code snippets>

```python
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

iii. The AI justifies this in `CONVERSION_NOTES.md` by citing the paper statement that decoding used behavior traces obtained by averaging 10 consecutive timestamps, and extends that same 10-frame averaging to neural traces as well.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not loaded from a timestamp file. The AI derives time from bin indices plus the nominal frame rate.

ii. <Code snippets>

```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. `CONVERSION_NOTES.md` describes the decoder input as “absolute time elapsed from session start, represented as a continuous time series in seconds.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes the midpoint time of each 10-frame bin in seconds and then offsets each trial by its position within the session, yielding absolute time-from-session-start rather than trial-relative time.

ii. <Code snippets>

```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The notes justify this directly: “The decoder input is not trial-relative time; it is absolute time-from-session-start in seconds, carried through as a 1-by-time array.”

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the input time series to the same 2-minute block structure and the same 10-frame bins used for neural data, with one scalar time value per neural bin.

ii. <Code snippets>

```python
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
...
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. This is justified implicitly by the AI’s decision to block and bin all streams identically before building the dataset.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/interframe_int.npy` used to infer missing camera frames.

ii. <Code snippets>

```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
...
interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. `CONVERSION_NOTES.md` says missing camera frames should be identified from `interframe_int.npy` and repaired before downstream use.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI repairs missing motion samples by using large interframe intervals to infer gaps, inserts `NaN`s into a repaired array, fills them by linear interpolation, averages motion energy in non-overlapping 10-frame bins, computes a global min and max across all binned motion values, and later normalizes each binned trial by that global min-max range.

ii. <Code snippets>

```python
nominal = float(np.median(interframe))
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
...
for gap in jumps:
    repaired[dst] = motion[src]
    ...
    if gap:
        repaired[dst:dst + gap] = np.nan
...
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

iii. The notes justify this by citing release notes about missing camera frames and a methods statement about averaging 10 timestamps, then choosing a global normalized scale before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global 20/40/60/80 percentiles on the 10-frame-averaged motion values, transforms those cut points into the corresponding global min-max normalized scale, then uses `np.digitize` to assign each time bin to one of five quintile categories.

ii. <Code snippets>

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

iii. `CONVERSION_NOTES.md` describes this as “globally normalized, and discretized into 5 equal-percentile bins,” with the caveat that the quintiles are computed after 10-frame averaging and min-max normalization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first repairs the motion signal to match the neural frame count, then applies the same 2-minute blocking and 10-frame averaging as for neural data, producing one motion label per neural time bin.

ii. <Code snippets>

```python
motion_full, missing_count = repair_motion_energy(session_dir, n_frames)
neural_trials, time_trials, motion_trials = session_to_trials(neural_full, motion_full, n_frames)
```

```python
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
```

iii. The notes justify this as synchronized behavior-neural alignment after missing-frame repair, followed by identical block-and-bin processing for all streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI explicitly repairs missing motion-energy samples by inferring dropped camera frames from `interframe_int.npy`, filling the missing positions by interpolation, and raising errors if the inferred number of missing frames does not match the neural-motion length difference. It also drops leftover session frames that do not fill a complete 2-minute block.

ii. <Code snippets>

```python
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
if usable_frames != session_duration_frames:
    print(
        f"  Dropping trailing frames: kept {usable_frames} of {session_duration_frames} for consecutive 2-minute blocks"
    )
```

iii. `CONVERSION_NOTES.md` cites release notes saying missing camera frames should be treated as missing or interpolated, and the notes summarize repaired-frame counts per session as a sanity check.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is rebuilding the neural signal session-by-session in `load_suite2p_dff`, especially the Suite2p preprocessing call and subsequent `dF/F` reconstruction. The first full-dataset pass over all 41 sessions dominates the runtime.

ii. <Code snippets>

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

iii. The trajectory explicitly says the heavy step is reconstructing `dF/F` “session by session” and later says the full conversion is the heaviest step because it has to rebuild `dF/F` for all 41 recording days.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not document this explicitly, but the remaining obvious Python loops are the missing-frame repair loop over `jumps`, the per-trial list construction in `session_to_trials`, and the per-trial normalization-plus-digitization loop when building `data["output"]`. The repair loop is already more efficient than repeated `np.insert`, but it is still a Python loop.

ii. <Code snippets>

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
```

```python
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
    output_trials.append(motion_bins[np.newaxis, :])
```

iii. There is no explicit justification in the notes. This answer is inferred from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats several passes over motion data: first during per-session repair and binning, then when concatenating all motion trials to compute global statistics, and then again when normalizing and digitizing each trial into output categories. It also makes multiple copies of the fluorescence arrays while computing `dF/F`.

ii. <Code snippets>

```python
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
...
all_motion = np.concatenate(
    [trial for record in session_records for trial in record["motion_trials"]]
).astype(np.float32, copy=False)
...
for motion_trial in record["motion_trials"]:
    motion_norm = (motion_trial - motion_min) / motion_scale
    motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

```python
Fcorr = F
...
dff_num = Fcorr.copy()
...
baseline = Fcorr
baseline -= dff_num
```

iii. The notes do not call this out. This is inferred from the implementation structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extra conversion summaries and metadata that are not used by the downstream decoder, including per-mouse summary statistics and missing-frame reports. Within the output pipeline, the global min-max normalization is also unnecessary for percentile digitization because the later discretization only depends on rank order.

ii. <Code snippets>

```python
summary = {
    "mode": mode,
    "nsessions": len(session_records),
    "subjects": subjects,
    "neurons_per_mouse": neurons_per_mouse,
    "duration_minutes_unique": sorted({round(value, 3) for value in duration_minutes}),
    "total_missing_behavior_frames": int(sum(missing_behavior_by_session.values())),
    ...
}
```

```python
motion_min = float(all_motion.min())
motion_max = float(all_motion.max())
motion_scale = motion_max - motion_min
...
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)
...
motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

iii. There is no explicit justification in the notes for these extra computations beyond documentation and sanity checks. The min-max normalization step is described in `CONVERSION_NOTES.md`, but it does not change percentile-bin assignments apart from the AI’s separate 10-frame averaging choice.
