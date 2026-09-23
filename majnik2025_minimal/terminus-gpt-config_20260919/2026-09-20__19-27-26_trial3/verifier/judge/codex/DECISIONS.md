# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates subject folders under `/app/data` whose names start with `jm` and that contain at least one date-like subdirectory. It then enumerates per-subject session directories matching `20*`. For each session it loads Suite2p deconvolved activity from `spks.npy`, cell labels from `iscell.npy`, motion energy from `motion_energy_glob.npy`, and camera timestamps from `tstamps.npy`. Trials are not loaded directly from disk; they are created later by slicing each continuous session into consecutive 60-second blocks after temporal averaging.

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir() and any(p.glob('20*')))
...
for subj_i, subject in enumerate(subjects):
    session_dirs = sorted(p for p in (ROOT / subject).glob('20*') if p.is_dir())
    for session_dir in session_dirs:
        p2 = session_dir / 'suite2p' / 'plane0'
        move = session_dir / 'move_deve'
        spks = np.load(p2 / 'spks.npy')
        iscell = np.load(p2 / 'iscell.npy')
        ...
        motion_raw = np.load(move / 'motion_energy_glob.npy')
        timestamps = np.load(move / 'tstamps.npy')
```

iii. In the trajectory, the agent first noted that the paper pointed toward baseline-corrected fluorescence from raw `F.npy`/`Fneu.npy` (steps 5 and 8), but later switched to loading `spks.npy` because the notebook presented it as an analysis-ready alternative and because downloading all `spks.npy` files was much cheaper than downloading all raw fluorescence files (step 14). It also justified loading timestamps so dropped camera frames could be reconstructed by interpolation (steps 4, 14, and 16).

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory. Every `/app/data/jm*` directory that contains at least one `20*` session directory becomes one subject, and the subject list is sorted alphabetically.

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir() and any(p.glob('20*')))
```

iii. The trajectory repeatedly refers to the dataset as six mice organized as `jm*` folders and says sessions will be enumerated chronologically/alphabetically within those subjects (steps 4, 6, and 14).

## 1-c. How are the data split into sessions?

i. Sessions are split by date-named subdirectories under each subject. The code uses all subject subdirectories matching `20*`, sorts them, and treats each one as a separate session.

ii.
```python
for subj_i, subject in enumerate(subjects):
    session_dirs = sorted(p for p in (ROOT / subject).glob('20*') if p.is_dir())
    for session_dir in session_dirs:
        ...
```

iii. The trajectory says it would “chronologically enumerate all 41 sessions” (step 14) and consistently describes each date folder as one daily recording session (steps 6 and 17).

## 1-d. How are the data split into trials?

i. The agent treats the recordings as continuous and creates artificial trials by splitting each temporally averaged session into consecutive, non-overlapping 60-second blocks. After 10-frame averaging at 30 Hz, each trial contains 180 bins. Any leftover bins that do not fill a full 60-second block are discarded by truncation.

ii.
```python
TRIAL_SECONDS = 60
BIN_SECONDS = AVERAGE_FRAMES / FS
BINS_PER_TRIAL = int(round(TRIAL_SECONDS / BIN_SECONDS))
...
n_bins = min(activity_binned.shape[1], len(motion_binned))
n_trials = n_bins // BINS_PER_TRIAL
n_use = n_trials * BINS_PER_TRIAL
...
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
```

iii. The trajectory explicitly says the requested trials are “consecutive, complete 60-s blocks” and that 10-frame averaging produces 180 samples per 60-second trial (steps 14 and 17).

## 1-e. How are trials filtered based on quality controls?

i. The code does not apply any trial-level quality-control filter. It only enforces that a session must contain at least two complete 60-second trials; otherwise it raises an error instead of producing output for that session.

ii.
```python
n_trials = n_bins // BINS_PER_TRIAL
n_use = n_trials * BINS_PER_TRIAL
if n_trials < 2:
    raise ValueError(f'fewer than two complete trials in {session_dir}')
```

iii. The trajectory ties this to the output-format requirement that sessions need at least two trials for decoder evaluation, not to any paper-defined trial-quality metric (steps 14 and 15).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from Suite2p’s `spks.npy`, not from raw fluorescence. The code also loads `iscell.npy` to decide which rows of `spks.npy` to keep.

ii.
```python
spks = np.load(p2 / 'spks.npy')
iscell = np.load(p2 / 'iscell.npy')
...
keep = (iscell[:, 0] == 1) & (iscell[:, 1] >= 0.5)
if not np.all(keep):
    spks = spks[keep]
```

iii. The trajectory shows an explicit change of mind: step 8 says the strongest paper-consistent choice would be to derive dF/F from `F.npy`, but step 14 says the agent instead chose `spks.npy` because the notebook offered it as a processed alternative and it avoided downloading far more raw data.

## 2-b. How is the `neural` data processed?

i. The agent does not perform neuropil subtraction or Suite2p baseline correction itself. Instead, it assumes `spks.npy` is already processed neural activity and applies only non-overlapping 10-frame averaging to reduce 30 Hz data to 3 Hz.

ii.
```python
# Identical paper processing for activity and behavior.
activity_binned = mean_blocks(np.asarray(spks, dtype=np.float32))
...
def mean_blocks(a, block=AVERAGE_FRAMES):
    n = a.shape[-1] // block * block
    return a[..., :n].reshape(*a.shape[:-1], n // block, block).mean(axis=-1)
```

iii. In step 14 the agent says it is “using deconvolved Suite2p activity” and in step 15 says it will apply “10-frame averaging” as the paper’s denoising step. The trajectory does not claim to reproduce the reference `F - 0.7 * Fneu` plus `dcnv.preprocess` pipeline; it explicitly chose an alternate neural source for convenience.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code filters neurons with Suite2p’s `iscell.npy`, keeping only rows with `iscell[:, 0] == 1` and cell probability `>= 0.5`. If all rows already satisfy that condition, no neurons are removed.

ii.
```python
iscell = np.load(p2 / 'iscell.npy')
...
keep = (iscell[:, 0] == 1) & (iscell[:, 1] >= 0.5)
if not np.all(keep):
    spks = spks[keep]
```

iii. The trajectory says the paper uses the default 0.5 Suite2p cell threshold and that the tracked rows in this dataset already appear curated, but the agent still chose to enforce the threshold in code “nevertheless” (docstring lines 6-9 and steps 5, 11, and 14).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code conceptually aligns neural data to the start of each consecutive 60-second block, not to a biological or task event. In practice it slices each session into contiguous trial windows after binning and records metadata claiming the alignment event is the start of each session block.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
...
'metadata': {
    'temporal_alignment_event': 'start of each consecutive 60-second session block',
    'off_start': 0.0,
    'off_end': float(TRIAL_SECONDS),
}
```

iii. The trajectory describes trials as artificial consecutive 60-second blocks and never identifies any stimulus or behavioral event for alignment (steps 5, 14, and 17). The metadata wording reflects that interpretation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins. With `FS = 30.0`, this yields 3 Hz sampling and a 333.33 ms time bin. The same temporal averaging is applied to neural activity and motion energy.

ii.
```python
FS = 30.0
AVERAGE_FRAMES = 10
BIN_SECONDS = AVERAGE_FRAMES / FS
...
activity_binned = mean_blocks(np.asarray(spks, dtype=np.float32))
motion_binned = mean_blocks(motion).astype(np.float32)
...
'time_bin_size': BIN_SECONDS * 1000.0,
```

iii. The trajectory says the paper averages neural and behavioral traces in non-overlapping 10-frame groups before decoding, turning 30 Hz into 3 Hz and yielding 180 bins per 60-second trial (steps 5, 14, and 17).

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a timestamp array. It is derived from the bin index, the assumed frame rate `FS = 30.0`, and the 10-frame bin width.

ii.
```python
FS = 30.0
AVERAGE_FRAMES = 10
...
elapsed = (np.arange(n_use, dtype=np.float32) * AVERAGE_FRAMES +
           (AVERAGE_FRAMES - 1) / 2) / FS
```

iii. The trajectory says sessions have fixed 30 Hz sampling and that 10-frame averaging defines the common 3 Hz time base; based on that, elapsed time can be generated directly from sample index rather than loaded from raw data (steps 11 and 14).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code computes elapsed time at the center of each 10-frame bin, not at the left edge. It then slices that 1D session-long vector into trial-length segments and stores each one as shape `(1, trial_timepoints)`.

ii.
```python
# Mean elapsed time of each underlying group of ten frames.
elapsed = (np.arange(n_use, dtype=np.float32) * AVERAGE_FRAMES +
           (AVERAGE_FRAMES - 1) / 2) / FS
...
ins.append(np.ascontiguousarray(elapsed[sl][None, :], dtype=np.float32))
```

iii. The trajectory does not dwell on the left-edge versus center choice, but the code comment says it is using the “mean elapsed time” of each underlying 10-frame group, which is the agent’s rationale encoded in the script itself.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction to the same binned sample index as the neural data. The code creates a single elapsed-time vector for the binned session, truncates it to the same number of usable bins as the neural and motion signals, and slices all three with the same per-trial indices.

ii.
```python
n_bins = min(activity_binned.shape[1], len(motion_binned))
n_trials = n_bins // BINS_PER_TRIAL
n_use = n_trials * BINS_PER_TRIAL
...
elapsed = (np.arange(n_use, dtype=np.float32) * AVERAGE_FRAMES +
           (AVERAGE_FRAMES - 1) / 2) / FS
...
sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
ins.append(np.ascontiguousarray(elapsed[sl][None, :], dtype=np.float32))
```

iii. In step 14 the agent says it will “encode elapsed session time as a 1×time input” after applying the same 10-frame averaging and trial splitting used for the neural and motion signals.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy labels are derived from `move_deve/motion_energy_glob.npy`. The code also uses `move_deve/tstamps.npy` as the alignment signal for reconstructing dropped camera frames before binning and discretization.

ii.
```python
motion_raw = np.load(move / 'motion_energy_glob.npy')
timestamps = np.load(move / 'tstamps.npy')
motion = restore_motion(motion_raw, timestamps, spks.shape[1])
```

iii. The trajectory repeatedly identifies `motion_energy_glob.npy` as the behavioral signal and says timestamps should be used to infer the locations of missing camera frames (steps 4, 11, 14, and 16).

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code first restores missing camera frames by interpolating `motion_energy_glob.npy` onto the neural-frame index implied by `tstamps.npy`. It then temporally averages motion in non-overlapping 10-frame bins, truncates to an integer number of 60-second trials, and finally discretizes the binned values into five within-session percentile bins.

ii.
```python
motion = restore_motion(motion_raw, timestamps, spks.shape[1])
...
motion_binned = mean_blocks(motion).astype(np.float32)
...
motion_binned = motion_binned[:n_use]
labels, edges = quintile_labels(motion_binned)
...
def quintile_labels(x):
    edges = np.percentile(x, [20, 40, 60, 80])
    return np.digitize(x, edges, right=False).astype(np.int64), edges
```

iii. Step 14 says motion should be “reconstruct[ed] ... at timestamp-indicated gaps by interpolation to the neural frame count,” then “average[d] ... in non-overlapping 10-frame bins as in the paper,” and then assigned “per-session quintile boundaries.”

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent uses five equal-frequency categories per session. It computes the 20th, 40th, 60th, and 80th percentiles of the binned motion trace for that session and uses `np.digitize` to assign labels `0` through `4`.

ii.
```python
def quintile_labels(x):
    # Internal percentile cut points; digitize returns exactly labels 0..4.
    edges = np.percentile(x, [20, 40, 60, 80])
    return np.digitize(x, edges, right=False).astype(np.int64), edges
```

iii. The trajectory explicitly says motion-energy classes should use “per-session quintile boundaries” (steps 14 and 17).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code aligns motion energy to neural frames before any averaging or trial splitting. If motion is shorter than the neural recording, it reconstructs the missing camera-frame indices from timestamp intervals, interpolates motion onto every neural frame index, averages the repaired signal and neural data with the same 10-frame bins, truncates them to the same usable length, and slices both with the same trial boundaries.

ii.
```python
def restore_motion(motion, timestamps, n_frames):
    ...
    intervals = np.diff(timestamps)
    dt = np.median(intervals)
    steps = np.maximum(1, np.rint(intervals / dt).astype(np.int64))
    x = np.concatenate(([0], np.cumsum(steps)))
    if x[-1] != n_frames - 1:
        raise ValueError(f'timestamp gaps imply {x[-1] + 1} frames, expected {n_frames}')
    return np.interp(np.arange(n_frames, dtype=np.float64), x, motion)
...
motion = restore_motion(motion_raw, timestamps, spks.shape[1])
activity_binned = mean_blocks(np.asarray(spks, dtype=np.float32))
motion_binned = mean_blocks(motion).astype(np.float32)
```

iii. The trajectory says motion streams that are shorter than neural streams should be restored by interpolation because timestamp gaps exactly explain the missing samples, and step 16 explains why the agent changed from rounding absolute timestamps to reconstructing frame indices from rounded inter-frame intervals.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are repaired by interpolation instead of dropping sessions or trials. If the inferred timestamp gaps do not exactly account for the expected neural-frame count, the code raises an error. Remainder bins that do not fill a complete 60-second trial are silently discarded by truncating to `n_use`.

ii.
```python
if len(motion) > n_frames:
    raise ValueError(f'camera has {len(motion)} samples but neural has {n_frames}')
...
if x[-1] != n_frames - 1:
    raise ValueError(f'timestamp gaps imply {x[-1] + 1} frames, expected {n_frames}')
...
n_trials = n_bins // BINS_PER_TRIAL
n_use = n_trials * BINS_PER_TRIAL
activity_binned = activity_binned[:, :n_use]
motion_binned = motion_binned[:n_use]
```

iii. The trajectory says sparse camera losses should be repaired rather than discarded (steps 14, 15, and 17), and step 16 adds an explicit correctness check that the reconstructed frame index must end exactly at the expected neural-frame count.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts of this code are loading the large `spks.npy` arrays for all 41 sessions, interpolating long motion traces when frames are missing, averaging large session-long arrays into 10-frame bins, and then materializing per-trial contiguous arrays. Unlike the human reference, this code does not spend time on Suite2p baseline preprocessing because it bypasses `F.npy`/`Fneu.npy`.

ii.
```python
spks = np.load(p2 / 'spks.npy')
...
motion = restore_motion(motion_raw, timestamps, spks.shape[1])
activity_binned = mean_blocks(np.asarray(spks, dtype=np.float32))
motion_binned = mean_blocks(motion).astype(np.float32)
...
ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
```

iii. The trajectory’s performance discussion centers on avoiding raw fluorescence downloads and reducing data volume through 10-frame averaging (steps 14 and 15). It does not mention any expensive baseline-correction stage, because the chosen pipeline does not include one.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized the main signal-processing steps (`np.interp`, reshape-and-mean binning, percentile computation). The remaining obvious loop that could be vectorized is the per-trial assembly loop, which slices one trial at a time and wraps each slice in `np.ascontiguousarray` before appending it to nested Python lists.

ii.
```python
ns, ins, outs = [], [], []
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
    ins.append(np.ascontiguousarray(elapsed[sl][None, :], dtype=np.float32))
    outs.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The trajectory specifically fixes an earlier non-vectorized timestamp-rounding approach by moving to a fully vectorized interpolation path (step 16). It does not discuss the trial loop as a bottleneck, so this vectorization opportunity is inferred from the final code.

## 6-c. What processing does the code repeat multiple times?

i. There is no large redundant preprocessing pass over the same data. The main repeated work is structural: for every session and every trial, the code repeats slice creation, contiguous copies, and list appends for `neural`, `input`, and `output`.

ii.
```python
for subj_i, subject in enumerate(subjects):
    ...
    for session_dir in session_dirs:
        ...
        for tr in range(n_trials):
            sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
            ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
            ins.append(np.ascontiguousarray(elapsed[sl][None, :], dtype=np.float32))
            outs.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The trajectory treats the pipeline as a single pass per session and does not call out any intentional duplicated preprocessing. The repeated work here is a consequence of building the nested session/trial structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extra metadata that the decoder does not need, especially `session_info` and per-session `motion_quintile_edges`. It also loads `iscell.npy` and performs a filter check even though the trajectory concluded the tracked rows were already curated in practice. Those steps are not required for the downstream decoder task.

ii.
```python
iscell = np.load(p2 / 'iscell.npy')
...
session_info.append({
    'subject': subject, 'session': session_dir.name,
    'n_neurons': int(spks.shape[0]), 'n_trials': n_trials,
    'original_neural_frames': int(spks.shape[1]),
    'original_motion_frames': int(len(motion_raw)),
    'interpolated_motion_frames': int(spks.shape[1] - len(motion_raw)),
    'motion_quintile_edges': edges.tolist(),
})
...
'metadata': {
    ...
    'session_info': session_info,
}
```

iii. Step 11 and step 14 say the rows already appear curated and constant across days, so the extra `iscell` enforcement is mostly a guard. The trajectory also emphasizes rich metadata for verification, even though those extra fields are not needed by the decoder itself.
