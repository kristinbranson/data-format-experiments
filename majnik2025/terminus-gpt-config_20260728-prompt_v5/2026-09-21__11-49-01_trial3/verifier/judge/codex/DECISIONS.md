# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every subject directory under `/app/data`, then every session directory under each subject, and keeps only sessions that contain both `suite2p/plane0` and `move_deve`. For each kept session it loads Suite2p files (`ops.npy`, `F.npy`, `Fneu.npy`, `iscell.npy`) and behavior files (`motion_energy_glob.npy`, `tstamps.npy`). Trials are not loaded directly; they are created later from the continuous session recordings.

ii.
```python
def list_sessions(data_root: Path):
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            s2p = sess_dir / 'suite2p' / 'plane0'
            mov = sess_dir / 'move_deve'
            if s2p.exists() and mov.exists():
                sessions.append((subj_dir.name, sess_dir.name, sess_dir))
    return sessions

def load_session(sess_dir: Path):
    s2p = sess_dir / 'suite2p' / 'plane0'
    mov = sess_dir / 'move_deve'
    ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
    F = np.load(s2p / 'F.npy').astype(np.float32)
    Fneu = np.load(s2p / 'Fneu.npy').astype(np.float32)
    iscell = np.load(s2p / 'iscell.npy')
    motion = np.load(mov / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
    return ops, F, Fneu, iscell, motion, tstamps
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset is organized as subject directories containing session directories, each with `suite2p/plane0` and `move_deve`, and that these are the relevant native sources. The notes also describe the motion-energy arrays as slightly shorter than imaging arrays, motivating the choice to load timestamps alongside the traces.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the names of the top-level directories under `/app/data`. The AI derives the final `subjects` list from the set of subject names attached to the enumerated sessions, then sorts them.

ii.
```python
sessions = list_sessions(data_root)
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes say `subject_idx` should come from the top-level mouse folder names and that each top-level directory corresponds to one mouse. No narrower filter such as `jm*` is applied in the final code.

## 1-c. How are the data split into sessions?

i. Each kept session is one subject subdirectory that contains both the Suite2p and motion directories. The dataset stores one converted session per such directory.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        s2p = sess_dir / 'suite2p' / 'plane0'
        mov = sess_dir / 'move_deve'
        if s2p.exists() and mov.exists():
            sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. The notes explicitly state: “Treat each session directory as one session in the target dataset.” The presence checks for `suite2p/plane0` and `move_deve` are the AI’s way of validating that a session directory is usable.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions and creates artificial trials as consecutive non-overlapping 60-second windows after binning. It computes the bin duration as `10 / fs` seconds, converts 60 seconds into `trial_len` bins, then slices each session into `n_trials = floor(n_bins / trial_len)` windows.

ii.
```python
dt = float(bin_size_frames / fs)
trial_len = int(round(trial_seconds / dt))
n_trials = neural_b.shape[1] // trial_len

for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(time_b[sl][None, :].astype(np.float32))
    output_trials.append(motion_bins[sl][None, :].astype(np.int64))
```

iii. In the notes, the AI calls these “60 s pseudo-trials” and justifies them as required by the decoder task because the native recordings are continuous rather than naturally trial-structured.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter. The AI discards any incomplete trailing segment that does not fill a full 60-second window and raises an error if a session would yield fewer than two trials.

ii.
```python
n_trials = neural_b.shape[1] // trial_len
if n_trials < 2:
    raise ValueError(f'Not enough 60 s trials in session {sess_dir.name}: {n_trials}')

usable = n_trials * trial_len
neural_b = neural_b[:, :usable]
motion_bins = motion_bins[:usable]
time_b = time_b[:usable]
```

iii. The justification in the notes is procedural rather than biological: the decoder requires at least two trials per session, and fixed-length pseudo-trials require dropping any incomplete tail so shapes stay consistent.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from Suite2p fluorescence and metadata: `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy`. `F` and `Fneu` provide the traces, `iscell` is used to select ROIs, and `ops` supplies `fs` and `neucoeff`.

ii.
```python
ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
F = np.load(s2p / 'F.npy').astype(np.float32)
Fneu = np.load(s2p / 'Fneu.npy').astype(np.float32)
iscell = np.load(s2p / 'iscell.npy')
...
fs = float(ops.get('fs', 30.0))
neucoeff = float(ops.get('neucoeff', 0.7))
cell_mask = iscell[:, 0].astype(bool)
F = F[cell_mask]
Fneu = Fneu[cell_mask]
```

iii. The notes justify this by saying the reference code uses Suite2p-style loading and that `iscell` is the clearest available curation signal. The notes also say the methods mention dF/F, so `F` and `Fneu` are the raw variables most suitable for reconstructing that signal.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil correction and then applies its own “dF/F-like” normalization. Specifically, it computes `Fcorr = F - neucoeff * Fneu`, estimates a per-neuron baseline as the 20th percentile across time, clips very small baselines to `1e-6`, and returns `(Fcorr - baseline) / baseline`. After that it averages the traces in non-overlapping 10-frame bins.

ii.
```python
def robust_dff(Fcorr: np.ndarray):
    baseline = np.percentile(Fcorr, 20, axis=1, keepdims=True)
    baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
    return (Fcorr - baseline) / baseline

Fcorr = F - neucoeff * Fneu
neural = robust_dff(Fcorr)
neural_b = moving_average_bin(neural, bin_size_frames).astype(np.float32)
```

iii. The justification in the notes is that the paper methods describe decoding from dF/F traces, but the raw data do not provide a direct dF/F array. The AI therefore chose a “dF/F-like reconstruction” after neuropil correction and said this was an approximation intended to match the methods text where possible.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using Suite2p `iscell`, keeping only ROIs whose first `iscell` column is truthy. No additional neuron-quality metric is used beyond that mask.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
F = F[cell_mask]
Fneu = Fneu[cell_mask]
```

iii. The notes repeatedly justify this by saying Track2p/Suite2p code appears to rely on `iscell`-based curation and that, absent a clearer paper-specific rule, `iscell` is the best available neuron-inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to session start rather than to an external stimulus or behavior event. After overlap trimming and binning, each trial is simply the next contiguous 60-second window from the session timeline.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'session start',
    'off_start': 0.0,
    'off_end': 60.0,
    ...
}
...
for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The notes justify this by saying the recordings are continuous and the decoder task requires pseudo-trials, so the appropriate alignment event is the start of the session and then consecutive within-session windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins. With `fs` taken from `ops`, this yields `dt = 10 / fs` seconds per bin, which is about 333.3 ms at 30 Hz. The same temporal binning is applied to neural activity, motion energy, and the derived session-time input.

ii.
```python
def moving_average_bin(x: np.ndarray, bin_size: int):
    n = x.shape[-1] // bin_size
    trimmed = x[..., : n * bin_size]
    new_shape = x.shape[:-1] + (n, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

neural_b = moving_average_bin(neural, bin_size_frames).astype(np.float32)
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
```

iii. The notes quote the methods statement that decoding used averages over 10 consecutive timestamps and say the conversion should match that denoising step for both neural and behavioral traces.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not use a stored raw time-in-seconds variable. It derives session time from the imaging frame index and the sampling rate `fs` from `ops.npy`. Although it loads `tstamps.npy`, the final input time series comes from `np.arange(n_overlap) / fs`.

ii.
```python
ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
...
fs = float(ops.get('fs', 30.0))
...
raw_time = np.arange(n_overlap, dtype=np.float32) / fs
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
```

iii. The notes and trajectory justify this explicitly: an earlier attempt used `tstamps.npy` directly, but that produced zero valid 60-second trials because the timestamp units were not suitable as elapsed seconds. The AI therefore switched to frame index divided by imaging rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI first computes per-frame elapsed session time from `np.arange(n_overlap) / fs`, then averages that time vector in the same non-overlapping 10-frame bins used for neural and motion data. The resulting binned times are then cut into 60-second pseudo-trials.

ii.
```python
raw_time = np.arange(n_overlap, dtype=np.float32) / fs
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
...
input_trials.append(time_b[sl][None, :].astype(np.float32))
```

iii. The notes justify the frame-index part because `tstamps.npy` did not behave like seconds. The use of the same averaging function as other streams is implied by the stated goal of keeping neural, input, and output on one shared binned time base.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the time input to neural data by trimming both to the same overlap length, applying the same 10-frame binning, and then slicing the same trial boundaries from all streams.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
neural = neural[:, :n_overlap]
motion = motion[:n_overlap]
tstamps = tstamps[:n_overlap]
...
for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(time_b[sl][None, :].astype(np.float32))
```

iii. The notes say the elapsed-time vector should be “after the same temporal binning used for neural/output,” and the trajectory records that the time axis was changed specifically to restore consistent 60-second trial segmentation.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion-energy output from `move_deve/motion_energy_glob.npy`, while also loading `tstamps.npy` to limit all streams to a common overlap. It does not use `interframe_int.npy`.

ii.
```python
motion = np.load(mov / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
...
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:n_overlap]
```

iii. The notes justify this by saying `motion_energy_glob.npy` is the scalar behavior signal described in the methods and that `tstamps` should define the behavior time base. The recorded trajectory also shows the AI explicitly chose overlap trimming after deciding the timestamps were not usable as elapsed seconds.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI trims motion energy to the shared overlap with neural data and timestamps, averages it in non-overlapping 10-sample bins, then discretizes the binned trace into five quantile bins within each session.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
motion = motion[:n_overlap]
...
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)
motion_bins, edges = compute_quantile_bins(motion_b, n_bins=5)
```

iii. The notes justify the 10-sample averaging by quoting the paper’s “10 consecutive timestamps” denoising step. They justify within-session binning as required by the decoder task and preferable for avoiding cross-session scale confounds.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes five session-specific quantile bins from the binned motion-energy values. It uses equally spaced quantiles from 0 to 1, adjusts any non-increasing bin edges upward by `1e-9`, then assigns labels with `np.digitize`.

ii.
```python
def compute_quantile_bins(values: np.ndarray, n_bins: int = 5):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins.astype(np.int64), edges
```

iii. The notes explicitly say the output should be “5 equiprobable bins” chosen within each session to satisfy the decoder specification while avoiding session-to-session scale differences. The small edge adjustment is an implementation safeguard rather than a separately documented conceptual decision.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by truncating both streams, and the timestamps array, to the same minimum sample count before binning and trialization. The binned motion and binned neural data are then sliced with identical trial boundaries.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
neural = neural[:, :n_overlap]
motion = motion[:n_overlap]
tstamps = tstamps[:n_overlap]
...
for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    output_trials.append(motion_bins[sl][None, :].astype(np.int64))
```

iii. The notes say the behavior trace is typically slightly shorter than imaging and that the chosen resolution was to “trim neural and behavior arrays to their overlapping sample count.” This was presented as the fix after a failed timestamp-based approach.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several small issues conservatively: it clips tiny dF/F baselines to avoid division by zero, trims all streams to their common overlap length when they disagree by a few samples, discards any incomplete trailing pseudo-trial, and errors out if a session has fewer than two full trials. It does not attempt to interpolate missing motion frames.

ii.
```python
baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
...
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
neural = neural[:, :n_overlap]
motion = motion[:n_overlap]
tstamps = tstamps[:n_overlap]
...
if n_trials < 2:
    raise ValueError(f'Not enough 60 s trials in session {sess_dir.name}: {n_trials}')
usable = n_trials * trial_len
```

iii. The notes justify overlap trimming by saying motion-energy arrays were slightly shorter than imaging arrays, and they justify the time-input rewrite because `tstamps` could not be used as elapsed seconds. The handling is framed as pragmatic shape reconciliation rather than a paper-derived correction method.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s documentation suggests the expensive parts are full-array per-session loading and overall per-session conversion work, while plotting is intentionally limited to the first two sessions in `--show-processing` mode to keep that path from becoming dominant. The code also records per-session and total elapsed times.

ii.
```python
t0 = time.time()
for idx, (subject, session_name, sess_dir) in enumerate(sessions):
    st = time.time()
    neural_trials, input_trials, output_trials, info = process_session(
        sess_dir, show_processing=show_processing and idx < 2
    )
    ...
    print(f'processed {subject}/{session_name}: ... time={time.time()-st:.2f}s')

print(f'total sessions={len(data["neural"])} total time={time.time()-t0:.2f}s')
```

iii. In Step 6 and Step 7 of the notes, the AI says full-array loading per session is acceptable for this dataset size, gives a measured sample runtime, and says limiting plots avoids plot generation dominating runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main temporal averaging step with reshape-and-mean. The remaining obvious Python loop is the per-trial list construction, plus the small loop that makes quantile edges strictly increasing.

ii.
```python
def moving_average_bin(x: np.ndarray, bin_size: int):
    n = x.shape[-1] // bin_size
    trimmed = x[..., : n * bin_size]
    new_shape = x.shape[:-1] + (n, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(time_b[sl][None, :].astype(np.float32))
    output_trials.append(motion_bins[sl][None, :].astype(np.int64))
```

iii. The notes explicitly claim “Vectorized non-overlapping bin averaging via reshape/mean” as a speedup. They do not call out additional vectorization opportunities beyond that, so the remaining opportunities are only implicit from the code structure.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same generic operations across streams within each session: trimming to overlap, binning, and then trial slicing. It also repeatedly casts arrays to `float32` or `int64` at multiple stages, and it applies the same `moving_average_bin` helper separately to neural activity, motion energy, and the constructed time axis.

ii.
```python
neural_b = moving_average_bin(neural, bin_size_frames).astype(np.float32)
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
...
neural_trials.append(neural_b[:, sl].astype(np.float32))
input_trials.append(time_b[sl][None, :].astype(np.float32))
output_trials.append(motion_bins[sl][None, :].astype(np.int64))
```

iii. There is no explicit prose justification for this repetition. It appears to be a straightforward implementation choice for keeping the three output streams synchronized while packaging them into the required nested session/trial structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several intermediates that are not used by the downstream decoder itself: `motion_b` is only an intermediate toward categorical labels and optional plots, `quantile_edges` are stored only in metadata, and `tstamps` are loaded only to constrain `n_overlap`. The plotting branch also generates figures that are not part of the saved dataset.

ii.
```python
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)
motion_bins, edges = compute_quantile_bins(motion_b, n_bins=5)
...
info = {
    ...
    'quantile_edges': edges.tolist(),
    'raw_frames_overlap': int(n_overlap),
    'duration_seconds_overlap': float(n_overlap / fs),
}
...
if show_processing:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    ...
```

iii. The notes justify some of this as documentation and sanity-check support rather than decoder necessity. The plots are explicitly described as a validation aid, and the extra session metadata is included for traceability.
