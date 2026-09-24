# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively discovers sorted subject and session directories under `data/`, but admits only sessions containing a fixed set of Suite2p and movement files. For each admitted session it loads `F`, `Fneu`, `spks`, `iscell`, `ops`, global motion energy, and video timestamps. Full mode uses every discovered valid session; sample mode takes the first two.

ii.
```python
for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        suite = sess_dir / 'suite2p' / 'plane0'
        move = sess_dir / 'move_deve'
        req = [suite/'F.npy', suite/'Fneu.npy', suite/'iscell.npy', suite/'ops.npy', suite/'spks.npy', move/'motion_energy_glob.npy', move/'tstamps.npy']
        if all(p.exists() for p in req):
            sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
iscell = np.load(suite / 'iscell.npy')
ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
spks = np.load(suite / 'spks.npy').astype(np.float32)
```

iii. The notes identify the hierarchy as `data/<subject>/<session>/...`, with Suite2p data in `suite2p/plane0` and behavior in `move_deve`. They report 6 subjects and 41 valid sessions. Requiring all expected files was intended to avoid incomplete sessions.

## 1-b. How are the data split into subjects?

i. The first-level directory name is the subject ID. After session discovery, unique subject IDs are sorted and mapped to integer indices; each retained session receives the corresponding index.

ii.
```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[subj])
```

iii. The AI states that directories such as `jm031` represent mice and reports six subjects. Alphabetical sorting provides stable ordering.

## 1-c. How are the data split into sessions?

i. Each second-level directory is treated as a daily recording/session, provided all required neural and movement files exist. Session order is subject-directory order followed by session-directory order.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    ...
    if all(p.exists() for p in req):
        sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. The notes describe the recordings as continuous daily sessions and report 41 valid sessions. The file-existence test was used as a validity check.

## 1-d. How are the data split into trials?

i. The AI creates non-overlapping consecutive 120-second pseudo-trials after 10-frame binning. Only complete windows are retained; the trailing partial window is silently dropped.

ii.
```python
WINDOW_SECONDS = 120.0
bins_per_window = max(1, int(round(window_seconds * fs_binned)))
n_windows = neural_b.shape[1] // bins_per_window
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The AI followed the paper's statement that reference decoding cross-validation used consecutive two-minute blocks. It reasoned that fixed pseudo-trials were necessary because the recordings have no native trials. This overlooked the task's explicit instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial signal-quality filter. Incomplete trailing windows are discarded, and an entire session is skipped if it produces fewer than two complete 120-second windows.

ii.
```python
n_windows = neural_b.shape[1] // bins_per_window
...
if len(neural_trials) < 2:
    continue
```

iii. The notes say there are no native trials or trial-quality markers. The two-window rule enforces the decoder format requirement that every session contain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived from Suite2p's deconvolved activity in `spks.npy`, selected using `iscell.npy`. Although `F.npy`, `Fneu.npy`, and `ops.npy` are loaded, they do not contribute to the final neural values.

ii.
```python
spks = np.load(suite / 'spks.npy').astype(np.float32)
...
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The AI initially planned fluorescence/dF/F because the methods specify it, but its approximate dF/F conversion gave below-chance decoder accuracy. It switched to native Suite2p `spks.npy` after sample validation accuracy improved from about 0.18 to 0.32, prioritizing benchmark performance over matching the stated reference processing.

## 2-b. How is the `neural` data processed?

i. Cells are selected, the activity is truncated to a common duration with behavior/timestamps, and every 10 consecutive spike samples are averaged. The result is divided into complete 120-second windows and cast to `float32`. The `compute_df_f` function exists but is never called.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
...
return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)
```

iii. Ten-frame averaging was justified by the paper's decoding method. The choice of spikes was justified by improved decoder validation, even though the AI acknowledged that the methods call for baseline-corrected fluorescence/dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are retained only when the first `iscell` column is greater than 0.5.

ii.
```python
CELL_THRESHOLD = 0.5
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The AI cites the methods statement that ROIs above Suite2p's default 0.5 cell-probability threshold were considered cells. Its notes report 20,445 retained cells and say spot checks matched this rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. The continuous binned signal is divided from session time zero into consecutive 120-second windows. Metadata describes alignment to session start, with offsets 0 to 120 seconds.

ii.
```python
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
```
```python
'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
'off_start': 0.0,
'off_end': WINDOW_SECONDS,
```

iii. The AI reasoned that continuous recordings have no natural stimulus-alignment event, so session start is the only meaningful origin and fixed windows supply decoder-compatible trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten raw frames are averaged per output time point. With the session sampling rate taken from `ops['fs']` (normally 30 Hz), the binned rate is 3 Hz and the bin duration is about 333.33 ms.

ii.
```python
BIN_FRAMES = 10
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
neural_b = bin_array_2d(neural, BIN_FRAMES)
...
'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values]))
```

iii. The methods explicitly say neural and behavioral traces were denoised by averaging 10 consecutive timestamps for decoding, so the AI applied the same binning to both streams.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Final time input is derived from the binned sample index and Suite2p's `ops['fs']`, not from the loaded `tstamps.npy` values.

ii.
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. The AI first tried binned raw timestamps, found their scale was not seconds, and changed to frame-rate-derived elapsed seconds. It considered that robust and consistent with the required session-start time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. An increasing zero-based sequence is divided by the post-binning sampling rate, cast to `float32`, sliced into the same windows as neural data, and expanded to shape `(1, time)`.

ii.
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
...
def make_time_input(t_binned):
    return t_binned[None, :].astype(np.float32)
```

iii. The AI validated that later windows continue at 120 seconds, 240 seconds, and so on rather than resetting, matching time elapsed from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural data are constructed at the same binned rate and sliced with the exact same window boundaries.

ii.
```python
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
input_trials.append(make_time_input(time_b[sl]))
```

iii. The AI spot-checked that the first trial spans approximately 0–119.67 seconds and that later trials preserve absolute time from session start.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from `move_deve/motion_energy_glob.npy`. `tstamps.npy` is loaded for alignment/trimming but its binned values are ultimately unused.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
```

iii. The notes identify global motion energy as the paper's scalar movement proxy from squared pixel-wise video-frame differences and therefore use the provided precomputed signal.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion is truncated to the shortest shared neural/motion/timestamp length, averaged over non-overlapping groups of 10 frames, sliced into complete 120-second windows, then discretized using percentile edges computed over all retained trials from all sessions.

ii.
```python
n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
return neural[:, :n], motion[:n], tstamps[:n]
```
```python
motion_b = bin_array_1d(motion, BIN_FRAMES)
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
```

iii. The AI justified trimming as alignment to the shared valid duration and global percentiles as producing globally balanced classes. It noted that this causes some sessions to contain only upper categories.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five classes are made using the 0, 20, 40, 60, 80, and 100 percent quantiles of all retained motion values pooled across every session. Interior four edges are passed to `np.digitize`, producing integer classes 0–4.

ii.
```python
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
edges[0] = -np.inf
edges[-1] = np.inf
...
b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. The AI interpreted “five equal-percentile bins” globally and used exact global balance as a sanity check. The task, however, explicitly required bins selected per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Neural, motion, and timestamp arrays are truncated at the end to their minimum common raw length. Neural and motion are then separately averaged in the same 10-frame groups and sliced with identical trial windows.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
...
motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The notes say microscope acquisition triggered video acquisition, so frame indices should align. For mismatches, the AI chose minimum-length trimming rather than reconstructing dropped camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions lacking any required file are omitted. Within a session, unequal neural, motion, and timestamp lengths are handled by truncating every stream to the minimum length. Partial 10-frame bins and incomplete final 120-second windows are discarded. Sessions with fewer than two complete windows are skipped.

ii.
```python
if all(p.exists() for p in req):
    sessions.append(...)
```
```python
n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
return neural[:, :n], motion[:n], tstamps[:n]
```

iii. The AI observed exact matches for most sessions and small or larger mismatches in others. It described truncation as retaining only the shared valid duration. It did not use `interframe_int.npy` to locate and interpolate dropped video frames.

## 6-a. What are the most time-consuming steps of the code?

i. In the implemented pipeline the dominant operations are loading multiple full-session arrays and averaging/serializing large arrays across 41 sessions. Optional diagnostic plotting also adds work for two sessions. Unlike the reference, the active path does not run Suite2p baseline correction.

ii.
```python
F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
...
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
```

iii. The notes contain empty runtime tables and no measured bottleneck analysis. They report per-session timings and a roughly 409 MB output, but do not explicitly justify which active step is slowest.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Session traversal is inherently file-oriented, but trial construction and output categorization use Python loops. Windowing could be reshaped into a trial axis, and category assignment could operate on concatenated arrays with split indices rather than nested session/trial loops. Directory discovery also uses loops, though vectorization would not materially help filesystem traversal.

ii.
```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```
```python
for sess in all_motion_trials:
    sess_out = []
    for m in sess:
        b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. The AI's Step 6 efficiency fields were left as placeholders, so it supplied no explicit vectorization justification. The existing loops favor clarity and nested target-format construction.

## 6-c. What processing does the code repeat multiple times?

i. Every session independently repeats file loading, cell masking, common-length trimming, 10-frame averaging, time-vector creation, and window slicing. Motion data are first stored continuously as trial arrays and then traversed again for pooled quantile computation and class conversion.

ii.
```python
for subj, sess_name, sess_dir in sessions:
    F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
    ...
    neural_trials, input_trials, motion_trials, bins_per_window = segment_session(...)
```
```python
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
...
for sess in all_motion_trials:
```

iii. The AI did not identify repeated processing in its notes. Most per-session repetition is necessary because files and sampling metadata are session-specific; the second motion traversal follows from needing global thresholds before labels can be assigned.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The active conversion loads `F.npy` and `Fneu.npy` but uses only `spks.npy`; it bins timestamps into `_t_b_raw` and immediately discards the result; and it keeps `compute_df_f` as dead code. Raw timestamps are otherwise used only to shorten all streams. With `--show-processing`, plots are generated but do not affect conversion.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
...
neural = spks[keep].astype(np.float32)
```
```python
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. The unused fluorescence loading is residue from the original dF/F plan, and `_t_b_raw` is residue from the abandoned timestamp-derived time input. The notes document the change in strategy but do not call out or remove these inefficiencies.
