# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers sorted `jm*` subject directories and their sorted session directories, retaining sessions containing both `suite2p/plane0` and `move_deve`. For every retained session it loads fluorescence, neuropil fluorescence, cell labels, Suite2p options, motion energy, timestamps, and interframe intervals. Full mode processes all discovered sessions; sample mode keeps the first two.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        if plane.exists() and move.exists():
            sessions.append({...})

F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
iscell = np.load(plane / 'iscell.npy')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
```

iii. The notes say the directory inspection found six `jm*` mice and 41 Suite2p/motion sessions. Loading these streams was justified by the paper's fluorescence-based decoding and the need to repair behavior/neural length mismatches.

## 1-b. How are the data split into subjects?

i. A subject is the name of a `jm*` directory. Unique names are sorted, mapped to integer indices, and each accepted session receives its subject index.

ii.
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
data['subject_idx'].append(subject_to_idx[sess['subject']])
```

iii. The notes identify the six directory names as the six mice reported by the paper.

## 1-c. How are the data split into sessions?

i. Every qualifying subdirectory of a mouse directory is one session, ordered lexicographically. Sessions producing fewer than two 60-second trials are omitted from the output.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    ...
if len(neural_trials) < 2:
    continue
data['neural'].append(neural_trials)
```

iii. The notes describe each dated folder as a daily continuous recording and report that all 41 discovered sessions survived in the full result.

## 1-d. How are the data split into trials?

i. Continuous sessions are split into consecutive, non-overlapping 60-second windows after 10-frame binning. At 30 Hz this is 180 bins per trial; incomplete tails are discarded.

ii.
```python
trial_len_bins = int(round(60.0 / (bin_size / float(ops.get('fs', 30.0)))))
n_trials = n_time // trial_len_bins
keep = n_trials * trial_len_bins
neural_trials = [neural[:, i * trial_len_bins:(i + 1) * trial_len_bins] for i in range(n_trials)]
```

iii. There are no native trials, and the task explicitly requires 60-second trials. The notes verify 180 binned samples per trial and 1,090 trials overall.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level signal quality filter. A whole session is excluded only if it yields fewer than two complete trials, and partial trailing windows are dropped.

ii.
```python
if n_trials < 2:
    return [], [], []
...
if len(neural_trials) < 2:
    continue
```

iii. The source has no native trial QC. The two-trial minimum comes directly from the target-format validation requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy`, with `ops.npy` supplying processing parameters. `iscell.npy` is loaded but does not contribute to the result.

ii.
```python
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
neural = compute_dff(F, Fneu,
    neuropil_coeff=float(ops.get('neucoeff', 0.7)),
    baseline_percentile=float(ops.get('prctile_baseline', 8.0)))
```

iii. The agent chose fluorescence rather than `spks.npy` because its reading of the methods said downstream analyses used baseline-corrected fluorescence/dF/F.

## 2-b. How is the `neural` data processed?

i. The agent subtracts neuropil, computes one 8th-percentile baseline per neuron over the entire session, replaces near-zero baselines, divides the baseline-subtracted trace by the absolute baseline, casts to float32, then averages non-overlapping groups of 10 frames.

ii.
```python
Fc = F - neuropil_coeff * Fneu
baseline = np.percentile(Fc, baseline_percentile, axis=1, keepdims=True).astype(np.float32)
baseline = np.where(np.abs(baseline) < 1e-3, 1e-3, baseline)
dff = (Fc - baseline) / np.abs(baseline)
neural_b = bin_time_series(neural, 10, axis=1, reducer='mean')
```

iii. The notes call this an “approximate Suite2p-style baseline-corrected fluorescence signal,” motivated by the absence of an explicit dF/F file and the paper's use of dF/F. They also cite the paper's 10-frame averaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed; every row of the provided tracked fluorescence arrays is retained. Although `iscell.npy` is loaded, it is unused.

ii.
```python
iscell = np.load(plane / 'iscell.npy')
...
# No indexing of F/Fneu by iscell
```

iii. The notes report that the supplied tracked ROIs already contain longitudinally matched cells and that all inspected ROIs pass the paper's Suite2p probability threshold of 0.5, so another filter would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Trials are contiguous windows from session start, and metadata calls the alignment event `session start`, with offsets 0 and 60 seconds.

ii.
```python
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The notes explain that recordings are continuous and trials are artificial, so session start/window boundaries are the only relevant alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged into each non-overlapping bin, giving 3 Hz or 333.33 ms. Short tails not filling a bin are discarded.

ii.
```python
bin_size = 10
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
...
'time_bin_size': 1000.0 * (10.0 / 30.0),
```

iii. This is justified by the methods statement that neural and behavior traces were denoised using averages of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is derived primarily from `tstamps.npy`, interpreted as day-based timestamps and converted to seconds relative to the first timestamp. `ops['fs']` and frame indices provide validation and fallback.

ii.
```python
dt_sec = float(np.median(dt) * 86400.0)
if 0.5 / fs < dt_sec < 1.5 / fs:
    t = (tstamps[:n_frames] - tstamps[0]) * 86400.0
    return np.asarray(t, dtype=np.float32)
return np.arange(n_frames, dtype=np.float32) / fs
```

iii. The notes infer day units from timestamp spans and differences, cross-check them against the 30-Hz rate, and use frame-rate time when the timestamp test fails.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Timestamps are zeroed at session start, multiplied by 86,400, validated against frame rate, then averaged in the same 10-frame bins as neural data. Thus values represent bin-mean (roughly center) times.

ii.
```python
t = (tstamps[:n_frames] - tstamps[0]) * 86400.0
time_b = bin_time_series(neural_times_sec[None, :], 10, axis=1, reducer='mean')[0]
inp = time_b[None, :].astype(np.float32)
```

iii. The agent wanted physically elapsed seconds and exact common binning; its sanity check reports exact agreement with its reconstruction.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. A time is assigned to every neural frame, then the time array and neural array are truncated/binned identically and sliced with identical trial boundaries.

ii.
```python
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
...
inp = inp[:, :keep]
```

iii. Common binning and slicing were intended to guarantee one input time for every neural sample.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output derives from `move_deve/motion_energy_glob.npy`; `tstamps.npy` is used for alignment. `interframe_int.npy` is loaded but not used by the algorithm.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
motion_aligned = align_motion_to_neural(motion, tstamps, neural_times_sec, ops)
```

iii. The notes identify motion energy as the requested behavioral variable and timestamps as the basis for repairing missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Nonfinite timestamp/value pairs are removed, motion is interpolated onto neural-frame times (or length-normalized as fallback), averaged over 10 frames, and discretized using five session-specific quantile bins.

ii.
```python
good = np.isfinite(mt) & np.isfinite(motion)
return np.interp(neural_times_sec, mt.astype(np.float32), mv.astype(np.float32)).astype(np.float32)
...
motion_b = bin_time_series(motion_aligned[None, :], 10, axis=1, reducer='mean')[0]
motion_disc, edges = discretize_into_quantile_bins(motion_b, n_bins=5)
```

iii. The agent cites the data warning permitting interpolation, the paper's 10-frame denoising, and the task's requirement for per-session five-percentile categorical output.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Quantiles from 0 to 1 in steps of 0.2 are computed separately over each session's binned motion. Exterior edges become infinite, repeated internal edges are nudged upward, and `np.digitize` produces integer labels 0–4.

ii.
```python
edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
edges[0] = -np.inf
edges[-1] = np.inf
if edges[i] <= edges[i - 1]:
    edges[i] = np.nextafter(edges[i - 1], np.inf)
labels = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
```

iii. This directly implements five equal-percentile bins selected per session and guards against tied quantile boundaries.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion timestamps are converted to seconds relative to their first value, then motion is linearly interpolated at neural frame times. If usable timestamps are unavailable, equal-length data is returned directly or normalized-index interpolation is used. Neural and aligned motion then receive matching 10-frame bins and trial slices.

ii.
```python
mt = (tstamps[:n_motion] - tstamps[0]) * 86400.0
return np.interp(neural_times_sec, mt.astype(np.float32), mv.astype(np.float32)).astype(np.float32)
...
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
```

iii. The notes prefer timestamp interpolation to dropping neural samples and report an exact converted-output reconstruction using this alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite motion/timestamp pairs are removed and missing camera samples are filled by interpolation onto neural time. Invalid/nonmonotonic timestamp cases fall back to direct or normalized-index resampling. Near-zero fluorescence baselines are clamped, incomplete bins/trials are truncated, and sessions with fewer than two trials are skipped.

ii.
```python
good = np.isfinite(mt) & np.isfinite(motion)
...
x_old = np.linspace(0, 1, num=len(motion), dtype=np.float32)
return np.interp(x_new, x_old, motion.astype(np.float32)).astype(np.float32)
...
baseline = np.where(np.abs(baseline) < 1e-3, 1e-3, baseline)
```

iii. The notes document 1–148 missing behavior frames in nine sessions and choose interpolation because the dataset README permits it and retaining all neural data is desirable.

## 6-a. What are the most time-consuming steps of the code?

i. The code times whole-session processing but does not profile individual stages. Based on its operations and notes, loading full fluorescence arrays and computing the per-neuron percentile/binned neural matrices dominate; optional plotting adds work for at most two sessions.

ii.
```python
t0 = time.time()
F, Fneu, iscell, ops, motion, tstamps, interframe = load_session_arrays(sess)
...
'process_time_sec': time.time() - t0,
```

iii. The notes explicitly identify full-session array loading as the main concern and report about 0.17 seconds per sample session, but provide no stage-level benchmark.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Numerical transforms are already vectorized. The remaining trial list-comprehension slices and serial session loop could be expressed as reshapes/batched or parallel work, though trial lists are ultimately required by the output format.

ii.
```python
neural_trials = [neural[:, i * trial_len_bins:(i + 1) * trial_len_bins] for i in range(n_trials)]
for i, sess in enumerate(sessions):
    neural_trials, input_trials, output_trials, info = process_session(sess, ...)
```

iii. The notes claim vectorization of neuropil correction, percentile estimation, binning, interpolation, and trial segmentation; they do not propose further loop optimization.

## 6-c. What processing does the code repeat multiple times?

i. Every session independently loads arrays, builds time coordinates, computes neural preprocessing, alignment, binning, quantiles, and trial lists. `process_session` receives a `show_processing` argument that is ignored in favor of the per-session dictionary flag. Arrays are also repeatedly cast while loading, computing, and slicing trials.

ii.
```python
for i, sess in enumerate(sessions):
    neural_trials, input_trials, output_trials, info = process_session(sess, ...)
...
def process_session(sess, show_processing=False):
    ...
    if show_processing:
```

iii. The agent does not document repeated processing as a concern; session-local repetition is necessary because parameters, lengths, and quantiles are session-specific.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `iscell.npy` and `interframe_int.npy` are loaded but never used. The `show_processing` parameter to `build_dataset` is unused, and optional figures are diagnostic only. Motion quantile edges and timing details are retained only in metadata, not consumed by decoder tensors.

ii.
```python
iscell = np.load(plane / 'iscell.npy')
interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
...
def build_dataset(sessions, show_processing=False):
```

iii. The notes do not acknowledge the two unused loads; they describe plots and metadata as sanity checks, so those are deliberate validation artifacts rather than decoder features.
