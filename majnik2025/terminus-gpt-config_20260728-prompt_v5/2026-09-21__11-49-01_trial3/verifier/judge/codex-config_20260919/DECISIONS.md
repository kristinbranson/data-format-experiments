# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates sorted top-level subject directories and sorted child session directories under `/app/data`. It retains only session directories containing both `suite2p/plane0` and `move_deve`. For every retained session it loads Suite2p metadata, fluorescence, neuropil fluorescence, cell labels, motion energy, and behavior timestamps. Full mode processes every discovered session; sample mode keeps the first two.

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

ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
F = np.load(s2p / 'F.npy').astype(np.float32)
Fneu = np.load(s2p / 'Fneu.npy').astype(np.float32)
iscell = np.load(s2p / 'iscell.npy')
motion = np.load(mov / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
```

iii. The notes say the observed hierarchy is subject/session, with each session containing Suite2p and motion files. The agent regarded required-subdirectory checks and sorted traversal as a deterministic way to include valid recordings.

## 1-b. How are the data split into subjects?

i. Each top-level directory containing at least one valid discovered session is treated as a subject. Subject names are deduplicated and sorted; a dictionary maps each name to `subject_idx`.

ii.
```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_to_idx[subject])
```

iii. The notes identify top-level mouse folders as the subject IDs and state that subject bookkeeping should come from those folder names.

## 1-c. How are the data split into sessions?

i. Every valid child directory below a subject is one session. Sessions are sorted by subject and directory name and appended separately to the target lists.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    ...
    sessions.append((subj_dir.name, sess_dir.name, sess_dir))
...
for idx, (subject, session_name, sess_dir) in enumerate(sessions):
    neural_trials, input_trials, output_trials, info = process_session(sess_dir, ...)
    data['neural'].append(neural_trials)
```

iii. The notes describe each directory as one continuous daily recording and explicitly plan to treat each as one target-format session.

## 1-d. How are the data split into trials?

i. After alignment and 10-frame binning, each continuous session is divided into consecutive, non-overlapping 60-second pseudo-trials. At 30 Hz and ten frames per bin, each trial has 180 bins. An incomplete tail is discarded.

ii.
```python
dt = float(bin_size_frames / fs)
trial_len = int(round(trial_seconds / dt))
n_trials = neural_b.shape[1] // trial_len
usable = n_trials * trial_len
neural_b = neural_b[:, :usable]
...
for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The task explicitly requests 60-second trials, while the source recordings have no natural trial structure. The notes therefore choose fixed consecutive windows and drop only the trailing incomplete window.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality filter. The code only discards incomplete trailing data and rejects an entire session if it would contain fewer than two complete trials.

ii.
```python
n_trials = neural_b.shape[1] // trial_len
if n_trials < 2:
    raise ValueError(f'Not enough 60 s trials in session {sess_dir.name}: {n_trials}')
usable = n_trials * trial_len
```

iii. The notes found no invalid-period or trial-level curation rule; the two-trial check enforces the downstream format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `F.npy` and `Fneu.npy`, with `iscell.npy` used to select rows. `ops.npy` supplies sampling rate and neuropil coefficient.

ii.
```python
ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
F = np.load(s2p / 'F.npy').astype(np.float32)
Fneu = np.load(s2p / 'Fneu.npy').astype(np.float32)
iscell = np.load(s2p / 'iscell.npy')
```

iii. The notes interpreted the paper's decoding signal as dF/F and chose fluorescence rather than `spks`, using available Suite2p fields to reconstruct a dF/F-like signal.

## 2-b. How is the `neural` data processed?

i. The code subtracts neuropil using `ops['neucoeff']` (default 0.7), computes a per-neuron session-wide 20th-percentile baseline, divides corrected fluorescence changes by that baseline, trims to the common stream length, and averages non-overlapping groups of 10 frames.

ii.
```python
def robust_dff(Fcorr: np.ndarray):
    baseline = np.percentile(Fcorr, 20, axis=1, keepdims=True)
    baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
    return (Fcorr - baseline) / baseline

Fcorr = F - neucoeff * Fneu
neural = robust_dff(Fcorr)
neural = neural[:, :n_overlap]
neural_b = moving_average_bin(neural, bin_size_frames).astype(np.float32)
```

iii. The agent says the methods explicitly use dF/F, so it selected a robust session-wise baseline normalization after standard neuropil correction, followed by the paper's ten-timestamp denoising.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only rows whose first `iscell` column is truthy are retained. No further neuron-quality thresholds are applied.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
F = F[cell_mask]
Fneu = Fneu[cell_mask]
```

iii. The notes state that Track2p/Suite2p loading and export code uses `iscell`, making this the clearest available neuron-curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Artificial trials are sequential windows referenced to session start; each trial slice is applied identically to neural, input, and output streams. Metadata describes session start and offsets 0 to 60 seconds.

ii.
```python
sl = slice(i * trial_len, (i + 1) * trial_len)
neural_trials.append(neural_b[:, sl].astype(np.float32))
...
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The notes explain that continuous recordings lack stimulus-aligned trials, so session-relative sequential segmentation is appropriate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural, motion, and constructed time arrays are averaged in non-overlapping ten-frame bins. With the per-session Suite2p sampling rate (30 Hz in these data), resolution is 1/3 second or 333.33 ms.

ii.
```python
neural_b = moving_average_bin(neural, bin_size_frames)
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0]
dt = float(bin_size_frames / fs)
...
data['metadata']['time_bin_size'] = float(np.median(dt_values) * 1000.0)
```

iii. This directly follows the methods statement that dF/F and behavior were denoised by averaging ten consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is constructed from imaging sample indices and `ops['fs']`; it is not taken from `tstamps.npy`, although that file is loaded and participates in determining the common length.

ii.
```python
fs = float(ops.get('fs', 30.0))
raw_time = np.arange(n_overlap, dtype=np.float32) / fs
```

iii. The first attempt used behavior timestamps but produced no 60-second trials because their units were unsuitable. The agent then used imaging frame rate for reliable elapsed seconds.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A frame-index time vector is divided by sampling frequency, then averaged over the same ten-frame groups as the other streams. Thus each saved value is the mean/center time of a bin rather than its left edge.

ii.
```python
raw_time = np.arange(n_overlap, dtype=np.float32) / fs
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
```

iii. The notes justify frame-derived time because the imaging frequency is known and behavior timestamp units were unreliable.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time uses the same common-length trimming, ten-frame binning, usable-tail truncation, and trial slices as neural data.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
...
time_b = time_b[:usable]
...
input_trials.append(time_b[sl][None, :].astype(np.float32))
```

iii. The mapping plan requires all streams to share one binned time base and spot-checks that time advances by the expected binned interval.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output values come from `move_deve/motion_energy_glob.npy`. `tstamps.npy` is loaded to constrain overlap, but its values are not used to resample or repair the motion trace.

ii.
```python
motion = np.load(mov / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
...
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
```

iii. The notes identify motion energy as the paper's precomputed frame-difference scalar and initially propose using timestamps to reconcile its small length mismatch.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion is truncated to the shared prefix, averaged in ten-sample non-overlapping bins, and converted to integer session-wise quintiles. It is finally truncated to complete trials.

ii.
```python
motion = motion[:n_overlap]
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)
motion_bins, edges = compute_quantile_bins(motion_b, n_bins=5)
motion_bins = motion_bins[:usable]
```

iii. The agent cites the paper's ten-timestamp averaging and the task's requirement for five equal-percentile bins selected per session. It chose overlap trimming to handle small length discrepancies.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Six boundaries (0, 20, 40, 60, 80, 100 percentiles) are computed separately for each session after averaging. Interior boundaries are passed to `np.digitize`, yielding labels 0–4. Non-increasing duplicate edges are nudged upward by `1e-9`.

ii.
```python
edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
for i in range(1, len(edges)):
    if edges[i] <= edges[i - 1]:
        edges[i] = edges[i - 1] + 1e-9
bins = np.digitize(values, edges[1:-1], right=False)
```

iii. Session-wise equal-percentile categories satisfy the decoder specification and reduce cross-session scale confounds; validation showed nearly balanced categories.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent assumes sample-by-sample correspondence at the beginning of the arrays, trims every stream to the shortest length, then applies identical binning, tail truncation, and trial slices. It does not detect or insert missing camera frames.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
neural = neural[:, :n_overlap]
motion = motion[:n_overlap]
...
output_trials.append(motion_bins[sl][None, :].astype(np.int64))
```

iii. The notes call this alignment to the overlapping valid range and chose it to resolve the observed two-sample mismatch between imaging and behavior arrays.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Length mismatches are handled by silently trimming all streams to their common prefix. Incomplete ten-frame bins and incomplete 60-second tails are dropped. A near-zero dF/F baseline is replaced with `1e-6`, repeated quantile edges are nudged, and sessions with fewer than two trials raise an error.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
...
trimmed = x[..., : n * bin_size]
...
baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
...
if n_trials < 2:
    raise ValueError(...)
```

iii. The agent viewed overlap trimming and incomplete-tail removal as a practical response to small off-by-one mismatches. Defensive baseline and edge adjustments avoid numerical failures.

## 6-a. What are the most time-consuming steps of the code?

i. In this implementation the main costs are loading full `.npy` arrays, computing per-neuron percentiles over full sessions, ten-frame reductions, and optional Matplotlib plots. The notes report conversion around 0.46 seconds per session in the sample and identify full-array loading as the main remaining inefficiency.

ii.
```python
F = np.load(s2p / 'F.npy').astype(np.float32)
...
baseline = np.percentile(Fcorr, 20, axis=1, keepdims=True)
...
if show_processing:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
```

iii. The notes specifically say full-array loading is acceptable at this dataset size and restrict diagnostic plots to the first two sessions to keep plotting from dominating runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Time-bin averaging is already vectorized with reshape/mean. Trial construction still loops over trials to create the required list-of-arrays structure, session discovery loops over directories, and duplicate quantile-edge repair loops over only six edges. Trial arrays could first be reshaped in bulk before conversion to lists, but the likely gain is small.

ii.
```python
return trimmed.reshape(new_shape).mean(axis=-1)
...
for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The notes highlight vectorized bin averaging as an implemented speedup and do not identify a consequential remaining Python loop.

## 6-c. What processing does the code repeat multiple times?

i. The same load, filtering, normalization, overlap, binning, quantile, and trial-splitting pipeline runs independently for every session. Within a session, `.astype(np.float32)` is repeated at loading, after binning, and again while making each trial.

ii.
```python
for idx, (subject, session_name, sess_dir) in enumerate(sessions):
    neural_trials, input_trials, output_trials, info = process_session(sess_dir, ...)
...
neural_b = moving_average_bin(...).astype(np.float32)
...
neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The notes do not flag repeated processing beyond the necessary per-session pipeline; the extra casts are defensive and generally no-ops after the first conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `tstamps.npy` is fully loaded but only its length is used. `motion_b` is truncated after categorical labels have already been made and is otherwise retained only for optional plotting. Diagnostic figures, raw-value summaries, quantile edges, and detailed session metadata are not decoder inputs. The imported `math` and `os` modules are unused.

ii.
```python
tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
...
motion_b = motion_b[:usable]
...
if show_processing:
    axes[2].plot(time_b[:500], motion_b[:500])
```

iii. The agent justifies optional plots and metadata as validation aids. It does not explicitly acknowledge the unused timestamp values or imports.
