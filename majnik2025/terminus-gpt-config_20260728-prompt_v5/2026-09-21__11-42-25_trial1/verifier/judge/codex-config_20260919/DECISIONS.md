# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers sorted `jm*` subject directories and sorted session directories, retaining a session only when both `F.npy` and `motion_energy_glob.npy` exist. Each retained session is then loaded independently. In addition to fluorescence and motion energy, it loads `Fneu.npy`, `iscell.npy`, `ops.npy`, and `tstamps.npy`.

ii.
```python
def discover_sessions(data_root: Path):
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            if (sess_dir / 'suite2p/plane0/F.npy').exists() and (sess_dir / 'move_deve/motion_energy_glob.npy').exists():
                sessions.append(sess_dir)
    return sessions
```
```python
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
```

iii. The notes say the raw hierarchy is subject, then dated session, with Suite2p neural files and `move_deve` behavior files. They report six mice and 41 sessions and justify fluorescence loading because the paper decodes dF/F rather than `spks.npy`.

## 1-b. How are the data split into subjects?

i. A subject is a top-level directory whose name starts with `jm`. Final subject names are the sorted unique parent-directory names of successfully processed sessions, and `subject_idx` maps every session result back to that list.

ii.
```python
subjects = sorted({r.subject for r in results})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.asarray([subject_to_idx[r.subject] for r in results], dtype=np.int64),
```

iii. The agent states that folders such as `jm031` are individual mice and confirms that the result contains six subjects.

## 1-c. How are the data split into sessions?

i. Every sorted child directory under a subject is treated as one daily session if the principal neural and motion files exist. One `SessionResult` becomes one target-format session.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    if (sess_dir / 'suite2p/plane0/F.npy').exists() and (sess_dir / 'move_deve/motion_energy_glob.npy').exists():
        sessions.append(sess_dir)
```
```python
'neural': [r.neural_trials for r in results],
```

iii. The notes identify dated folders as sessions and preserve one session per recording day. Requiring expected files prevents malformed/non-session folders from being processed.

## 1-d. How are the data split into trials?

i. After 10-frame averaging, each continuous session is split into consecutive, non-overlapping 60-second windows. The number of bins is calculated using the inferred binned sampling rate; any incomplete tail is discarded.

ii.
```python
bins_per_trial = int(round(trial_sec * fs_binned))
n_trials = neural.shape[1] // bins_per_trial
usable = n_trials * bins_per_trial
neural = neural[:, :usable]
neural_trials = [neural[:, i*bins_per_trial:(i+1)*bins_per_trial] for i in range(n_trials)]
```

iii. The agent notes that the recordings are continuous and have no native trials, so the task-required 60-second trials must be constructed. Its checks found 180 bins per trial and 19–30 complete trials per session.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial quality filter. Only complete 60-second windows are retained; incomplete trailing data are dropped. Dataset validation also requires at least two trials per session, but it asserts rather than filters.

ii.
```python
n_trials = neural.shape[1] // bins_per_trial
usable = n_trials * bins_per_trial
neural = neural[:, :usable]
```
```python
assert len(data['neural'][s]) >= 2
```

iii. The notes describe incomplete trailing bins as an edge case and say they are consistently removed from neural, input, and output. They do not identify any bad-trial flag in the source data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from Suite2p `F.npy` and `Fneu.npy`; `iscell.npy` determines which rows are retained, and `ops.npy` supplies sampling and baseline parameters.

ii.
```python
iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
keep = iscell[:, 0] > iscell_thr
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```

iii. The agent chose fluorescence-derived dF/F rather than `spks.npy` because the paper explicitly describes decoding dF/F traces. It interpreted the Track2p/Suite2p context as supporting an `iscell > 0.5` filter.

## 2-b. How is the `neural` data processed?

i. The code subtracts 0.7 times neuropil fluorescence, shifts each neuron upward if needed, estimates a piecewise-constant 8th-percentile baseline in non-overlapping 60-second blocks, floors the baseline, computes `(F-baseline)/baseline`, and averages non-overlapping groups of 10 frames.

ii.
```python
fcorr = F - neuropil_coeff * Fneu
min_per_neuron = fcorr.min(axis=1, keepdims=True)
shift = np.maximum(0.0, 1.0 - min_per_neuron)
fc = fcorr + shift
```
```python
for start in range(0, n, win):
    end = min(n, start + win)
    b = np.percentile(fc[:, start:end], prctile_baseline, axis=1, keepdims=True)
    baseline[:, start:end] = b.astype(np.float32)
dff = (fc - baseline) / baseline
neural_b = moving_average_nonoverlap(dff, bin_size)
```

iii. The agent says a naive dF/F normalization produced unstable values and near-chance sample decoding. It introduced this “stable running low-percentile baseline informed by Suite2p metadata,” plus positivity and baseline floors, to improve numerical stability and decoder accuracy.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with Suite2p `iscell[:, 0] > 0.5` are retained. There is no later neuron-level filtering.

ii.
```python
iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
keep = iscell[:, 0] > iscell_thr
F = F[keep, :n_common]
Fneu = Fneu[keep, :n_common]
```

iii. The agent believed the README’s `iscell_thr = 0.5` example and the paper’s focus on cells justified excluding non-cell ROIs. It reported 20,445 included ROI-session observations after this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Neural data begin at the session start and are cut into consecutive 60-second windows; corresponding input and output use identical slices. Metadata describes session start and continuous windows.

ii.
```python
neural_trials = [neural[:, i*bins_per_trial:(i+1)*bins_per_trial] for i in range(n_trials)]
```
```python
'temporal_alignment_event': 'session start; continuous session split into contiguous 60-second windows',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The agent explains that native data are continuous rather than event-locked, making session start the natural origin and contiguous windows the task-required unit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Raw 30 Hz values are averaged in non-overlapping groups of 10, giving approximately 3 Hz or 333.33 ms bins. Both neural and motion streams are rebinned before trial splitting and motion discretization.

ii.
```python
neural_b = moving_average_nonoverlap(dff, bin_size)
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
```
```python
'time_bin_size': float(results[0].metadata['time_bin_size_sec'] * 1000.0)
```

iii. This matches the methods statement that dF/F and behavior were denoised by averaging 10 consecutive timestamps. The agent’s full validation found every trial had 180 bins.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The code initially derives time from `move_deve/tstamps.npy`. When the timestamp scale appears implausible, as it does for these data, it falls back to the binned frame index and `ops['fs']`.

ii.
```python
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
time_b, sec_per_bin = compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback=fs)
```
```python
t = np.arange(n, dtype=np.float64) * expected
```

iii. The agent first encountered zero trials due to timestamp units, then added plausibility checks and a frame-rate fallback. It describes the final variable as elapsed seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Timestamps are grouped in 10s and averaged, shifted so the first binned value is zero, conditionally converted from milliseconds, and checked against the expected `10/fs` interval. Implausible timing is replaced by `arange(n) * 10/fs`.

ii.
```python
trimmed = np.asarray(tstamps[: n * bin_size], dtype=np.float64).reshape(n, bin_size)
t = trimmed.mean(axis=1)
t = t - t[0]
```
```python
if (not np.isfinite(sec_per_bin)) or sec_per_bin <= 0 or sec_per_bin > 10 * expected or sec_per_bin < 0.1 * expected:
    t = np.arange(n, dtype=np.float64) * expected
```

iii. The stated goal is robust unit handling: the source timestamps did not behave as seconds, so the agent preferred a deterministic frame-rate fallback over producing invalid trial lengths.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Neural, timestamps, and motion are first truncated to the same raw length. The resulting binned time array is limited to the neural bin count and split with exactly the same trial boundaries.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
time_b = time_b[:, :n_bins]
```
```python
input_trials = [inp[:, i*bins_per_trial:(i+1)*bins_per_trial] for i in range(n_trials)]
```

iii. The notes emphasize frame-level alignment and validate that each neural/input/output trial has the same temporal length.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output values come from `move_deve/motion_energy_glob.npy`. `tstamps.npy` participates only in selecting a common length and timing, not in calculating motion magnitude. The code does not load `interframe_int.npy`.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
```

iii. The agent identifies `motion_energy_glob.npy` as the video-derived behavioral variable and treats its sample order as frame-synchronous with imaging.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion is truncated to the shortest stream, averaged over non-overlapping 10-frame groups, then categorized using five within-session equal-percentile bins.

ii.
```python
motion = motion[:n_common]
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
motion_labels, motion_edges = digitize_equal_percentile(motion_b, n_bins=5)
```

iii. The agent says both streams must receive the paper’s 10-frame smoothing and that percentile edges must be session-specific per the decoder instructions. For mismatched lengths it chose common-length truncation as a simple alignment policy.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Six quantile edges from 0% through 100% are computed separately for each session after smoothing. The four interior edges are passed to `np.digitize`, producing integer labels 0–4. Repeated edges are forced nondecreasing.

ii.
```python
edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
edges = np.maximum.accumulate(edges)
labels = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
```

iii. This directly follows the requirement for five equal-percentile bins selected per session. The notes confirm near-20% global class fractions before incomplete-trial truncation.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code assumes common frame order and truncates neural fluorescence, motion, and timestamps to their minimum raw length. Neural and motion are then separately averaged into 10-frame bins, restricted to the neural bin count, and sliced with identical trial indices. It does not insert values for dropped camera frames.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
F = F[keep, :n_common]
motion = motion[:n_common]
```
```python
output_trials = [out[:, i*bins_per_trial:(i+1)*bins_per_trial] for i in range(n_trials)]
```

iii. The notes call the arrays frame-level and “nearly” matched, and justify truncating rare mismatches to a common valid length. They do not account for the fact that an interior dropped video frame shifts all subsequent behavioral samples.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/mismatched tail data are handled by truncating every stream to the shortest raw length. Incomplete 10-frame bins and incomplete 60-second trials are dropped. Assertions check final shapes and minimum trial count. There is no interpolation of dropped motion frames.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
```
```python
n = arr.shape[-1] // bin_size
trimmed = arr[..., : n * bin_size]
```

iii. The agent describes discrepancies as rare 0–1-frame mismatches and considers common-length truncation safe. Its critical review checked that shorter sessions still generated complete, aligned target arrays.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant computation is per-session percentile-baseline construction across every neuron and frame, followed by loading large `F.npy`/`Fneu.npy` arrays. The agent’s notes explicitly flag full-array loading as the main item to monitor, although measured conversion was fast (about 0.18 seconds per session in its sample estimate).

ii.
```python
for start in range(0, n, win):
    chunk = fc[:, start:end]
    b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
```

iii. The agent attributes efficiency to vectorized NumPy for most transforms and says full-array I/O/memory may be the practical cost. It did not provide a formal profiler breakdown.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over baseline windows could be replaced by a reshape plus percentile operation when full fixed-size blocks are used. Trial construction also uses list comprehensions that could be created as a reshaped view before conversion to the required list structure. Subject/session iteration is inherently needed for separate files and session-local thresholds.

ii.
```python
for start in range(0, n, win):
    end = min(n, start + win)
    chunk = fc[:, start:end]
    b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
```
```python
neural_trials = [np.asarray(neural[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.float32) for i in range(n_trials)]
```

iii. The notes broadly claim that dF/F, smoothing, discretization, and trial splitting are vectorized, but do not acknowledge the explicit baseline-window loop. They correctly describe reshape-based temporal averaging as already vectorized.

## 6-c. What processing does the code repeat multiple times?

i. Loading, cell selection, dF/F calculation, 10-frame averaging, time construction, quantile computation, trialization, and metadata creation are repeated once per session. This repetition is necessary because files, neurons, baselines, and output thresholds are session-specific; no expensive transform is redundantly recomputed for the same session in the main conversion.

ii.
```python
for i, sess_dir in enumerate(sessions, start=1):
    res = load_session(sess_dir)
    results.append(res)
```

iii. The agent intentionally computes percentile edges separately per session and reports no cache because the full run is already quick. The later sanity check recomputes session 0 outside the conversion script, but that is validation rather than production processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full conversion, baseline and positivity-shift arrays are produced even though only their medians enter metadata; the full arrays are then discarded after each session. With `--show-processing`, plotting reads and renders sample trials solely for diagnostics. Most other computed fields are retained in either target arrays or session metadata.

ii.
```python
dff, baseline, shift = robust_df_over_f(...)
'baseline_median': float(np.median(baseline)),
'shift_median': float(np.median(shift)),
```
```python
if args.show_processing and plot_budget > 0:
    maybe_plot_session(res, outpath.parent)
```

iii. The agent treats plots as optional diagnostic output and baseline/shift summaries as traceability checks. It does not identify other intentionally discarded processing in its notes.
