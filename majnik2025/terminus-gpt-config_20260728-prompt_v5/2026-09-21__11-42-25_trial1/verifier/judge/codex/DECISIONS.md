# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories whose names start with `jm`, then scans each subject directory for session subdirectories that contain both `suite2p/plane0/F.npy` and `move_deve/motion_energy_glob.npy`. It loads each session independently. Within each session it loads `iscell.npy`, `F.npy`, `Fneu.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy`.

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
iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI justified this as matching the observed directory structure: one mouse per `jm*` folder, one recording per dated subdirectory, with Suite2p outputs and behavior files inside each session. The trajectory also shows it explicitly inspected the data tree before deciding to discover sessions this way.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the parent directory names of discovered sessions. The final `subjects` list is the sorted set of unique subject names encountered during session processing, and `subject_idx` maps each session back to that subject.

ii.
```python
return SessionResult(
    session_id=meta['session_id'],
    subject=meta['subject'],
    ...
)
```

```python
subjects = sorted({r.subject for r in results})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.asarray([subject_to_idx[r.subject] for r in results], dtype=np.int64),
```

iii. In Step 2 and Step 5 notes, the AI states that each top-level `jm*` directory corresponds to one mouse. The trajectory shows it computed dataset summaries grouped by these subject folders and carried that grouping through to the output.

## 1-c. How are the data split into sessions?

i. Each qualifying dated subdirectory inside a subject folder is treated as one session. The AI flattens all sessions into a single ordered list, then stores per-session neural/input/output lists in that order.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        if (sess_dir / 'suite2p/plane0/F.npy').exists() and (sess_dir / 'move_deve/motion_energy_glob.npy').exists():
            sessions.append(sess_dir)
```

```python
'neural': [r.neural_trials for r in results],
'input': [r.input_trials for r in results],
'output': [r.output_trials for r in results],
```

iii. The AI’s Step 2 notes say the data are organized by subject and then by dated session folder, and each dated folder contains one daily recording. The trajectory shows it relied on that folder structure rather than any separate session index file.

## 1-d. How are the data split into trials?

i. Sessions are split into contiguous non-overlapping 60-second trials after temporal binning. The number of bins per trial is computed from the binned sampling rate, then any incomplete trailing bins are discarded.

ii.
```python
def split_trials(neural, inp, out, fs_binned, trial_sec=60.0):
    bins_per_trial = int(round(trial_sec * fs_binned))
    n_trials = neural.shape[1] // bins_per_trial
    usable = n_trials * bins_per_trial
    neural = neural[:, :usable]
    inp = inp[:, :usable]
    out = out[:, :usable]
    neural_trials = [np.asarray(neural[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.float32) for i in range(n_trials)]
    input_trials = [np.asarray(inp[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.float32) for i in range(n_trials)]
    output_trials = [np.asarray(out[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.int64) for i in range(n_trials)]
    return neural_trials, input_trials, output_trials, bins_per_trial
```

iii. In Step 4 and Step 5 notes, the AI says the dataset has no natural trial structure, so trials must be derived as contiguous 60-second windows for the decoder task. The trajectory shows it initially misunderstood session duration, then corrected the notes after sample validation showed that 60-second windows produced 180 bins/trial and about 20 or 30 trials/session depending on session length.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit trial-quality filter. It only trims all streams to a common usable length, drops any incomplete trailing bins/trials, and later asserts that each session retains at least two trials.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
...
neural = neural[:, :usable]
inp = inp[:, :usable]
out = out[:, :usable]
```

```python
for s in range(len(data['neural'])):
    assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s])
    assert len(data['neural'][s]) >= 2
```

iii. In the notes the AI treats the recordings as continuous and does not describe any trial-wise QC metric. Its justifications focus on preserving alignment and meeting the decoder requirement of at least two trials per session, rather than filtering trials for noise or behavior quality.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p fluorescence traces `F.npy` and `Fneu.npy`, after filtering ROIs with `iscell.npy`. It also reads `ops.npy` to get frame rate and baseline parameters used during processing.

ii.
```python
iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
keep = iscell[:, 0] > iscell_thr

F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```

iii. The Step 4 and Step 5 notes say the paper refers to dF/F-like fluorescence traces, not `spks.npy`, so the AI chose to start from fluorescence rather than deconvolved spikes. It also argued that Track2p/Suite2p conventions suggest using the `iscell` mask to keep only putative cells.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil correction (`F - 0.7 * Fneu`), then runs its own `robust_df_over_f` routine. That routine shifts traces upward if needed, estimates a per-neuron low-percentile baseline in 60-second windows, floors the baseline to avoid tiny denominators, computes dF/F, and finally averages the result in non-overlapping 10-frame bins.

ii.
```python
fcorr = F - neuropil_coeff * Fneu
dff, baseline, shift = robust_df_over_f(
    fcorr,
    fs=fs,
    win_baseline_sec=float(ops.get('win_baseline', 60.0)),
    prctile_baseline=float(ops.get('prctile_baseline', 8.0)),
)

neural_b = moving_average_nonoverlap(dff, bin_size)
```

```python
def robust_df_over_f(fcorr, fs, win_baseline_sec=60.0, prctile_baseline=8.0, eps=1e-3):
    min_per_neuron = fcorr.min(axis=1, keepdims=True)
    shift = np.maximum(0.0, 1.0 - min_per_neuron)
    fc = fcorr + shift
    ...
    b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
    ...
    dff = (fc - baseline) / baseline
    return dff.astype(np.float32), baseline.astype(np.float32), shift.astype(np.float32)
```

iii. The notes show the AI explicitly chose fluorescence-derived dF/F because it read the paper as describing decoding from dF/F rather than deconvolved spikes. The trajectory further shows that an earlier, simpler normalization gave near-chance decoding, so the AI replaced it with this “more stable running low-percentile baseline informed by Suite2p metadata.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters ROIs using the Suite2p cell classifier and keeps only rows where `iscell[:, 0] > 0.5`. It does not apply additional QC to traces after that.

ii.
```python
iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
keep = iscell[:, 0] > iscell_thr
...
F = F[keep, :n_common]
Fneu = Fneu[keep, :n_common]
```

iii. In Step 3 through Step 5 notes, the AI repeatedly justifies this as a standard Suite2p/Track2p curation rule and cites the Track2p README example using `iscell_thr = 0.5`. The trajectory shows this was a deliberate choice, not an accidental side effect.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to session start. It treats sessions as continuous recordings and creates trials as consecutive 60-second windows relative to the beginning of the session, not around a stimulus event.

ii.
```python
'metadata': {
    'task_description': 'Decode session-wise motion energy from barrel-cortex calcium activity in longitudinal mouse recordings.',
    'time_bin_size': float(results[0].metadata['time_bin_size_sec'] * 1000.0) if results else None,
    'temporal_alignment_event': 'session start; continuous session split into contiguous 60-second windows',
    'off_start': 0.0,
    'off_end': 60.0,
```

iii. The notes justify this by saying there is no native trial or event alignment in the source dataset. The trajectory shows the AI explicitly contrasted the paper’s continuous recordings with the decoder task’s requirement for 60-second trials and chose session start as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins neural and motion data by averaging every 10 consecutive frames into one bin. With the 30 Hz recordings this yields about 333.3 ms bins and no further temporal rebinning.

ii.
```python
def moving_average_nonoverlap(arr, bin_size):
    n = arr.shape[-1] // bin_size
    trimmed = arr[..., : n * bin_size]
    new_shape = arr.shape[:-1] + (n, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

```python
neural_b = moving_average_nonoverlap(dff, bin_size)
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
```

iii. The AI’s Step 3 to Step 5 notes repeatedly cite the paper statement that both neural traces and behavior traces were denoised by averaging over 10 consecutive timestamps before decoding. That is the main justification it gives for the rebinning choice.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI derives the time input from `tstamps.npy` when those timestamps look plausible, with fallback to the imaging frame rate from `ops.npy` if the timestamp scale looks wrong.

ii.
```python
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
...
time_b, sec_per_bin = compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback=fs)
```

```python
def compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback):
    ...
    expected = bin_size / fs_fallback
    if (not np.isfinite(sec_per_bin)) or sec_per_bin <= 0 or sec_per_bin > 10 * expected or sec_per_bin < 0.1 * expected:
        t = np.arange(n, dtype=np.float64) * expected
        sec_per_bin = expected
```

iii. The trajectory shows the AI originally tried to derive time directly from timestamps, then discovered the timestamp scale issue during sample conversion and added the fallback logic. In Step 10 notes it explicitly says a timestamp scaling issue briefly caused zero derived trials and this was resolved with plausibility checks plus frame-rate fallback.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI averages timestamps within each 10-frame bin, subtracts the first binned value so time starts at zero, converts units if they look like milliseconds, and falls back to a synthetic regular time axis based on `ops['fs']` when the inferred bin spacing looks implausible.

ii.
```python
trimmed = np.asarray(tstamps[: n * bin_size], dtype=np.float64).reshape(n, bin_size)
t = trimmed.mean(axis=1)
t = t - t[0]
sec_per_bin = float(np.median(np.diff(t))) if len(t) > 1 else np.nan

if np.isfinite(sec_per_bin) and sec_per_bin > 1.0:
    t = t / 1000.0
    sec_per_bin = sec_per_bin / 1000.0
```

```python
expected = bin_size / fs_fallback
if (not np.isfinite(sec_per_bin)) or sec_per_bin <= 0 or sec_per_bin > 10 * expected or sec_per_bin < 0.1 * expected:
    t = np.arange(n, dtype=np.float64) * expected
    sec_per_bin = expected
```

iii. The AI’s justification is almost entirely practical: in the trajectory it reports that timestamp-derived timing initially broke trialization, so it added heuristics and a fallback so timing would remain sensible and compatible with the neural bins.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI first truncates fluorescence, motion, and timestamps to a common raw length. It then bins the time series to the same number of 10-frame bins as the neural data and splits it into the same 60-second trial boundaries.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
...
motion = motion[:n_common]
tstamps = tstamps[:n_common]
```

```python
n_bins = neural_b.shape[1]
motion_b = motion_b[:n_bins]
time_b, sec_per_bin = compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback=fs)
time_b = time_b[:, :n_bins]
...
neural_trials, input_trials, output_trials, bins_per_trial = split_trials(
    neural_b, time_b, output_b, fs_binned=fs_binned, trial_sec=trial_sec
)
```

iii. In Step 4 and Step 5 notes, the AI says all streams should be aligned at the frame level and trimmed to a shared valid interval. The trajectory shows this was a conscious response to small length mismatches between imaging and behavior arrays.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion-energy signal is derived from `motion_energy_glob.npy`. The AI also loads `tstamps.npy` for alignment, but it does not use `interframe_int.npy`.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
```

iii. In the notes the AI identifies `motion_energy_glob.npy` as the behavioral variable required by the task and `tstamps.npy` as the time base used to keep behavior aligned with the imaging frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI truncates the motion trace to the common raw length shared with fluorescence and timestamps, averages it in non-overlapping 10-frame bins, and then discretizes the binned values into five session-specific quantile bins. It does not interpolate missing video frames.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
...
motion = motion[:n_common]
...
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
...
motion_labels, motion_edges = digitize_equal_percentile(motion_b, n_bins=5)
output_b = motion_labels[None, :]
```

iii. The notes justify the 10-frame averaging by reference to the paper’s decoder preprocessing, and justify per-session discretization because the task asked for five equal-percentile bins per session. The trajectory shows the AI intentionally chose truncation over dropped-frame interpolation after observing only small length mismatches.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes five equal-percentile bins separately within each session using `np.quantile`, forces the edges to be monotonic to avoid repeated-value issues, and then labels each time bin with `np.digitize`.

ii.
```python
def digitize_equal_percentile(x, n_bins=5):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.maximum.accumulate(edges)
    labels = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    return labels, edges.astype(np.float32)
```

iii. In Step 5 notes, the AI explicitly states that output discretization should use five equal-percentile bins computed separately within each session after smoothing/alignment, because that is what the decoder task requested.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to the neural data by truncating fluorescence, motion energy, and timestamps to the minimum shared raw length, then binning both streams identically and splitting them into matching trial windows. It does not repair dropped camera frames.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])

F = F[keep, :n_common]
Fneu = Fneu[keep, :n_common]
motion = motion[:n_common]
tstamps = tstamps[:n_common]
```

```python
neural_b = moving_average_nonoverlap(dff, bin_size)
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
...
neural_trials, input_trials, output_trials, bins_per_trial = split_trials(
    neural_b, time_b, output_b, fs_binned=fs_binned, trial_sec=trial_sec
)
```

iii. In Step 4 notes, the AI says behavior should be aligned “using shared frame index/timestamps” and that rare mismatches should be handled “by truncating to the common valid length.” The trajectory shows it preferred this simpler alignment rule after inspecting the data and deciding the mismatches were small.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles data issues with heuristics rather than explicit repair. Length mismatches are handled by truncating all arrays to `n_common`; implausible timestamps trigger a fallback to frame-rate-based time; partial trailing data are discarded when trializing. It does not use `interframe_int.npy` to interpolate missing motion-energy frames.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
...
expected = bin_size / fs_fallback
if (not np.isfinite(sec_per_bin)) or sec_per_bin <= 0 or sec_per_bin > 10 * expected or sec_per_bin < 0.1 * expected:
    t = np.arange(n, dtype=np.float64) * expected
    sec_per_bin = expected
```

```python
n_trials = neural.shape[1] // bins_per_trial
usable = n_trials * bins_per_trial
neural = neural[:, :usable]
inp = inp[:, :usable]
out = out[:, :usable]
```

iii. The trajectory documents two concrete justifications: timestamp units were unreliable, so the AI added a plausibility fallback, and behavior arrays could differ slightly in length from imaging, so it truncated to a shared interval. The notes present those as practical fixes rather than paper-matched processing.

## 6-a. What are the most time-consuming steps of the code?

i. The AI does not provide a formal profile, but its notes and code indicate that the main costs are loading full session arrays from disk and computing the baseline/dF/F transform inside `robust_df_over_f`. Optional plotting is extra overhead when enabled.

ii.
```python
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
```

```python
for start in range(0, n, win):
    end = min(n, start + win)
    chunk = fc[:, start:end]
    b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
    baseline[:, start:end] = b.astype(np.float32)
```

iii. In Step 6 notes the AI specifically remarks that loading full `F.npy` and `Fneu.npy` per session could be a cost, and that it used vectorized NumPy operations where possible. The runtime summaries in the trajectory show per-session processing dominated by the per-session loading/preprocessing path, not by later bookkeeping.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loop is the Python loop over baseline windows in `robust_df_over_f`. The trial-splitting list comprehensions also create repeated Python-level slicing/copying, though the core binning and discretization are already vectorized.

ii.
```python
for start in range(0, n, win):
    end = min(n, start + win)
    chunk = fc[:, start:end]
    b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
    baseline[:, start:end] = b.astype(np.float32)
```

```python
neural_trials = [np.asarray(neural[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.float32) for i in range(n_trials)]
input_trials = [np.asarray(inp[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.float32) for i in range(n_trials)]
output_trials = [np.asarray(out[:, i*bins_per_trial:(i+1)*bins_per_trial], dtype=np.int64) for i in range(n_trials)]
```

iii. The AI’s Step 6 notes say it already vectorized the main signal-processing operations and considered the script acceptably fast. That implies the remaining obvious vectorization target is the baseline loop, not the averaging/discretization path.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated processing inside the core conversion path beyond per-session processing by design. The code computes one session result at a time, then later loops over those results again to assemble the final dataset and validate it.

ii.
```python
for i, sess_dir in enumerate(sessions, start=1):
    res = load_session(sess_dir)
    results.append(res)
```

```python
data = build_dataset(results)
validate_dataset(data)
```

iii. The AI did not document any substantial duplicate computation in the script itself. Its notes emphasize that the main operations were already vectorized and the full run finished quickly, so repeated work was not a major concern.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes some extra processing that is not needed by downstream decoder training: optional plotting, computation of extra session metadata, and storage of summary statistics such as motion-energy bin edges, baseline medians, shift medians, and source paths. The decoder only needs the standardized arrays plus basic metadata.

ii.
```python
if args.show_processing and plot_budget > 0:
    maybe_plot_session(res, outpath.parent)
    plot_budget -= 1
```

```python
meta = {
    'session_id': f'{sess_dir.parent.name}/{sess_dir.name}',
    ...
    'motion_bin_edges': motion_edges,
    'baseline_median': float(np.median(baseline)),
    'shift_median': float(np.median(shift)),
    'source_session_path': str(sess_dir),
    'tstamp_start_sec': float(tstamps[0]) if len(tstamps) else 0.0,
    'tstamp_end_sec': float(tstamps[-1]) if len(tstamps) else 0.0,
}
```

iii. The notes show the AI deliberately kept richer per-session metadata for traceability and debugging, and the workflow required optional diagnostic plots. Those steps are useful for validation, but not required by downstream decoding once the final arrays have been built.
