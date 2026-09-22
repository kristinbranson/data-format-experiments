# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by iterating over sorted subject directories (starting with `jm`) in `/app/data`, then sorted session subdirectories within each. It checks that both `suite2p/plane0/F.npy` and `move_deve/motion_energy_glob.npy` exist before including a session. For each session it loads `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`.

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
def load_session(sess_dir: Path, ...):
    iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
    F = np.load(plane / 'F.npy').astype(np.float32)
    Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md that the directory structure follows the standard convention of subject folders containing session subfolders with suite2p output and motion energy files. The existence check ensures only complete sessions are processed.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data root, sorted alphabetically. Each subject directory name becomes the subject identifier.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
```
```python
subjects = sorted({r.subject for r in results})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a sorted subdirectory within a subject's folder. A flat list of all sessions across subjects is built, and each session is processed independently.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    if (sess_dir / 'suite2p/plane0/F.npy').exists() and (sess_dir / 'move_deve/motion_energy_glob.npy').exists():
        sessions.append(sess_dir)
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as contiguous, non-overlapping 60-second segments of each continuous recording. After 10-frame binning (30 Hz / 10 = 3 Hz), each trial has 180 time bins. Remainder bins that don't fill a complete trial are discarded.

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

iii. Per instruction, trials are defined as 60-second non-overlapping segments. Since the recording has no stimulus-driven trial structure, fixed-length segmentation is the appropriate approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are included. The AI validates that each session has at least 2 trials.

ii.
```python
assert len(data['neural'][s]) >= 2
```

iii. There is no natural trial structure or quality metric for trials in this dataset, so no filtering beyond completeness is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `iscell.npy` (cell classification) from `suite2p/plane0/`. The AI also loads `ops.npy` to get frame rate and baseline parameters.

ii.
```python
iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces, cell classification, and processing parameters.

## 2-b. How is the `neural` data processed?

i. The AI applies: (1) cell filtering via `iscell[:,0] > 0.5`, (2) neuropil subtraction `F - 0.7 * Fneu`, (3) a custom dF/F computation (`robust_df_over_f`) that uses a running low-percentile baseline with a 60-second window and 8th percentile, (4) non-overlapping 10-frame averaging (binning).

The custom dF/F function shifts traces upward to ensure positivity, computes a block-wise percentile baseline, floors the baseline at 5% of the median, then computes `(fc - baseline) / baseline`.

ii.
```python
keep = iscell[:, 0] > iscell_thr  # iscell_thr = 0.5

fcorr = F - neuropil_coeff * Fneu  # neuropil_coeff = 0.7

def robust_df_over_f(fcorr, fs, win_baseline_sec=60.0, prctile_baseline=8.0, eps=1e-3):
    min_per_neuron = fcorr.min(axis=1, keepdims=True)
    shift = np.maximum(0.0, 1.0 - min_per_neuron)
    fc = fcorr + shift
    win = max(1, int(round(fs * win_baseline_sec)))
    n = fc.shape[1]
    baseline = np.empty_like(fc, dtype=np.float32)
    for start in range(0, n, win):
        end = min(n, start + win)
        chunk = fc[:, start:end]
        b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
        baseline[:, start:end] = b.astype(np.float32)
    med = np.median(fc, axis=1, keepdims=True)
    floor = np.maximum(eps, 0.05 * med)
    baseline = np.maximum(baseline, floor).astype(np.float32)
    dff = (fc - baseline) / baseline
    return dff.astype(np.float32), baseline.astype(np.float32), shift.astype(np.float32)

neural_b = moving_average_nonoverlap(dff, bin_size)  # bin_size = 10
```

iii. The AI noted in CONVERSION_NOTES.md that a naive dF/F implementation initially produced near-chance decoding, so it replaced it with a more stable running low-percentile baseline. The AI's notes reference Suite2p metadata parameters (baseline=maximin, win_baseline=60, prctile_baseline=8) as informing the implementation, but the actual code does NOT use suite2p's `dcnv.preprocess` - it implements its own version.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `iscell[:,0] > 0.5`, keeping only ROIs classified as cells by suite2p's classifier. This reduces the total neuron count from the full ROI set to 20,445 across all sessions.

ii.
```python
iscell = np.load(plane / 'iscell.npy', allow_pickle=True)
keep = iscell[:, 0] > iscell_thr  # iscell_thr = 0.5
F = F[keep, :n_common]
Fneu = Fneu[keep, :n_common]
```

iii. The AI justified this by noting that the paper says "We considered all ROIs above the default threshold of 0.5 as true cells" and that the Track2p README example uses `iscell_thr = 0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'session start; continuous session split into contiguous 60-second windows',
'off_start': 0.0,
'off_end': 60.0,
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing the 30 Hz data to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
def moving_average_nonoverlap(arr, bin_size):
    n = arr.shape[-1] // bin_size
    trimmed = arr[..., : n * bin_size]
    new_shape = arr.shape[:-1] + (n, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

neural_b = moving_average_nonoverlap(dff, bin_size)  # bin_size = 10
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
```

iii. The methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is derived from `tstamps.npy` (timestamps from the behavioral video), with a fallback to frame-rate-derived computation if timestamps are implausible.

ii.
```python
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
...
def compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback):
    n = len(tstamps) // bin_size
    trimmed = np.asarray(tstamps[: n * bin_size], dtype=np.float64).reshape(n, bin_size)
    t = trimmed.mean(axis=1)
    t = t - t[0]
    sec_per_bin = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    if np.isfinite(sec_per_bin) and sec_per_bin > 1.0:
        t = t / 1000.0
        sec_per_bin = sec_per_bin / 1000.0
    expected = bin_size / fs_fallback
    if (not np.isfinite(sec_per_bin)) or sec_per_bin <= 0 or sec_per_bin > 10 * expected or sec_per_bin < 0.1 * expected:
        t = np.arange(n, dtype=np.float64) * expected
        sec_per_bin = expected
    return t[None, :].astype(np.float32), float(sec_per_bin)
```

iii. The AI's CONVERSION_NOTES.md documents that timestamp scaling initially caused zero derived trials and was fixed with plausibility checks plus frame-rate fallback.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The timestamps are averaged in 10-frame bins (matching the neural binning), subtracted by the first value to get time from session start, checked for unit plausibility (converted from ms to s if needed), and if still implausible, replaced with a synthetic time series computed from frame rate.

ii. See code snippet in 3-a above.

iii. The AI noted that timestamp units were not clearly in seconds, requiring robust handling.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same `tstamps.npy` array that is truncated to `n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])` along with neural and motion energy data, ensuring alignment. Then both are binned the same way.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
F = F[keep, :n_common]
...
tstamps = tstamps[:n_common]
...
time_b, sec_per_bin = compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback=fs)
time_b = time_b[:, :n_bins]
```

iii. Truncation to common length ensures all streams have identical indexing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) truncation to common length with neural data, (2) non-overlapping 10-frame bin averaging, (3) discretization into 5 equal-percentile bins per session.

ii.
```python
motion = motion[:n_common]
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
motion_labels, motion_edges = digitize_equal_percentile(motion_b, n_bins=5)

def digitize_equal_percentile(x, n_bins=5):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.maximum.accumulate(edges)
    labels = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    return labels, edges.astype(np.float32)
```

iii. The discretization into 5 equal-percentile bins is done per session as specified in the instructions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Percentile edges are computed at 0%, 20%, 40%, 60%, 80%, 100% of the binned motion energy within each session, then `np.digitize` assigns each value to one of 5 bins (0-4). Monotonicity of edges is enforced via `np.maximum.accumulate`.

ii.
```python
def digitize_equal_percentile(x, n_bins=5):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.maximum.accumulate(edges)
    labels = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    return labels, edges.astype(np.float32)
```

iii. Equal-percentile binning ensures approximately equal class frequencies within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy array is truncated to the common length `n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])` before any processing. After binning, it is further truncated to match the number of neural bins. No dropped-frame interpolation is performed.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
...
motion = motion[:n_common]
...
motion_b = moving_average_nonoverlap(motion[None, :], bin_size)[0]
n_bins = neural_b.shape[1]
motion_b = motion_b[:n_bins]
```

iii. The AI chose truncation to common length rather than interpolating dropped frames. This means any dropped video frames cause a temporal misalignment between neural and motion energy data for subsequent frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles length mismatches between neural, motion energy, and timestamp arrays by truncating all to the minimum common length. Remainder bins that don't fill a complete 60-second trial are discarded. No dropped-frame interpolation is performed.

ii.
```python
n_common = min(F.shape[1], motion.shape[0], tstamps.shape[0])
F = F[keep, :n_common]
Fneu = Fneu[keep, :n_common]
motion = motion[:n_common]
tstamps = tstamps[:n_common]
```

iii. The AI documented that timestamp scaling issues and edge cases were fixed iteratively, and that truncation to common length is used for alignment.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the custom `robust_df_over_f` baseline computation, which iterates over non-overlapping windows and computes percentiles for each. The `np.percentile` calls on large arrays and the Python-level loop contribute to processing time. Loading `.npy` files is I/O bound but relatively fast.

ii.
```python
for start in range(0, n, win):
    end = min(n, start + win)
    chunk = fc[:, start:end]
    b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
    baseline[:, start:end] = b.astype(np.float32)
```

iii. The total conversion for all 41 sessions completed in ~37.6 seconds, suggesting this is not a major bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The baseline computation loop in `robust_df_over_f` iterates over non-overlapping windows using a Python for-loop. This could potentially be vectorized using reshape-based operations similar to the binning function. The trial-splitting uses list comprehensions which are reasonably efficient.

ii.
```python
for start in range(0, n, win):
    end = min(n, start + win)
    chunk = fc[:, start:end]
    b = np.percentile(chunk, prctile_baseline, axis=1, keepdims=True)
    baseline[:, start:end] = b.astype(np.float32)
```

iii. With sessions having ~36,000 frames and a 1800-frame window (60s * 30Hz), this loop runs only ~20 iterations per session, so the impact is minimal.

## 6-c. What processing does the code repeat multiple times?

i. The AI processes each session independently in a single pass, so there is no repeated processing. The `compute_time_binned_from_tstamps` function does some redundant computation (binning timestamps, checking units, then potentially falling back to synthetic time), but this is not computationally expensive.

ii. N/A

iii. The code is reasonably efficient with no significant repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `tstamps.npy` and performs complex timestamp processing with unit detection and fallback logic, when a simpler frame-index-based time computation would suffice (since the frame rate is known and constant at 30 Hz). The baseline and shift values from `robust_df_over_f` are stored in per-session metadata but not used in downstream analysis.

ii.
```python
tstamps = np.load(move / 'tstamps.npy').astype(np.float32)
...
time_b, sec_per_bin = compute_time_binned_from_tstamps(tstamps, bin_size, fs_fallback=fs)
```

iii. The timestamp-based approach adds complexity without clear benefit when the frame rate is known.
