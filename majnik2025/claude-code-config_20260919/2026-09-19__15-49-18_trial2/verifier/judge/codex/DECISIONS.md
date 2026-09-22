# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter uses a fixed six-mouse list, enumerates every directory below each mouse in sorted order, and converts all 41 resulting sessions (unless `--sample` is requested). Each worker loads the session's Suite2p fluorescence/cell/ops arrays and motion-energy/inter-frame-interval arrays. The continuous recordings are later exposed as trial lists.

ii.
```python
SUBJECTS = list(SUBJECT_INFO.keys())

def list_sessions(subject):
    d = os.path.join(DATA_DIR, subject)
    return sorted(f.path for f in os.scandir(d) if f.is_dir())

session_dirs = []
for s in SUBJECTS:
    session_dirs.extend(list_sessions(s))
sessions = list(ex.map(_worker, [(sd, args.neural_scaling, args.neucoeff)
                                 for sd in session_dirs]))
```

iii. The agent states that these are the six paper mice and verifies 41 sessions, 1,090 trials, and 20,445 session-summed neurons. Sorted directories give deterministic chronological ordering, while process workers reduce wall time.

## 1-b. How are the data split into subjects?

i. Subject identity comes from the hard-coded `SUBJECT_INFO` keys (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`). Each converted session stores its parent-directory subject, which is mapped back to an integer `subject_idx`.

ii.
```python
subject = os.path.basename(os.path.dirname(session_dir))
...
'subjects': SUBJECTS,
'subject_idx': np.array([SUBJECTS.index(s['info']['subject'])
                         for s in sessions], dtype=np.int64),
```

iii. The notes map these folder IDs to paper mice A-F and report the expected 7, 7, 7, 7, 6, and 7 sessions.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject is one daily session and becomes one top-level element of `neural`, `input`, and `output`.

ii.
```python
def list_sessions(subject):
    d = os.path.join(DATA_DIR, subject)
    return sorted(f.path for f in os.scandir(d) if f.is_dir())
...
'neural': [s['neural'] for s in sessions],
'input': [s['input'] for s in sessions],
'output': [s['output'] for s in sessions],
```

iii. The session names are dates, so lexical sorting is chronological. The agent treats the released recording directories as authoritative when paper prose and actual durations differ.

## 1-d. How are the data split into trials?

i. After 10-frame binning, each continuous session is cut into consecutive, non-overlapping 60-second trials. At the measured 30 Hz this is 180 binned samples per trial; an incomplete tail would be dropped.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
n_trials = dff_binned.shape[1] // bins_per_trial
n_used = n_trials * bins_per_trial
...
for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural.append(np.ascontiguousarray(neural_all[:, sl]))
```

iii. Sixty seconds is required by the task. The notes explain that the dataset has no natural behavioral trials and that contiguous blocks also resemble the paper's blockwise decoding split.

## 1-e. How are trials filtered based on quality controls?

i. No complete trials or sessions are quality-filtered. Dropped camera samples are repaired before trial cutting; only a hypothetical trailing segment shorter than 60 seconds is excluded.

ii.
```python
n_trials = dff_binned.shape[1] // bins_per_trial
n_used = n_trials * bins_per_trial
dff_binned = dff_binned[:, :n_used]
me_binned = me_binned[:n_used]
```

iii. The agent found no task-specific bad-trial criterion, complete neural/behavioral coverage, and at most 0.41% isolated missing camera frames. Actual recordings divide exactly into 20 or 30 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from Suite2p `F.npy` and `Fneu.npy`; `ops.npy` supplies sampling rate, frame count, and neuropil coefficient. `iscell.npy` is loaded for a curation assertion.

ii.
```python
F = np.load(os.path.join(p, 'F.npy'))
Fneu = np.load(os.path.join(p, 'Fneu.npy'))
iscell = np.load(os.path.join(p, 'iscell.npy'))
ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
return F, Fneu, float(ops['fs']), int(ops['nframes']), float(ops['neucoeff'])
```

iii. The paper and repository use baseline-corrected fluorescence rather than `spks.npy` for decoding. The released arrays are already Track2p matched.

## 2-b. How is the `neural` data processed?

i. The code subtracts neuropil (`F - neucoeff*Fneu`), Gaussian-smooths in time, applies the 60-second maximin baseline, subtracts that baseline, and averages 10 consecutive frames. By default it then subtracts each neuron's whole-session mean and divides every neuron by one pooled session standard deviation before trial slicing.

ii.
```python
Fc = F - neucoeff * Fneu
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, int(win_baseline * fs))
Flow = maximum_filter1d(Flow, int(win_baseline * fs))
return Fc - Flow, Flow
...
dff_binned = bin_average(dff).astype(np.float32)
scale = float(dff_binned.std())
neural_all = (dff_binned - dff_binned.mean(axis=1, keepdims=True)) / scale
```

iii. The baseline recipe and 0.7 coefficient are justified as Suite2p defaults and were checked against paper figures. The extra centering/scaling was chosen to improve optimization and SVD initialization; the agent calls it information-preserving for the biased linear decoder and reports several decoder comparisons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed during conversion. Instead, the code asserts that every released ROI is marked as a cell and verifies fluorescence dimensions/frame counts.

ii.
```python
assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROIs'
assert F.shape == Fneu.shape
assert F.shape[1] == ops['nframes']
```

iii. The agent verified that all released ROIs already pass `iscell` and are Track2p-matched across every day of their mouse, so applying another unavailable criterion would be inappropriate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Binned neural data is sliced into contiguous 60-second blocks beginning at session start. Metadata describes session start as the global reference and each trial's own start as its local alignment, with offsets 0 to 60 seconds.

ii.
```python
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural.append(np.ascontiguousarray(neural_all[:, sl]))
...
'temporal_alignment_event':
    'start of the imaging session ... each trial is aligned to its own start',
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. The agent explains that spontaneous continuous recording has no stimulus or response event, making fixed block starts the only applicable alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural and motion-energy traces are averaged over non-overlapping groups of 10 raw 30-Hz frames, yielding 3 Hz or 333.33 ms samples. Short tails are discarded.

ii.
```python
BIN_FRAMES = 10
return x[..., :n].reshape(*x.shape[:-1], n // k, k).mean(axis=-1)
...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. This directly follows the paper's stated 10-timestamp averaging for all decoding analyses; a tested 1-second alternative was rejected because it departed from the paper without meaningful gain.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is constructed from the binned sample index and the session's `ops['fs']`; no stored timestamp stream is used.

ii.
```python
bin_dt = BIN_FRAMES / fs
time_all = (np.arange(n_used) + 0.5) * bin_dt
```

iii. With a constant acquisition rate, index times bin duration supplies elapsed session time. The half-bin offset represents each averaged bin by its center.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The global binned index is shifted by one half and multiplied by 10/fs seconds, converted to float32, and reshaped to `(1, time)` within each trial. It remains continuous across trial boundaries.

ii.
```python
time_all = (np.arange(n_used) + 0.5) * bin_dt
...
inputs.append(time_all[sl][None, :].astype(np.float32))
```

iii. The agent deliberately uses bin centers, producing 0.167 to 1199.833/1799.833 seconds, and independently checked the saved ramps.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One center timestamp is generated for every binned neural column, and exactly the same trial slice is applied to both.

ii.
```python
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural.append(np.ascontiguousarray(neural_all[:, sl]))
inputs.append(time_all[sl][None, :].astype(np.float32))
```

iii. Shared indexing makes each time value describe the center of the corresponding 10-frame neural average; round-trip assertions/plots checked this mapping.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is based on `move_deve/motion_energy_glob.npy`; `interframe_int.npy` locates dropped camera triggers so that motion energy can be placed on the imaging frame grid.

ii.
```python
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
itv = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The raw motion-energy file is the paper's precomputed summed squared pixel-wise frame difference. Camera timing is needed because a few video frames are absent.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Inter-frame intervals are normalized by their median and cumulatively summed to reconstruct imaging-frame indices. Out-of-range camera samples are ignored, the first placeholder and missing frames are linearly interpolated, the aligned trace is averaged in 10-frame bins, and its values are converted to session-specific quintile labels.

ii.
```python
steps = np.round(itv / np.median(itv)).astype(np.int64)
idx = np.concatenate([[0], np.cumsum(steps)])
out = np.full(n_frames, np.nan)
out[idx[idx < n_frames]] = me[idx < n_frames]
out[0] = np.nan
missing = np.isnan(out)
out[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing), out[~missing])
me_binned = bin_average(me)
labels_all, edges = discretize_quantiles(me_binned)
```

iii. This follows the dataset README's interpolation option and the paper's bin averaging. The agent verified that gaps are single-frame, corrected alignment improves neural-behavior correlation, and the first motion value is only a placeholder.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds are the 20th, 40th, 60th, and 80th percentiles of each session's complete binned motion-energy trace. `searchsorted(..., side='right')` assigns integer labels 0-4.

ii.
```python
qs = np.linspace(0, 100, n_bins + 1)[1:-1]
edges = np.percentile(x, qs)
labels = np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. Per-session equal-percentile bins are explicitly required. The agent reports exactly 20% per class and confirmed labels with an independent rank-based calculation.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera-frame locations are reconstructed on the imaging grid, missing samples are interpolated, and then both signals undergo identical non-overlapping 10-frame averaging, truncation, and trial slicing.

ii.
```python
me, me_missing = load_motion_energy(session_dir, n_frames)
dff_binned = bin_average(dff)
me_binned = bin_average(me)
...
neural.append(neural_all[:, sl])
outputs.append(labels_all[sl][None, :])
```

iii. The microscope triggered the camera, supporting framewise alignment once dropped triggers are restored. Cross-correlation, shifted negative controls, and corrected-versus-naive comparisons supported the result.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames and the initial placeholder are represented as NaN and linearly interpolated; video samples extending beyond imaging are dropped. Shape/content assertions fail loudly on unexpected data, and partial terminal trials would be truncated.

ii.
```python
keep = idx < n_frames
out[idx[keep]] = me[keep]
out[0] = np.nan
missing = np.isnan(out)
if missing.all():
    raise ValueError(...)
out[missing] = np.interp(...)
```

iii. This preserves fixed trial lengths while following the README. The notes enumerate the relevant edge cases and report final checks for finite values, shapes, and class ranges.

## 6-a. What are the most time-consuming steps of the code?

i. Whole-session maximin baseline filtering is the dominant computation. Loading `ops.npy` is the notable I/O expense because it contains large images, and final pickling/stats traverse hundreds of MB.

ii.
```python
ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The agent measured baseline work at roughly 0.3-0.8 seconds per session and notes that each 85-MB ops object is loaded for only three scalar fields. Eight process workers reduced the full run to about six seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Array-heavy operations are already vectorized: baseline filters act across all neurons, binning uses reshape/mean, interpolation uses `np.interp`, and sessions are parallelized. The remaining trial loop mainly creates the required nested list and could be reshaped in bulk before list construction, but would not remove the need for list objects.

ii.
```python
return x[..., :n].reshape(*x.shape[:-1], n // k, k).mean(axis=-1)
...
for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
```

iii. The notes explicitly credit vectorized filtering/averaging and process-level parallelism; behavior and trial assembly take under 0.05 seconds per session, so further trial-loop optimization has little value.

## 6-c. What processing does the code repeat multiple times?

i. `list_sessions(subject)` rescans/sorts a mouse directory for every converted session to compute `day_idx`, despite sessions having already been enumerated. With `--show-processing`, selected sessions are fully converted once for plotting and again in the normal conversion. Summary and validation passes also traverse all trials after assembly.

ii.
```python
day_idx = list_sessions(subject).index(session_dir)
...
if args.show_processing:
    plot_processing(sd, ...)
...
sessions = list(ex.map(_worker, ...))
```

iii. The agent did not emphasize these repetitions because directory scans and validation are small relative to baseline filtering; diagnostic reconversion is optional and supports visual verification.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Loading `ops.npy` materializes large mean/registration images although only `fs`, `nframes`, and `neucoeff` are retained. `F0` is returned for optional plots but discarded in ordinary conversion, and the per-bin missing-frame fraction is computed only to summarize metadata. Optional diagnostic plots also reconvert sessions and are not decoder inputs.

ii.
```python
ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
...
dff, F0 = f_processing(F, Fneu, fs, neucoeff)
missing_binned = bin_average(me_missing.astype(np.float64))
...
if os.environ.get('T2P_KEEP_DIAG'):
    diag = dict(F=F, F0=F0, ...)
```

iii. The notes explicitly identify the oversized ops load as an inefficiency. The other discarded intermediates support diagnostics and provenance, so they are deliberate but unnecessary to downstream decoder training.
