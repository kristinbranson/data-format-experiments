# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` for sorted `jm*` subject directories and sorted child directories containing `suite2p`. Each daily recording is processed as one session. Per session it loads `F.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `tstamps.npy`; full mode maps all 41 session jobs across up to six worker processes.

ii.
```python
for subject in sorted(os.listdir(DATA_ROOT)):
    ...
    for sdir in sorted(glob.glob(os.path.join(subj_dir, '*'))):
        if os.path.isdir(sdir) and os.path.isdir(os.path.join(sdir, 'suite2p')):
            sessions.append((subject, os.path.basename(sdir), sdir))
...
F = np.load(os.path.join(p0, 'F.npy'))
ops = np.load(os.path.join(p0, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(p0, 'iscell.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
```

iii. The notes say the released hierarchy is subject/day/suite2p plus `move_deve`, contains six mice and 41 daily recordings, and already stores Track2p-matched cells. Sorting makes the ordering deterministic; the suite2p-directory test avoids unrelated directories.

## 1-b. How are the data split into subjects?

i. A subject is a top-level directory whose name begins with `jm`. Subject names are sorted, and `subject_idx` is constructed by looking up each processed session’s subject in that sorted list.

ii.
```python
if not os.path.isdir(subj_dir) or not subject.startswith('jm'):
    continue
...
subjects = sorted(set(r['subject'] for r in results))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The agent states that the six `jm*` folders are the six mice and verifies session counts per mouse against the released data and paper.

## 1-c. How are the data split into sessions?

i. Every sorted subject child directory with a `suite2p` directory is a session/daily recording. The `Pool.map` result order preserves the session-job order, so neural, input, output, and `subject_idx` remain aligned.

ii.
```python
for sdir in sorted(glob.glob(os.path.join(subj_dir, '*'))):
    if os.path.isdir(sdir) and os.path.isdir(os.path.join(sdir, 'suite2p')):
        sessions.append((subject, os.path.basename(sdir), sdir))
...
results = pool.map(process_session, jobs, chunksize=1)
```

iii. The notes identify each dated folder as one daily recording and report 41 sessions, with 6–7 sessions per mouse.

## 1-d. How are the data split into trials?

i. Continuous recordings are divided into consecutive, non-overlapping 60-second blocks after 10-frame binning. At 30 Hz this gives 180 bins per trial. Only complete trials are retained.

ii.
```python
bins_per_trial = int(round(TRIAL_SECONDS * fs / BIN_FRAMES))
ntrials = nbins // bins_per_trial
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
```

iii. There are no natural experimental trials, and the task explicitly requires 60-second trials. The notes verify that all actual session lengths are exact trial multiples, so the full conversion loses no frames.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. Incomplete tail bins or trials would be dropped by integer truncation, but the actual sessions have no remainder. Shape, finiteness, and label-range assertions are run afterward.

ii.
```python
ntrials = nbins // bins_per_trial
...
assert T == r['bins_per_trial']
assert np.isfinite(r['neural'][tr]).all()
assert np.isfinite(r['input'][tr]).all()
```

iii. The notes say the paper specifies no trial rejection and that all recordings divide exactly into complete trials. Thus no data-driven trial exclusion was justified.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The saved neural signal is derived from suite2p `F.npy` only. Although `f_processing` accepts `Fneu`, the caller does not load or pass `Fneu`, and the selected neuropil coefficient is zero. `ops.npy` supplies acquisition and baseline parameters, while `iscell.npy` is used only for a curation assertion.

ii.
```python
F = np.load(os.path.join(p0, 'F.npy'))
ops = np.load(os.path.join(p0, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(p0, 'iscell.npy'))
...
dF = f_processing(F.astype(np.float32), fs=fs,
                  baseline=ops.get('baseline', 'maximin'), ...)
```

iii. The agent says the repository’s `DataManagement.F_processing` defaults to `neucoeff=0`, so the paper’s baseline-corrected fluorescence should use `F` without neuropil subtraction. This differs from the human reference, which uses both `F` and `Fneu` with coefficient 0.7.

## 2-b. How is the `neural` data processed?

i. `F` is converted to float32, a maximin baseline is estimated using Gaussian smoothing followed by 60-second minimum and maximum filters, and the baseline is subtracted. No division by baseline and no neuropil subtraction occur. The result is then averaged over non-overlapping groups of 10 frames and cast to float32.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
return Fc - Flow
...
dF_binned = bin_trace(dF).astype(np.float32)
```

iii. The notes characterize this as a verbatim reimplementation of repository `F_processing` (`maximin`, sigma 10 frames, window 60 s, `neucoeff=0`) and cite the paper’s instruction to average dF/F over 10 timestamps. The human reference instead first computes `F - 0.7*Fneu`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed during conversion. The code asserts that each released ROI is either marked as a cell or has classifier probability above 0.5, and that `iscell` and `F` have equal row counts.

ii.
```python
assert np.all((iscell[:, 0] == 1) | (iscell[:, 1] > 0.5)), \
    f'{sess_dir}: uncurated ROIs present'
assert iscell.shape[0] == F.shape[0]
```

iii. The agent concluded that suite2p cell curation and Track2p cross-day matching were already applied when the released files were created; filtering again would incorrectly discard or re-index cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is the first imaging frame/session start. Trials are simply successive 60-second windows, with metadata offsets 0 to 60 seconds; there is no stimulus-event realignment.

ii.
```python
sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
...
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The experiment is continuous spontaneous behavior without a stimulus or task event, so the notes justify session start and consecutive segmentation as the only applicable alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The original 30 Hz data are averaged in non-overlapping 10-frame bins, producing 3 Hz data with 333.33 ms bins. The same binning is applied to neural and aligned motion-energy streams.

ii.
```python
BIN_FRAMES = 10
...
return x[..., :T].reshape(newshape).mean(axis=-1)
...
'time_bin_size': float(BIN_FRAMES / results[0]['fs'] * 1000.0)
```

iii. The agent quotes the paper’s decoding method of averaging neural and behavior traces over 10 consecutive timestamps and verifies 180 bins per 60-second trial.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is constructed rather than read directly: it uses the binned sample index, constant `BIN_FRAMES=10`, and session-specific `ops['fs']`.

ii.
```python
fs = float(ops['fs'])
bin_centre_frames = np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0
time_s = (bin_centre_frames / fs).astype(np.float32)
```

iii. The notes say a fixed imaging clock makes index-derived elapsed time equivalent to timestamps and intentionally choose bin centers (0.15, 0.483..., seconds) rather than bin left edges.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For every retained 10-frame bin, the code computes its center-frame position `(10*i + 4.5)`, divides by the imaging frequency to obtain seconds, casts to float32, and slices the continuous session vector into trial arrays of shape `(1, 180)`.

ii.
```python
bin_centre_frames = np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0
time_s = (bin_centre_frames / fs).astype(np.float32)
...
input_trials.append(time_s[sl][None, :].copy())
```

iii. The agent’s independent checks confirm a constant one-third-second step and end values half a bin before session end.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector has one value at the center of every neural bin and is sliced with exactly the same trial slice. It remains elapsed session time across trial boundaries rather than resetting each trial.

ii.
```python
sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
input_trials.append(time_s[sl][None, :].copy())
```

iii. The notes report explicit boundary and random-entry checks showing the input indices correspond to the same raw-frame groups as the neural bins.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output starts from the provided `move_deve/motion_energy_glob.npy`. Camera `tstamps.npy` is used to locate missing camera samples relative to the imaging frame grid. The methods describe the provided signal as summed squared pixelwise differences between consecutive video frames.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
me_frames, n_interp = align_motion_energy(me_raw, ts, nframes)
```

iii. The notes say the camera was hardware-triggered by the microscope and the dataset README identifies timestamps/interframe intervals as the means of detecting dropped frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Raw motion energy is mapped to the imaging grid, missing values and the undefined first value are linearly interpolated, the trace is averaged in the same 10-frame bins as neural data, and session-specific quintile labels are calculated.

ii.
```python
full[idx[keep]] = me[keep]
full[0] = np.nan
full[nanmask] = np.interp(np.flatnonzero(nanmask), np.flatnonzero(good), full[good])
...
me_binned = bin_trace(me_frames)
me_labels, quant_edges = discretize_quantiles(me_binned, N_OUTPUT_BINS)
```

iii. The agent justifies interpolation as preserving synchronous frame indexing and bins before categorization because averaging category labels would be meaningless.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds are the 20th, 40th, 60th, and 80th percentiles of each session’s complete binned motion-energy trace. `np.searchsorted(..., side='right')` assigns integer labels 0–4, with equality assigned to the higher category.

ii.
```python
edges = np.quantile(x, np.arange(1, nbins) / nbins)
labels = np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The task requires five equal-percentile bins selected per session. The agent verifies approximately 20% of retained samples in each category and stores each session’s thresholds in metadata.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When camera and imaging counts match, indices are identical. Otherwise, median timestamp spacing and rounded gap multiples reconstruct camera positions on the imaging grid; all missing locations are linearly interpolated. Motion and neural traces are then identically binned and identically trial-sliced.

ii.
```python
ifi = np.diff(np.asarray(ts, dtype=np.float64))
med = np.median(ifi)
steps = np.maximum(np.round(ifi / med).astype(np.int64), 1)
idx = np.concatenate([[0], np.cumsum(steps)])
...
assert me_binned.size == nbins
output_trials.append(me_labels[sl][None, :].copy())
```

iii. The notes explain that hardware triggering normally gives frame identity, while timestamp gaps identify drops. Independent checks include sessions with 116 and 148 dropped camera frames and report exact reconstruction.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames and the undefined first motion-energy sample are linearly interpolated. The code asserts neural frame counts against `ops`, validates ROI counts/curation, checks aligned behavior length, and checks saved trial shapes, finite values, and category ranges. Incomplete final bins/trials would be truncated.

ii.
```python
assert F.shape[1] == nframes
...
full[nanmask] = np.interp(...)
...
assert me_binned.size == nbins
assert np.isfinite(r['neural'][tr]).all()
```

iii. The notes document nine sessions with dropped camera frames and validate interpolation independently. They also note one suite2p `badframes` entry is retained because the paper does not prescribe removing it.

## 6-a. What are the most time-consuming steps of the code?

i. The maximin baseline filtering over every neuron and frame is the main compute cost. Loading large arrays and optional plotting are secondary. Sessions are parallelized across up to six processes.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
...
with Pool(nproc) as pool:
    results = pool.map(process_session, jobs, chunksize=1)
```

iii. Per-session timings in the notes attribute roughly 0.24–1.11 seconds to baseline processing and under 0.1 seconds to binning/behavior; the full 41-session conversion took 6.8 seconds with multiprocessing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial construction and per-trial validation use Python loops and could be reshaped/vectorized internally, although the required output is ultimately a nested list. `session_info` also repeatedly scans prior results to compute day indices, and subject lookup uses repeated `list.index`; both could use counters/maps. Major numerical operations (binning, timestamp alignment, filtering, quantiles) are already vectorized.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    ...
...
'day_index_within_subject': sum(1 for q in results[:i]
                                if q['subject'] == r['subject']),
```

iii. The agent explicitly emphasizes vectorized reshape binning and multiprocessing; it does not identify these small remaining loops as bottlenecks. Given only 20–30 trials and 41 sessions, their cost is minor.

## 6-c. What processing does the code repeat multiple times?

i. Every session independently repeats loading, baseline filtering, binning, alignment, quantile computation, and trial slicing, as required for session-specific signals and thresholds. It also traverses every trial again for sanity checks and concatenates outputs to calculate class counts. Optional processing plots concatenate and rescan trial data.

ii.
```python
for si, r in enumerate(results):
    for tr in range(len(r['neural'])):
        ...
    o = np.concatenate([x.ravel() for x in r['output']])
...
out_all = np.concatenate([o.ravel() for o in res['output']])
```

iii. The notes favor these repetitions as validation: they report independent recomputation and shape/statistical checks. The repeated passes are inexpensive relative to baseline filtering.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `--show-processing` optionally produces eight-panel diagnostic plots and computes z-scored neural data solely for display; these are not stored in the converted dataset. Timings, quantile diagnostics, class counts, and extensive sanity statistics are calculated mainly for logs/metadata, not decoder features. `iscell` and much of `ops` are validation/parameter inputs rather than saved signals.

ii.
```python
zs = (allneural - allneural.mean(axis=1, keepdims=True)) / \
     (allneural.std(axis=1, keepdims=True) + 1e-9)
ax[3].imshow(zs, ...)
...
if show_processing:
    plot_processing(...)
```

iii. The agent says plots and independent checks were used to catch temporal shifts and confirm processing. They are intentionally diagnostic and optional; the z-scored display data are correctly discarded rather than substituted for the reference-faithful neural signal.
