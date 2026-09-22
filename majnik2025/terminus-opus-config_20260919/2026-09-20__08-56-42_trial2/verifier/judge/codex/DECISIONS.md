# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `jm*` subject directory under `/app/data`, sorts each subject's session directories, and loads each session's `ops.npy`, `F.npy`, `Fneu.npy`, `iscell.npy`, `motion_energy_glob.npy`, and, when camera samples are missing, `interframe_int.npy`. Full mode processes all 6 mice and 41 sessions.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))
for si, subject in enumerate(subjects):
    sessions = list_sessions(subject)
    for sd in sessions:
        sess = process_session(sd, drop_neurons=drop)
```
```python
F = np.load(os.path.join(p, 'F.npy'))
Fneu = np.load(os.path.join(p, 'Fneu.npy'))
iscell = np.load(os.path.join(p, 'iscell.npy'))
```

iii. The agent justified this from the released directory convention and the example loading notebook. It reports retaining all 6 mice, all 41 daily recordings, and 1,090 constructed trials.

## 1-b. How are the data split into subjects?

i. A subject is each sorted directory whose name begins with `jm`; the subject's position in that list is stored once per session in `subject_idx`.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))
data['subject_idx'].append(si)
```

iii. The notes state that the six `jm031` through `jm046` folders are the six mice and optionally map them to paper labels A-F.

## 1-c. How are the data split into sessions?

i. Each immediate subdirectory of a subject is treated as one daily session and sorted lexicographically/chronologically. Each becomes one top-level element of `neural`, `input`, and `output`.

ii.
```python
def list_sessions(subject):
    return sorted(p for p in glob.glob(os.path.join(DATA_ROOT, subject, '*'))
                  if os.path.isdir(p))
```

iii. Session names are dated directories, so sorting gives deterministic daily order. Metadata records the date and day index.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping 60-second blocks after 10-frame binning. At 30 Hz this is 180 bins per trial; only complete blocks are used.

ii.
```python
bin_sec = BIN_SIZE / fs
bins_per_trial = int(round(TRIAL_SEC / bin_sec))
ntrials = nbins // bins_per_trial
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
```

iii. The task explicitly requires 60-second trials, while the source recordings have no natural trials. The notes verify that all actual sessions divide exactly, yielding 20 or 30 trials without loss.

## 1-e. How are trials filtered based on quality controls?

i. No complete 60-second trial or session is quality-filtered. The formula would discard an incomplete tail, but the supplied sessions have none.

ii.
```python
ntrials = nbins // bins_per_trial
nused = ntrials * bins_per_trial
me_used = me_binned[:nused]
```

iii. The agent states that all behavior and neural streams were complete after camera-gap repair and that the 36,000/54,000-frame sessions divide exactly into complete trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p `F.npy` and `Fneu.npy`, with processing parameters and sampling rate read from `ops.npy`; `iscell.npy` is checked for curation.

ii.
```python
ops = load_ops(session_dir)
F, Fneu, iscell = load_traces(session_dir)
Fc = F - neucoeff * Fneu
```

iii. The agent identified these as the reference code's raw fluorescence, neuropil fluorescence, saved preprocessing parameters, and ROI classification.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil (`F - neucoeff*Fneu`), performs maximin baseline correction, averages non-overlapping groups of 10 frames, then additionally z-scores every neuron across the whole session before splitting into trials and casting to float32.

ii.
```python
dff = compute_dff(F, Fneu, fs,
                  neucoeff=float(ops.get('neucoeff', 0.7)),
                  baseline=str(ops.get('baseline', 'maximin')),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
dff_binned = bin_average(dff)
neural = zscore_rows(dff_binned).astype(np.float32)
```

iii. The maximin pipeline and 10-frame averaging were justified from the paper/reference code. The extra z-score was justified from Track2p raster visualization preprocessing and because the supplied decoder performs SVD without internal normalization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It asserts every released ROI has `iscell[:,0] == 1`. For each mouse, if a neuron has an all-zero `F` trace on any day, that neuron is removed from every session of that mouse (five unique neurons total).

ii.
```python
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in released data'
drop = find_zero_neurons(sessions)
keep = np.ones(F.shape[0], dtype=bool)
keep[drop_neurons] = False
F, Fneu = F[keep], Fneu[keep]
```

iii. The agent reasoned that the data were already Suite2p/Track2p curated, while zero traces represent failed extraction and would create invalid z-scores. Cross-day removal preserves matched row identity across a mouse.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trial zero begins at session onset and each subsequent artificial block begins at `k*60` seconds; neural arrays are sliced on those common bin boundaries. Metadata calls the alignment event the start of each 60-second block, with offsets 0 and 60 seconds.

ii.
```python
sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. The recordings are continuous spontaneous activity without stimulus-locked trials, so fixed block starts are the only applicable alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are mean-aggregated into each bin, giving 333.33 ms (3 Hz). Both neural and motion-energy streams use the same rebinning.

ii.
```python
BIN_SIZE = 10
return x[..., :n].reshape(newshape).mean(axis=-1)
bin_ms = 1000.0 * BIN_SIZE / 30.0
```

iii. The paper says decoding denoised dF/F and behavior by averaging 10 consecutive timestamps; applying it to both preserves alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from the post-binning sample index and `ops['fs']`, rather than loaded from a timestamp array. It represents seconds from session onset at bin centers.

ii.
```python
bin_sec = BIN_SIZE / fs
tvec = (np.arange(nused) + 0.5) * bin_sec
```

iii. Hardware acquisition is treated as uniformly sampled at the saved 30-Hz rate, making indices sufficient; the agent chose bin centers as the natural timestamp for bin averages.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It multiplies each binned index plus one half by 10/fs seconds, slices this continuous session ramp by trial, adds a variable dimension, and casts it to float32.

ii.
```python
tvec = (np.arange(nused) + 0.5) * bin_sec
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The half-bin offset was explicitly documented as representing bin-center time; no other transformation is applied.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The same trial slice is applied to the time vector and neural matrix, so each input timestamp labels the corresponding 10-frame neural average.

ii.
```python
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The notes report value-level spot checks and trial plots confirming identical 180-bin axes.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output derives from `move_deve/motion_energy_glob.npy`; `interframe_int.npy` is additionally read when the motion series is shorter than imaging.

ii.
```python
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The motion file is the paper's precomputed global video motion-energy trace, and interframe intervals locate dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is cast to float64, camera gaps are reconstructed by linear interpolation (or excess samples truncated), it is averaged in 10-frame bins, restricted to complete trials, and converted to categorical labels.

ii.
```python
me_full = np.interp(np.arange(nframes), idx, me)
me_binned = bin_average(me)
me_used = me_binned[:nused]
labels = quantile_bin(me_used, NCLASSES)
```

iii. Float conversion avoids integer arithmetic issues; interpolation follows the data README and restores one behavior value per imaging frame; binning matches paper decoding preprocessing.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Within each session, ordinal ranks of the used binned values are divided into five equal-count groups labeled 0-4. Ties are deliberately split by original order to guarantee exactly 20% per class.

ii.
```python
r = rankdata(x, method='ordinal')
lab = ((r - 1) * nclasses) // len(x)
return lab.astype(np.int64)
```

iii. The agent interpreted “five equal-percentile bins, selected per session” as requiring equal counts and chose ranks to handle ties deterministically.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera frames are assumed hardware-triggered one-to-one with imaging. Missing positions are inferred from interframe intervals and interpolated onto the imaging-frame grid; neural and behavior are then identically 10-frame averaged and sliced.

ii.
```python
nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
idx = np.arange(len(me)) + np.concatenate([[0], np.cumsum(nmiss)])
me_full = np.interp(np.arange(nframes), idx, me)
assert me_binned.shape[0] == nbins
output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. The agent cites microscope hardware triggering and verified that inferred gaps explain the deficits in almost all sessions; a uniform-stretch fallback handles inconsistent gap metadata.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion traces are gap-filled by interpolation; if inferred gaps do not span the expected endpoint, acquired samples are uniformly mapped across the imaging timeline. Long traces are truncated. All-zero neural rows are removed across the mouse. Assertions enforce ROI status, matched binned lengths, finiteness, shapes, and label range.

ii.
```python
if idx[-1] != nframes - 1:
    idx = np.linspace(0, nframes - 1, len(me))
    info['fallback_uniform'] = True
me_full = np.interp(np.arange(nframes), idx, me)
```

iii. These choices avoid silent misalignment and NaNs while preserving the common imaging timeline. The notes document camera deficits, zero traces, and independent value checks.

## 6-a. What are the most time-consuming steps of the code?

i. The agent identifies full-array I/O (especially `F`, `Fneu`, and large `ops.npy`) and maximin baseline filtering as dominant; measured neural processing takes about 0.3-1.2 seconds per session.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The notes attribute the cost to filtering every neuron across the full session and reading roughly 130 MB per session; total conversion was reported as 41.1 seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent reports no important unvectorized numerical loop. Session, subject, and trial loops are structural; all heavy within-session math is vectorized. The trial loop could be replaced by reshaping/transposing, and gap bookkeeping could be further vectorized, but neither is a major bottleneck.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. The notes explicitly say “all operations vectorised” and “none significant,” with filtering and I/O dominating runtime.

## 6-c. What processing does the code repeat multiple times?

i. `F.npy` is read once via memory mapping for every session during `find_zero_neurons`, then read again normally when that session is processed. Session lookup `sessions.index(sd)` also repeatedly searches the session list. Neural and motion arrays are each traversed again for summary assertions.

ii.
```python
F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
# Later:
F = np.load(os.path.join(p, 'F.npy'))
```

iii. The agent acknowledged the preliminary zero-neuron scan but considered memory mapping an optimization that avoids a costly full extra read; it did not identify other repetition as significant.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `process_session` retains a `_raw` dictionary containing full raw/intermediate arrays for plotting, even in normal full conversion; `main` then discards it with the session object. It also loads the very large full `ops.npy` dictionary although only a handful of scalar fields are used, and scans `F.npy` separately for zero rows.

ii.
```python
_raw=dict(F=F, dff=dff, dff_binned=dff_binned, neural=neural,
          me=me, me_binned=me_binned, labels=labels, tvec=tvec)
...
del sess
```

iii. `_raw` was intentionally kept for optional processing plots and sanity checks, but in runs without `--show-processing` it has no downstream use. The agent otherwise claimed no significant unnecessary processing.
