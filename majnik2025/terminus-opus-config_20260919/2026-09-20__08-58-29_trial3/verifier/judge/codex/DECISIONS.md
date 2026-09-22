# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory under `/app/data` as a subject, then every directory under each subject as a session, sorting both levels. For every session it loads `suite2p/plane0/F.npy` and `iscell.npy`, plus `move_deve/motion_energy_glob.npy` and `interframe_int.npy`. The full run processes 41 sessions; sample mode deliberately selects two.

ii. Code snippets:
```python
for subj in sorted(d for d in os.listdir(DATA_ROOT)
                   if os.path.isdir(os.path.join(DATA_ROOT, d))):
    subj_dir = os.path.join(DATA_ROOT, subj)
    for sess in sorted(d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))):
        out.append((subj, sess, os.path.join(subj_dir, sess)))

F = np.load(os.path.join(plane, 'F.npy'))
iscell = np.load(os.path.join(plane, 'iscell.npy'))
me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The notes say the released directory structure contains six mouse directories and 41 daily recordings, with fluorescence already exported in suite2p form and motion energy in `move_deve`. The agent intentionally avoids `Fneu.npy`, `ops.npy`, and unrelated tracking ground-truth files because its chosen processing does not use them.

## 1-b. How are the data split into subjects?

i. Each first-level directory under the data root is treated as one mouse. Unique subject names from the selected sessions are sorted, and each session receives the index of its mouse in that list.

ii. Code snippets:
```python
subjects = sorted({s[0] for s in sessions})
data['subject_idx'].append(subjects.index(subj))
```

iii. The notes identify the six directories `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` as the six mice and report that this produces the expected subject and session counts.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject is one session; each becomes one element of the outer `neural`, `input`, and `output` lists.

ii. Code snippets:
```python
for sess in sorted(d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))):
    out.append((subj, sess, os.path.join(subj_dir, sess)))
...
data['neural'].append(neural)
data['input'].append(inp)
data['output'].append(out)
```

iii. The agent states that these are daily recordings and validates the expected distribution of 7, 7, 7, 7, 6, and 7 sessions across the mice.

## 1-d. How are the data split into trials?

i. Each continuous session is divided from its start into consecutive, non-overlapping 60-second trials. At 30 Hz and after 10-frame averaging, each trial contains 180 bins. The code requires the session to divide exactly rather than silently dropping a partial trial.

ii. Code snippets:
```python
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FS))
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES
ntrials = nbins // TRIAL_BINS
assert ntrials * TRIAL_BINS == nbins
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
```

iii. The decoder task explicitly requires 60-second trials. The notes establish that all recordings have 36,000 or 54,000 frames, so they divide exactly into 20 or 30 trials and no data is lost.

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered. Exact divisibility, output/input/neural shapes, and finiteness are asserted; all 1,090 constructed trials are retained.

ii. Code snippets:
```python
assert ntrials * TRIAL_BINS == nbins
assert data['neural'][s][k].shape == (nn, TRIAL_BINS)
assert np.isfinite(data['neural'][s][k]).all()
assert np.isfinite(data['input'][s][k]).all()
```

iii. The data are continuous rather than naturally trial-based, `badframes` is empty, and every session tiles exactly into 60-second blocks, so the agent found no trial-level exclusion criterion to apply.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The delivered neural data is derived only from `suite2p/plane0/F.npy`. `iscell.npy` is loaded for a curation assertion, not as a signal. Although `F_processing` accepts `Fneu`, the agent passes `None` and uses `neucoeff=0.0`.

ii. Code snippets:
```python
F = np.load(os.path.join(plane, 'F.npy'))
dF = F_processing(F, None, fs=FS)
...
if neucoeff:
    Fc = F - neucoeff * Fneu
else:
    Fc = F.astype(np.float32, copy=True)
```

iii. The agent interpreted the Track2p GUI function's default/call path as authoritative: `neucoeff=0.0`, making `Fneu.npy` unnecessary. It explicitly considered and rejected the suite2p ops value of 0.7 as relevant only to suite2p deconvolution.

## 2-b. How is the `neural` data processed?

i. Raw fluorescence is copied to float32, smoothed with a temporal Gaussian (sigma 10 frames), baseline-estimated with a 60-second minimum filter followed by a maximum filter, and baseline-subtracted. It is then averaged in non-overlapping groups of 10 frames and cast to float32. It is not divided by the baseline despite some documentation calling it dF/F.

ii. Code snippets:
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
return Fc - Flow
...
dF_binned = bin_frames(dF).astype(np.float32)
```

iii. The agent says this copies `DataManagement.F_processing` and matches the paper's baseline-corrected fluorescence and its decoding denoising by averaging 10 timestamps. Independent spot checks were reported as equal to its implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed during conversion. The code asserts that every released ROI has `iscell[:,0] == 1`; all are retained, with one brain-region index per row.

ii. Code snippets:
```python
assert iscell.shape[0] == nneurons
assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'
data['brain_region_idx'].append(np.zeros(info['n_neurons'], dtype=np.int64))
```

iii. According to the notes and data README, the release already contains suite2p-classified cells tracked and row-matched across all days, so applying the original selection again would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is session start. Binned neural data is cut into contiguous slices beginning at bin zero; trial `k` covers seconds `60k` through `60(k+1)`.

ii. Code snippets:
```python
sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
...
'temporal_alignment_event': (
    'Start of the imaging session; recordings are continuous and are cut into '
    'consecutive non-overlapping 60 s trials.'),
```

iii. There is no experimental event or natural trial onset in this task-free continuous recording, so the agent uses session start and the artificial block boundaries required by the decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged without overlap, yielding 3 Hz data and 333.333 ms bins for both neural activity and behavior.

ii. Code snippets:
```python
BIN_FRAMES = 10
return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)
...
'time_bin_size': 1000.0 * BIN_FRAMES / FS
```

iii. The notes cite the paper's decoding method, which averages both dF/F and behavior in bins of 10 consecutive timestamps, and emphasize applying identical binning to preserve alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthesized from the imaging-frame/bin index and the nominal 30-Hz sampling rate; no timestamp file is used.

ii. Code snippets:
```python
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
```

iii. The agent uses the documented/reference nominal rate of 30 Hz and describes the input as elapsed session time required by the decoder task.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each 10-frame bin, the agent computes the center of its covered sample indices: `(10*b + 4.5)/30` seconds. Values are cast to float32 when stored, so the first bin is 0.15 seconds and spacing is one third second.

ii. Code snippets:
```python
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
input_trials.append(t_bins[sl].astype(np.float32)[None, :])
```

iii. The agent deliberately represents each averaged bin by its center and validates strict continuity and the expected one-third-second increment across concatenated trials.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The same binned index slice is applied to `t_bins` and `dF_binned` for each trial, producing one time value per neural column.

ii. Code snippets:
```python
sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
input_trials.append(t_bins[sl].astype(np.float32)[None, :])
```

iii. The notes report checks that concatenated trials reproduce the session and that every input advances by exactly 1/3 second.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from `move_deve/motion_energy_glob.npy`; `move_deve/interframe_int.npy` supplies timing gaps used to reconstruct missing camera-frame positions.

ii. Code snippets:
```python
me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
```

iii. The notes identify motion energy as the precomputed videography behavior stream and inter-frame intervals as the evidence needed to locate dropped camera frames on the imaging grid.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If camera and imaging lengths differ, inter-frame intervals are divided by their median and rounded to recover frame steps; samples are placed on an imaging-length array. Missing samples and the initial zero artifact are linearly interpolated. The aligned signal is averaged over the same 10-frame bins as neural data, then discretized per session.

ii. Code snippets:
```python
med = np.median(ifi)
steps = np.round(ifi / med).astype(np.int64)
idx = np.concatenate([[0], np.cumsum(steps)])
full = np.full(nframes, np.nan)
full[idx] = me
full[0] = np.nan
bad = np.isnan(full)
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
me_binned = bin_frames(me_full)
```

iii. The camera was hardware-triggered by the microscope, while nine sessions have documented dropped camera frames. The agent reports verifying that reconstructed final indices match the neural final frame in every mismatch session. It treats the always-zero first motion-energy value as undefined because no preceding video frame exists.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds at the 20th, 40th, 60th, and 80th percentiles are computed independently for each session's binned motion-energy values. `searchsorted(..., side='right')` assigns integer classes 0 through 4.

ii. Code snippets:
```python
edges = np.quantile(x, np.arange(1, nq) / nq)
labels = np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. This directly implements the request for five equal-percentile bins selected per session. The agent verifies approximately 20% of samples in every class (exact for these data).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Hardware-synchronized camera samples are mapped to imaging frame indices, missing camera frames are interpolated, both streams are separately averaged over identical 10-frame windows, and identical trial slices are applied to neural and output arrays.

ii. Code snippets:
```python
assert idx[-1] == nframes - 1
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
me_binned = bin_frames(me_full)
...
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
output_trials.append(me_labels[sl].astype(np.int64)[None, :])
```

iii. The notes cite microscope triggering and independently reconstruct/recompare output labels. The explicit index assertion is intended to prevent silent behavioral/neural drift.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are reconstructed from inter-frame intervals and linearly interpolated; the undefined first motion-energy point is also interpolated. Unexpected ROI status, failed index reconstruction, partial sessions, nonfinite converted values, or shape mismatches cause assertions rather than silent repair. No session has a partial trial.

ii. Code snippets:
```python
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
assert idx[-1] == nframes - 1
assert ntrials * TRIAL_BINS == nbins
assert np.isfinite(data['neural'][s][k]).all()
```

iii. The agent follows the data README's recommendation to interpolate missing camera values and uses fail-fast checks for anomalies it cannot justify repairing. It documents nine affected sessions and reports independent equality checks.

## 6-a. What are the most time-consuming steps of the code?

i. The temporal Gaussian and 60-second minimum/maximum filtering of every neuron's full trace dominate computation. Loading the large fluorescence arrays is the main I/O cost; optional plotting adds substantial sample-run overhead.

ii. Code snippets:
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The notes time neural preprocessing at roughly 0.2–0.7 seconds per sample session and identify these filters as dominant. The full 41-session conversion took 33.9 seconds with cached files and plotting disabled.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Timepoint-level binning, quantile assignment, and missing-value interpolation are already vectorized. The remaining loops are over sessions and trials; the trial loop could be replaced by reshape/splitting for the numeric arrays, but nested lists of separate trial matrices are required by the target format, so some iteration/materialization remains useful.

ii. Code snippets:
```python
return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)
labels = np.searchsorted(edges, x, side='right').astype(np.int64)
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
```

iii. The agent explicitly says it vectorized binning and discretization and avoided per-timepoint loops. It regarded remaining per-trial list assembly as negligible and structurally appropriate.

## 6-c. What processing does the code repeat multiple times?

i. The same load, baseline correction, binning, alignment, quantiling, and trial assembly pipeline is necessarily repeated once per session. Within a session, the main conversion does not recompute a signal. Optional visualization and final sanity summaries traverse or concatenate converted arrays again.

ii. Code snippets:
```python
for i, (subj, sess, sdir) in enumerate(sessions):
    neural, inp, out, info = process_session(subj, sess, sdir, show_processing=show)
...
allout = np.concatenate([o[0] for s in data['output'] for o in s])
allin = np.concatenate([i[0] for s in data['input'] for i in s])
```

iii. The notes do not identify harmful repeated conversion work. Repeated passes at the end are characterized as sanity checks, while optional plotting exists to expose every processing stage.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `iscell.npy` is loaded only to assert prior curation; aggregate arrays/statistics are built only for console sanity checks. With `--show-processing`, raw and intermediate streams are plotted for at most two sessions and not used in the pickle. The normal full run avoids loading unused `Fneu.npy`, `ops.npy`, `stat.npy`, and `spks.npy`.

ii. Code snippets:
```python
iscell = np.load(os.path.join(plane, 'iscell.npy'))
assert np.all(iscell[:, 0] == 1)
...
allneu = np.concatenate([n.ravel()[::101] for s in data['neural'] for n in s])
...
show = args.show_processing and i < 2
```

iii. The agent justifies the discarded assertion and summaries as validation. It deliberately removes larger unnecessary work—especially loading neuropil and ops under its chosen `neucoeff=0.0` interpretation—and makes diagnostic figures opt-in.
