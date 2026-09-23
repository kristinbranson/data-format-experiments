# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data`, keeps subject directories whose names start with `jm`, then keeps session subdirectories that contain `suite2p`. Each session is processed independently, and `main()` uses `multiprocessing.Pool` to run `process_session()` over all discovered sessions. Within each session, the code loads calcium data from `F.npy`, metadata from `ops.npy`, ROI curation flags from `iscell.npy`, and behavior from `motion_energy_glob.npy` plus `tstamps.npy`.

ii. 
```python
def list_sessions():
    sessions = []
    for subject in sorted(os.listdir(DATA_ROOT)):
        subj_dir = os.path.join(DATA_ROOT, subject)
        if not os.path.isdir(subj_dir) or not subject.startswith('jm'):
            continue
        for sdir in sorted(glob.glob(os.path.join(subj_dir, '*'))):
            if os.path.isdir(sdir) and os.path.isdir(os.path.join(sdir, 'suite2p')):
                sessions.append((subject, os.path.basename(sdir), sdir))
    return sessions
...
F = np.load(os.path.join(p0, 'F.npy'))
ops = np.load(os.path.join(p0, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(p0, 'iscell.npy'))
...
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
...
with Pool(nproc) as pool:
    results = pool.map(process_session, jobs, chunksize=1)
```

iii. In `CONVERSION_NOTES.md`, the AI says the released dataset is already the track2p output in suite2p format, so scanning the `jm*` folders and their session subdirectories is the natural way to include all 6 subjects and all 41 sessions. The trajectory shows it deliberately added multiprocessing because it expected the per-session neural preprocessing to dominate runtime.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory name. Any directory in `/app/data` whose name starts with `jm` is treated as one mouse, and the subject list is sorted alphabetically.

ii.
```python
for subject in sorted(os.listdir(DATA_ROOT)):
    subj_dir = os.path.join(DATA_ROOT, subject)
    if not os.path.isdir(subj_dir) or not subject.startswith('jm'):
        continue
...
subjects = sorted(set(r['subject'] for r in results))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The notes justify this by citing the dataset README and the paper’s naming convention: each `jm*` directory is one mouse, with `jm031..jm046` corresponding to mice A..F.

## 1-c. How are the data split into sessions?

i. Sessions are split by subject subdirectory. Every child directory under a subject folder that contains a `suite2p` directory is treated as one session, and those directories are sorted lexicographically.

ii.
```python
for sdir in sorted(glob.glob(os.path.join(subj_dir, '*'))):
    if os.path.isdir(sdir) and os.path.isdir(os.path.join(sdir, 'suite2p')):
        sessions.append((subject, os.path.basename(sdir), sdir))
```

iii. The notes say each `/app/data/<subject>/<date>_a/` directory is one daily recording session, so keeping every such directory preserves the full longitudinal series.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous and creates artificial trials by cutting each session into consecutive, non-overlapping 60 s blocks after temporal binning. With `BIN_FRAMES = 10` at 30 Hz, this becomes 180 bins per trial.

ii.
```python
BIN_FRAMES = 10
TRIAL_SECONDS = 60.0
...
bins_per_trial = int(round(TRIAL_SECONDS * fs / BIN_FRAMES))    # 180
ntrials = nbins // bins_per_trial
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(time_s[sl][None, :].copy())
    output_trials.append(me_labels[sl][None, :].copy())
```

iii. In the notes, the AI explicitly says the paper has no natural trial structure and that the 60 s trial segmentation is imposed by the decoder task. It also notes that the actual sessions are exact multiples of 60 s, so “no data is dropped.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a separate trial-quality filter. Every full 60 s block is kept. The only implicit exclusion would be an incomplete tail if a session length were not divisible by 60 s after binning, because `ntrials` is computed with floor division.

ii.
```python
ntrials = nbins // bins_per_trial
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    ...
```

iii. The notes justify this by saying the paper’s recordings are continuous rather than trial-based, so there is no trial-level curation rule to inherit. The AI instead focused its QC on camera-frame alignment and on shape/finiteness assertions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the code, `neural` is derived from `suite2p/plane0/F.npy` only. `ops.npy` and `iscell.npy` are also loaded, but only for parameters and assertions; `Fneu.npy` is not loaded by the script at all.

ii.
```python
p0 = os.path.join(sess_dir, 'suite2p', 'plane0')
F = np.load(os.path.join(p0, 'F.npy'))
ops = np.load(os.path.join(p0, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(p0, 'iscell.npy'))
...
dF = f_processing(F.astype(np.float32), fs=fs,
                  baseline=ops.get('baseline', 'maximin'),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
```

iii. The notes and trajectory justify omitting `Fneu.npy` by appealing to the reference `F_processing` function in `track2p/gui/data_management.py`, which they interpret as using `neucoeff=0` and therefore “no neuropil subtraction.”

## 2-b. How is the `neural` data processed?

i. The AI applies a custom `f_processing()` baseline subtraction to `F.npy`: Gaussian smoothing over time, then a 60 s minimum filter and a 60 s maximum filter, and finally subtracts that baseline (`Fc - Flow`). After that, it averages the result into non-overlapping 10-frame bins. It does not call suite2p’s `dcnv.preprocess`, and it does not subtract neuropil.

ii.
```python
def f_processing(F, Fneu=None, fs=30.0, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    Fc = F if (neucoeff == 0.0 or Fneu is None) else F - neucoeff * Fneu

    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    return Fc - Flow
...
dF = f_processing(F.astype(np.float32), fs=fs,
                  baseline=ops.get('baseline', 'maximin'),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
dF_binned = bin_trace(dF).astype(np.float32)
```

iii. The AI repeatedly justifies this as a “verbatim re-implementation” of the paper’s `F_processing` step, and the notes say this matches the paper’s “baseline corrected fluorescence traces” plus the paper’s 10-frame averaging used in decoding analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out during conversion. It assumes the released dataset is already pre-curated and only asserts that every row passes the suite2p `iscell` criterion and that `iscell` has the same number of rows as `F`.

ii.
```python
iscell = np.load(os.path.join(p0, 'iscell.npy'))
...
assert np.all((iscell[:, 0] == 1) | (iscell[:, 1] > 0.5)), f'{sess_dir}: uncurated ROIs present'
assert iscell.shape[0] == F.shape[0]
```

iii. The notes justify this by saying the distributed files are already the track2p-matched, suite2p-curated cells, so repeating the original track2p cross-day filtering is unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns the data to recording-session start, not to a stimulus or behavioral event. Trials are simply consecutive 60 s windows from the beginning of the continuous recording.

ii.
```python
'metadata': {
    'temporal_alignment_event': (
        'start of the recording session (first two-photon imaging frame); trials are '
        'consecutive non-overlapping 60 s blocks of the continuous recording'),
    'off_start': 0.0,
    'off_end': float(TRIAL_SECONDS),
}
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
```

iii. The notes say there is no trial/event alignment in the source experiment, so session start is the only meaningful global alignment event once 60 s trials are imposed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the neural and behavioral streams by averaging every 10 frames, so the stored data are at 3 Hz with a nominal bin size of 333.33 ms.

ii.
```python
BIN_FRAMES = 10
...
def bin_trace(x, nframes_per_bin=BIN_FRAMES):
    T = x.shape[-1] // nframes_per_bin * nframes_per_bin
    newshape = x.shape[:-1] + (T // nframes_per_bin, nframes_per_bin)
    return x[..., :T].reshape(newshape).mean(axis=-1)
...
dF_binned = bin_trace(dF).astype(np.float32)
me_binned = bin_trace(me_frames)
...
'time_bin_size': float(BIN_FRAMES / results[0]['fs'] * 1000.0)
```

iii. The justification in the notes is directly from the methods text: the paper says decoding analyses averaged “10 consecutive timestamps,” so the AI applies the same binning to both streams.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In the actual code, the time input is not loaded from a raw timestamp array. It is computed from the bin index and `ops['fs']`, using the center of each 10-frame bin. The notes loosely describe this as coming from “frame index / `tstamps.npy`,” but `tstamps.npy` is not used for the input computation.

ii.
```python
fs = float(ops['fs'])
...
bin_centre_frames = np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0
time_s = (bin_centre_frames / fs).astype(np.float32)
```

iii. The notes justify this as the required decoder input: elapsed time from the beginning of the session. They also explicitly prefer bin-center timestamps so the time input sits “in the middle of the raw samples.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI creates one time value per binned sample, using bin centers in seconds, casts the result to `float32`, and then slices that 1-D session-long vector into 60 s trial segments.

ii.
```python
bin_centre_frames = np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0
time_s = (bin_centre_frames / fs).astype(np.float32)
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    input_trials.append(time_s[sl][None, :].copy())
```

iii. The notes justify this as a straightforward transformation from constant frame rate to elapsed seconds, and specifically defend the bin-center convention as avoiding a temporal offset relative to the averaged neural bins.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The input time vector is aligned bin-for-bin with the neural data because both are created on the same binned session grid and then cut with the same trial slices.

ii.
```python
nbins = dF_binned.shape[1]
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(time_s[sl][None, :].copy())
    output_trials.append(me_labels[sl][None, :].copy())
```

iii. The notes justify this by saying the time input is continuous across the whole session and should therefore share the exact same bin boundaries as the neural and output streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from `move_deve/motion_energy_glob.npy`, with `move_deve/tstamps.npy` used to reconstruct missing camera frames and align the behavior to the imaging frame grid.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
me_frames, n_interp = align_motion_energy(me_raw, ts, nframes)
```

iii. The notes justify this by saying `motion_energy_glob.npy` is the paper’s precomputed behavior signal, while `tstamps.npy` provides the frame-timing information needed to recover dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first maps motion-energy samples onto the two-photon frame grid. If camera and imaging frame counts differ, it estimates missing positions from rounded timestamp gaps; then it fills missing positions by linear interpolation. It also forces the first sample to `NaN` and interpolates that because the first motion-energy value is defined as 0 only because there is no previous frame. After alignment, it averages the trace into 10-frame bins and then discretizes it.

ii.
```python
def align_motion_energy(me, ts, nframes):
    me = np.asarray(me, dtype=np.float64)
    ncam = me.size
    if ncam == nframes:
        idx = np.arange(nframes)
    else:
        ifi = np.diff(np.asarray(ts, dtype=np.float64))
        med = np.median(ifi)
        steps = np.maximum(np.round(ifi / med).astype(np.int64), 1)
        idx = np.concatenate([[0], np.cumsum(steps)])

    full = np.full(nframes, np.nan)
    keep = idx < nframes
    full[idx[keep]] = me[keep]
    full[0] = np.nan
    ...
    full[nanmask] = np.interp(np.flatnonzero(nanmask), np.flatnonzero(good), full[good])
    return full, int(nanmask.sum())
...
me_frames, n_interp = align_motion_energy(me_raw, ts, nframes)
me_binned = bin_trace(me_frames)
me_labels, quant_edges = discretize_quantiles(me_binned, N_OUTPUT_BINS)
```

iii. The notes justify this in three ways: the camera is hardware-triggered to the imaging system, dropped frames are recoverable from `tstamps.npy`, and the first motion-energy sample is “meaningless” because it has no preceding video frame.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes each session’s binned motion-energy trace into 5 equal-percentile bins using session-specific quantile thresholds.

ii.
```python
def discretize_quantiles(x, nbins=N_OUTPUT_BINS):
    edges = np.quantile(x, np.arange(1, nbins) / nbins)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
...
me_labels, quant_edges = discretize_quantiles(me_binned, N_OUTPUT_BINS)
```

iii. The notes explicitly justify per-session quintiles as matching the decoder task requirement for 5 equal-percentile bins while respecting session-to-session scale changes in motion energy.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to the neural data on the imaging-frame grid before binning. The AI assumes identity alignment when the camera and imaging frame counts already match; otherwise it uses timestamp gaps to infer where camera frames are missing, interpolates onto the full imaging grid, bins both streams in the same 10-frame windows, and then applies the same per-trial slices.

ii.
```python
me_frames, n_interp = align_motion_energy(me_raw, ts, nframes)
me_binned = bin_trace(me_frames)
...
nbins = dF_binned.shape[1]
assert me_binned.size == nbins
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    output_trials.append(me_labels[sl][None, :].copy())
```

iii. The notes justify this by citing the dataset README: the camera was hardware-triggered by the microscope, so frame-wise alignment is the default and timestamp-based interpolation is only needed for dropped video frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI’s main missing-data handling is for behavior, not neural data. Missing camera frames are inferred from timestamp gaps and linearly interpolated. The first motion-energy sample is also treated as undefined and interpolated. The script additionally uses assertions to catch malformed sessions (`F.shape` vs `ops['nframes']`, `iscell` length, finite trial arrays, label ranges). There is no special imputation for neural values.

ii.
```python
assert F.shape[1] == nframes, f'{sess_dir}: F has {F.shape[1]} frames, ops says {nframes}'
assert np.all((iscell[:, 0] == 1) | (iscell[:, 1] > 0.5)), f'{sess_dir}: uncurated ROIs present'
...
full = np.full(nframes, np.nan)
...
full[0] = np.nan
...
full[nanmask] = np.interp(np.flatnonzero(nanmask), np.flatnonzero(good), full[good])
...
assert me_binned.size == nbins
...
assert np.isfinite(r['neural'][tr]).all()
assert np.isfinite(r['input'][tr]).all()
```

iii. The notes say the dataset’s main inconsistency is dropped camera frames, that the decoder cannot accept `NaN`, and that interpolation is the cleanest way to preserve alignment while still using the full recording.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies baseline correction of the fluorescence traces as the dominant cost. It also records smaller timing buckets for file loading, neural binning, and behavior processing.

ii.
```python
timings = {}
...
timings['load'] = time.time() - t0
...
dF = f_processing(...)
timings['dff'] = time.time() - t
...
dF_binned = bin_trace(dF).astype(np.float32)
timings['bin_neural'] = time.time() - t
...
me_frames, n_interp = align_motion_energy(me_raw, ts, nframes)
me_binned = bin_trace(me_frames)
timings['behaviour'] = time.time() - t
```

iii. In the notes, the AI explicitly says the dF baseline filtering dominates runtime and justifies multiprocessing primarily on that basis.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the heaviest operations (`bin_trace`, timestamp-to-frame reconstruction inside `align_motion_energy`, and multiprocessing across sessions). The main remaining Python loops are the per-trial slicing loop in `process_session()`, the small plotting loops, and the per-session/per-trial sanity-check loops in `main()`.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(time_s[sl][None, :].copy())
    output_trials.append(me_labels[sl][None, :].copy())
...
for i in range(nshow):
    ax[0].plot(...)
...
for si, r in enumerate(results):
    for tr in range(len(r['neural'])):
        ...
```

iii. The notes do not call out these leftover loops directly; instead they justify the current implementation by saying the expensive parts were already vectorized and parallelized, and that baseline filtering still dominates.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several small aggregations and derived computations: trial outputs are concatenated to compute class fractions, all trials are concatenated again inside `plot_processing()`, subject metadata are recomputed during `session_info` assembly, and per-trial shape/finiteness checks iterate back over the already-built trial lists.

ii.
```python
'class_fractions': np.bincount(np.concatenate([o.ravel() for o in output_trials]),
                               minlength=N_OUTPUT_BINS).tolist(),
...
allneural = np.concatenate(res['neural'], axis=1)
...
out_all = np.concatenate([o.ravel() for o in res['output']])
...
'day_index_within_subject': sum(1 for q in results[:i] if q['subject'] == r['subject']),
...
for si, r in enumerate(results):
    for tr in range(len(r['neural'])):
        ...
```

iii. The notes mostly justify these repetitions as diagnostics and documentation. The trajectory emphasizes validation and sanity checks, so the repeated passes appear intentional rather than accidental.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only used for logging, plotting, or metadata and are not consumed by the downstream decoder: timing measurements, `quantile_edges`, `class_fractions`, the optional plotting pipeline (including the z-scored raster), and the large `session_info` metadata block. The decoder only needs the converted arrays plus a few basic metadata fields.

ii.
```python
timings = {}
...
'quantile_edges': quant_edges.tolist(),
'class_fractions': np.bincount(...).tolist(),
'timings': timings,
'total_time': time.time() - t0,
...
if show_processing:
    plot_processing(res, F, dF, me_raw, me_frames, me_binned, me_labels, quant_edges, time_s)
...
zs = (allneural - allneural.mean(axis=1, keepdims=True)) / (allneural.std(axis=1, keepdims=True) + 1e-9)
...
'session_info': [
    {'subject': r['subject'], ... 'motion_energy_quintile_edges': r['quantile_edges'], ...}
    for i, r in enumerate(results)],
```

iii. The notes justify this extra work as part of a deliberately heavy validation/documentation process: the AI wanted visual checks, timing breakdowns, and detailed session metadata to support its claim that the conversion matched the paper.
