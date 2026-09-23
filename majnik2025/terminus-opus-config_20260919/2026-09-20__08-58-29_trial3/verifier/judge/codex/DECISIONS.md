# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by scanning every top-level directory under `/app/data`, then every session subdirectory within each subject directory. For each session it loads neural traces from `suite2p/plane0/F.npy`, cell labels from `iscell.npy`, and behavioral data from `move_deve/motion_energy_glob.npy` plus `interframe_int.npy`. Trials are not loaded directly from disk; they are created later by splitting each processed session into consecutive 60 s chunks.

ii.
```python
def list_sessions():
    """Return [(subject, session_name, session_dir), ...] sorted by subject then date."""
    out = []
    for subj in sorted(d for d in os.listdir(DATA_ROOT)
                       if os.path.isdir(os.path.join(DATA_ROOT, d))):
        subj_dir = os.path.join(DATA_ROOT, subj)
        for sess in sorted(d for d in os.listdir(subj_dir)
                           if os.path.isdir(os.path.join(subj_dir, d))):
            out.append((subj, sess, os.path.join(subj_dir, sess)))
    return out
```

```python
plane = os.path.join(sdir, 'suite2p', 'plane0')
F = np.load(os.path.join(plane, 'F.npy'))
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
md = os.path.join(sdir, 'move_deve')
me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI says the released dataset is organized as subject/session folders and that the session folders contain suite2p outputs plus motion-energy files. In the trajectory, it explicitly cites the data README and notebook as the basis for scanning subject folders and session subdirectories.

## 1-b. How are the data split into subjects?

i. Subjects are the distinct top-level directories returned by `list_sessions()`, sorted alphabetically. Later, the subject list is reconstructed from the loaded sessions as sorted unique subject IDs.

ii.
```python
subjects = sorted({s[0] for s in sessions})
...
data = {
    ...
    'subjects': subjects,
    'subject_idx': [],
    ...
}
```

iii. The AI’s notes say the released data has six subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) corresponding to mice A-F in the paper, and that subject folder names should populate `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each session is one session directory inside a subject folder. The AI sorts session directory names within each subject and treats each as one converted session.

ii.
```python
for subj in sorted(d for d in os.listdir(DATA_ROOT)
                   if os.path.isdir(os.path.join(DATA_ROOT, d))):
    subj_dir = os.path.join(DATA_ROOT, subj)
    for sess in sorted(d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))):
        out.append((subj, sess, os.path.join(subj_dir, sess)))
```

iii. The justification in the notes is that each recording day is stored as its own session subfolder and sorting preserves chronological order.

## 1-d. How are the data split into trials?

i. The AI treats each continuous session as a sequence of non-overlapping 60 s trials after temporal binning. At 30 Hz with 10-frame binning, each trial is 180 bins. It asserts that each session length is an exact multiple of 180 bins and slices the binned arrays into consecutive fixed-length segments.

ii.
```python
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FS))          # 1800
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES                # 180
...
nbins = dF_binned.shape[1]
...
ntrials = nbins // TRIAL_BINS
assert ntrials * TRIAL_BINS == nbins, (
    f'{subj}/{sess}: {nbins} bins is not a multiple of {TRIAL_BINS}')
...
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(t_bins[sl].astype(np.float32)[None, :])
    output_trials.append(me_labels[sl].astype(np.int64)[None, :])
```

iii. In `CONVERSION_NOTES.md`, the AI states there is no native trial structure in the recordings, so the decoder-task requirement is satisfied by cutting the continuous sessions into consecutive 60 s trials. It also notes that the public sessions are exact multiples of 20 or 30 minutes, so no partial trial handling is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter trials. Every 60 s segment is kept if the session can be split cleanly into 180-bin trials.

ii.
```python
ntrials = nbins // TRIAL_BINS
assert ntrials * TRIAL_BINS == nbins, (
    f'{subj}/{sess}: {nbins} bins is not a multiple of {TRIAL_BINS}')
...
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(t_bins[sl].astype(np.float32)[None, :])
    output_trials.append(me_labels[sl].astype(np.int64)[None, :])
```

iii. The notes say the released data is already curated and that all session lengths are exact multiples of the requested trial size, so there is no additional trial-quality rejection step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from `suite2p/plane0/F.npy` only. It also loads `iscell.npy` to assert all ROIs are already valid cells, but it does not load or use `Fneu.npy` in the actual conversion path.

ii.
```python
plane = os.path.join(sdir, 'suite2p', 'plane0')
F = np.load(os.path.join(plane, 'F.npy'))
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
dF = F_processing(F, None, fs=FS)
```

iii. The AI’s notes justify this by saying the released data already contains tracked, curated cells and that the track2p GUI’s `F_processing` defaults to `neucoeff=0.0`, so neuropil traces are unnecessary for the chosen dF computation. The trajectory explicitly states that neuropil subtraction with `neucoeff=0.7` was considered and rejected.

## 2-b. How is the `neural` data processed?

i. The AI computes baseline-corrected fluorescence using a local copy of the track2p GUI’s `F_processing` logic: optional neuropil subtraction, Gaussian smoothing, `minimum_filter1d`, `maximum_filter1d`, and subtraction of the resulting baseline. In practice it uses `neucoeff=0.0`, so `Fc` is just `F`. It then averages the result in non-overlapping 10-frame bins and stores those binned dF values as `neural`.

ii.
```python
def F_processing(F, Fneu=None, fs=FS, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    if neucoeff:
        Fc = F - neucoeff * Fneu
    else:
        Fc = F.astype(np.float32, copy=True)

    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    return Fc - Flow
```

```python
dF = F_processing(F, None, fs=FS)
dF_binned = bin_frames(dF).astype(np.float32)
```

iii. In the notes, the AI argues that this exactly matches the track2p GUI implementation used by the paper, and that the paper describes using baseline-corrected fluorescence with default Suite2p parameters. The trajectory shows it explicitly chose this route over using `spks.npy` or neuropil-subtracted traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied during conversion. Instead, the AI asserts that every ROI already has `iscell[:, 0] == 1` and relies on the released data already being restricted to track2p-tracked cells that passed Suite2p’s cell classifier.

ii.
```python
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
assert iscell.shape[0] == nneurons
assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'
```

iii. The notes say the released arrays already contain only cells tracked across all days and already filtered by Suite2p’s classifier threshold, so further filtering would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns the neural data to session start. There is no stimulus or task event; the continuous recording is simply cut into consecutive non-overlapping 60 s blocks beginning at time 0 of the session.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': (
        'Start of the imaging session; recordings are continuous and are cut into '
        'consecutive non-overlapping 60 s trials.'),
    'off_start': 0.0,
    'off_end': TRIAL_SECONDS,
    ...
}
```

iii. The notes repeatedly justify this by pointing out that the source experiment is spontaneous continuous recording with no natural trial event, so session start is the only sensible alignment anchor for the requested decoder format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame non-overlapping temporal bins. With a 30 Hz sampling rate, that is 333.33 ms per bin (3 Hz). Both neural and behavioral data are rebinned this way.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
...
def bin_frames(x, nbin=BIN_FRAMES):
    x = np.asarray(x)
    T = x.shape[-1]
    nb = T // nbin
    x = x[..., :nb * nbin]
    return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)
```

```python
'time_bin_size': 1000.0 * BIN_FRAMES / FS,
```

iii. The AI cites the paper’s decoding methods saying dF/F and behavior were averaged in bins of 10 consecutive timestamps, and uses that as the reason to bin both streams before trialization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not loaded from a raw timestamp array. It is derived from bin indices plus the assumed frame rate and bin size.

ii.
```python
nbins = dF_binned.shape[1]
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
```

iii. The AI’s justification in the notes is that the session runs at nominal 30 Hz and the decoder input requested by the task is elapsed time from session start, so computing it from frame/bin index is sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one time value per 10-frame bin using the bin center in seconds, not the left edge. It uses `(bin_start_frame + 4.5) / 30` for each bin, then converts to `float32`.

ii.
```python
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
...
input_trials.append(t_bins[sl].astype(np.float32)[None, :])
```

iii. The AI’s notes and sanity checks say this reflects the fact that neural and motion-energy values were averaged across 10-frame windows, so the input time should represent the center of each averaged bin.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time to the already binned neural data by creating exactly one time value per neural bin, slicing those time bins with the same per-trial slices used for `neural`, and storing them with matching `(1, 180)` trial shapes.

ii.
```python
sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
input_trials.append(t_bins[sl].astype(np.float32)[None, :])
```

iii. The notes say the input is strictly increasing in 1/3 s steps and that concatenating trial inputs reproduces the full-session timeline, which the AI used as an alignment sanity check.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `output` is derived from the per-session motion-energy trace in `move_deve/motion_energy_glob.npy` and the camera inter-frame intervals in `move_deve/interframe_int.npy`.

ii.
```python
md = os.path.join(sdir, 'move_deve')
me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The notes say motion energy is already precomputed from videography, and that `interframe_int.npy` is needed to locate missing camera frames whenever the motion-energy array is shorter than the imaging trace.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first reconstructs motion energy on the imaging-frame grid. If the motion-energy vector is shorter than the neural recording, it infers the true frame index of each camera sample from `interframe_int.npy` relative to the median interval, places the observed values onto the full imaging grid, marks missing locations as `NaN`, treats `me[0]` as missing as well, and fills missing values by linear interpolation. It then averages the motion-energy trace in non-overlapping 10-frame bins and discretizes the binned signal into 5 per-session quantile bins.

ii.
```python
def align_motion_energy(me, ifi, nframes):
    me = np.asarray(me, dtype=np.float64)
    if len(me) == nframes:
        idx = np.arange(nframes)
    else:
        med = np.median(ifi)
        steps = np.round(ifi / med).astype(np.int64)
        idx = np.concatenate([[0], np.cumsum(steps)])
        ...

    full = np.full(nframes, np.nan)
    full[idx] = me
    full[0] = np.nan
    bad = np.isnan(full)
    good = ~bad
    full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
    return full, int(bad.sum()), idx
```

```python
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
me_binned = bin_frames(me_full)
me_labels, edges = quantile_discretize(me_binned)
```

iii. The notes justify this as a more faithful implementation of the data README’s instruction to use `interframe_int.npy` to identify missing frames and either treat them as missing or interpolate over them. The trajectory also says the first sample is always zero because there is no preceding frame, so it should be treated as an artifact.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After 10-frame averaging, the AI discretizes motion energy into five equal-percentile bins within each session. It computes the 20th/40th/60th/80th percentile thresholds from that session’s binned motion-energy trace and assigns labels 0-4 using `np.searchsorted`.

ii.
```python
def quantile_discretize(x, nq=NQUANTILES):
    """Discretize x into nq equal-percentile bins (thresholds from this session)."""
    edges = np.quantile(x, np.arange(1, nq) / nq)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
```

iii. The notes explicitly say this follows the decoder-task requirement that motion energy be converted into five equal-percentile bins per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes the camera and microscope are hardware-synchronized, so camera frame `i` corresponds to imaging frame `i` when there are no drops. When frames are missing, it reconstructs sample indices from `interframe_int.npy`, expands the trace onto the full imaging frame grid, linearly interpolates missing points, and only then bins the behavior and neural streams identically.

ii.
```python
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
me_binned = bin_frames(me_full)
...
dF_binned = bin_frames(dF).astype(np.float32)
...
sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
output_trials.append(me_labels[sl].astype(np.int64)[None, :])
```

iii. The notes say the microscope triggers the camera, so the base alignment is frame-for-frame, and that dropped camera frames are the only exception. The AI uses its interpolation step and trial-concatenation plots as the main sanity checks for this alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing camera frames by reconstructing frame indices from `interframe_int.npy`, placing motion-energy samples on a full-length array, and interpolating gaps. It also treats `me[0]` as missing because it is an artifact of frame differencing. It asserts that all ROIs are already cells via `iscell`, and it asserts that each session can be tiled exactly into 60 s trials.

ii.
```python
if len(me) == nframes:
    idx = np.arange(nframes)
else:
    med = np.median(ifi)
    steps = np.round(ifi / med).astype(np.int64)
    idx = np.concatenate([[0], np.cumsum(steps)])
    assert idx[-1] == nframes - 1, (
        f'frame-index reconstruction failed: {idx[-1]} != {nframes - 1}')
    assert len(idx) == len(me)

full = np.full(nframes, np.nan)
full[idx] = me
full[0] = np.nan
bad = np.isnan(full)
good = ~bad
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
```

```python
assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'
...
assert ntrials * TRIAL_BINS == nbins, (
    f'{subj}/{sess}: {nbins} bins is not a multiple of {TRIAL_BINS}')
```

iii. The notes explicitly discuss missing camera frames, occasional timestamp glitches, and the `me[0]` artifact. The AI’s stated rationale is to repair obvious behavioral-stream problems while failing loudly on structural inconsistencies rather than silently discarding or misaligning data.

## 6-a. What are the most time-consuming steps of the code?

i. The AI treats neural baseline correction as the main cost. The Gaussian smoothing plus min/max baseline filters dominate runtime; file loading is secondary.

ii.
```python
t1 = time.time()
dF = F_processing(F, None, fs=FS)
t_dff = time.time() - t1
t1 = time.time()
dF_binned = bin_frames(dF).astype(np.float32)
t_bin = time.time() - t1
```

iii. In `CONVERSION_NOTES.md`, the AI reports that the baseline-correction step dominates per-session runtime and explicitly calls out `gaussian_filter` and the min/max filters as the bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s code is already mostly vectorized in the expensive parts. It uses vectorized frame binning, vectorized quantile discretization, and vectorized interpolation for missing motion-energy frames, avoiding the repeated `np.insert` loop used in the human reference. The remaining explicit loop is the per-trial assembly loop, which is simple bookkeeping rather than a heavy numeric bottleneck.

ii.
```python
def bin_frames(x, nbin=BIN_FRAMES):
    x = np.asarray(x)
    T = x.shape[-1]
    nb = T // nbin
    x = x[..., :nb * nbin]
    return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)
```

```python
bad = np.isnan(full)
good = ~bad
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
```

```python
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(t_bins[sl].astype(np.float32)[None, :])
    output_trials.append(me_labels[sl].astype(np.int64)[None, :])
```

iii. The notes claim vectorization was an explicit optimization goal, and the trajectory says the agent intentionally replaced loop-based motion-energy repair with a frame-grid reconstruction plus `np.interp`.

## 6-c. What processing does the code repeat multiple times?

i. There is little repeated heavy processing. The main repetition in the conversion path is per-session execution of the same neural preprocessing and motion-energy alignment pipeline, plus per-trial slicing of already processed session arrays. Optional plotting also concatenates trial outputs back to session-level traces for sanity checks, but that is not part of the saved deliverable.

ii.
```python
for i, (subj, sess, sdir) in enumerate(sessions):
    show = args.show_processing and i < 2
    neural, inp, out, info = process_session(subj, sess, sdir, show_processing=show)
    data['neural'].append(neural)
    data['input'].append(inp)
    data['output'].append(out)
```

```python
cat_out = np.concatenate([o[0] for o in output_trials])
cat_in = np.concatenate([i[0] for i in input_trials])
cat_neu = np.concatenate([n[0] for n in neural_trials])
```

iii. The notes emphasize that the expensive numeric operations are done once per session and then reused for trial slicing. The extra concatenations only exist inside the optional plotting/sanity-check path.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion includes several pieces of work that are useful for verification but not used by the downstream decoder itself: loading `iscell.npy` only to assert cell status, recording per-session metadata such as quantile edges and missing-frame counts, timing the pipeline, and optionally generating detailed diagnostic plots that reassemble trials into full-session views.

ii.
```python
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'
```

```python
info = {
    'subject': subj,
    'session': sess,
    'n_neurons': int(nneurons),
    'n_frames': int(nframes),
    'n_bins': int(nbins),
    'n_trials': int(ntrials),
    'n_missing_camera_frames': int(n_missing),
    'motion_energy_quantile_edges': edges.tolist(),
    'duration_s': float(nframes / FS),
}
```

```python
if show_processing:
    plot_processing(subj, sess, F, dF, dF_binned, me_raw, me_full, me_binned,
                    me_labels, edges, t_bins, sample_idx, nframes,
                    neural_trials, input_trials, output_trials)
```

iii. The AI’s notes describe these as sanity-check and documentation aids. They are retained in metadata or figures for auditability, but the decoder only consumes `neural`, `input`, `output`, subject/region indices, and high-level metadata.
