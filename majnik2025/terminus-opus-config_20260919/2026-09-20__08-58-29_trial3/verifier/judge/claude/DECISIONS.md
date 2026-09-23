# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All subject directories are discovered by listing the data root directory and filtering for subdirectories (no `startswith('jm')` filter, unlike the reference -- but all directories happen to match). Within each subject, session subdirectories are discovered and sorted. For each session, the code loads `suite2p/plane0/F.npy` (raw fluorescence), `suite2p/plane0/iscell.npy` (cell classification), `move_deve/motion_energy_glob.npy` (motion energy), and `move_deve/interframe_int.npy` (interframe intervals). Notably, `Fneu.npy` is NOT loaded because the AI chose `neucoeff=0.0` (no neuropil subtraction).

ii.
```python
def list_sessions():
    out = []
    for subj in sorted(d for d in os.listdir(DATA_ROOT)
                       if os.path.isdir(os.path.join(DATA_ROOT, d))):
        subj_dir = os.path.join(DATA_ROOT, subj)
        for sess in sorted(d for d in os.listdir(subj_dir)
                           if os.path.isdir(os.path.join(subj_dir, d))):
            out.append((subj, sess, os.path.join(subj_dir, sess)))
    return out

# In process_session:
F = np.load(os.path.join(plane, 'F.npy'))
iscell = np.load(os.path.join(plane, 'iscell.npy'))
me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The AI documented in CONVERSION_NOTES.md that subject directories contain session subfolders, each with suite2p output and motion energy files. The directory traversal pattern is standard for this dataset layout.

## 1-b. How are the data split into subjects?

i. Subjects correspond to top-level directories in the data root, sorted alphabetically. No `startswith('jm')` filter is applied, but all directories in the data root are subject directories.

ii.
```python
for subj in sorted(d for d in os.listdir(DATA_ROOT)
                   if os.path.isdir(os.path.join(DATA_ROOT, d))):
```

iii. The CONVERSION_NOTES document that there are 6 subject directories (jm031 through jm046), consistent with the paper's report of 6 mice.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, sorted alphabetically (which gives chronological order since folders are named YYYY-MM-DD_a).

ii.
```python
for sess in sorted(d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))):
    out.append((subj, sess, os.path.join(subj_dir, sess)))
```

iii. The CONVERSION_NOTES document 41 total sessions (7,7,7,7,6,7 per subject), consistent with the data.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive non-overlapping 60-second segments. After 10-frame binning (30 Hz to 3 Hz), each trial is 180 bins. The code asserts that the total number of bins is an exact multiple of 180 (no remainder frames are discarded).

ii.
```python
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES  # 180
ntrials = nbins // TRIAL_BINS
assert ntrials * TRIAL_BINS == nbins, (
    f'{subj}/{sess}: {nbins} bins is not a multiple of {TRIAL_BINS}')
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
```

iii. The CONVERSION_NOTES state that sessions are exact multiples of 1800 frames (36000 or 54000), so no data is lost. The assertion verifies this.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All trials from all sessions are included.

ii. N/A (no filtering code)

iii. The CONVERSION_NOTES state there is no natural trial structure and no basis for quality-based trial exclusion. The paper's recordings are continuous spontaneous activity with no stimulus-driven trial structure.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived solely from `suite2p/plane0/F.npy` (raw fluorescence). The AI chose `neucoeff=0.0` (following the track2p GUI's `F_processing` defaults), so `Fneu.npy` is not used.

ii.
```python
F = np.load(os.path.join(plane, 'F.npy'))
# ...
dF = F_processing(F, None, fs=FS)  # neucoeff=0.0, Fneu not needed
```

iii. The CONVERSION_NOTES document that the track2p GUI's `F_processing` function has `neucoeff=0.0` as its default, meaning neuropil is not subtracted for the dF/F used in analyses. The AI explicitly investigated the discrepancy between `ops.npy` (`neucoeff=0.7`, used by suite2p for deconvolution) and the GUI function default (`neucoeff=0.0`), and chose to follow the reference code.

## 2-b. How is the `neural` data processed?

i. The AI reimplemented the track2p GUI's `F_processing` function using scipy filters: Gaussian smoothing (sigma=10 frames) followed by minimum filter (window=1800 frames = 60s) and maximum filter (same window) to compute the baseline (`Flow`), which is subtracted from `F` to produce baseline-corrected dF. No neuropil subtraction is applied (`neucoeff=0.0`). The result is then averaged in non-overlapping 10-frame bins.

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
    return Fc - Flow

# Called as:
dF = F_processing(F, None, fs=FS)
dF_binned = bin_frames(dF).astype(np.float32)
```

iii. The CONVERSION_NOTES state this function is "copied verbatim from the reference `track2p/gui/data_management.py::DataManagement.F_processing`" and that with `neucoeff=0.0` the neuropil trace is unused. The paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. However, the code explicitly loads `iscell.npy` and asserts that all ROIs are classified as cells (`iscell[:,0]==1`), verifying that the released data already contains only curated cells.

ii.
```python
iscell = np.load(os.path.join(plane, 'iscell.npy'))
assert iscell.shape[0] == nneurons
assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'
```

iii. The CONVERSION_NOTES explain that the released data already contains only suite2p-classified cells (prob > 0.5) tracked across all days by track2p, so no further filtering is needed. The assertion serves as a sanity check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since the recordings are continuous and trials are contiguous 60-second segments starting from t=0, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': (
    'Start of the imaging session; recordings are continuous and are cut into '
    'consecutive non-overlapping 60 s trials.'),
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The CONVERSION_NOTES document that there is no stimulus event to align to; the recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged in non-overlapping bins of 10 frames, reducing the sampling rate from 30 Hz to 3 Hz (333.33 ms time bins). This matches the paper's decoding preprocessing.

ii.
```python
BIN_FRAMES = 10
def bin_frames(x, nbin=BIN_FRAMES):
    x = np.asarray(x)
    T = x.shape[-1]
    nb = T // nbin
    x = x[..., :nb * nbin]
    return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)

# Applied to both streams:
dF_binned = bin_frames(dF).astype(np.float32)
me_binned = bin_frames(me_full)
```

iii. The CONVERSION_NOTES cite the paper's Methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index within the session and the known frame rate/bin size. The values represent bin centers (not left edges), named `time_from_session_start_s`.

ii.
```python
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
```

iii. The CONVERSION_NOTES state that since the frame rate is constant at 30 Hz, computing time from bin indices is equivalent to reading timestamps from the data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A simple arithmetic calculation: for each bin `b`, the time is `(b * 10 + 4.5) / 30` seconds, representing the center of the bin. The first bin has time 0.15s, and the last bin of a 30-minute session has time ~1799.82s. Time runs continuously across trials within a session.

ii.
```python
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
# Then sliced per trial:
input_trials.append(t_bins[sl].astype(np.float32)[None, :])
```

iii. No additional justification given beyond stating it gives seconds from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is derived from the same bin indices as the neural data, so alignment is inherent. Each time bin center corresponds exactly to the average time of the 10 frames that were averaged for that neural data bin.

ii. (Same slicing is used for neural and time: `t_bins[sl]` and `dF_binned[:, sl]`)

iii. Alignment is guaranteed by construction since both use the same bin indexing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy` (global motion energy from behavioral video) and `move_deve/interframe_int.npy` (interframe intervals used to identify and handle dropped camera frames).

ii.
```python
me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The CONVERSION_NOTES document that motion energy is a pre-computed global metric from the behavioral video, and the interframe intervals are needed to identify dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) Motion energy samples are placed on the imaging frame grid by reconstructing frame indices from interframe intervals (median/round/cumsum), (2) me[0]=0 artifact (no preceding frame for difference) is treated as missing and interpolated, (3) all missing/dropped frames are linearly interpolated, (4) the aligned trace is averaged in 10-frame bins, then (5) discretized into 5 equal-percentile (quintile) bins per session using `np.quantile` + `np.searchsorted`.

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
    full = np.full(nframes, np.nan)
    full[idx] = me
    full[0] = np.nan  # first-frame artifact (me[0] == 0)
    bad = np.isnan(full)
    good = ~bad
    full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
    return full, int(bad.sum()), idx

def quantile_discretize(x, nq=NQUANTILES):
    edges = np.quantile(x, np.arange(1, nq) / nq)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
```

iii. The CONVERSION_NOTES explain that the camera is hardware-triggered by the microscope, so frame i of the video corresponds to imaging frame i. When camera frames are dropped, the true frame index is reconstructed from inter-frame intervals. The me[0]=0 artifact is documented as "no preceding video frame to difference against."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) per session. Thresholds are the 20th, 40th, 60th, and 80th percentiles of the binned motion energy within each session. Values are assigned to bins 0-4 using `np.searchsorted` with `side='right'`.

ii.
```python
def quantile_discretize(x, nq=NQUANTILES):
    edges = np.quantile(x, np.arange(1, nq) / nq)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
```

iii. The CONVERSION_NOTES document that quintile thresholds are computed within each session, as specified in the Decoder Task instructions ("discretized into five equal-percentile bins, selected per session").

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera is hardware-triggered by the microscope, giving a 1:1 correspondence between camera and imaging frames. When camera frames are dropped (motion energy array shorter than neural frames), frame indices are reconstructed from interframe intervals and missing values are linearly interpolated. After alignment, both streams are binned identically (10-frame averages), ensuring temporal alignment is maintained through to the trial-level output.

ii.
```python
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
me_binned = bin_frames(me_full)  # same binning as neural
```

iii. The CONVERSION_NOTES detail the alignment approach and document that the cumulative sum of rounded interframe intervals correctly reconstructs frame indices for all sessions with dropped frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of data issues are handled: (1) Dropped camera frames are detected via interframe interval analysis and linearly interpolated. (2) The me[0]=0 artifact (motion energy at frame 0 is always 0 because there is no preceding frame) is treated as missing and interpolated. (3) An assertion verifies that session lengths are exact multiples of the trial length (no partial trials need to be discarded). Additionally, `iscell.npy` is checked to verify all ROIs are classified as cells.

ii.
```python
# Dropped frames + me[0] artifact:
full[0] = np.nan  # first-frame artifact
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])

# Exact division assertion:
assert ntrials * TRIAL_BINS == nbins

# Cell classification check:
assert np.all(iscell[:, 0] == 1)
```

iii. The CONVERSION_NOTES document 9 of 41 sessions with 1-148 missing camera frames, all successfully handled by the interpolation approach. The me[0] artifact is documented as a known property of the motion energy computation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction in `F_processing`, which applies Gaussian smoothing, minimum filtering, and maximum filtering over the full session length for every neuron using scipy. The code includes timing instrumentation that reports per-session load/dff/bin times.

ii.
```python
if verbose:
    print(f'[load {t_load:.1f}s dff {t_dff:.1f}s bin {t_bin:.1f}s total {time.time()-t0:.1f}s]')
```

iii. The CONVERSION_NOTES report approximately 0.2-0.7s per session for the baseline correction, depending on session size.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop is the per-trial slicing loop, which iterates over trials to extract slices from pre-computed arrays. This is already efficient since it only performs array slicing. There are no computationally expensive loops that need vectorization -- the heavy operations (filtering, binning) are already vectorized.

ii.
```python
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
```

iii. The CONVERSION_NOTES note that binning and discretization are vectorized, and the per-session processing time is already fast (~0.8s/session).

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once, and the discretization is done in a single pass per session.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `iscell.npy` for every session solely to assert that all ROIs are classified as cells -- this is a sanity check rather than processing, and could be omitted after initial verification. The `plot_processing` function generates 8-panel figures when `--show-processing` is enabled, which is only used during development. No significant unnecessary computation is performed.

ii. N/A

iii. N/A
