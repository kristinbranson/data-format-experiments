# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the `/app/data` directory for subdirectories starting with `jm` (subjects), then within each subject finds session directories that contain a `suite2p` subfolder. For each session, it loads `F.npy` and `ops.npy` (and `iscell.npy` for validation) from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. Unlike the human reference, it does NOT load `Fneu.npy` (because it uses neuropil coefficient 0) and uses `tstamps.npy` instead of `interframe_int.npy` for dropped frame detection.

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

# In process_session:
F = np.load(os.path.join(p0, 'F.npy'))
ops = np.load(os.path.join(p0, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(p0, 'iscell.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
```

iii. The AI justifies loading `ops.npy` to read session-specific parameters (fs, nframes, baseline settings) and `iscell.npy` to assert that all ROIs are valid tracked cells. The use of `tstamps.npy` (instead of `interframe_int.npy`) is justified as providing the full camera timestamps needed to reconstruct dropped frame positions via cumulative IFI ratios.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data root, sorted alphabetically. This matches the reference approach.

ii.
```python
for subject in sorted(os.listdir(DATA_ROOT)):
    subj_dir = os.path.join(DATA_ROOT, subject)
    if not os.path.isdir(subj_dir) or not subject.startswith('jm'):
        continue
```

iii. The `jm*` prefix convention is consistent across the dataset. The AI also maintains a `SUBJECT_LETTER` mapping to the paper's mouse identifiers (A-F).

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder that contain a `suite2p` subfolder, sorted alphabetically (chronologically by date). This adds a validation check compared to the reference, which includes all subdirectories.

ii.
```python
for sdir in sorted(glob.glob(os.path.join(subj_dir, '*'))):
    if os.path.isdir(sdir) and os.path.isdir(os.path.join(sdir, 'suite2p')):
        sessions.append((subject, os.path.basename(sdir), sdir))
```

iii. Checking for the `suite2p` subfolder ensures only valid recording sessions are included.

## 1-d. How are the data split into trials?

i. Trials are consecutive, non-overlapping 60-second blocks of the continuous recording. At 30 Hz with 10-frame bins, each trial is 180 bins. Sessions of 36000 frames yield 20 trials and sessions of 54000 frames yield 30 trials. Since both are exact multiples of 1800 frames, no data is discarded. This matches the reference approach.

ii.
```python
bins_per_trial = int(round(TRIAL_SECONDS * fs / BIN_FRAMES))    # 180
ntrials = nbins // bins_per_trial
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(time_s[sl][None, :].copy())
    output_trials.append(me_labels[sl][None, :].copy())
```

iii. The 60-second trial duration is prescribed by the Decoder Task. The paper has no natural trial structure (continuous recordings), so fixed-length segmentation is the appropriate approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All trials from all sessions are included. This matches the reference.

ii. N/A (no filtering code)

iii. The paper performs no trial-based quality control (continuous recordings with no stimulus-driven trials). The data README mentions only dropped camera frames, which are handled by interpolation rather than trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `F.npy` only (raw fluorescence from suite2p). Unlike the reference, `Fneu.npy` (neuropil fluorescence) is NOT loaded or used because the AI sets `neucoeff=0` (no neuropil subtraction). The AI also loads `ops.npy` for processing parameters.

ii.
```python
F = np.load(os.path.join(p0, 'F.npy'))
ops = np.load(os.path.join(p0, 'ops.npy'), allow_pickle=True).item()
# ...
dF = f_processing(F.astype(np.float32), fs=fs,
                  baseline=ops.get('baseline', 'maximin'),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
```

iii. The AI justifies using only `F.npy` because the reference track2p code (`F_processing` in `data_management.py`) defaults to `neucoeff=0`, meaning no neuropil subtraction. The AI reads processing parameters from each session's `ops.npy` rather than hardcoding them.

## 2-b. How is the `neural` data processed?

i. The AI applies baseline correction using the "maximin" method reimplemented with scipy: gaussian filter (sigma=10 frames), then minimum filter (60s window), then maximum filter (60s window), then subtracts this baseline from F. Critically, no neuropil subtraction is applied (`neucoeff=0`). This differs from the reference which uses `neucoeff=0.7` with suite2p's `dcnv.preprocess`.

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
    # ...
    return Fc - Flow
```

iii. The AI justifies this as a "verbatim re-implementation" of the track2p reference code's `F_processing`, which uses `neucoeff=0` by default. The CONVERSION_NOTES state: "dF/F used in the paper = suite2p baseline-corrected fluorescence with default parameters ... and no neuropil subtraction, exactly as F_processing."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied beyond what was already done upstream. The AI asserts that all ROIs in the data pass `iscell` checks (classifier prob > 0.5) and are track2p-tracked cells. This matches the reference.

ii.
```python
assert np.all((iscell[:, 0] == 1) | (iscell[:, 1] > 0.5)), f'{sess_dir}: uncurated ROIs present'
assert iscell.shape[0] == F.shape[0]
```

iii. The released data already contains only iscell>0.5 ROIs tracked across all days (verified by assertions). The paper applies no further neuron filtering beyond this.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the recording session. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. This matches the reference.

ii.
```python
'temporal_alignment_event': (
    'start of the recording session (first two-photon imaging frame); trials are '
    'consecutive non-overlapping 60 s blocks of the continuous recording'),
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing from 30 Hz to 3 Hz (333.33 ms per bin). This matches the reference exactly.

ii.
```python
BIN_FRAMES = 10
def bin_trace(x, nframes_per_bin=BIN_FRAMES):
    T = x.shape[-1] // nframes_per_bin * nframes_per_bin
    newshape = x.shape[:-1] + (T // nframes_per_bin, nframes_per_bin)
    return x[..., :T].reshape(newshape).mean(axis=-1)
# ...
dF_binned = bin_trace(dF).astype(np.float32)
me_binned = bin_trace(me_frames)
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." Both streams are binned before discretization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the bin index and the known frame rate/bin size. It is not derived from any raw data variable directly. This matches the reference.

ii.
```python
bin_centre_frames = np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0
time_s = (bin_centre_frames / fs).astype(np.float32)
```

iii. Since the frame rate is constant at 30 Hz, computing time from bin indices is equivalent to using timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes the time at the **center** of each bin: `t = (10*bin + 4.5) / 30` seconds. This differs from the reference, which uses the **left edge** of each bin: `t = (bin * 10) / 30`. The result is a constant offset of 0.15 seconds (half a bin width).

ii.
```python
# AI code (bin center):
bin_centre_frames = np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0
time_s = (bin_centre_frames / fs).astype(np.float32)
# First bin: (0*10 + 4.5)/30 = 0.15s

# Reference code (left edge):
# t = ((s + np.arange(trial_frames)) * BIN_FRAMES / FS).astype(np.float32)
# First bin: (0 * 10)/30 = 0.0s
```

iii. The AI chose bin centers as a more physically meaningful representation of the time within a bin (the averaged signal represents the center of the averaging window). The CONVERSION_NOTES document the time range as `[0.15, 1199.85]` or `[0.15, 1799.85]` seconds.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time and neural data are inherently aligned since both are indexed by the same bin indices. Each bin's time value corresponds to the same temporal window as the neural data for that bin. This matches the reference.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(time_s[sl][None, :].copy())
```

iii. Both time and neural data use the same slicing of the binned session data, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Camera timestamps from `tstamps.npy` are used to detect and handle dropped frames. This differs from the reference which uses `interframe_int.npy` instead of `tstamps.npy`.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
me_frames, n_interp = align_motion_energy(me_raw, ts, nframes)
```

iii. The motion energy file contains pre-computed global motion energy. The timestamps file provides camera frame times needed to identify dropped frames and align the video signal to the imaging frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Align camera frames to imaging frames using timestamps — when `n_cam != n_frames`, compute cumulative steps from IFI ratios relative to the median IFI to find which imaging frames each camera frame maps to; (2) treat the first motion energy sample (always 0, undefined) as NaN and linearly interpolate all missing/NaN values; (3) average into 10-frame bins; (4) discretize into 5 equal-percentile bins (quintiles) per session using `np.quantile` at [0.2, 0.4, 0.6, 0.8].

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
    full[0] = np.nan  # motion energy of the very first frame is undefined (=0)
    nanmask = np.isnan(full)
    if nanmask.any():
        good = ~nanmask
        full[nanmask] = np.interp(np.flatnonzero(nanmask), np.flatnonzero(good), full[good])
    return full, int(nanmask.sum())

def discretize_quantiles(x, nbins=N_OUTPUT_BINS):
    edges = np.quantile(x, np.arange(1, nbins) / nbins)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
```

iii. The AI explains that the first motion energy sample is always 0 by construction (no preceding frame to difference), so it is treated as undefined. The timestamp-based alignment uses the ratio of each inter-frame interval to the median to determine how many imaging frames elapsed between camera captures, which identifies exactly where dropped frames occurred.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles), with edges computed per session using `np.quantile`. Labels are assigned using `np.searchsorted`, producing integer labels 0-4. This is functionally equivalent to the reference's `np.percentile` + `np.digitize` approach.

ii.
```python
def discretize_quantiles(x, nbins=N_OUTPUT_BINS):
    edges = np.quantile(x, np.arange(1, nbins) / nbins)  # [0.2, 0.4, 0.6, 0.8]
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
```

iii. Per-session quintiles ensure approximately 20% of bins in each class, as required by the instructions ("five equal-percentile bins, selected per session").

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera is hardware-triggered by the microscope, so camera frame i corresponds to imaging frame i when no frames are dropped. When camera frames are fewer than imaging frames, the AI uses the camera timestamps to reconstruct the frame mapping via cumulative IFI ratios and linearly interpolates missing values. After alignment, both streams are binned identically and sliced into the same trials. The reference uses `interframe_int.npy` with a threshold of `dt * 1000 > 0.04` instead.

ii.
```python
# In align_motion_energy:
ifi = np.diff(np.asarray(ts, dtype=np.float64))
med = np.median(ifi)
steps = np.maximum(np.round(ifi / med).astype(np.int64), 1)
idx = np.concatenate([[0], np.cumsum(steps)])
# Then interpolate missing frames
# ...
# Both streams binned and sliced identically:
me_binned = bin_trace(me_frames)
assert me_binned.size == nbins  # same as dF_binned.shape[1]
```

iii. The timestamp-based approach is more robust than a fixed threshold because it adapts to the actual median inter-frame interval of each session. The AI validates alignment with the assertion that `me_binned.size == nbins`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of missing data are handled: (1) Dropped camera frames (9/41 sessions) are detected via timestamps and linearly interpolated; (2) The first motion energy sample (always 0, undefined) is treated as NaN and interpolated. Sessions are exact multiples of 1800 frames, so no incomplete trials are created. The reference handles dropped frames similarly but does not special-case the first sample.

ii.
```python
full[0] = np.nan  # motion energy of the very first frame is undefined (=0)
nanmask = np.isnan(full)
if nanmask.any():
    good = ~nanmask
    full[nanmask] = np.interp(np.flatnonzero(nanmask), np.flatnonzero(good), full[good])
# ...
assert me_binned.size == nbins
```

iii. Linear interpolation for dropped frames is recommended by the data README. The AI additionally identifies that `motion_energy_glob[0]` is always exactly 0 (no preceding frame to compute a difference), and treats it as missing to avoid biasing the lowest quintile.

## 6-a. What are the most time-consuming steps of the code?

i. The baseline correction (`f_processing`) is the most time-consuming step, taking 0.24-1.11 seconds per session depending on the number of neurons. The AI uses scipy filters (CPU) rather than suite2p's dcnv.preprocess (which can use GPU). The code uses multiprocessing (6 workers) to parallelize across sessions.

ii.
```python
# Timing tracked per session:
t = time.time()
dF = f_processing(F.astype(np.float32), fs=fs, ...)
timings['dff'] = time.time() - t
# ...
# Multiprocessing:
with Pool(nproc) as pool:
    results = pool.map(process_session, jobs, chunksize=1)
```

iii. The CONVERSION_NOTES report that the full conversion takes 6.8 seconds wall clock with 6 workers. The baseline filtering dominates because it involves sliding window operations over the full session length for every neuron.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-cutting loop iterates over trials to slice arrays, which could potentially be done with a single reshape operation (similar to the binning). However, since trial counts vary per session and the loop only runs 20-30 times, the impact is negligible.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(time_s[sl][None, :].copy())
    output_trials.append(me_labels[sl][None, :].copy())
```

iii. The AI notes in CONVERSION_NOTES that binning is already vectorized via reshape. The trial-cutting loop is the only remaining loop but it runs a small number of iterations.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once, and the multiprocessing architecture ensures each session's processing is independent.

ii. N/A

iii. The single-pass design (load -> process -> bin -> discretize -> cut trials) avoids redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and asserts `iscell.npy` for validation purposes, even though the data is already curated and no filtering is performed based on it. Additionally, the extensive metadata (session_info with detailed per-session statistics) goes beyond what the decoder uses but serves documentation purposes.

ii.
```python
iscell = np.load(os.path.join(p0, 'iscell.npy'))
assert np.all((iscell[:, 0] == 1) | (iscell[:, 1] > 0.5))
assert iscell.shape[0] == F.shape[0]
```

iii. The iscell loading is a sanity check to verify the data is pre-curated, not a processing step. The overhead is minimal.
