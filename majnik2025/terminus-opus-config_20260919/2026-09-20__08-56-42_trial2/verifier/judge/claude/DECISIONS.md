# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as sorted directories starting with `jm` in the data root. Sessions are sorted subdirectories within each subject. For each session, the agent loads `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. The agent also pre-scans all sessions of a mouse to identify zero-trace neurons before processing.

ii.
```python
def list_sessions(subject):
    return sorted(p for p in glob.glob(os.path.join(DATA_ROOT, subject, '*'))
                  if os.path.isdir(p))

def load_ops(session_dir):
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'),
                   allow_pickle=True).item()

def load_traces(session_dir):
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    return F, Fneu, iscell
```

iii. The agent follows the data organization documented in the data README and `load_data.ipynb`. Loading `ops.npy` allows reading the actual processing parameters (fs, neucoeff, baseline method) used by the authors rather than hardcoding them. Loading `iscell.npy` enables asserting that all ROIs are classified as cells.

## 1-b. How are the data split into subjects?

i. Subjects correspond to sorted directories starting with `jm` in the data root directory.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))
```

iii. Each `jm*` directory represents one mouse. Sorting ensures deterministic order. Six subjects are found: jm031-jm046.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject directory, each corresponding to one daily recording.

ii.
```python
def list_sessions(subject):
    return sorted(p for p in glob.glob(os.path.join(DATA_ROOT, subject, '*'))
                  if os.path.isdir(p))
```

iii. Each subdirectory (named by date, e.g. `2023-10-18_a`) contains the suite2p output and motion energy files for one imaging session. Sorting by name gives chronological order.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as consecutive, non-overlapping 60-second blocks of the continuous recording, yielding 180 bins per trial (60s * 30Hz / 10 frames per bin). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
bin_sec = BIN_SIZE / fs
bins_per_trial = int(round(TRIAL_SEC / bin_sec))
ntrials = nbins // bins_per_trial
nused = ntrials * bins_per_trial
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. The 60-second trial duration is specified in the task instructions. Since 36000 and 54000 frames divide exactly into 1800-frame (60s) blocks after 10-frame binning, zero frames are actually discarded.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second blocks are retained.

ii. N/A

iii. The paper analyses whole continuous recordings with no trial structure, so there are no trial-level quality criteria to apply.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (processing parameters including fs, neucoeff, baseline method) from `suite2p/plane0/`.

ii.
```python
ops = load_ops(session_dir)
fs = float(ops['fs'])
F, Fneu, iscell = load_traces(session_dir)
```

iii. These are the standard suite2p output files. The `ops.npy` file stores the exact parameters used by the authors in their suite2p processing pipeline.

## 2-b. How is the `neural` data processed?

i. Four processing steps: (1) neuropil subtraction (`Fc = F - 0.7 * Fneu`), (2) maximin baseline correction (gaussian smoothing with sigma=10, then minimum filter with 60s window, then maximum filter), (3) averaging into 10-frame bins (333.33 ms), (4) z-scoring each neuron across the entire session.

ii.
```python
def compute_dff(F, Fneu, fs, neucoeff=0.7, baseline='maximin',
                sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    return Fc - Flow

dff = compute_dff(F, Fneu, fs, neucoeff=float(ops.get('neucoeff', 0.7)), ...)
dff_binned = bin_average(dff)
neural = zscore_rows(dff_binned).astype(np.float32)
```

iii. The dF/F computation is copied from `track2p/gui/data_management.py::F_processing`, which is identical to suite2p's `dcnv.preprocess`. Parameters are read from `ops.npy` (neucoeff=0.7, baseline='maximin', sig_baseline=10, win_baseline=60). The 10-frame binning matches the paper's Methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps". Z-scoring is taken from the track2p raster preprocessing (`zscore_all_f_t2p` in `raster_wd.py`), which normalizes each neuron for visualization and analysis.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps: (1) Assert all ROIs have `iscell[:,0] == 1` (already applied upstream by the authors). (2) Remove neurons with an identically-zero F trace on any session of a mouse, applied consistently across all sessions of that mouse to maintain the tracked population structure.

ii.
```python
def find_zero_neurons(subject_sessions):
    bad = set()
    for sd in subject_sessions:
        F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        z = np.where(~np.asarray(F).any(axis=1))[0]
        bad.update(z.tolist())
    return sorted(bad)

# In process_session:
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in released data'
keep = np.ones(F.shape[0], dtype=bool)
if drop_neurons is not None:
    keep[drop_neurons] = False
F, Fneu = F[keep], Fneu[keep]
```

iii. The iscell assertion verifies the released data is already curated (suite2p probability > 0.5). Zero-trace neuron removal mirrors `rem_zero_rows` in the reference raster preprocessing code, which removes neurons that would produce NaN after z-scoring. Five neurons across 4 mice are affected (8 neuron-sessions total).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Trial k starts at k*60 seconds after the beginning of the session. There is no event-based alignment since the recording is continuous with no stimulus events.

ii.
```python
data['metadata'] = dict(
    ...
    temporal_alignment_event=('start of each 60 s block of the continuous recording '
                              '(block k starts at k*60 s after session onset); '
                              'recordings are continuous, there are no behavioural trials'),
    off_start=0.0,
    off_end=TRIAL_SEC,
    ...
)
```

iii. The paper describes continuous recordings with no stimulus or task. The 60-second trial blocks are artificial divisions for the decoder, aligned to session onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, taking the 30 Hz sampling rate to 3 Hz (333.33 ms bins). This rebinning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10
def bin_average(x, bin_size=BIN_SIZE):
    x = np.asarray(x, dtype=np.float64)
    n = (x.shape[-1] // bin_size) * bin_size
    newshape = x.shape[:-1] + (n // bin_size, bin_size)
    return x[..., :n].reshape(newshape).mean(axis=-1)

# Applied to both streams:
dff_binned = bin_average(dff)
me_binned = bin_average(me)

# Metadata:
time_bin_size=1000.0 * BIN_SIZE / 30.0  # 333.33 ms
```

iii. The paper's Methods state: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps". Both streams are binned together before discretization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and the known frame rate and bin size: `time = (bin_index + 0.5) * (BIN_SIZE / fs)`, giving seconds from session start at the center of each time bin.

ii.
```python
tvec = (np.arange(nused) + 0.5) * bin_sec  # bin_sec = BIN_SIZE / fs
```

iii. Since the frame rate is constant at 30 Hz and bin size is 10 frames, time can be computed directly from bin indices. The +0.5 places the time at the center of each bin.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Minimal processing: multiply bin index (with +0.5 center offset) by the bin duration (10/30 s = 0.333 s). The result is in seconds from session start. For 20-minute sessions, time ranges from ~0.17 to ~1199.83 s; for 30-minute sessions, ~0.17 to ~1799.83 s.

ii.
```python
bin_sec = BIN_SIZE / fs  # 10/30 = 0.333... s
tvec = (np.arange(nused) + 0.5) * bin_sec
...
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The time is continuous across trials within a session (not reset per trial), as required by the task specification ("Time elapsed from the beginning of the session").

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is constructed from the same bin indices used to slice the neural data, so alignment is exact by construction. Bin k of the time vector corresponds to bin k of the neural and output data.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. All three data streams (neural, input, output) use the same slice of the same bin-indexed timeline, ensuring perfect temporal alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (global motion energy per camera frame) and `interframe_int.npy` (inter-frame intervals for detecting dropped camera frames), both in the `move_deve` subdirectory.

ii.
```python
md = os.path.join(session_dir, 'move_deve')
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
...
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal (sum of squared pixel-wise differences between consecutive video frames, as described in the Methods). The interframe interval file is needed to identify and repair dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) Dropped camera frames are detected from interframe intervals (positions where `round(ifi/median(ifi)) - 1 > 0`) and repaired by linear interpolation (`np.interp`) to match the neural data length. (2) The trace is averaged into 10-frame bins (same as neural). (3) The binned signal is discretized into 5 equal-count bins using rank-based discretization (`rankdata` with `method='ordinal'`), computed per session.

ii.
```python
# Dropped frame detection and interpolation:
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
med = np.median(ifi)
nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
idx = np.arange(len(me)) + np.concatenate([[0], np.cumsum(nmiss)])
me_full = np.interp(np.arange(nframes), idx, me)

# Binning:
me_binned = bin_average(me)

# Discretization:
def quantile_bin(x, nclasses=NCLASSES):
    r = rankdata(x, method='ordinal')
    lab = ((r - 1) * nclasses) // len(x)
    return lab.astype(np.int64)

me_used = me_binned[:nused]
labels = quantile_bin(me_used, NCLASSES)
```

iii. The rank-based discretization guarantees exactly equal class counts (20% each) even when there are tied motion energy values. Percentile edges are computed on the binned motion energy of the whole session (before splitting into trials), as required by the task specification ("selected per session").

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Rank-based discretization: each sample is ranked ordinally, then mapped to one of 5 classes via `((rank - 1) * 5) // n_samples`. This guarantees exactly 20% of samples per class in every session.

ii.
```python
def quantile_bin(x, nclasses=NCLASSES):
    r = rankdata(x, method='ordinal')  # 1..n
    lab = ((r - 1) * nclasses) // len(x)
    return lab.astype(np.int64)
```

iii. The task specifies "five equal-percentile bins, selected per session". The rank-based approach exactly satisfies "equal-percentile" by ensuring each bin contains exactly 1/5 of the session's samples. Discretization is applied to the portion of the session used for trials (after discarding remainder bins), so class counts are computed on the same data that enters the decoder.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video camera is hardware-triggered by the microscope, so camera frame i corresponds to imaging frame i (1:1). When camera frames are dropped (motion energy array shorter than neural), gaps are detected from interframe intervals and filled by linear interpolation. After repair, both streams have the same length and are binned together, then sliced into trials using the same indices.

ii.
```python
me, me_info = load_motion_energy(session_dir, nframes)  # nframes = F.shape[1]
me_binned = bin_average(me)
assert me_binned.shape[0] == nbins  # same as neural

# Same slicing for all streams:
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. The 1:1 frame correspondence is documented in the Methods ("Videos were recorded at 30 Hz, with the microscope acquisition acting as a trigger for camera frame acquisition"). Dropped frames are the only source of misalignment and are corrected before any further processing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms: (1) Dropped camera frames are detected and interpolated via interframe intervals. A fallback to uniform stretch is used if inferred gaps don't exactly explain the deficit. (2) Neurons with identically-zero F traces (extraction failure) are removed across all sessions of a mouse. (3) An iscell assertion catches unexpected non-cell ROIs. (4) uint64 motion energy is cast to float64 before arithmetic. (5) All neural values are asserted to be finite. (6) Remainder bins that don't fill a complete trial are discarded.

ii.
```python
# Dropped frame fallback:
if idx[-1] != nframes - 1:
    idx = np.linspace(0, nframes - 1, len(me))
    info['fallback_uniform'] = True

# Zero-trace neurons:
z = np.where(~np.asarray(F).any(axis=1))[0]

# Finiteness check:
assert np.isfinite(data['neural'][s][k]).all()
```

iii. The CONVERSION_NOTES document that 38/41 sessions have exact gap-to-deficit matches, while 3 jm046 sessions have a timestamp hiccup without actual data loss. The code handles all these cases robustly.

## 6-a. What are the most time-consuming steps of the code?

i. The maximin baseline correction (gaussian filtering + min/max filters over the full session for every neuron) is the dominant cost (~0.3-1.4 s/session depending on number of neurons and session length). Loading the .npy files is also I/O-bound but fast (~0.05-0.1 s/session). Total conversion time for all 41 sessions is ~41 seconds.

ii. Per-session timing is printed:
```python
t0 = time.time()
dff = compute_dff(F, Fneu, fs, ...)
dff_binned = bin_average(dff)
neural = zscore_rows(dff_binned).astype(np.float32)
t_neural = time.time() - t0
```

iii. The agent reports timing per session in the output, showing neural processing (0.3-1.4s) dominating over loading (0.0-0.1s) and behaviour processing (<0.05s).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. All major processing is already vectorized (numpy/scipy operations on full arrays). The trial-splitting loop is inherently sequential but operates on pre-computed arrays via slicing. The `find_zero_neurons` function loops over sessions but uses mmap_mode='r' for efficiency. No significant loops remain to vectorize.

ii. N/A (no inefficient loops present)

iii. The agent deliberately vectorized all operations. The per-trial slicing loop is O(n_trials) with constant-time slices, which is inherently efficient.

## 6-c. What processing does the code repeat multiple times?

i. The `find_zero_neurons` function reads F.npy for each session of a mouse before processing begins, and then F.npy is read again in `load_traces` during processing. This means F.npy is loaded twice per session. However, `find_zero_neurons` uses mmap_mode='r' to minimize the overhead of the first pass.

ii.
```python
# First pass (find_zero_neurons):
F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')

# Second pass (load_traces):
F = np.load(os.path.join(p, 'F.npy'))
```

iii. The double read is a minor inefficiency mitigated by mmap_mode. The alternative (reading once and caching all sessions' F arrays in memory) would increase memory usage significantly.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `_raw` dictionary stored in each session's result contains intermediate arrays (raw F, dF/F, dff_binned, neural, me, me_binned, labels, tvec) used only for `--show-processing` plots and sanity checks. These are deleted after processing (`del sess`) but temporarily increase peak memory usage.

ii.
```python
return dict(...,
            _raw=dict(F=F, dff=dff, dff_binned=dff_binned, neural=neural,
                      me=me, me_binned=me_binned, labels=labels, tvec=tvec))
...
del sess  # after appending trial data and optional plotting
```

iii. The _raw data is necessary for the `--show-processing` visualization mode and was used during development for sanity checking. It is not included in the final pickle output.
