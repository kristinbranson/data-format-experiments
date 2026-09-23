# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all subject directories under `/app/data` whose names start with `jm`, sorts them, then enumerates each subject's session subdirectories in sorted order. For each session it loads `ops.npy`, `F.npy`, `Fneu.npy`, and `iscell.npy` from `suite2p/plane0`, and it loads `motion_energy_glob.npy` plus `interframe_int.npy` from `move_deve` when processing behavior. It does not preload the whole dataset into one large array; it processes one session at a time and appends per-session trial lists into the output dict.

ii. 
```python
subjects = sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))

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

iii. In `CONVERSION_NOTES.md`, the AI says this mirrors `data/load_data.ipynb` and the standard Track2p/Suite2p directory layout. The trajectory also shows it deliberately surveyed the folder structure first and concluded the data are organized by subject/session with Suite2p neural files and `move_deve` behavior files.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories whose names start with `jm`, sorted alphabetically.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))
```

iii. The AI's notes state that the dataset contains six subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and that each corresponds to one mouse. The trajectory shows it explicitly counted six mice during dataset exploration.

## 1-c. How are the data split into sessions?

i. Each session is a subdirectory of a subject directory. Sessions are sorted lexicographically, which the AI treats as chronological order because the folder names are date strings.

ii.
```python
def list_sessions(subject):
    """Chronologically sorted session directories of one subject."""
    return sorted(p for p in glob.glob(os.path.join(DATA_ROOT, subject, '*'))
                  if os.path.isdir(p))
```

iii. In `CONVERSION_NOTES.md`, the AI says the session directories are daily recordings and describes them as chronologically sorted. The trajectory shows it inspected session naming and day counts per mouse before implementing `list_sessions`.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions with no native trial structure and creates artificial trials as consecutive, non-overlapping 60-second blocks. Because the data are first averaged into 10-frame bins at 30 Hz, each trial has 180 bins.

ii.
```python
BIN_SIZE = 10
TRIAL_SEC = 60.0

bin_sec = BIN_SIZE / fs
bins_per_trial = int(round(TRIAL_SEC / bin_sec))
ntrials = nbins // bins_per_trial
nused = ntrials * bins_per_trial

for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. The AI justifies this in the script docstring and notes as following the task specification to split sessions into 60-second trials. The notes also say the original paper analyzed continuous recordings rather than behavioral trials, so fixed-length segmentation was introduced for the decoder task.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-level quality filtering. All complete 60-second blocks are kept.

ii.
```python
ntrials = nbins // bins_per_trial
nused = ntrials * bins_per_trial

for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    ...
```

iii. The notes say there are no behavioral trials in the source data and all sessions divide exactly into 20 or 30 one-minute trials after 10-frame binning, so no partial or low-quality trials needed to be removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from Suite2p fluorescence traces `F.npy` and `Fneu.npy`, with `ops.npy` providing preprocessing parameters and `iscell.npy` used only for sanity-checking the released ROI curation.

ii.
```python
def load_traces(session_dir):
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    return F, Fneu, iscell

ops = load_ops(session_dir)
fs = float(ops['fs'])
...
dff = compute_dff(F, Fneu, fs,
                  neucoeff=float(ops.get('neucoeff', 0.7)),
                  baseline=str(ops.get('baseline', 'maximin')),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
```

iii. The notes say this follows the paper's dF/F definition and Track2p/Suite2p outputs. The trajectory shows the AI explicitly inspected `ops.npy` and chose to use its stored Suite2p parameters, especially `neucoeff=0.7`, rather than the Track2p GUI display default.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F-like traces by subtracting neuropil (`F - neucoeff * Fneu`) and then applying a maximin baseline procedure equivalent to Track2p `F_processing` / Suite2p `dcnv.preprocess`. It then averages the result into 10-frame bins and additionally z-scores each neuron across the full session before saving it as `neural`.

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

dff = compute_dff(F, Fneu, fs,
                  neucoeff=float(ops.get('neucoeff', 0.7)),
                  baseline=str(ops.get('baseline', 'maximin')),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
dff_binned = bin_average(dff)
neural = zscore_rows(dff_binned).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies maximin baseline correction as matching the paper and Track2p code, and it justifies z-scoring by citing Track2p raster preprocessing plus the decoder's lack of internal normalization. The trajectory explicitly says it added z-scoring because the decoder performs SVD on raw neural matrices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI assumes all released ROIs are already valid cells (`iscell[:,0] == 1`) and asserts that condition. It then drops any neuron whose raw `F` trace is identically zero on any session of a mouse, and removes that neuron from all sessions of that mouse so neuron rows remain matched across days.

ii.
```python
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in released data'
keep = np.ones(F.shape[0], dtype=bool)
if drop_neurons is not None:
    keep[drop_neurons] = False
F, Fneu = F[keep], Fneu[keep]

def find_zero_neurons(subject_sessions):
    bad = set()
    for sd in subject_sessions:
        F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        z = np.where(~np.asarray(F).any(axis=1))[0]
        bad.update(z.tolist())
    return sorted(bad)
```

iii. The notes justify this as mirroring Track2p raster preprocessing's removal of rows that would become NaN after z-scoring, and the trajectory says the AI found a handful of zero-F rows and chose to remove them across the mouse to preserve the tracked-cell identity across sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to artificial 60-second block boundaries within each session. In metadata it describes the alignment event as the start of each 60-second block in the continuous recording rather than a natural behavioral event.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))

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

iii. The notes justify this by saying the recordings are continuous with no task event, so the only practical alignment for the forced trialization is the start of each synthetic 60-second block.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral signals by averaging non-overlapping groups of 10 imaging frames. At 30 Hz this yields 3 Hz data, or 333.33 ms bins.

ii.
```python
BIN_SIZE = 10

def bin_average(x, bin_size=BIN_SIZE):
    x = np.asarray(x, dtype=np.float64)
    n = (x.shape[-1] // bin_size) * bin_size
    newshape = x.shape[:-1] + (n // bin_size, bin_size)
    return x[..., :n].reshape(newshape).mean(axis=-1)

dff_binned = bin_average(dff)
me_binned = bin_average(me)

bin_ms = 1000.0 * BIN_SIZE / 30.0
```

iii. The notes cite the paper's statement that both dF/F and behavior were "averag[ed] in bins of 10 consecutive timestamps" for decoding, and the trajectory shows the AI treating this as a key consistency requirement.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. In code, the input time is not read from a dedicated raw timestamp variable; it is generated from the binned sample index and the imaging frame rate. The notes also mention frame index / `tstamps.npy` as contextual timing information, but the script itself uses only `fs` and bin index.

ii.
```python
fs = float(ops['fs'])
bin_sec = BIN_SIZE / fs
...
tvec = (np.arange(nused) + 0.5) * bin_sec
```

iii. The AI's notes justify this as time elapsed from session start on a uniform 30 Hz acquisition, so deriving time analytically from bin index is sufficient even though raw video timestamps exist.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one continuous time vector per session at bin centers, not bin left edges. It uses `(bin_index + 0.5) * bin_duration`, then slices that vector into trials.

ii.
```python
bin_sec = BIN_SIZE / fs
tvec = (np.arange(nused) + 0.5) * bin_sec
...
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The notes describe this as "time elapsed from the beginning of the session" and the trajectory shows the AI deliberately choosing a continuous time-in-session regressor. No separate justification for using bin centers rather than left edges was written beyond this continuous-time interpretation.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the time input by deriving it on the same binned timeline as the neural data and slicing both with the same trial boundaries, so each trial's time vector and neural matrix have the same number of bins.

ii.
```python
nbins = neural.shape[1]
assert me_binned.shape[0] == nbins

tvec = (np.arange(nused) + 0.5) * bin_sec

for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The notes and trajectory emphasize joint 10-frame binning of neural and behavioral streams before trialization, and the alignment of the input is a direct consequence of being generated on that same binned index.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives the output from `move_deve/motion_energy_glob.npy`, using `interframe_int.npy` to infer dropped camera frames before aligning the signal to imaging frames.

ii.
```python
md = os.path.join(session_dir, 'move_deve')
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
...
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The notes cite the paper's motion-energy definition and the data README's advice that dropped frames should be reconstructed from inter-frame intervals. The trajectory shows this was a major exploration topic before implementation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI casts motion energy to float, repairs missing camera frames by inferring gap counts from `interframe_int.npy` relative to the median interval, linearly interpolates onto the full imaging-frame timeline, averages into 10-frame bins, trims to whole trials, and only then discretizes into five classes.

ii.
```python
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
...
med = np.median(ifi)
nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
idx = np.arange(len(me)) + np.concatenate([[0], np.cumsum(nmiss)])
if idx[-1] != nframes - 1:
    idx = np.linspace(0, nframes - 1, len(me))
    info['fallback_uniform'] = True
...
me_full = np.interp(np.arange(nframes), idx, me)
...
me_binned = bin_average(me)
...
me_used = me_binned[:nused]
labels = quantile_bin(me_used, NCLASSES)
```

iii. The AI justifies this in the notes as a more robust implementation of the camera/imaging 1:1 alignment described in the methods and data README. The trajectory shows it explicitly checked mismatch cases and added the `fallback_uniform` safeguard after finding a few timestamp hiccups.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses a rank-based quantile binning function that assigns exactly equal numbers of samples to the five classes within each session. This is not percentile-edge-plus-`digitize`; it is ordinal-rank partitioning into quintiles.

ii.
```python
def quantile_bin(x, nclasses=NCLASSES):
    """Discretise into nclasses equal-percentile bins (exactly equal counts)."""
    r = rankdata(x, method='ordinal')          # 1..n
    lab = ((r - 1) * nclasses) // len(x)
    return lab.astype(np.int64)

me_used = me_binned[:nused]
labels = quantile_bin(me_used, NCLASSES)
```

iii. The notes justify this as satisfying the task requirement for five equal-percentile bins per session and ensuring exactly balanced class counts even when ties are present.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes camera and imaging frames should be 1:1 aligned, repairs only the missing-camera-frame cases, then bins the repaired motion signal with the same 10-frame averaging used for neural data and slices it with identical trial boundaries.

ii.
```python
me, me_info = load_motion_energy(session_dir, nframes)
me_binned = bin_average(me)
...
nbins = neural.shape[1]
assert me_binned.shape[0] == nbins
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. The notes justify this by citing hardware triggering of the camera from the microscope, and the trajectory shows the AI used the developmental PC1-vs-motion correlation pattern from the paper as a sanity check that the alignment was correct.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several minor data issues: dropped camera frames are reconstructed by interpolation; if camera samples exceed imaging frames they are truncated; if inferred gaps do not exactly explain a deficit it falls back to a uniform interpolation mapping; all-zero neurons are removed per mouse; and only whole 60-second trials are kept.

ii.
```python
if len(me) == nframes:
    ...
if len(me) > nframes:
    return me[:nframes], info
...
if idx[-1] != nframes - 1:
    idx = np.linspace(0, nframes - 1, len(me))
    info['fallback_uniform'] = True
...
def find_zero_neurons(subject_sessions):
    ...
    z = np.where(~np.asarray(F).any(axis=1))[0]
...
ntrials = nbins // bins_per_trial
nused = ntrials * bins_per_trial
```

iii. The notes justify the dropped-frame logic from the data README and the zero-neuron removal from Track2p's z-score preprocessing behavior. The trajectory shows the AI added the fallback logic after identifying sessions where timestamp irregularities did not correspond to actual missing frames.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies maximin dF/F preprocessing as the main computational cost, with file I/O as the other notable cost. Its notes say there are no major inefficiencies beyond those expected heavy steps.

ii.
```python
dff = compute_dff(F, Fneu, fs,
                  neucoeff=float(ops.get('neucoeff', 0.7)),
                  baseline=str(ops.get('baseline', 'maximin')),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
...
F = np.load(os.path.join(p, 'F.npy'))
Fneu = np.load(os.path.join(p, 'Fneu.npy'))
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. In the notes it reports maximin filtering at roughly 0.3 s to 1.2 s per session and says the dominant remaining cost is reading large `.npy` files from disk.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code already replaces the obvious per-gap insertion loop with vectorized interpolation. It does not call out any remaining important vectorization opportunities; the only obvious remaining Python loops are over sessions/trials when packaging outputs.

ii.
```python
nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
idx = np.arange(len(me)) + np.concatenate([[0], np.cumsum(nmiss)])
...
me_full = np.interp(np.arange(nframes), idx, me)

for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. The notes explicitly say there are "none significant" because the dominant operations are already vectorized and the main cost is the maximin filtering rather than Python loops.

## 6-c. What processing does the code repeat multiple times?

i. The AI does not describe repeated processing as a significant concern. In the actual code, the clearest repeated work is that each subject's `F.npy` files are scanned once in `find_zero_neurons()` and then the same sessions are loaded again in `process_session()` for real preprocessing.

ii.
```python
def find_zero_neurons(subject_sessions):
    for sd in subject_sessions:
        F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        z = np.where(~np.asarray(F).any(axis=1))[0]
        bad.update(z.tolist())

for sd in sessions:
    sess = process_session(sd, drop_neurons=drop)
```

iii. The notes say "Code inefficiencies identified: none significant" and frame the zero-neuron scan as an acceptable extra pass because it uses memory-mapped reads and avoids loading `ops.npy`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI again says there is no significant unnecessary processing, but the code does always construct a `_raw` bundle containing large intermediate arrays (`F`, `dff`, `dff_binned`, `me`, etc.) for plotting and sanity checks, and that bundle is usually discarded immediately after `process_session()` returns.

ii.
```python
return dict(neural=neural_trials, input=input_trials, output=output_trials,
            nneurons=neural.shape[0], nframes=nframes, fs=fs,
            ntrials=ntrials, bins_per_trial=bins_per_trial,
            bin_sec=bin_sec, me_info=me_info,
            _raw=dict(F=F, dff=dff, dff_binned=dff_binned, neural=neural,
                      me=me, me_binned=me_binned, labels=labels, tvec=tvec))

...
if args.show_processing and nplotted < 2:
    plot_processing(sess, f'{subject}_{os.path.basename(sd)}')
...
del sess
```

iii. The notes justify this implicitly as support for plotting and validation, and they do not treat it as a meaningful downstream inefficiency. There is no separate explicit justification beyond "none significant."
