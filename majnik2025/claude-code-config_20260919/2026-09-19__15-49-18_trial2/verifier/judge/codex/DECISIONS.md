# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six expected subject IDs in `SUBJECT_INFO`/`SUBJECTS`, scans each subject directory for session subdirectories, and then loads each session's raw neural and behavior arrays from `suite2p/plane0` and `move_deve`. Trial structure is created later from each fully loaded session after preprocessing and binning.

ii.
```python
SUBJECT_INFO = {
    'jm031': ('A', 7),
    'jm032': ('B', 7),
    'jm038': ('C', 8),
    'jm039': ('D', 8),
    'jm040': ('E', 9),
    'jm046': ('F', 8),
}
SUBJECTS = list(SUBJECT_INFO.keys())

def list_sessions(subject):
    d = os.path.join(DATA_DIR, subject)
    return sorted(f.path for f in os.scandir(d) if f.is_dir())

def load_session_raw(session_dir):
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
    ...
    return F, Fneu, float(ops['fs']), int(ops['nframes']), float(ops['neucoeff'])

session_dirs = []
for s in SUBJECTS:
    session_dirs.extend(list_sessions(s))
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset contains exactly six mice (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), that the released suite2p folders already contain Track2p-tracked cells, and that loading should follow the paper/repo structure by reading `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, and motion-energy files from each session folder.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by folder name, but the agent does not discover them generically with a `jm*` scan. It fixes subject identity and subject order from the hard-coded `SUBJECT_INFO` mapping and uses folder names only when building session paths and per-session metadata.

ii.
```python
SUBJECT_INFO = {
    'jm031': ('A', 7),
    'jm032': ('B', 7),
    'jm038': ('C', 8),
    'jm039': ('D', 8),
    'jm040': ('E', 9),
    'jm046': ('F', 8),
}
SUBJECTS = list(SUBJECT_INFO.keys())

...
'subjects': SUBJECTS,
'subject_idx': np.array([SUBJECTS.index(s['info']['subject'])
                         for s in sessions], dtype=np.int64),
```

iii. The notes say folder names like `jm031` map to mice and that the chosen order matches paper mice A through F. The agent also justified adding paper mouse labels and starting postnatal days from Fig. 5B.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject directory. Sessions are sorted lexicographically/chronologically and converted one session at a time.

ii.
```python
def list_sessions(subject):
    """Session directories of one subject, chronologically sorted."""
    d = os.path.join(DATA_DIR, subject)
    return sorted(f.path for f in os.scandir(d) if f.is_dir())

...
for s in SUBJECTS:
    session_dirs.extend(list_sessions(s))
```

iii. In the notes, the agent describes the on-disk layout as `data/<subject>/<YYYY-MM-DD>_a/...` and treats each dated subdirectory as one daily recording session.

## 1-d. How are the data split into trials?

i. The agent treats the recordings as continuous sessions with no natural task trials, so it creates artificial non-overlapping 60 s trials after neural and motion-energy binning. It computes `bins_per_trial` from `TRIAL_SEC * fs / BIN_FRAMES`, floors to a whole number of trials, and discards any tail that does not fill a complete trial.

ii.
```python
TRIAL_SEC = 60.0
BIN_FRAMES = 10

bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
n_trials = dff_binned.shape[1] // bins_per_trial
n_used = n_trials * bins_per_trial
dff_binned = dff_binned[:, :n_used]
me_binned = me_binned[:n_used]

for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural.append(np.ascontiguousarray(neural_all[:, sl]))
    inputs.append(time_all[sl][None, :].astype(np.float32))
    outputs.append(labels_all[sl][None, :])
```

iii. The notes explicitly justify `Trial = 60 s = 180 bins` because the task requires 60-second trials and the original recording is continuous rather than trial-structured.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply trial-level quality filtering. All full-length 60 s trial blocks are kept. The only handling of bad data is interpolation of missing motion-energy frames and dropping a tail if a future session does not divide evenly into full trials.

ii.
```python
n_trials = dff_binned.shape[1] // bins_per_trial
n_used = n_trials * bins_per_trial
dff_binned = dff_binned[:, :n_used]
me_binned = me_binned[:n_used]
missing_binned = missing_binned[:n_used]
```

iii. `CONVERSION_NOTES.md` says: "No trial/session curation: there is no task, so no trial can be 'bad'; every session has complete neural and behavioural coverage. The only defect is dropped camera frames ... interpolated rather than dropped so that all trials keep the same length."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives neural data from raw fluorescence and neuropil fluorescence in suite2p output: `F.npy` and `Fneu.npy`. It also reads `ops.npy` for `fs`, `nframes`, and `neucoeff`, and checks `iscell.npy`, but does not use `spks.npy`.

ii.
```python
def load_session_raw(session_dir):
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
    ...
    return F, Fneu, float(ops['fs']), int(ops['nframes']), float(ops['neucoeff'])
```

iii. The notes state "Neural signal = dF/F, not `spks`" and map `suite2p/plane0/F.npy`, `Fneu.npy`, and `ops['neucoeff','fs']` to the `neural` field.

## 2-b. How is the `neural` data processed?

i. Neural processing is: neuropil subtraction (`F - neucoeff * Fneu`), maximin baseline subtraction implemented manually with SciPy (`gaussian_filter`, `minimum_filter1d`, `maximum_filter1d`), 10-frame non-overlapping averaging, truncation to whole trials, then an extra decoder-conditioning step that subtracts each neuron's session mean and divides the whole session by one pooled session standard deviation. The saved arrays are `float32`.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff, baseline='maximin',
                 sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    return Fc - Flow, Flow

dff, F0 = f_processing(F, Fneu, fs, neucoeff)
dff_binned = bin_average(dff).astype(np.float32)

if neural_scaling == 'center':
    scale = float(dff_binned.std())
    neural_all = (dff_binned - dff_binned.mean(axis=1, keepdims=True)) / scale
...
neural_all = neural_all.astype(np.float32)
```

iii. The notes justify the first half as matching the reference repo's `F_processing` with suite2p defaults and explicitly reject dividing by `F0`. They separately justify the extra centering/scaling as "information-preserving" for the provided linear decoder and say it improved validation accuracy and SVD conditioning, so `center` was chosen as the default.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not perform additional neuron filtering. It asserts that all released ROIs already satisfy `iscell[:,0] == 1` and assumes Track2p's cross-day matching/curation has already been applied in the released suite2p folders.

ii.
```python
iscell = np.load(os.path.join(p, 'iscell.npy'))
...
assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROIs'
```

iii. The notes say the released data already contains only Track2p-tracked, `iscell`-curated neurons with matched row order across days, so "No neuron curation" is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. The agent does not align to a stimulus or behavioral event. It keeps the continuous session timeline, cuts it into consecutive 60 s blocks, and describes the alignment event in metadata as the start of the imaging session while also saying each trial is aligned to its own start.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural.append(np.ascontiguousarray(neural_all[:, sl]))

...
'temporal_alignment_event':
    'start of the imaging session (first 2-photon frame); the continuous '
    'session is cut into consecutive non-overlapping 60 s trials, and each '
    'trial is aligned to its own start',
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. The notes justify this by saying the recording has no task structure, the decoder input is elapsed session time, and the 60 s trials are artificial contiguous chunks rather than event-locked trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame bins at a 30 Hz acquisition rate, so each bin is 333.33 ms. The agent applies non-overlapping temporal averaging to both neural and motion-energy streams before trial cutting and output discretization.

ii.
```python
BIN_FRAMES = 10

def bin_average(x, k=None):
    k = BIN_FRAMES if k is None else k
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(*x.shape[:-1], n // k, k).mean(axis=-1)

dff_binned = bin_average(dff).astype(np.float32)
me_binned = bin_average(me)

'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. The notes say this matches the paper's statement that both dF/F and behavior were averaged in bins of 10 consecutive timestamps "for all decoding analysis."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The agent does not use a raw timestamp file for decoder input time. It derives time from the binned frame index together with `fs` and `BIN_FRAMES`.

ii.
```python
bin_dt = BIN_FRAMES / fs
time_all = (np.arange(n_used) + 0.5) * bin_dt
```

iii. The notes map the input to "frame index" and describe it as `(bin_index + 0.5) * 10 / 30` seconds.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent computes a continuous session-time ramp at the center of each binned time bin, then slices it into trials. This is a synthetic time variable rather than a transformed raw sensor signal.

ii.
```python
bin_dt = BIN_FRAMES / fs
time_all = (np.arange(n_used) + 0.5) * bin_dt

for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    inputs.append(time_all[sl][None, :].astype(np.float32))
```

iii. The notes justify using bin-center time because the decoder input is task-specific and the saved neural/behavior outputs are both averaged over each 10-frame bin.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it uses the same `n_used` truncation, the same 10-frame bin size, and the same per-trial slices as the neural matrix, so each input sample corresponds to one neural time bin.

ii.
```python
n_used = n_trials * bins_per_trial
dff_binned = dff_binned[:, :n_used]
...
time_all = (np.arange(n_used) + 0.5) * bin_dt

for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural.append(np.ascontiguousarray(neural_all[:, sl]))
    inputs.append(time_all[sl][None, :].astype(np.float32))
```

iii. The notes say the input ramp was sanity-checked against the expected ramp and report that the concatenated saved `input` matched the expected time values.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy` and aligned using `move_deve/interframe_int.npy`.

ii.
```python
md = os.path.join(session_dir, 'move_deve')
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
itv = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. The notes explicitly map `motion_energy_glob.npy` plus `interframe_int.npy` to the output field and cite the README/methods discussion of dropped frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent reconstructs camera-frame positions on the imaging frame grid from inter-frame intervals, places each motion-energy sample on that grid, treats missing values and the placeholder first sample as invalid, linearly interpolates them, then averages into 10-frame bins.

ii.
```python
nominal = np.median(itv)
steps = np.round(itv / nominal).astype(np.int64)
idx = np.concatenate([[0], np.cumsum(steps)])

out = np.full(n_frames, np.nan)
keep = idx < n_frames
out[idx[keep]] = me[keep]
out[0] = np.nan

missing = np.isnan(out)
out[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing),
                         out[~missing])

me_binned = bin_average(me)
```

iii. The notes justify this by saying the microscope triggers the camera, dropped triggers appear as doubled inter-frame intervals, `motion_energy_glob[0]` is a placeholder, and the README explicitly allows interpolation over missing video frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent discretizes the binned motion-energy trace into five within-session equal-percentile bins. It computes the 20th, 40th, 60th, and 80th percentiles and then assigns class labels 0-4 using those edges.

ii.
```python
def discretize_quantiles(x, n_bins=N_QUANTILES):
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(x, qs)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges

labels_all, edges = discretize_quantiles(me_binned)
```

iii. The notes say quintiles were chosen per session because the task requires five equal-percentile bins and because that guarantees each class occupies 20% of each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent aligns motion energy to neural data by rebuilding a length-`n_frames` motion-energy vector on the imaging frame grid, discarding camera samples that extend past the final 2-photon frame, interpolating missing imaging-frame positions, then applying the same 10-frame binning, truncation, and trial slicing used for neural data.

ii.
```python
def load_motion_energy(session_dir, n_frames):
    ...
    idx = np.concatenate([[0], np.cumsum(steps)])
    out = np.full(n_frames, np.nan)
    keep = idx < n_frames
    out[idx[keep]] = me[keep]
    ...
    out[missing] = np.interp(...)
    return out, missing

me, me_missing = load_motion_energy(session_dir, n_frames)
me_binned = bin_average(me)
...
me_binned = me_binned[:n_used]
...
outputs.append(labels_all[sl][None, :])
```

iii. The notes justify this with the paper/README claim that the camera is microscope-triggered, plus empirical checks showing the drop-corrected alignment outperformed a naive alignment in sessions with many dropped frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases explicitly: it asserts expected suite2p shapes and `iscell` status, reconstructs and interpolates missing motion-energy frames, treats `motion_energy_glob[0]` as missing, discards motion-energy samples past the final imaging frame, raises if all motion values would be missing, truncates partial trailing trial data, and runs final structural assertions on all saved trials.

ii.
```python
assert np.all(iscell[:, 0] == 1)
assert F.shape == Fneu.shape
assert F.shape[1] == ops['nframes']
assert len(itv) == len(me) - 1

out = np.full(n_frames, np.nan)
keep = idx < n_frames
out[idx[keep]] = me[keep]
out[0] = np.nan

missing = np.isnan(out)
if missing.all():
    raise ValueError(f'{session_dir}: no motion energy could be aligned')
out[missing] = np.interp(...)

n_trials = dff_binned.shape[1] // bins_per_trial
n_used = n_trials * bins_per_trial

for i in range(ns):
    ...
    assert np.isfinite(data['neural'][i][tr]).all()
    assert data['output'][i][tr].min() >= 0 and data['output'][i][tr].max() < N_QUANTILES
```

iii. The notes' "Edge cases handled" section explains these cases explicitly, especially placeholder first motion-energy samples, dropped camera triggers, extra camera frames beyond `nframes`, and future uneven trial boundaries.

## 6-a. What are the most time-consuming steps of the code?

i. The agent identifies dF/F computation, especially the maximin baseline filtering across full sessions, as the dominant compute cost. It also notes that loading `ops.npy` is wasteful because the file is large although only a few fields are used.

ii.
```python
dff, F0 = f_processing(F, Fneu, fs, neucoeff)
dff_binned = bin_average(dff).astype(np.float32)

with ProcessPoolExecutor(max_workers=workers) as ex:
    sessions = list(ex.map(_worker,
                           [(sd, args.neural_scaling, args.neucoeff)
                            for sd in session_dirs]))
```

iii. In Step 6 of the notes, the agent says "the maximin baseline over (n_neurons × 54 000) is the dominant cost (~0.8 s)" and that `ops.npy` is about 85 MB per session even though only `fs`, `nframes`, and `neucoeff` are needed.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent's effective decision is that the obvious hot loops were already vectorized: binning uses `reshape(...).mean(...)`, missing-frame filling uses `np.interp`, and sessions are parallelized across processes. The main remaining Python loops are the unavoidable loops over sessions and over trial slices needed to produce the required nested list structure.

ii.
```python
def bin_average(x, k=None):
    k = BIN_FRAMES if k is None else k
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(*x.shape[:-1], n // k, k).mean(axis=-1)

out[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing),
                         out[~missing])

for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural.append(np.ascontiguousarray(neural_all[:, sl]))
    inputs.append(time_all[sl][None, :].astype(np.float32))
    outputs.append(labels_all[sl][None, :])
```

iii. The notes explicitly say "all filtering/averaging is vectorised" and present process-level parallelism plus vectorized averaging as the intended speedups.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats some nonessential work: `convert_session` rescans the subject directory to compute `day_idx`, `plot_processing` reruns `convert_session` for diagnostic figures, and the broader agent workflow reran conversion under multiple scaling/binning variants while tuning the final choice.

ii.
```python
day_idx = list_sessions(subject).index(session_dir)

def plot_processing(session_dir, neural_scaling, outfile):
    os.environ['T2P_KEEP_DIAG'] = '1'
    sess, d = convert_session(session_dir, neural_scaling=neural_scaling, verbose=False)
    del os.environ['T2P_KEEP_DIAG']
```

iii. The notes mention separate diagnostic plotting, repeated validation experiments, and variant testing (`none`, `global`, `center`, `zscore`, different bin sizes), which explains why some processing was intentionally rerun during development.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent does extra work beyond what the downstream decoder needs: it loads the full `ops.npy` object to extract only a few metadata fields, computes and sometimes stores diagnostic-only quantities (`F0`, `me_missing`, `missing_binned`, `edges`, `time_all` under diagnostics), enriches metadata with postnatal-day and mouse-label annotations, and optionally regenerates full processing figures.

ii.
```python
ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()

diag = None
if os.environ.get('T2P_KEEP_DIAG'):
    diag = dict(F=F, F0=F0, dff=dff, me=me, me_missing=me_missing,
                dff_binned=dff_binned, me_binned=me_binned,
                labels_all=labels_all, edges=edges, time_all=time_all,
                bins_per_trial=bins_per_trial, fs=fs)

info = {
    'mouse_label': label,
    'postnatal_day': p0 + day_idx,
    'frac_frames_interpolated': float(me_missing.mean()),
    'motion_energy_quintile_edges': edges.tolist(),
    'dff_scale_divisor': scale,
    'frac_bins_with_interpolated_frames': float((missing_binned > 0).mean()),
}
```

iii. The notes explicitly call out `ops.npy` as larger than necessary for the needed fields and describe the diagnostics and metadata as part of validation and documentation rather than decoder-required outputs.
