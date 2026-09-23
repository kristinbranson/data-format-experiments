# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates sessions by taking a fixed list of known subject IDs from `SUBJECT_INFO`, scanning each subject directory for session subdirectories, and then loading each session on demand during conversion. For each session it loads neural fluorescence from `suite2p/plane0/F.npy`, imaging rate from `ops.npy`, and motion energy from `move_deve/motion_energy_glob.npy`; if camera frames are missing it also loads `move_deve/tstamps.npy`. Trials are not loaded directly from disk because the dataset is continuous; they are created later by splitting each converted session into 60 s blocks.

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

def list_sessions(data_root=DATA_ROOT):
    sessions = []
    for subject in sorted(SUBJECT_INFO.keys()):
        subj_dir = os.path.join(data_root, subject)
        for sess_dir in sorted(f.path for f in os.scandir(subj_dir) if f.is_dir()):
            sessions.append((subject, os.path.basename(sess_dir), sess_dir))
    return sessions

def load_traces(session_dir):
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))

def load_fs(session_dir):
    ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'),
                  allow_pickle=True).item()
    return float(ops['fs'])

me_raw = np.load(os.path.join(session_dir, 'move_deve',
                              'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI says it used the six released mice explicitly, kept subject/chronological order, reused the dataset notebook’s `F.npy` loader, and used `tstamps.npy` because the dataset README says dropped camera frames can be reconstructed from timestamps.

## 1-b. How are the data split into subjects?

i. Subjects are the six hard-coded mouse IDs in `SUBJECT_INFO`, sorted lexicographically. The code does not discover subjects by scanning for `jm*` directories; it assumes the released dataset is exactly the known six animals.

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

for subject in sorted(SUBJECT_INFO.keys()):
    subj_dir = os.path.join(data_root, subject)
```

iii. The AI justifies this in the notes by saying the released dataset has six known mice with a verified subject-to-paper mapping, so it treated those six IDs as the subject set.

## 1-c. How are the data split into sessions?

i. Each session is a subject subdirectory, sorted chronologically by directory name, and stored as a `(subject, session_name, session_dir)` tuple. One output session is produced per recording-day directory.

ii. 
```python
def list_sessions(data_root=DATA_ROOT):
    sessions = []
    for subject in sorted(SUBJECT_INFO.keys()):
        subj_dir = os.path.join(data_root, subject)
        for sess_dir in sorted(f.path for f in os.scandir(subj_dir) if f.is_dir()):
            sessions.append((subject, os.path.basename(sess_dir), sess_dir))
    return sessions
```

iii. The notes say the dataset organization is one folder per mouse and one dated subfolder per recording day, so sorted subdirectories give deterministic subject/chronological session order.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous and creates artificial trials by splitting each binned session into consecutive, non-overlapping 60 s blocks. With 10-frame bins at 30 Hz, each trial has 180 bins. The code would drop any trailing incomplete block, although the AI states the real sessions are exact multiples so no bins are actually discarded.

ii. 
```python
TRIAL_LEN_S = 60.0
BINS_PER_TRIAL = int(round(TRIAL_LEN_S / BIN_SIZE_S))   # 180

n_bins = dff_binned.shape[1]
n_trials = n_bins // BINS_PER_TRIAL
n_dropped_bins = n_bins - n_trials * BINS_PER_TRIAL

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural.append(np.ascontiguousarray(dff_binned[:, sl]))
    inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
    outputs.append(me_labels[sl][None, :].astype(np.int64))
```

iii. The notes say there is no natural trial structure in the spontaneous recordings, so 60 s blocks are imposed by the decoder task. The AI also argues this is consistent with the paper’s use of consecutive temporal blocks for decoding splits.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial-level exclusion. It keeps all sessions and all artificial 60 s trials, only requiring that each session produce at least two trials. It records how many bins would be dropped if a partial trial existed, but says that never happens in this dataset.

ii. 
```python
n_trials = n_bins // BINS_PER_TRIAL
n_dropped_bins = n_bins - n_trials * BINS_PER_TRIAL

for s in range(n_sessions):
    nt = len(data['neural'][s])
    assert nt >= 2, f'session {s} has only {nt} trials'
```

iii. In the notes, the AI explicitly says “No session/trial exclusion” and justifies it by saying all 41 sessions have at least 20 trials, neural data are finite, and behavior coverage is at least 99.6% of frames in every session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural data are derived from `suite2p/plane0/F.npy` only. Although `f_processing` can accept `Fneu`, the conversion path does not load or use `Fneu.npy` when producing the final dataset.

ii. 
```python
def load_traces(session_dir):
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))

F = load_traces(session_dir)
...
dff, baseline = f_processing(F, fs=fs, return_baseline=True)
```

iii. The AI’s notes say the released data are already Track2p-tracked ROIs and that the only dF/F implementation it trusted in the reference code was `track2p/gui/data_management.py:F_processing`, which it interpreted as baseline correction on `F` with `neucoeff=0`.

## 2-b. How is the `neural` data processed?

i. The AI computes neural activity by applying a custom reimplementation of Track2p’s `F_processing`: cast `F` to `float32`, optionally subtract neuropil only if `neucoeff` is nonzero, smooth with `gaussian_filter`, take a 60 s `minimum_filter1d` followed by `maximum_filter1d` to estimate the maximin baseline, subtract that baseline, then average into non-overlapping 10-frame bins. It does not divide by baseline.

ii. 
```python
def f_processing(F, Fneu=None, fs=FS, neucoeff=NEUCOEFF, baseline='maximin',
                 sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE,
                 prctile_baseline=8.0, return_baseline=False):
    Fc = F.astype(np.float32, copy=True)
    if neucoeff:
        Fc = Fc - neucoeff * Fneu.astype(np.float32)

    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    dff = Fc - Flow
    return dff

dff, baseline = f_processing(F, fs=fs, return_baseline=True)
dff_binned = bin_mean(dff).astype(np.float32)
```

iii. The notes say this is a “verbatim port” of `track2p/gui/data_management.py:F_processing`, and the AI defends it as matching the paper’s phrase “baseline corrected fluorescence traces” better than dividing by baseline. It also reports trying alternative variants and keeping this one because it believed it best matched the paper/code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI assumes the released files already contain Track2p-tracked, `iscell>0.5` neurons, and then adds an extra curation step: for each subject, it drops any ROI whose `F.npy` trace is identically zero on at least one session, and removes that ROI from every session of that subject.

ii. 
```python
def find_bad_neurons(subject_sessions):
    bad = None
    for _, _, sess_dir in subject_sessions:
        F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        b = np.asarray(F).std(axis=1) == 0
        bad = b if bad is None else (bad | b)
    return bad

for subject in subjects_used:
    subj_sessions = [s for s in all_sessions if s[0] == subject]
    bad = find_bad_neurons(subj_sessions)
    keep_masks[subject] = ~bad
...
if keep_neurons is not None:
    F = F[keep_neurons]
```

iii. The notes justify this by saying such ROIs fell outside the imaged field on some day, giving all-zero traces that carry no signal, and that removing them for every day preserves cross-day neuron matching within a subject.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the data as continuous spontaneous activity with no stimulus event. It therefore aligns neural trials to the start of each artificial 60 s block, and in metadata describes the temporal alignment event as the start of the 60 s trial.

ii. 
```python
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural.append(np.ascontiguousarray(dff_binned[:, sl]))

data['metadata'] = {
    'temporal_alignment_event': (
        'Start of the 60 s trial. Recordings are continuous spontaneous activity '
        'with no task events, so each session is simply cut into consecutive, '
        'non-overlapping 60 s blocks starting at the first imaging frame.'),
    'off_start': 0.0,
    'off_end': TRIAL_LEN_S,
}
```

iii. The notes explicitly say there is no task event and the only meaningful alignment is the session/trial start of the imposed 60 s blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral streams by averaging non-overlapping windows of 10 frames at 30 Hz, yielding 3 Hz data with a 333.33 ms time bin.

ii. 
```python
FS = 30.0
BIN_FRAMES = 10
BIN_SIZE_S = BIN_FRAMES / FS
BIN_SIZE_MS = BIN_SIZE_S * 1000.0

def bin_mean(x, bin_frames=BIN_FRAMES, nan_aware=False):
    n = x.shape[-1] // bin_frames
    v = x[..., :n * bin_frames].reshape(*x.shape[:-1], n, bin_frames)
    if nan_aware:
        return np.nanmean(v, axis=-1)
    return v.mean(axis=-1)

dff_binned = bin_mean(dff).astype(np.float32)
me_binned = bin_mean(me_frames, nan_aware=True)
```

iii. The AI cites the Methods sentence about “averaging in bins of 10 consecutive timestamps” for both dF/F and behavior and says the same binning must be applied before motion-energy discretization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not read from a dedicated raw timestamp variable. It is derived from the session-wide bin index together with the imaging sampling rate `fs` from `ops.npy` and the fixed 10-frame bin width.

ii. 
```python
def load_fs(session_dir):
    ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'),
                  allow_pickle=True).item()
    return float(ops['fs'])

fs = load_fs(session_dir)
...
bin_centre_s = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
```

iii. The notes say `ops['fs']=30` for every session and there are no needed per-frame experiment timestamps, so reconstructing time from bin index is sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a continuous session-wide time axis at the center of each 10-frame bin, then slices it into trials. The formula is `(bin_index * 10 + 4.5) / fs`, so the first value is 0.15 s rather than 0.0 s.

ii. 
```python
# time (s) elapsed from session start at the centre of each bin
bin_centre_s = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
```

iii. In the notes, the AI explicitly justifies using bin centers, calling the input “time from session start (s) at bin centres” and treating that as the most natural representation for binned data.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is built on the same binned session grid as the neural data and sliced with the same trial boundaries, so each neural bin and each time bin share the same index within a trial.

ii. 
```python
n_bins = dff_binned.shape[1]
...
bin_centre_s = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural.append(np.ascontiguousarray(dff_binned[:, sl]))
    inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
```

iii. The notes say the time axis is continuous across trial borders and is checked by concatenating per-trial inputs and verifying constant 1/3 s steps.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`. To align it to imaging frames, the AI also uses `move_deve/tstamps.npy` to reconstruct where dropped camera frames occurred.

ii. 
```python
me_raw = np.load(os.path.join(session_dir, 'move_deve',
                              'motion_energy_glob.npy')).astype(np.float64)
...
ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
```

iii. The notes say the dataset README recommends reconstructing dropped frames from timestamps, and the AI chose `tstamps.npy` instead of relying only on precomputed inter-frame intervals.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI maps motion-energy samples onto the imaging frame grid, leaving missing camera frames as `NaN`, explicitly sets frame 0 to `NaN` because the first frame-difference value is an artifact, averages the result into 10-frame bins with `np.nanmean`, interpolates only if an entire binned window is missing, and then discretizes the binned continuous trace.

ii. 
```python
me = np.full(n_frames, np.nan, dtype=np.float64)
me[frame_idx] = me_raw
me[0] = np.nan

me_binned = bin_mean(me_frames, nan_aware=True)
if np.any(np.isnan(me_binned)):
    bad = np.isnan(me_binned)
    good = ~bad
    me_binned[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good),
                               me_binned[good])
    me_info['n_interpolated_bins'] = int(bad.sum())
else:
    me_info['n_interpolated_bins'] = 0

me_labels, me_edges = discretize_quantiles(me_binned)
```

iii. The notes justify this by saying the camera is microscope-triggered, dropped frames should be treated as missing rather than inventing frame-level values immediately, and frame 0 is not a real motion measurement because no preceding frame exists.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes per-session quintile thresholds from the binned motion-energy values at the 20th, 40th, 60th, and 80th percentiles, then assigns labels 0–4 using `np.searchsorted(..., side='right')`.

ii. 
```python
def discretize_quantiles(x, n_quantiles=N_QUANTILES):
    edges = np.percentile(x, np.arange(1, n_quantiles) * (100.0 / n_quantiles))
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
```

iii. The notes say this follows the decoder instruction “five equal-percentile bins, selected per session” and normalizes away arbitrary per-session motion-energy scale differences.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by reconstructing the imaging-frame index of each camera frame from timestamp gaps, placing the observed motion-energy value at that imaging frame, leaving dropped camera frames as `NaN`, and then binning on the same 10-frame grid as the neural data.

ii. 
```python
ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
d = np.diff(ts)
step = np.median(d)
n_steps = np.round(d / step).astype(int)
frame_idx = np.concatenate([[0], np.cumsum(n_steps)])
...
me = np.full(n_frames, np.nan, dtype=np.float64)
me[frame_idx] = me_raw
...
assert me_binned.shape[0] == n_bins
```

iii. The notes justify this by citing the paper’s hardware triggering and the README’s statement that dropped frames can be recovered from timestamps. The AI says this avoids gross temporal misalignment and verified that reconstructed camera-frame spans match imaging-frame counts.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI adds several robustness rules: it truncates extra trailing camera frames if they ever exist, reconstructs missing camera-frame positions from `tstamps.npy`, treats dropped frames and the first motion-energy frame as `NaN`, interpolates only fully empty 10-frame bins, drops subject-wide all-zero ROIs, and would drop any trailing partial trial if present.

ii. 
```python
if n_cam > n_frames:
    me_raw = me_raw[:n_frames]

me = np.full(n_frames, np.nan, dtype=np.float64)
me[frame_idx] = me_raw
me[0] = np.nan

if np.any(np.isnan(me_binned)):
    bad = np.isnan(me_binned)
    good = ~bad
    me_binned[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good),
                               me_binned[good])

def find_bad_neurons(subject_sessions):
    ...
    b = np.asarray(F).std(axis=1) == 0

n_dropped_bins = n_bins - n_trials * BINS_PER_TRIAL
```

iii. The notes repeatedly frame these as edge-case handling: dropped video frames are a known issue from the README, motion-energy frame 0 is an artifact, and all-zero ROIs correspond to neurons outside the field of view on some days.

## 6-a. What are the most time-consuming steps of the code?

i. The AI says the expensive part is the neural baseline estimation in `f_processing`, specifically the `gaussian_filter`, `minimum_filter1d`, and `maximum_filter1d` operations over full-session fluorescence traces. It characterizes the rest of the pipeline as relatively light.

ii. 
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
...
t1 = time.time()
dff, baseline = f_processing(F, fs=fs, return_baseline=True)
dff_binned = bin_mean(dff).astype(np.float32)
t_neural = time.time() - t1
```

iii. In Step 6 of the notes, the AI says there were “no material” inefficiencies besides these filter operations and reports they take about 1.2 s for the largest session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s stated position is that there are no material remaining vectorization opportunities. It already vectorizes temporal binning with reshape/mean, and the remaining Python loops are mostly over sessions or trial slices used to build nested lists for the target format.

ii. 
```python
def bin_mean(x, bin_frames=BIN_FRAMES, nan_aware=False):
    n = x.shape[-1] // bin_frames
    v = x[..., :n * bin_frames].reshape(*x.shape[:-1], n, bin_frames)
    if nan_aware:
        return np.nanmean(v, axis=-1)
    return v.mean(axis=-1)

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural.append(np.ascontiguousarray(dff_binned[:, sl]))
    inputs.append(bin_centre_s[sl][None, :].astype(np.float32))
    outputs.append(me_labels[sl][None, :].astype(np.int64))
```

iii. The notes explicitly say “Code inefficiencies identified: none material” and argue that binning is already fully vectorized and trial construction is just slicing already computed arrays.

## 6-c. What processing does the code repeat multiple times?

i. The AI’s answer is effectively that it avoids repeated heavy processing: each session’s neural and motion traces are converted once at session level, then trials are created by slicing the already binned arrays. It does not rerun baseline correction or rebinned motion processing per trial.

ii. 
```python
dff, baseline = f_processing(F, fs=fs, return_baseline=True)
dff_binned = bin_mean(dff).astype(np.float32)
...
me_frames, me_info = motion_energy_on_imaging_frames(session_dir, n_frames)
me_binned = bin_mean(me_frames, nan_aware=True)
me_labels, me_edges = discretize_quantiles(me_binned)

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural.append(np.ascontiguousarray(dff_binned[:, sl]))
```

iii. The notes say one of the intentional speedups was “trials produced by slicing the already-binned matrix (no recomputation per trial).”

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra work for diagnostics and plotting that is not needed for the final converted dataset. In particular it always computes and returns the baseline, and it stores a large `_raw` dictionary of intermediates (`F`, `baseline`, `dff`, `dff_binned`, `me_frames`, `me_binned`, `me_labels`, `bin_centre_s`) so `--show-processing` can plot them, then removes `_raw` before saving. It also computes timing and class-fraction diagnostics that are mainly for reporting.

ii. 
```python
dff, baseline = f_processing(F, fs=fs, return_baseline=True)
...
out = {
    ...
    'class_fractions': np.bincount(me_labels, minlength=N_QUANTILES) / n_bins,
    'timing': {'load': t_load, 'neural': t_neural, 'behaviour': t_behav,
               'total': time.time() - t0},
    '_raw': {'F': F, 'baseline': baseline, 'dff': dff, 'dff_binned': dff_binned,
             'me_frames': me_frames, 'me_binned': me_binned,
             'me_labels': me_labels, 'bin_centre_s': bin_centre_s},
}
...
if args.show_processing and n_plotted < 2:
    plot_processing(f'{subject}_{sess_name}', res)
...
res.pop('_raw')
```

iii. The notes say the script carries these intermediates to support the required `--show-processing` visual checks and detailed sanity checking, not because the downstream pickle format needs them.
