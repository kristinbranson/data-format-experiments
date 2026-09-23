# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs, scans every session subdirectory for each subject, and loads session-level continuous recordings rather than any pre-existing trial files. For each session it loads neural fluorescence from `suite2p/plane0/F.npy`, plus `iscell.npy` and `ops.npy`; it only loads `Fneu.npy` if `neucoeff != 0`. It loads behavior from `move_deve/motion_energy_glob.npy` and `interframe_int.npy`. Trials are created later by splitting the processed session arrays into 60-second blocks.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def find_sessions(data_root=DATA_ROOT):
    sessions = []
    for subject in SUBJECTS:
        subject_dir = os.path.join(data_root, subject)
        names = sorted(f.name for f in os.scandir(subject_dir) if f.is_dir())
        for name in names:
            sessions.append((subject, name, os.path.join(subject_dir, name)))
    return sessions
```

```python
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float32)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()

if neucoeff != 0.0:
    Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float32)
else:
    Fneu = np.float32(0.0)

raw = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(md, 'interframe_int.npy')).astype(np.float64)
```

iii. In `CONVERSION_NOTES.md`, the AI says it used the published directory convention and the README’s fixed mouse order. It also justifies skipping `Fneu.npy` in the default path because it decided the reference `F_processing()` call uses `neucoeff=0.0`, so neuropil data is not needed unless the user overrides that default.

## 1-b. How are the data split into subjects?

i. Subjects are not discovered dynamically. They are defined by the fixed list `['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`, and this same order is used for `subjects` and `subject_idx`.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
MOUSE_LETTER = dict(zip(SUBJECTS, 'ABCDEF'))
```

```python
data = {
    'subjects': SUBJECTS, 'subject_idx': [],
    ...
}
...
data['subject_idx'].append(SUBJECTS.index(subject))
```

iii. The notes say this order comes from the dataset README, where alphabetical order corresponds to mice A-F. The AI chose the fixed list to preserve that order explicitly.

## 1-c. How are the data split into sessions?

i. Sessions are taken as all subdirectories inside each hard-coded subject folder, sorted lexicographically, and each sorted subdirectory becomes one session.

ii.
```python
def find_sessions(data_root=DATA_ROOT):
    sessions = []
    for subject in SUBJECTS:
        subject_dir = os.path.join(data_root, subject)
        names = sorted(f.name for f in os.scandir(subject_dir) if f.is_dir())
        for name in names:
            sessions.append((subject, name, os.path.join(subject_dir, name)))
    return sessions
```

iii. The AI’s notes say the session directories are daily recordings named by date, so alphabetical sorting yields chronological order within each mouse.

## 1-d. How are the data split into trials?

i. The AI treats each session as continuous data and splits it into consecutive non-overlapping 60-second trials after 10-frame binning. With `FS=30` and `BIN_FRAMES=10`, each trial has `180` time bins.

ii.
```python
TRIAL_SECONDS = 60.0
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FS / BIN_FRAMES))   # 180 bins
```

```python
n_trials = n_bins // BINS_PER_TRIAL
...
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
    input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int8))
```

iii. The notes say the experiment has no natural trial structure, so 60-second blocks are an artificial segmentation imposed by the decoder task.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply quality filtering based on trial content. It only requires at least two complete 60-second trials per session and drops any incomplete trailing block.

ii.
```python
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2, f'{session_dir}: only {n_trials} complete 60 s trials'
if n_trials * BINS_PER_TRIAL != n_bins and verbose:
    print(f'    note: dropping {n_bins - n_trials * BINS_PER_TRIAL} trailing bins '
          f'(incomplete 60 s trial)')
```

```python
'trial_curation': (
    'every complete 60 s block of each session is kept; incomplete trailing blocks would be '
    'dropped ...'),
```

iii. In the notes, the AI says there are no experimental trials to curate, so every complete 60-second block is retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives the default neural signal from `F.npy` plus `ops.npy` and `iscell.npy` for checks. `Fneu.npy` is only loaded when `neucoeff != 0`; in the default path it is replaced by scalar `0.0`.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float32)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()

if neucoeff != 0.0:
    Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float32)
else:
    Fneu = np.float32(0.0)
```

iii. The AI’s notes argue that the track2p GUI computes its “dF/F0” trace with `F_processing(..., neucoeff=0.0)`, so default conversion should not subtract neuropil.

## 2-b. How is the `neural` data processed?

i. The AI copies the reference `F_processing()` implementation into its script, applies `Fc = F - neucoeff * Fneu`, then performs a `maximin` baseline operation using Gaussian smoothing followed by minimum and maximum filters. By default it outputs `F - Flow` with `neucoeff=0.0`, not a divided `dF/F`. It then averages the result in non-overlapping 10-frame bins. The code also exposes optional `divide` and `zscore` modes, but the default conversion uses `subtract`.

ii.
```python
def F_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin', sig_baseline=10.0,
                 win_baseline=60.0, prctile_baseline: float = 8):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == "maximin":
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    F = Fc - Flow
    return F, Flow
```

```python
dff, Flow = F_processing(F, Fneu, fs=ops['fs'], neucoeff=neucoeff,
                         baseline=ops.get('baseline', 'maximin'),
                         sig_baseline=ops.get('sig_baseline', SIG_BASELINE),
                         win_baseline=ops.get('win_baseline', WIN_BASELINE),
                         prctile_baseline=ops.get('prctile_baseline', PRCTILE_BASELINE))
...
dff_binned = bin_time(dff).astype(np.float32)
```

iii. The notes justify this as the authors’ own GUI implementation of the trace labeled `dF/F0`. The AI explicitly notes a discrepancy between the paper wording and the code, and chooses to follow the code path with `neucoeff=0.0`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI performs two quality-control steps. First, it asserts that all loaded ROIs already satisfy `iscell[:,0] == 1` and `iscell[:,1] > 0.5`. Second, it detects neurons whose fluorescence trace is identically zero on any day and removes those neurons from every session of that mouse.

ii.
```python
assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROI'
assert np.all(iscell[:, 1] > 0.5), f'{session_dir}: ROI below the 0.5 iscell threshold'
...
failed = np.all(F == 0, axis=1)
```

```python
def drop_failed_neurons(data, session_info, session_subjects, session_names, failed_masks):
    ...
    for isess, subject in enumerate(session_subjects):
        bad = bad_per_subject[subject]
        if not bad.any():
            continue
        keep = ~bad
        data['neural'][isess] = [t[keep] for t in data['neural'][isess]]
        data['brain_region_idx'][isess] = data['brain_region_idx'][isess][keep]
```

iii. The notes say the exported data are already curated for `iscell > 0.5` and all-day tracking, but the AI judged all-zero fluorescence rows to be failed signal extraction rather than true silence and therefore removed them across the whole mouse to preserve row correspondence across days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the imaging session, not to any task event. It bins the continuous session from the first 2-photon frame and then slices consecutive 60-second windows.

ii.
```python
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
...
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
```

```python
'temporal_alignment_event': (
    'start of the imaging session (first 2-photon frame); there is no trial structure in '
    'the experiment, so each session is cut into consecutive, non-overlapping 60 s trials'),
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The notes say there is no natural trial event in this spontaneous-behavior dataset, so session start is the only meaningful alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral streams by averaging every 10 consecutive frames. At 30 Hz this gives a 333.33 ms bin size.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FS
```

```python
def bin_time(x, bin_frames=BIN_FRAMES):
    x = np.asarray(x)
    n = (x.shape[-1] // bin_frames) * bin_frames
    x = x[..., :n]
    new_shape = x.shape[:-1] + (n // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
```

iii. The AI’s notes cite the paper’s statement that both dF/F and behavior should be averaged in bins of 10 consecutive timestamps for decoding.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw timestamp file. It is synthesized from the global binned sample index, `BIN_FRAMES`, and the nominal frame rate `FS`.

ii.
```python
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
```

iii. The notes justify this by saying the imaging runs at a fixed nominal 30 Hz and the decoder input only needs elapsed session time, so bin index is sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time as the center of each 10-frame bin, in seconds from session start, then casts each per-trial time vector to `float32`.

ii.
```python
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
...
input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
```

iii. The notes say the bin-center convention avoids a visible temporal shift when over-plotting binned neural traces against raw 30 Hz traces.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is defined on the same binned session axis as the neural data and is sliced with the same trial windows, so each time sample corresponds one-to-one with one neural bin.

ii.
```python
dff_binned = bin_time(dff).astype(np.float32)
...
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
...
sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
```

iii. The notes describe this as alignment by common binning and identical trial slicing, rather than by a separate timestamp file.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion output from `move_deve/motion_energy_glob.npy` and `move_deve/interframe_int.npy`.

ii.
```python
raw = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(md, 'interframe_int.npy')).astype(np.float64)
```

iii. In the notes, the AI says `motion_energy_glob.npy` holds the precomputed motion-energy trace and `interframe_int.npy` is used to reconstruct where camera frames were dropped relative to 2-photon frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI reconstructs motion energy onto the imaging frame grid by using interframe intervals to infer the 2-photon frame index of each camera sample. It fills a full-length frame vector with `NaN`, inserts raw motion samples at observed frame indices, marks the first sample as missing, linearly interpolates all missing positions, averages the result in 10-frame bins, and then discretizes those binned values.

ii.
```python
med = np.median(ifi)
n_intervals = np.round(ifi / med).astype(np.int64)
frame_index = np.concatenate([[0], np.cumsum(n_intervals)])

motion = np.full(n_frames, np.nan)
inside = frame_index < n_frames
motion[frame_index[inside]] = raw[inside]
motion[0] = np.nan

idx = np.arange(n_frames)
good = ~np.isnan(motion)
motion = np.interp(idx, idx[good], motion[good])
```

```python
motion_binned = bin_time(motion)
labels, edges = discretize_quantiles(motion_binned)
```

iii. The notes justify this using the dataset README: the camera is triggered by the microscope, dropped camera frames should be treated as missing values, and interpolation is allowed.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes binned motion energy into five within-session equal-percentile bins using the 20th, 40th, 60th, and 80th percentiles. Labels are integers `0` through `4`.

ii.
```python
def discretize_quantiles(x, n_bins=N_QUANTILES):
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1])
    labels = np.digitize(x, edges, right=False).astype(np.int8)
    return labels, edges
```

```python
'output_values': [['q1 (lowest 20%)', 'q2', 'q3 (middle 20%)', 'q4',
                   'q5 (highest 20%)']],
```

iii. The notes say per-session percentile binning was chosen because absolute motion-energy scale varies strongly across sessions and the task explicitly requires five equal-percentile bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by reconstructing motion onto the 2-photon frame grid first, then applying exactly the same 10-frame binning and the same 60-second trial slicing as the neural signal.

ii.
```python
motion, n_missing, motion_raw, frame_index = load_session_motion(session_dir, n_frames)
...
dff_binned = bin_time(dff).astype(np.float32)
motion_binned = bin_time(motion)
...
sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
output_trials.append(labels[sl][None, :].astype(np.int8))
```

iii. The notes justify this by the dataset’s hardware triggering: camera frame `i` should correspond to imaging frame `i` except where triggers were missed, so rebuilding the motion trace on the imaging frame grid is the alignment step.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues explicitly. Missing behavior-camera frames are reconstructed as gaps on the 2-photon frame grid and linearly interpolated. The first motion-energy sample is treated as a boundary artifact and interpolated. Camera samples that map beyond the last imaging frame are ignored by the `inside = frame_index < n_frames` mask. All-zero fluorescence traces are interpreted as failed extraction and those neurons are removed from every session of the affected mouse. Incomplete trailing trial fragments would be dropped.

ii.
```python
motion = np.full(n_frames, np.nan)
inside = frame_index < n_frames
motion[frame_index[inside]] = raw[inside]
motion[0] = np.nan
...
motion = np.interp(idx, idx[good], motion[good])
```

```python
failed = np.all(F == 0, axis=1)
...
keep = ~bad
data['neural'][isess] = [t[keep] for t in data['neural'][isess]]
```

```python
if n_trials * BINS_PER_TRIAL != n_bins and verbose:
    print(f'    note: dropping {n_bins - n_trials * BINS_PER_TRIAL} trailing bins '
          f'(incomplete 60 s trial)')
```

iii. The notes say interpolation over dropped behavior frames is sanctioned by the dataset README. The AI further argues that all-zero fluorescence rows are missing data rather than true inactivity and should be removed to preserve the cross-day tracked-neuron identity.

## 6-a. What are the most time-consuming steps of the code?

i. The AI treats neural preprocessing as the main cost: loading `F.npy` and running the `maximin` baseline filters inside `F_processing()`. Motion reconstruction, binning, and splitting are comparatively light.

ii.
```python
t0 = time.time()
dff, Flow, F, ops, iscell, failed = load_session_neural(session_dir, neucoeff=neucoeff,
                                                        dff_mode=dff_mode)
t_neural = time.time() - t0

t0 = time.time()
motion, n_missing, motion_raw, frame_index = load_session_motion(session_dir, n_frames)
t_motion = time.time() - t0
...
t_bin = time.time() - t0
```

```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. In `CONVERSION_NOTES.md`, the AI says the only heavy computation is the maximin baseline operation over the full neuron-by-time matrix; it reports motion handling as negligible.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes the main operations. Binning is done with `reshape(...).mean(...)`, missing motion samples are filled with `np.interp`, and baseline filtering uses SciPy array operations. The remaining explicit loops are mostly over sessions or trials; the only notable non-vectorized bookkeeping loop is the per-subject rescan in `drop_failed_neurons()`.

ii.
```python
def bin_time(x, bin_frames=BIN_FRAMES):
    x = np.asarray(x)
    n = (x.shape[-1] // bin_frames) * bin_frames
    x = x[..., :n]
    new_shape = x.shape[:-1] + (n // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
```

```python
idx = np.arange(n_frames)
good = ~np.isnan(motion)
motion = np.interp(idx, idx[good], motion[good])
```

```python
for subject in bad_per_subject:
    for subj, name, sdir in find_sessions():
        if subj != subject or name in processed_names[subject]:
            continue
        F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        bad_per_subject[subject] |= np.all(np.asarray(F) == 0, axis=1)
```

iii. The notes explicitly claim the main hotspots were already vectorized and that Python-loop interpolation was avoided in favor of `np.interp`.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats some work around failed-neuron curation. After initial per-session processing, `drop_failed_neurons()` rescans all sessions for each subject and may reopen `F.npy` files that were not part of the current run, especially in `--sample` mode. The script also carries a large `_raw` bundle through `process_session()` and then uses it again for optional plotting/sanity checks.

ii.
```python
for subject in bad_per_subject:
    for subj, name, sdir in find_sessions():
        if subj != subject or name in processed_names[subject]:
            continue
        F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        bad_per_subject[subject] |= np.all(np.asarray(F) == 0, axis=1)
```

```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    ...
    '_raw': {'F': F, 'Flow': Flow, 'dff': dff, 'motion': motion, 'motion_raw': motion_raw,
             'frame_index': frame_index, 'dff_binned': dff_binned,
             'motion_binned': motion_binned, 'labels': labels, 'edges': edges,
             'bin_centre_time': bin_centre_time},
}
```

iii. The notes justify the rescan as necessary to make `--sample` obey the rule “drop a neuron if it fails on any day of that mouse,” even when not all days are being converted.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and carries several intermediates that are not saved in the final dataset: raw fluorescence `F`, baseline `Flow`, reconstructed continuous motion, raw motion samples, frame indices, full-session binned arrays, quantile edges, and bin-center times inside `_raw`. It also loads `iscell.npy` and `ops.npy` mainly for assertions and metadata. These are useful for plotting and sanity checks but are discarded before writing the final pickle.

ii.
```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'info': info,
    'failed': failed,
    # extras kept only for plotting / sanity checks
    '_raw': {'F': F, 'Flow': Flow, 'dff': dff, 'motion': motion, 'motion_raw': motion_raw,
             'frame_index': frame_index, 'dff_binned': dff_binned,
             'motion_binned': motion_binned, 'labels': labels, 'edges': edges,
             'bin_centre_time': bin_centre_time},
}
```

```python
data['neural'].append(res['neural'])
data['input'].append(res['input'])
data['output'].append(res['output'])
...
del res
```

iii. The notes explicitly say these extras are kept only for plotting and sanity checks. They are not part of the final decoder data structure.
