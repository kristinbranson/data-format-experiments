# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates the dataset with `find_sessions()`, which walks a **hard-coded** list of the six
subject folders (`SUBJECTS = ['jm031', ..., 'jm046']`, taken from `data/README.md`, which states that
the alphabetical order maps onto mouse A…F in the paper) and, inside each, every subdirectory sorted
alphabetically (= chronologically, since folders are named `YYYY-MM-DD_a`). This yields 41
(subject, session_name, session_dir) tuples. For each session it loads:

* `suite2p/plane0/F.npy` — raw fluorescence of the Track2p-tracked cells (n_neurons × n_frames);
* `suite2p/plane0/iscell.npy` and `ops.npy` — used only to *assert* the curation invariants and to
  read the processing parameters (`fs`, `baseline`, `win_baseline`, `sig_baseline`, `prctile_baseline`);
* `suite2p/plane0/Fneu.npy` — **only if** `neucoeff != 0` (skipped in the default configuration, saving
  ~150 MB of I/O per session);
* `move_deve/motion_energy_glob.npy` and `move_deve/interframe_int.npy` — behaviour.

Sessions are processed one at a time and released (`del res`) so peak memory stays near one session.
`--sample` selects two specific sessions (`jm031/2023-10-22_a`, `jm046/2024-09-07_a`), deliberately
chosen because both exercise the dropped-camera-frame path.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def find_sessions(data_root=DATA_ROOT):
    """Return a list of (subject, session_name, session_dir), chronologically per subject."""
    sessions = []
    for subject in SUBJECTS:
        subject_dir = os.path.join(data_root, subject)
        names = sorted(f.name for f in os.scandir(subject_dir) if f.is_dir())
        for name in names:
            sessions.append((subject, name, os.path.join(subject_dir, name)))
    return sessions

def load_session_neural(session_dir, neucoeff=NEUCOEFF, dff_mode='subtract'):
    s2p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float32)
    iscell = np.load(os.path.join(s2p, 'iscell.npy'))
    ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
    ...
    if neucoeff != 0.0:
        Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float32)
    else:
        Fneu = np.float32(0.0)   # not needed; avoids reading ~150 MB per session

def load_session_motion(session_dir, n_frames):
    md = os.path.join(session_dir, 'move_deve')
    raw = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
    ifi = np.load(os.path.join(md, 'interframe_int.npy')).astype(np.float64)
```

iii. From CONVERSION_NOTES Step 1/2: the file layout was read from `data/README.md` and confirmed by
inspection; `load_data.ipynb:load_traces` (the official loader) loads exactly
`suite2p/plane0/F.npy`, and the notebook comment says "for more proper analysis compute dF/F the way
as described in the paper". `stat.npy` and `spks.npy` are deliberately not used (single brain region,
no anatomy needed; the paper's functional analyses use baseline-corrected fluorescence, not
deconvolved spikes). The subject list is hard-coded so that the mouse-letter and postnatal-day
mapping from Fig. 5B can be attached to each session in `metadata['session_info']`.

## 1-b. How are the data split into subjects?

i. One subject per `jm*` folder; six subjects, in alphabetical order, which the dataset README says
corresponds to mouse A–F in the paper. `data['subjects']` is the list of folder names and
`data['subject_idx'][isess] = SUBJECTS.index(subject)`. A per-subject letter and the postnatal day of
the first recording (read off Fig. 5B) are stored in the metadata.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
MOUSE_LETTER = dict(zip(SUBJECTS, 'ABCDEF'))
FIRST_PDAY = {'jm031': 7, 'jm032': 7, 'jm038': 8, 'jm039': 8, 'jm040': 9, 'jm046': 8}
...
data['subject_idx'].append(SUBJECTS.index(subject))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2: "For cross-referencing with the paper the subjects are named in
alphabetically increasing order (jm031 - mouse A … jm046 - mouse F)" (dataset README). All 6 mice are
kept: the paper says "a full dataset of 6 mice … we used this dataset for all subsequent analyses",
so no mouse is excluded.

## 1-c. How are the data split into sessions?

i. One session per date subdirectory of a subject folder (`YYYY-MM-DD_a`), sorted so that sessions run
chronologically within a mouse. 41 sessions total (7, 7, 7, 7, 6, 7). Each session becomes one entry
of the `neural`/`input`/`output` lists. A per-session `day_index` and `postnatal_day` are recorded.

ii.
```python
names = sorted(f.name for f in os.scandir(subject_dir) if f.is_dir())
...
day = day_counter.get(subject, 0)
day_counter[subject] = day + 1
res['info']['day_index'] = day
res['info']['postnatal_day'] = FIRST_PDAY[subject] + day
```

iii. CONVERSION_NOTES Step 2/3: "Each subject folder contains a number of session folders, each
corresponding to one recording day … the '_a' at the end of the folder name can be ignored"
(README); recordings are daily, so alphabetical date sorting = chronological order, and the day index
gives the postnatal day via Fig. 5B. All sessions are kept; none is excluded.

## 1-d. How are the data split into trials?

i. The experiment has **no** trial structure (continuous spontaneous recordings in the dark), so
trials are artificial: after 10-frame binning, each session is cut into consecutive, non-overlapping
blocks of 180 bins = 1800 frames = 60 s, starting at the first imaging frame. Any incomplete trailing
block would be dropped with a printed note; in practice there are none because 36,000 and 54,000
frames are exact multiples of 1800. This gives 20 trials for the two 20-minute mice and 30 trials for
the four 30-minute mice, 1090 trials in total. An assertion requires ≥2 trials per session.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FS / BIN_FRAMES))   # 180 bins
FRAMES_PER_TRIAL = BINS_PER_TRIAL * BIN_FRAMES                 # 1800 frames
...
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2, f'{session_dir}: only {n_trials} complete 60 s trials'
if n_trials * BINS_PER_TRIAL != n_bins and verbose:
    print(f'    note: dropping {n_bins - n_trials * BINS_PER_TRIAL} trailing bins '
          f'(incomplete 60 s trial)')

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
    input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int8))
```

iii. CONVERSION_NOTES Step 5 decision 5: "Trials = consecutive, non-overlapping 60 s blocks (180 bins)
from the session start, as the task requires. The experiment has no trial structure, so the alignment
event is the session start; this also mirrors the paper's own use of consecutive time blocks
(2-minute) for cross-validation." Sanity check 2 confirms `n_trials × 180 × 10 == n_frames` for every
session, i.e. no frame is silently lost.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality filtering.** The paper describes none (there are no trials in the
experiment, and no behavioural-state or artefact rejection is mentioned anywhere in the paper or
code). The only trial-level rule is structural: an incomplete trailing 60 s block would be discarded
(never triggered). All 1090 trials of all 41 sessions are kept. Sessions are also not filtered: every
session has ≥20 trials, far above the required minimum of 2.

ii.
```python
'trial_curation': (
    'every complete 60 s block of each session is kept; incomplete trailing blocks would be '
    'dropped (there are none: 36000 and 54000 frames are exact multiples of 1800)'),
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules: none in the paper (no trials)") and Step 5
decision 9 ("All 6 mice / 41 sessions / 1090 trials are kept — no session is excluded"). The AI
explicitly searched the paper/code for activity- or SNR-based rejection rules and found none, and
argued that inventing one would be an unjustified departure from the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from **`suite2p/plane0/F.npy`** only — the raw fluorescence traces of the
Track2p-tracked cells. `Fneu.npy` is deliberately *not* used, because the AI set the neuropil
coefficient to 0.0 (the default in the authors' own `F_processing`, see 2-b). `ops.npy` supplies the
processing parameters (`fs`, `baseline`, `win_baseline`, `sig_baseline`, `prctile_baseline`) and
`iscell.npy` is used only for assertions. `spks.npy` (deconvolved) and `stat.npy` are not used.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float32)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
...
if neucoeff != 0.0:
    Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float32)
else:
    Fneu = np.float32(0.0)
```

iii. CONVERSION_NOTES Step 5 decision 1: "Use `F.npy` (tracked cells) and compute dF/F with the
authors' `F_processing` rather than `spks.npy`: the paper states all functional analyses (including
decoding) used baseline-corrected fluorescence, not deconvolved spikes" ("We used baseline corrected
fluorescence traces as our dF/F … for all subsequent analyses"). The reason `Fneu` is not needed is
Step 4's resolution of the neuropil-coefficient discrepancy (see 2-b).

## 2-b. How is the `neural` data processed?

i. Three steps.
1. **dF/F** using a verbatim copy of the reference codebase's own
   `DataManagement.F_processing` (`code/track2p/gui/data_management.py:185`): neuropil subtraction
   `Fc = F − neucoeff·Fneu` with **neucoeff = 0.0**, then a *maximin* baseline
   (`gaussian_filter(σ=10 frames)` → `minimum_filter1d(win=60 s·fs)` → `maximum_filter1d(win)`),
   and `dff = Fc − Flow`. Note this subtracts but does **not** divide by the baseline, exactly as the
   reference implementation does. Parameters are read from each session's `ops.npy`
   (`baseline='maximin'`, `win_baseline=60`, `sig_baseline=10`, `prctile_baseline=8`, `fs=30`).
2. **Temporal binning**: mean over 10 consecutive frames (see 2-e).
3. Stored as `float32`; trials are `np.ascontiguousarray` slices.
`--neucoeff` and `--dff-mode {subtract,divide,zscore}` exist only to reproduce the Step-12 sensitivity
table; the default path is `neucoeff=0.0`, `dff_mode='subtract'`.

ii.
```python
NEUCOEFF = 0.0
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
...
dff, Flow = F_processing(F, Fneu, fs=ops['fs'], neucoeff=neucoeff,
                         baseline=ops.get('baseline', 'maximin'),
                         sig_baseline=ops.get('sig_baseline', SIG_BASELINE),
                         win_baseline=ops.get('win_baseline', WIN_BASELINE),
                         prctile_baseline=ops.get('prctile_baseline', PRCTILE_BASELINE))
...
dff_binned = bin_time(dff).astype(np.float32)
```

iii. CONVERSION_NOTES Step 4 documents the discrepancy explicitly: the paper says "baseline corrected
fluorescence traces as our dF/F (using the default Suite2p parameters)", `ops['neucoeff']` is 0.7
(suite2p's default, used for `spks`), but the authors' own dF/F call is
`F_processing(F=self.all_f_t2p[i], Fneu=self.all_fneu[i], fs=self.all_ops[i]['fs'])`, i.e. it takes
the function's default **neucoeff = 0.0**. The AI chose to follow the reference *code* over the
suite2p default, and then ran an end-to-end sensitivity analysis (Step 12) covering all readings:
`neucoeff=0` → 0.315/0.304/0.300/0.306, `neucoeff=0.7` → 0.299, `(F−F0)/F0` → 0.286, z-scored → 0.329.
It kept the reference-faithful choice, arguing the spread is small and that z-scoring would destroy
the amplitude information the paper's own analyses rely on. Correctness of the copy was verified by
re-deriving the baseline with an independent `sliding_window_view` implementation (agreement 1.3e-4).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two things.
1. **No new curation** is added on top of what the released data already contains. The AI *verified*
   (as hard assertions, every session) that suite2p's `iscell[:,0] == 1` and `iscell[:,1] > 0.5` hold
   for every ROI, and that every ROI was tracked by Track2p on all days of that mouse — i.e. the
   paper's stated curation ("all ROIs above the default threshold of 0.5") was already applied at
   export.
2. **One added curation step**: 8 ROIs (across 4 mice) have an identically-zero `F` trace on at least
   one day while their neuropil trace is normal — a failed signal extraction, i.e. *missing* data
   rather than silence. These are dropped from **every** session of the affected mouse, to preserve
   the dataset's defining property that row *i* is the same tracked neuron on every recording day.
   This removes 56 of 20,445 neuron-sessions (0.27 %); per-mouse counts go
   221/370/685/746/541/435 → 220/367/682/746/541/434.

ii.
```python
assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROI'
assert np.all(iscell[:, 1] > 0.5), f'{session_dir}: ROI below the 0.5 iscell threshold'
assert F.shape[1] == ops['nframes'], f'{session_dir}: F/ops frame-count mismatch'
assert ops['fs'] == FS, f'{session_dir}: unexpected frame rate {ops["fs"]}'
...
failed = np.all(F == 0, axis=1)
...
def drop_failed_neurons(data, session_info, session_subjects, session_names, failed_masks):
    bad_per_subject = {}
    for subject, name, mask in zip(session_subjects, session_names, failed_masks):
        prev = bad_per_subject.get(subject)
        bad_per_subject[subject] = mask if prev is None else (prev | mask)
    # days not processed in this run (--sample) still have to be inspected
    for subject in bad_per_subject:
        for subj, name, sdir in find_sessions():
            if subj != subject or name in processed_names[subject]:
                continue
            F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
            bad_per_subject[subject] |= np.all(np.asarray(F) == 0, axis=1)
    for isess, subject in enumerate(session_subjects):
        bad = bad_per_subject[subject]
        keep = ~bad
        data['neural'][isess] = [t[keep] for t in data['neural'][isess]]
        data['brain_region_idx'][isess] = data['brain_region_idx'][isess][keep]
```

iii. CONVERSION_NOTES Step 1/5 decision 3: "iscell>0.5 + all-day tracking were already applied at
export (verified) ⇒ No further neuron filtering is needed or justified", and "No activity-based or
SNR-based rejection is described anywhere in the paper/code". The extra drop came out of the
iteration log (Step 10, Issue 1): sanity check 17 flagged 8 zero-variance neurons; the AI traced them
to all-zero `F` rows with normal `Fneu`, classified them as missing data, and removed them mouse-wide
so a `--sample` run remains an exact subset of the full run (Step 10, Issue 6).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or task event in this experiment, so the alignment event is the **start of the
imaging session (first 2-photon frame)**. Trials are contiguous, non-overlapping 60 s segments of the
continuous recording, so trial *k* covers bins `[180k, 180(k+1))` and no event-based re-slicing is
performed. The metadata records `off_start = 0.0` and `off_end = 60.0` (the trial window measured from
the start of its own 60 s block).

ii.
```python
'temporal_alignment_event': (
    'start of the imaging session (first 2-photon frame); there is no trial structure in '
    'the experiment, so each session is cut into consecutive, non-overlapping 60 s trials'),
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
...
sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
```

iii. CONVERSION_NOTES Step 3/5: "no trial structure in the experiment (continuous 20/30 min
spontaneous recordings) … the alignment event is the session start". Panel 8 of the
`--show-processing` figure verifies that re-concatenating the trials exactly reproduces the
session-level arrays (`neural=True, input=True, output=True`), i.e. no gap, overlap or off-by-one at
trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning from the 30 Hz acquisition to **3 Hz**: the mean over 10 consecutive frames, giving
a **333.333 ms** time bin (`metadata['time_bin_size'] = 333.333`). The *same* binning, with the same
bin edges, is applied to the neural trace and to the motion-energy trace, and it is applied **before**
the motion energy is discretised. A trailing partial bin would be dropped (none occurs). Trials are
therefore 180 bins long, identical for every trial and session. The nominal 30 Hz from `ops['fs']` is
used rather than the measured 29.76 Hz (median camera interframe interval 33.60 ms), as the reference
code does.

ii.
```python
BIN_FRAMES = 10           # paper: denoise by averaging 10 consecutive timestamps
BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FS          # 333.333 ms

def bin_time(x, bin_frames=BIN_FRAMES):
    """Average consecutive `bin_frames` samples along the last axis, dropping any remainder."""
    x = np.asarray(x)
    n = (x.shape[-1] // bin_frames) * bin_frames
    x = x[..., :n]
    new_shape = x.shape[:-1] + (n // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
...
dff_binned = bin_time(dff).astype(np.float32)   # (n_neurons, n_bins)
motion_binned = bin_time(motion)                # (n_bins,)
assert motion_binned.shape[0] == n_bins
labels, edges = discretize_quantiles(motion_binned)
```

iii. CONVERSION_NOTES Step 3/5 decision 4, quoting the Methods verbatim: "For all decoding analysis we
slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive
timestamps." The AI notes that the binning must precede discretisation (averaging class labels would
be meaningless) and that using nominal 30 Hz makes the "60 s" trial actually 60.48 s of wall-clock
time (0.8 % long), which it documents in Step 4 and confirms in sanity check 8. Panel 3 of the
processing figure overlays the 30 Hz and binned traces to show there is no temporal shift.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from any stored raw variable — no per-frame timestamp exists for the 2-photon
stream. It is computed arithmetically from the **global bin index within the session** and the nominal
30 Hz frame rate, as the **bin-centre** time in seconds from the first imaging frame:
`t(b) = (10·b + 4.5) / 30`. Values run 0.15 → 1199.82 s (20-min sessions) and 0.15 → 1799.82 s
(30-min sessions). The clock runs continuously *across* trials within a session and resets at each new
session. It is a single time-varying input named `time_in_session_s`, stored `float32` with shape
(1, 180) per trial.

ii.
```python
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
...
input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
...
'input_names': ['time_in_session_s'],
```

iii. CONVERSION_NOTES Step 5 (variable mapping) and decision 8: the Decoder Task specifies "Time
elapsed from the beginning of the session in seconds. Time-varying." The camera timestamps
(`tstamps.npy`) exist but are for the behaviour camera, and the AI chose nominal 30 Hz over the
measured 29.76 Hz "as the reference code does (e.g. `win = int(win_baseline*fs)`)", documenting the
0.8 % consequence. Sanity check 7 verifies the time equals `(10b+4.5)/30` and that successive trial
starts are exactly 60 s apart; check 8 verifies the nominal session end is within 1 % of the measured
camera timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Essentially none beyond the arithmetic above: no smoothing, no normalisation, no binning (the value
is defined directly on the binned grid). Two deliberate sub-decisions: (a) the *centre* of each 10-frame
bin is used rather than its left edge, which is the value the binned neural/behavioural average
actually corresponds to; (b) the nominal 30 Hz is used, so the stored time is 0.8 % shorter than
wall-clock. Storage as `float32` introduces ≤6e-5 s of quantisation near 1800 s, which the AI measured
and documented as irrelevant at a 333 ms bin size.

ii.
```python
bin_centre_time = (np.arange(n_bins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
```

iii. CONVERSION_NOTES Step 4 (frame-rate discrepancy: `ops['fs']=30` vs a measured 33.60 ms interval)
and Step 10 Check 5 / Issue 3: the "time increases by exactly 1/3 s" check initially failed purely
because of float32 resolution at 1800 s; the AI confirmed this was a property of the check's
tolerance, not of the conversion, and left the representation unchanged.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: the time vector is indexed on the *same* binned grid as the neural matrix and
sliced with the *same* `slice(180k, 180(k+1))`, so element *j* of the input is the time of column *j*
of the neural matrix for every trial. No interpolation or resampling is involved and no offset is
possible.

ii.
```python
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(dff_binned[:, sl]))
    input_trials.append(bin_centre_time[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int8))
```
and the check inside `plot_processing`:
```python
input_cat = np.concatenate(result['input'], axis=1)[0]
ok_in = np.allclose(input_cat, tb[:nb])
```

iii. CONVERSION_NOTES Step 7/10: panel 8 of the processing figure and sanity check 7 both confirm the
per-trial time vectors reconcatenate into the session-level time vector and that trial *k* starts at
exactly 60·*k* + 0.15 s.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. From `move_deve/motion_energy_glob.npy` (the pre-computed global motion energy, one `uint64` value
per **camera** frame), together with `move_deve/interframe_int.npy` (the diff of the camera
timestamps), which is used to work out which 2-photon frame each camera sample belongs to. The number
of 2-photon frames comes from the neural array / `ops['nframes']`.

ii.
```python
md = os.path.join(session_dir, 'move_deve')
raw = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(md, 'interframe_int.npy')).astype(np.float64)
assert len(ifi) == len(raw) - 1, f'{session_dir}: tstamps/motion length mismatch'
```

iii. CONVERSION_NOTES Step 2/3: the Methods define motion energy as the summed squared pixel-wise
difference of consecutive video frames, and the released `motion_energy_glob.npy` already *is* that
quantity, so it is used as provided. The dataset README states that "the indices of missing frames can
be obtained by looking at `tstamps.npy` or `interframe_int.npy`", which is why the interframe
intervals are loaded.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four steps.
1. **Placement on the 2-photon frame grid.** The camera is hardware-triggered by the microscope, so
   camera sample *i* corresponds to imaging frame *i* *unless* triggers were missed. The number of
   trigger intervals spanned by each interframe interval is `round(ifi / median(ifi))`, and the
   cumulative sum of those gives each camera sample's true frame index. Samples whose index falls
   beyond the last imaging frame (the camera over-ran the acquisition in 3 jm046 sessions) are
   discarded rather than shifting the whole trace.
2. **Missing-value marking and interpolation.** Frames with no camera sample are `NaN`; additionally
   frame 0 is forced to `NaN` because `motion_energy_glob[0] == 0` in every session (frame
   differencing has no predecessor). All `NaN`s are filled by `np.interp` (linear).
3. **Binning**: the same 10-frame mean as the neural data (see 2-e).
4. **Discretisation** into 5 per-session quintiles (see 4-c).
Missing frames total 1–149 in 12 of 41 sessions (≤0.4 % of a session, worst case jm032 2023-10-22).

ii.
```python
med = np.median(ifi)
n_intervals = np.round(ifi / med).astype(np.int64)
frame_index = np.concatenate([[0], np.cumsum(n_intervals)])

motion = np.full(n_frames, np.nan)
inside = frame_index < n_frames
motion[frame_index[inside]] = raw[inside]

# The first motion-energy sample is a boundary artefact (no preceding video frame): it is 0
# in every session. Treat it as missing.
motion[0] = np.nan
n_missing = int(np.isnan(motion).sum())

idx = np.arange(n_frames)
good = ~np.isnan(motion)
motion = np.interp(idx, idx[good], motion[good])
...
motion_binned = bin_time(motion)
labels, edges = discretize_quantiles(motion_binned)
```

iii. CONVERSION_NOTES Step 2/5 decision 7 and Step 10 Check 5: the camera/2p correspondence follows
from the Methods ("the microscope acquisition acting as a trigger for camera frame acquisition, also
allowing for simple synchronisation across the two modalities"), and interpolation over missing frames
is explicitly sanctioned by the dataset README ("treated as missing values … or they can be
interpolated over"). The AI measured the median interframe interval (3.3603e-5 in the file's units =
33.60 ms) and verified that `len(motion_energy) + n_missing_triggers` equals the imaging frame count
exactly in every affected session. Treating `motion_energy[0]` as missing is justified because
"otherwise a spurious minimum would be introduced into the first bin and would bias the lowest
quintile". Sanity checks 9 and 13 re-derive the missing-frame count from `tstamps.npy` independently
and trace one hand-picked bin back to the raw camera samples.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into **5 equal-percentile (quintile) bins whose edges are computed within each session** from that
session's *binned* motion-energy trace: the 20/40/60/80th percentiles are used as the four interior
edges and `np.digitize(..., right=False)` produces labels 0–4, stored as `int8`. The edges are recorded
per session in `metadata['session_info'][i]['quantile_edges']`. By construction each class holds 20 %
of the bins in every session, which the verification log confirms (0.200 × 5 in all 41 sessions).

ii.
```python
N_QUANTILES = 5
def discretize_quantiles(x, n_bins=N_QUANTILES):
    """Digitise into `n_bins` equal-percentile bins; returns (labels int8, thresholds)."""
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1])
    labels = np.digitize(x, edges, right=False).astype(np.int8)
    return labels, edges
...
labels, edges = discretize_quantiles(motion_binned)
...
'output_names': ['motion_energy_quintile'],
'output_values': [['q1 (lowest 20%)', 'q2', 'q3 (middle 20%)', 'q4', 'q5 (highest 20%)']],
```

iii. CONVERSION_NOTES Step 5 decision 6: the Decoder Task requires "Motion energy, discretized into
five equal-percentile bins, selected per session." The AI adds the substantive reason for per-session
edges: absolute motion energy varies hugely across sessions (session medians 5.9e5 → 1.8e6), so global
edges would make the label largely a session identifier; per-session quintiles also guarantee exactly
20 % chance per class. Discretisation deliberately happens *after* binning. Sanity checks 10–12
reproduce the labels with an independent `np.searchsorted`/`np.quantile` implementation (100.0 %
identical) and confirm the class fractions. Step 12 notes the honest consequence: because motion
energy has a long "animal still" floor, the lower three thresholds are separated by only ~10–20 %, so
classes 0–2 are intrinsically hard to separate — a property of the prescribed task, not of the
conversion.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame-for-frame, via the hardware trigger. After step 1 of 4-b the motion array has exactly
`n_frames` entries on the imaging grid, is binned with the same `bin_time` call as the neural data, and
is sliced with the same per-trial `slice`, so bin *j* of the output is the same 333 ms window as column
*j* of the neural matrix. `assert motion_binned.shape[0] == n_bins` enforces this. Alignment was
verified empirically by cross-correlating the population dF/F with the motion energy across all 41
sessions: the median peak lag is 0 bins, and in every session with a non-negligible correlation the
peak is at lag 0 or +1 bin (consistent with calcium kinetics).

ii.
```python
motion, n_missing, motion_raw, frame_index = load_session_motion(session_dir, n_frames)
...
dff_binned = bin_time(dff).astype(np.float32)
motion_binned = bin_time(motion)
n_bins = dff_binned.shape[1]
assert motion_binned.shape[0] == n_bins
...
output_trials.append(labels[sl][None, :].astype(np.int8))
```

iii. CONVERSION_NOTES Step 3/4 ("camera acquisition is triggered by the 2p microscope, so camera frame
*i* corresponds to imaging frame *i*; no further alignment") and Step 10 sanity check 14 (the
cross-correlation test, plus `cache/alignment_check_out.txt`). Panels 4 and 7 of the processing figure
show the reconstructed trace with interpolated frames marked and the motion energy overlaid on the
neural raster.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct issues were found and handled, each documented:
1. **Dropped camera triggers** (8–12 sessions, 1–148 frames, ≤0.4 %): located from the interframe
   intervals, marked `NaN`, linearly interpolated.
2. **Camera over-running the acquisition** (jm046 2024-09-05/07/08: 3/10/3 surplus samples): samples
   mapping past the last imaging frame are dropped, so the trace is not shifted.
3. **`motion_energy_glob[0] == 0`** in all 41 sessions (frame-differencing boundary artefact): treated
   as missing and interpolated.
4. **Failed ROIs**: 8 neurons with an identically-zero `F` trace on ≥1 day (normal `Fneu`) are removed
   from all sessions of that mouse (see 2-c).
5. **Incomplete trailing 60 s block**: would be dropped with a printed note (never occurs).
Hard assertions guard the invariants (`len(ifi) == len(raw) - 1`, `F.shape[1] == ops['nframes']`,
`ops['fs'] == 30`, `iscell` thresholds, `motion_binned.shape[0] == n_bins`, `n_trials >= 2`), so any
mismatch fails loudly rather than silently mis-aligning the data. Unresolvable discrepancies (per-mouse
cell counts differing from Fig. 5B; 30-minute sessions vs the Methods' "20 minutes"; nominal vs
measured frame rate) are documented in Step 4/9 rather than patched.

ii.
```python
assert len(ifi) == len(raw) - 1, f'{session_dir}: tstamps/motion length mismatch'
assert F.shape[1] == ops['nframes'], f'{session_dir}: F/ops frame-count mismatch'
assert ops['fs'] == FS, f'{session_dir}: unexpected frame rate {ops["fs"]}'
assert motion_binned.shape[0] == n_bins
assert n_trials >= 2, f'{session_dir}: only {n_trials} complete 60 s trials'
...
inside = frame_index < n_frames          # drop camera samples past the last imaging frame
motion[frame_index[inside]] = raw[inside]
motion[0] = np.nan                        # boundary artefact
motion = np.interp(idx, idx[good], motion[good])
...
failed = np.all(F == 0, axis=1)
...
'missing_data_handling': (
    'camera frames dropped by the behaviour camera (<=0.4% of frames in 8/41 sessions) are '
    'located from the camera timestamps and linearly interpolated; the first motion-energy '
    'sample (always 0, a frame-difference boundary artefact) is interpolated as well'),
```

iii. CONVERSION_NOTES Step 10 Check 5 ("Check for edge cases") enumerates all of the above, and the
iteration log records that issues 3 and 4 were found by the AI's own sanity checks and fixed, with the
whole pipeline re-run afterwards. The dataset README is cited as authority for interpolating dropped
camera frames.

## 6-a. What are the most time-consuming steps of the code?

i. The AI instrumented the code with per-stage timers printed for every session. The dominant cost is
`load_session_neural` — reading `F.npy` plus the maximin baseline
(`gaussian_filter`/`minimum_filter1d`/`maximum_filter1d`) — at 0.28 s/session for the 221-neuron
20-minute sessions up to ~0.87 s for the 746-neuron 30-minute sessions. Motion-energy processing is
<0.01 s, binning and trial splitting 0.02–0.05 s, and the pickle write 0.5 s. Total: **0.9 s/session,
37.4 s for all 41 sessions**, well under the 15-minute budget, so no parallelism was added.

ii.
```python
t0 = time.time()
dff, Flow, F, ops, iscell, failed = load_session_neural(...)
t_neural = time.time() - t0
...
t_motion = time.time() - t0
...
t_bin = time.time() - t0
...
print(f'    neurons={info["n_neurons"]:4d} frames={n_frames} bins={n_bins} '
      f'trials={n_trials} missing_cam={n_missing:3d} '
      f'[load+dff {t_neural:.2f}s, motion {t_motion:.2f}s, bin/split {t_bin:.2f}s]')
...
print(f'\nConversion done in {t_conv:.1f} s ({t_conv / max(len(sessions), 1):.1f} s/session)')
```

iii. CONVERSION_NOTES Step 6/7: "The only heavy computation is the maximin baseline; `scipy.ndimage`'s
filters are already O(n) per neuron and run on the whole (n_neurons × n_frames) matrix at once."
Step 7 extrapolated 0.9 s/session (accounting for the 36k- vs 54k-frame sessions) to ~60 s for the full
run; the measured full run was 37 s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI reports that the loops that would normally be the bottleneck were vectorised up front:
binning is a `reshape(...).mean(-1)` rather than a per-bin loop, the missing-camera-frame fill is a
single `np.interp` rather than a per-gap insertion loop, and the maximin baseline runs on the whole
`(n_neurons, n_frames)` matrix in one call. The loops that remain are (a) the per-session loop in
`main()` — inherently serial I/O, deemed not worth parallelising at 0.9 s/session; (b) the per-trial
`for k in range(n_trials)` loop in `process_session`, which only creates 20–30 array views and costs
0.02–0.05 s; and (c) inside `drop_failed_neurons`, a loop that calls `find_sessions()` once per subject
and, in `--sample` mode only, materialises other days' `F.npy` via `np.asarray` on a memory-map — this
one is *not* flagged in the notes, but it is a no-op in the default `--full` run because every session
has already been processed.

ii.
```python
def bin_time(x, bin_frames=BIN_FRAMES):
    n = (x.shape[-1] // bin_frames) * bin_frames
    x = x[..., :n]
    new_shape = x.shape[:-1] + (n // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)      # vectorised, no loop
...
motion = np.interp(idx, idx[good], motion[good])   # vectorised, no per-gap loop
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "binning by `reshape(...).mean(-1)` instead of a
loop (vectorised); `np.interp` for the missing camera frames instead of per-gap loops; float32
throughout the neural pipeline (halves memory; verified against float64: max deviation 8.9e-5 in dF/F
units)". The AI's stated reason for not parallelising is simply that the total runtime (37 s) does not
justify it.

## 6-c. What processing does the code repeat multiple times?

i. Very little. The AI structured the code so each session is loaded and processed exactly once, with
`Fneu.npy` skipped entirely when `neucoeff == 0`. Two small repeats exist and are not flagged in the
notes: (a) `drop_failed_neurons()` re-calls `find_sessions()` once per subject inside its loop and, in
`--sample` mode, re-reads the mouse's other `F.npy` files to recompute the zero-trace mask that the
main loop already computed for the processed sessions (in `--full` mode this branch does nothing);
(b) `bin_time` is called separately for the neural and motion streams, which is necessary rather than
redundant. The AI's `--dff-mode`/`--neucoeff` sensitivity analyses re-run the whole conversion, but
those are separate invocations recorded in `cache/`, not repeated work in the default path.

ii.
```python
# the only genuine repeat, and only under --sample:
for subject in bad_per_subject:
    for subj, name, sdir in find_sessions():
        if subj != subject or name in processed_names[subject]:
            continue
        F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        bad_per_subject[subject] |= np.all(np.asarray(F) == 0, axis=1)
```

iii. CONVERSION_NOTES Step 10 Issue 6 explains why this re-read exists at all: "after the neuron drop,
`--sample` and `--full` disagreed on the neuron count (the drop rule needs all days of a mouse). Fix:
`drop_failed_neurons()` now inspects the mouse's remaining days lazily, so a sample conversion is an
exact subset of the full one." The AI accepted the redundant read as the price of making `--sample` a
true subset of `--full`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items, none of which the AI lists explicitly as waste (its Step 6 efficiency notes focus on
what it *avoided*):
1. `process_session` always builds a `_raw` dict holding the full-resolution `F`, `Flow`, `dff`,
   `motion`, `motion_raw`, `frame_index`, `dff_binned`, `motion_binned`, `labels` and
   `bin_centre_time`, even when `--show-processing` is off. Nothing but `plot_processing` reads it; it
   is freed per session by `del res`, so the cost is peak memory (~150 MB) rather than compute.
2. `F_processing` always computes and returns the baseline `Flow`, and `load_session_neural` returns
   `F`, `ops` and `iscell`; outside plotting and the `--dff-mode divide` path these are used only for
   assertions.
3. `motion_raw` and `frame_index` are retained purely for panel 4 of the figure.
Conversely, the AI *removed* one genuinely unnecessary operation: `Fneu.npy` (~150 MB/session) is not
read at all in the default configuration, halving the I/O. No discarded computation is on the critical
path — the whole conversion runs in 37 s.

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
...
    if args.show_processing and isess < 2:
        plot_processing(res, f'processing_{subject}_{name}.png')
    del res
```

iii. CONVERSION_NOTES Step 6: "`Fneu.npy` (~150 MB/session) is only needed when `neucoeff != 0`; it is
not read in the default configuration. Halves the I/O." and "sessions processed and released one at a
time (`del res`), so peak memory stays near the size of one session (~150 MB) plus the accumulating
output (~412 MB)." The retained `_raw` intermediates are the deliberate price of being able to plot
and spot-check every processing step from the same in-memory objects that produced the output.
