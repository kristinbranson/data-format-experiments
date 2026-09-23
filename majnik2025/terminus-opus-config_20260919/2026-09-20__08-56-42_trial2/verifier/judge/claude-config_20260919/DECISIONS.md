# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. `DATA_ROOT = '/app/data'` is scanned for subject directories whose name starts with `jm`, sorted alphabetically. For each subject, `list_sessions()` globs the immediate subdirectories (one per recording day) and sorts them, which is also chronological because the folders are named `YYYY-MM-DD_a`. For each session the AI loads six raw arrays: `suite2p/plane0/F.npy` (raw fluorescence), `Fneu.npy` (neuropil), `iscell.npy` (curation flags, used only for an assertion), `suite2p/plane0/ops.npy` (for `fs`, `neucoeff`, `baseline`, `sig_baseline`, `win_baseline`), `move_deve/motion_energy_glob.npy` (behaviour) and `move_deve/interframe_int.npy` (only when camera frames are missing). `spks.npy` and `stat.npy` are deliberately not loaded. All 6 subjects / 41 sessions / 20389 neuron-sessions / 1090 trials are loaded in a single pass, with a preceding lightweight `mmap` pass per subject (`find_zero_neurons`) over that subject's `F.npy` files. There is no trial structure in the raw data — trials are created afterwards by segmentation (see 1-d).

ii.
```python
DATA_ROOT = '/app/data'

def list_sessions(subject):
    """Chronologically sorted session directories of one subject."""
    return sorted(p for p in glob.glob(os.path.join(DATA_ROOT, subject, '*'))
                  if os.path.isdir(p))

def load_ops(session_dir):
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'),
                   allow_pickle=True).item()

def load_traces(session_dir):
    """Raw suite2p traces of the track2p-tracked cells (as in data/load_data.ipynb)."""
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    return F, Fneu, iscell
```
```python
subjects = sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))
...
for si, subject in enumerate(subjects):
    sessions = list_sessions(subject)
    ...
    drop = find_zero_neurons(sessions)
    for sd in sessions:
        sess = process_session(sd, drop_neurons=drop)
```
```python
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
...
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. From CONVERSION_NOTES.md Step 1/2: the loading "mirrors `data/load_data.ipynb`", which reads `suite2p/plane0/F.npy`, and the track2p GUI (`data_management.py`), which additionally reads `Fneu`, `ops` and `iscell`. The AI documented the directory layout (`/app/data/<subject>/<YYYY-MM-DD>_a/{suite2p/plane0, move_deve}`) and the data README's statement that subject folders map to paper mice A–F alphabetically. It checked file dtypes and shapes (F/Fneu/spks float32 `(n_neurons, n_frames)`; `iscell` `(n_neurons, 2)`; `motion_energy_glob` uint64 per camera frame; `tstamps` in units of 1000 s) and verified 41 sessions with 36000 frames (jm031, jm032) or 54000 frames (rest). `ground_truth.csv` files present for 3 subjects were explicitly identified as manual-tracking ground truth and excluded as not needed.

## 1-b. How are the data split into subjects (mice)?

i. One subject per top-level directory in `/app/data` whose name begins with `jm`, sorted alphabetically: `['jm031','jm032','jm038','jm039','jm040','jm046']` → 6 subjects. `subject_idx` is the index of the subject for each session, appended in the same loop order as the sessions, and stored as an `int64` array. The AI also carries the paper's mouse letters (A–F) into metadata via `SUBJECT_LETTER`. In `--sample` mode the `subjects` list is still all six even though only jm031 is processed.

ii.
```python
SUBJECT_LETTER = {'jm031': 'A', 'jm032': 'B', 'jm038': 'C',
                  'jm039': 'D', 'jm040': 'E', 'jm046': 'F'}
...
subjects = sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('jm'))
...
for si, subject in enumerate(subjects):
    ...
    for sd in sessions:
        ...
        data['subject_idx'].append(si)
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
...
subject_letters=[SUBJECT_LETTER.get(s, '?') for s in subjects],
```

iii. CONVERSION_NOTES.md Step 2 cites the data README: "For each subject there is a folder corresponding to the subject id" and "the subjects are named in alphabetically increasing order (jm031 - mouse A, jm032 - mouse B ... jm046 - mouse F)". The `jm` prefix filter excludes any non-subject entries. The AI checked this against the paper's "a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days" and confirmed 6 subjects. It also noted the subject boundary matters for neuron identity: track2p rows are matched only *within* a mouse, so the zero-trace neuron drop is computed per mouse.

## 1-c. How are the data split into sessions?

i. One session per subdirectory of a subject folder (one daily recording), sorted alphabetically = chronologically (`YYYY-MM-DD_a`). This gives 7/7/7/7/6/7 = 41 sessions. Every session becomes one entry in `neural`/`input`/`output`/`brain_region_idx`/`subject_idx`; no session is excluded. Per-session provenance (subject, mouse letter, path, date, day index, n_neurons, n_frames, fs, duration, n_trials, dropped camera frames, dropped neurons) is recorded in `metadata['session_info']`.

ii.
```python
def list_sessions(subject):
    return sorted(p for p in glob.glob(os.path.join(DATA_ROOT, subject, '*'))
                  if os.path.isdir(p))
```
```python
session_info.append(dict(
    subject=subject, mouse_letter=SUBJECT_LETTER.get(subject, '?'),
    session_dir=sd, date=os.path.basename(sd),
    day_index=sessions.index(sd), n_neurons=int(sess['nneurons']),
    n_frames=int(sess['nframes']), fs=sess['fs'],
    duration_s=float(sess['nframes'] / sess['fs']),
    n_trials=int(sess['ntrials']),
    n_dropped_camera_frames=int(sess['me_info']['n_missing']),
    n_dropped_neurons=int(len(drop))))
```

iii. Data README: "Each subject folder contains a number of session folders, each corresponding to one recording day... the name of the folder corresponds to the recording date in the YYYY-MM-DD format (the '_a' in the end of the folder name can be ignored)". CONVERSION_NOTES.md Step 4 records a discrepancy the AI investigated: the Methods say "each session lasted 20 minutes", but the data contain both 20-min (36000 frames; jm031, jm032) and 30-min (54000 frames; the other four) recordings. The AI resolved this in favour of the data ("the paper statement is a simplification") and kept both durations, letting sessions contribute 20 or 30 trials respectively. Step 5 decision 8: "All 6 mice / 41 sessions / 1090 trials retained; no session excluded (all have complete behaviour and neural data)."

## 1-d. How are the data split into trials?

i. The recordings are continuous spontaneous activity with no behavioural trial structure, so trials are defined artificially as consecutive, non-overlapping 60-second blocks of the session, per the task specification. Because binning is done first (10 frames → 333.33 ms), a trial is 180 bins: `bins_per_trial = round(TRIAL_SEC / bin_sec) = round(60 / (10/30)) = 180`. `ntrials = nbins // bins_per_trial`; a trailing partial block would be dropped, but 3600 and 5400 bins divide exactly into 20 and 30 trials, so zero frames are discarded. Total 1090 trials (14 sessions × 20 + 27 × 30).

ii.
```python
TRIAL_SEC = 60.0        # trial duration in seconds (task specification)
...
bin_sec = BIN_SIZE / fs
bins_per_trial = int(round(TRIAL_SEC / bin_sec))
ntrials = nbins // bins_per_trial
nused = ntrials * bins_per_trial
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. CONVERSION_NOTES.md Step 3 ("Trial curation rules"): "the paper analyses whole continuous recordings (there are no behavioural trials). For this task, sessions are cut into consecutive non-overlapping 60 s trials; 36000/54000-frame sessions divide exactly into 20/30 trials, so no partial trial is discarded." Step 5 decision 4 notes this is "compatible with the paper, which uses continuous recordings and splits only for cross-validation (2-min blocks)". Step 10 Check 5 (edge cases) explicitly verified "36000/54000 frames -> 3600/5400 bins -> exactly 20/30 trials of 180 bins; no partial trial, no discarded frame", and that the last trial ends at 1199.833 s / 1799.833 s.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Every 60-second block of every session is retained (1090/1090). The only trial-level exclusion mechanism in the code is the implicit dropping of a trailing partial block (`nbins // bins_per_trial`), which never triggers on this dataset. Sessions with dropped camera frames are repaired rather than excluded, so no trial is lost to behavioural data gaps. Assertions in `main()` check every trial for finiteness, shape consistency and output range rather than removing trials.

ii.
```python
ntrials = nbins // bins_per_trial      # trailing partial block would be dropped
nused = ntrials * bins_per_trial
```
```python
for k in range(len(data['neural'][s])):
    nk, tk = data['neural'][s][k].shape
    assert nk == nn == len(data['brain_region_idx'][s])
    assert data['input'][s][k].shape[1] == tk
    assert data['output'][s][k].shape[1] == tk
    assert np.isfinite(data['neural'][s][k]).all()
    assert np.isfinite(data['input'][s][k]).all()
    assert data['output'][s][k].min() >= 0 and data['output'][s][k].max() < NCLASSES
```

iii. CONVERSION_NOTES.md Step 3 states there are no behavioural trials to curate; Step 5 decision 8 states "no session excluded (all have complete behaviour and neural data)". The reference paper and code define no trial-quality criterion (the paper's only segmentation is into 2-minute blocks for cross-validation), and the AI found no per-trial quality variable in the released data, so it chose to keep everything and repair the only data defect it found (dropped camera frames) instead of discarding trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy` (raw ROI fluorescence) and `suite2p/plane0/Fneu.npy` (neuropil fluorescence), both `(n_neurons, n_frames)` float32 arrays of the track2p-tracked cells. `suite2p/plane0/ops.npy` supplies the processing parameters (`fs=30`, `neucoeff=0.7`, `baseline='maximin'`, `sig_baseline=10.0`, `win_baseline=60.0`) rather than the AI hard-coding them, and `iscell.npy` is loaded only to assert that the released ROIs are all classified as cells. `spks.npy` (suite2p deconvolved spikes) is explicitly **not** used.

ii.
```python
def load_traces(session_dir):
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    return F, Fneu, iscell
...
ops = load_ops(session_dir)
fs = float(ops['fs'])
F, Fneu, iscell = load_traces(session_dir)
...
dff = compute_dff(F, Fneu, fs,
                  neucoeff=float(ops.get('neucoeff', 0.7)),
                  baseline=str(ops.get('baseline', 'maximin')),
                  sig_baseline=float(ops.get('sig_baseline', 10.0)),
                  win_baseline=float(ops.get('win_baseline', 60.0)))
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping maps `F.npy`, `Fneu.npy` and the `ops` parameters to `neural`. The Methods state "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses", and the data README's own `load_data.ipynb` comment says "for more proper analysis compute dF/F the way as described in the paper (or alternatively use spks.npy)". The AI chose the dF/F route over `spks.npy` because the paper's decoding analysis uses dF/F. It read parameters from the authors' own `ops.npy` so that the processing reproduces what the authors actually ran.

## 2-b. How is the `neural` data processed?

i. Four steps, in order:
1. **Neuropil subtraction + maximin baseline correction** (`compute_dff`), a line-by-line copy of `track2p/gui/data_management.py::F_processing`, which is functionally identical to `suite2p.extraction.dcnv.preprocess`: `Fc = F - 0.7*Fneu`; `Flow = gaussian_filter(Fc, [0, 10])`; `Flow = minimum_filter1d(Flow, 60*fs)`; `Flow = maximum_filter1d(Flow, 60*fs)`; `dff = Fc - Flow`.
2. **Temporal binning**: non-overlapping averages of 10 consecutive frames (30 Hz → 3 Hz, 333.33 ms bins), after upcasting to float64.
3. **Per-neuron z-scoring across the whole session** (`zscore_rows`), matching `RasterWindow.preprocessing()` in the reference repo, then cast to float32.
4. **Segmentation** into 180-bin trials.

The resulting `neural` has mean 0.0000, std 1.0000, min −4.61, max 29.21 over the full dataset.

ii.
```python
def compute_dff(F, Fneu, fs, neucoeff=0.7, baseline='maximin',
                sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8.0):
    """Copied from track2p/gui/data_management.py::F_processing (identical to
    suite2p.extraction.dcnv.preprocess)."""
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    ...
    return Fc - Flow

def bin_average(x, bin_size=BIN_SIZE):
    """Average consecutive bin_size samples along the last axis (drops remainder)."""
    x = np.asarray(x, dtype=np.float64)
    n = (x.shape[-1] // bin_size) * bin_size
    newshape = x.shape[:-1] + (n // bin_size, bin_size)
    return x[..., :n].reshape(newshape).mean(axis=-1)

def zscore_rows(x):
    """z-score each neuron across the session (track2p raster preprocessing)."""
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
```
```python
dff = compute_dff(F, Fneu, fs, neucoeff=float(ops.get('neucoeff', 0.7)), ...)
dff_binned = bin_average(dff)
neural = zscore_rows(dff_binned).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 1–3. (1) dF/F: "maximin baseline-corrected, neuropil-subtracted fluorescence (suite2p defaults, `neucoeff=0.7` from the authors' own ops), matching `F_processing` in the reference code and 'baseline corrected fluorescence traces as our dF/F' in the Methods". Step 4 records that the track2p GUI's *display* default is `neucoeff=0.0` whereas the authors' stored `ops['neucoeff']` is 0.7 (the suite2p default referred to by "default Suite2p parameters"); the AI sanity-checked both and found corr(PC1, motion) barely changes (0.776 vs 0.761 on jm046 last day). (2) Binning: "exactly the denoising the authors used for all decoding analyses" — Methods: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps". (3) z-scoring: "matches reference raster preprocessing (`zscore` after binning) and puts all sessions/neurons on a comparable scale for the shared decoder (the decoder does SVD on raw values, so scale normalisation matters). Computed on the whole session, so no trial-level leakage of trial identity." As a validation, Step 4/10 show that PC1 of the processed neural data reproduces Fig. 7D's developmental increase in correlation with motion, and Step 10 Check 4 reproduces Fig. 7C ridge-regression R² (≈0 early, 0.40–0.75 on late days).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two layers. (1) **Inherited curation**, verified rather than re-applied: the released data already contain only ROIs with suite2p `iscell` probability > 0.5 *and* tracked by track2p across all days of the mouse. The AI asserts `iscell[:,0] == 1` for every ROI in every session rather than filtering. (2) **Zero-trace removal**: `find_zero_neurons` scans all sessions of a mouse and collects any neuron whose `F` trace is identically zero on *any* day; those neurons are dropped from **all** sessions of that mouse so the row-matched tracked population stays consistent across days. This removed 5 neurons (jm031:1, jm032:3, jm046:1) = 56 neuron-sessions, leaving 20389 of 20445. No other neuron filtering (no SNR, activity-rate or ROI-shape criterion) is applied.

ii.
```python
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in released data'
keep = np.ones(F.shape[0], dtype=bool)
if drop_neurons is not None:
    keep[drop_neurons] = False
F, Fneu = F[keep], Fneu[keep]
```
```python
def find_zero_neurons(subject_sessions):
    """Neurons with an identically-zero F trace on any session of a mouse."""
    bad = set()
    for sd in subject_sessions:
        F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        z = np.where(~np.asarray(F).any(axis=1))[0]
        bad.update(z.tolist())
    return sorted(bad)
```

iii. CONVERSION_NOTES.md Step 1: the reference `RasterWindow.preprocessing()` sets `rem_zero_rows = True` and removes rows that become NaN after z-scoring — i.e. exactly the neurons the AI drops. Step 5 decision 7: "drop neurons with an identically-zero F trace on any session of that mouse (8 neuron-sessions, 5 neurons); everything else already curated by suite2p iscell>0.5 + track2p tracking." Step 3 cites the Methods, "We considered all ROIs above the default threshold of 0.5 as true cells", and records the AI's verification that `iscell[:,0]` mean is exactly 1.0 in every session, so no further threshold need be applied. Step 10 Check 5 gives the mechanistic reason: an identically-zero trace gives 0/0 in the z-score, and dropping per-mouse (rather than per-session) preserves the dataset's defining property that row *i* is the same neuron on every day. Step 9 acknowledges the resulting neuron count (498 ± 181/mouse) is 5% below the paper's stated 526 ± 190 and attributes most of the gap to the released curated output rather than to this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural event to align to — the recording is continuous. Trials are therefore aligned to the start of each 60-second block, i.e. trial *k* of a session covers `[k*60, (k+1)*60)` seconds after session onset, with no pre-event window and no gaps or overlap. Slicing is done on the binned time axis with a single slice shared by `neural`, `input` and `output`, so the three streams are aligned by construction. Metadata records `temporal_alignment_event = 'start of each 60 s block of the continuous recording (block k starts at k*60 s after session onset); recordings are continuous, there are no behavioural trials'`, `off_start = 0.0`, `off_end = 60.0`.

ii.
```python
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int64))
```
```python
temporal_alignment_event=('start of each 60 s block of the continuous recording '
                          '(block k starts at k*60 s after session onset); '
                          'recordings are continuous, there are no behavioural trials'),
off_start=0.0,
off_end=TRIAL_SEC,
trial_duration_s=TRIAL_SEC,
```

iii. CONVERSION_NOTES.md Step 5 decision 4: "Trials = consecutive 60 s blocks, as required by the task; alignment event = session start (trial k starts at k*60 s), `off_start=0`, `off_end=60`. This is compatible with the paper, which uses continuous recordings and splits only for cross-validation (2-min blocks)." Because a single slice index is applied to all three streams, the AI argued alignment cannot drift; it verified this visually in `processing_<session>_trials.png` (neural raster, input ramp and output quintiles plotted on the same 180-bin axis for the first, middle and last trial) and functionally by reproducing Fig. 7C/7D, which "would collapse under any misalignment".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning from the native 30 Hz acquisition to 3 Hz. `BIN_SIZE = 10` consecutive frames are averaged (non-overlapping, remainder dropped), giving a **333.33 ms** bin, identical for every trial and session. `metadata['time_bin_size'] = 1000*10/30 = 333.33` ms. Crucially the *same* `bin_average` is applied to the motion-energy trace before it is discretised, so both streams share one time axis and one bin count (3600 bins for 20-min sessions, 5400 for 30-min), and a 60 s trial is 180 bins. Discretisation of motion energy happens strictly after binning.

ii.
```python
BIN_SIZE = 10           # frames averaged together (paper: bins of 10 timestamps)

def bin_average(x, bin_size=BIN_SIZE):
    x = np.asarray(x, dtype=np.float64)
    n = (x.shape[-1] // bin_size) * bin_size
    newshape = x.shape[:-1] + (n // bin_size, bin_size)
    return x[..., :n].reshape(newshape).mean(axis=-1)
```
```python
dff_binned = bin_average(dff)
neural = zscore_rows(dff_binned).astype(np.float32)
...
me, me_info = load_motion_energy(session_dir, nframes)
me_binned = bin_average(me)
...
nbins = neural.shape[1]
assert me_binned.shape[0] == nbins
...
bin_ms = 1000.0 * BIN_SIZE / 30.0
data['metadata'] = dict(..., time_bin_size=bin_ms, bin_size_frames=BIN_SIZE,
                        imaging_rate_hz=30.0, ...)
```

iii. CONVERSION_NOTES.md Step 3 quotes the Methods directly: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" (also matching the calcium-event-rate analysis, "we first denoised the traces slightly by averaging using a bin size of 10 frames"). Step 5 decision 2: "Binning = 10 frames (333.33 ms): exactly the denoising the authors used for all decoding analyses; also reduces trial length to a decoder-friendly 180 bins." Step 10 Check 3(d) confirms the formula matches `RasterWindow.preprocessing()`'s `np.mean(f.reshape(f.shape[0], -1, bin_size), axis=2)` and that it is applied to both dF/F and motion energy. The explicit assert `me_binned.shape[0] == nbins` enforces that the two streams remain the same length.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Not derived from any stored raw variable. It is computed analytically from the bin index and the (constant) frame rate read from `ops['fs']`: `t[i] = (i + 0.5) * BIN_SIZE / fs`, i.e. the **centre** of bin *i* in seconds since the start of that session. There is exactly one input, named `time_in_session_s`, and it is time-varying with shape `(1, 180)` per trial. Values run continuously across trials within a session (they are *not* reset at each trial boundary), spanning [0.167, 1199.833] s for 20-min sessions and [0.167, 1799.833] s for 30-min sessions. The AI separately examined `move_deve/tstamps.npy` during exploration and confirmed the recording durations, but did not use it to build the input.

ii.
```python
input_names=['time_in_session_s'],
...
bin_sec = BIN_SIZE / fs
...
# ---- input: time elapsed from the beginning of the session (bin centres)
tvec = (np.arange(nused) + 0.5) * bin_sec
...
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "frame index / `tstamps.npy` → `input[0]` = `time_in_session_s`; bin centre time = (bin_index + 0.5) * (10/30) s from session start, time-varying; continuous, 0 … 1200/1800 s". The decoder task specifies "Time elapsed from the beginning of the session in seconds. Time-varying." The AI's justification for deriving it from the index rather than `tstamps.npy` is that the 2-photon imaging rate is a fixed 30 Hz (`ops['fs']`, confirmed by the Methods: "Imaging rate was 30 Hz (resonant scanner)"), whereas `tstamps.npy` is the *camera* clock with occasional dropouts; the imaging frame grid is the canonical time base to which the behaviour is aligned. Step 2 notes the `tstamps` cross-check: last value 1.2097 (×1000 s = 1209.7 s ≈ 20 min for 36000-frame sessions) and 1.8145 (1814.5 s ≈ 30 min for 54000-frame sessions).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Essentially none beyond the arithmetic in 3-a. The vector is built once per session for all used bins (`nused = ntrials * bins_per_trial`), then sliced per trial and cast to `float32` to give shape `(1, 180)`. No normalisation, rescaling, standardisation or per-trial re-zeroing is applied — the decoder sees absolute seconds since session onset. No smoothing or interpolation is involved.

ii.
```python
tvec = (np.arange(nused) + 0.5) * bin_sec
...
input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. CONVERSION_NOTES.md Step 5 gives no additional transform for this variable; the AI treated the requirement literally ("Time elapsed from the beginning of the session in seconds"). Step 5's planned sanity checks include "Input time ranges: [0.167, 1199.8] s or [0.167, 1799.8] s", and Step 10 Check 2 verified specific entries against the raw formula: input at (session 0, trial 0, t 0) = 0.1667, (22, 17, 99) = 1053.1667, (40, 29, 179) = 1799.8333 — all PASS. The choice of bin *centres* rather than left edges is documented in the mapping table and makes the input value the mean time of the samples averaged into that bin, consistent with the fact that `neural` and the motion energy in that bin are also averages over the same 10 frames.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction, on the identical bin grid. `tvec` is built with one entry per binned neural timepoint, using the same `bin_sec` (from the same `ops['fs']`) and the same origin (frame 0 of the session), and is then sliced with the *same* `slice` object used for `neural` and `output`. So `input[s][k][0, j]` is the session time of `neural[s][k][:, j]` with no offset. Trial *k* starts at `k*180*bin_sec = k*60` s, so the input also encodes the trial index unambiguously.

ii.
```python
bin_sec = BIN_SIZE / fs
...
tvec = (np.arange(nused) + 0.5) * bin_sec
...
for k in range(ntrials):
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :].astype(np.int64))
```
```python
assert data['input'][s][k].shape[1] == tk   # tk = neural n_timepoints
```

iii. CONVERSION_NOTES.md Step 7 Processing Plots Review: "input time vector is a clean ramp with 60 s trial boundaries at the expected bin indices (every 180 bins)" and "per-trial figure shows neural raster, input and output on the same 180-bin axis, all aligned". Step 10 Check 5 verified "the last trial of a session ends at 1199.833 s / 1799.833 s (last bin centre)", ruling out an off-by-one at the session end. The shared-slice construction is the AI's structural argument that misalignment is impossible, backed by the per-trial shape assertions in `main()`.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. From `move_deve/motion_energy_glob.npy` — the authors' pre-computed global motion energy (uint64, one value per *camera* frame; the sum of squared pixel-wise differences between consecutive video frames). When that array is shorter than the number of imaging frames, `move_deve/interframe_int.npy` (the diff of the camera timestamps) is additionally loaded to locate the dropped frames. `tstamps.npy` was inspected during exploration but is not used in the conversion. There is exactly one output, `motion_energy_quintile`.

ii.
```python
def load_motion_energy(session_dir, nframes):
    md = os.path.join(session_dir, 'move_deve')
    me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
    info = {'n_camera_samples': int(len(me)), 'n_missing': int(nframes - len(me))}
    if len(me) == nframes:
        info['gap_positions'] = []
        return me, info
    ...
    ifi = np.load(os.path.join(md, 'interframe_int.npy'))
```
```python
output_names=['motion_energy_quintile'],
output_values=[['very low (0-20%)', 'low (20-40%)', 'medium (40-60%)',
                'high (60-80%)', 'very high (80-100%)']],
```

iii. CONVERSION_NOTES.md Step 3 quotes the Methods: "we first took each two consecutive frames, computed their pixelwise difference. We then squared all individual pixel-wise values and summed across pixels. This yielded a scalar value quantifying the motion of the mouse at each time point", and "We used the global movements of the mouse as a proxy of its arousal state." The data README confirms `move_deve` "Contains the processed behavioural data (motion energy extracted from videography of spontaneous behaviour 'motion_energy_glob.npy')" and that dropped-frame indices "can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy'". Step 2 also records that the first value of `motion_energy_glob` is 0 because there is no preceding frame.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three stages before discretisation:
1. **Cast to float64** (the raw array is uint64) to avoid overflow/truncation in subsequent arithmetic.
2. **Dropped-camera-frame repair** onto the imaging-frame grid. If `len(me) < nframes`, the number of missing frames in each interframe interval is inferred as `max(round(ifi/median(ifi)) - 1, 0)`; the acquired samples are placed at their true imaging-frame indices `idx[i] = i + sum(nmiss[:i])`, and `np.interp` linearly fills the gaps. If the inferred gaps do not exactly account for the deficit (`idx[-1] != nframes-1`), the code falls back to uniformly stretching the acquired samples over the imaging timeline and records `fallback_uniform`. If `len(me) > nframes`, the trace is truncated.
3. **Temporal binning**: the same `bin_average` (10 frames → 333.33 ms) used for the neural data.

Then the binned trace is discretised (see 4-c).

ii.
```python
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
info = {'n_camera_samples': int(len(me)), 'n_missing': int(nframes - len(me))}
if len(me) == nframes:
    info['gap_positions'] = []
    return me, info
if len(me) > nframes:                      # more camera than imaging frames
    info['gap_positions'] = []
    return me[:nframes], info
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
med = np.median(ifi)
nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
# index of each acquired camera sample in the full imaging-frame timeline
idx = np.arange(len(me)) + np.concatenate([[0], np.cumsum(nmiss)])
if idx[-1] != nframes - 1:
    idx = np.linspace(0, nframes - 1, len(me))
    info['fallback_uniform'] = True
idx = np.clip(idx, 0, nframes - 1)
me_full = np.interp(np.arange(nframes), idx, me)
info['gap_positions'] = np.where(nmiss > 0)[0].tolist()
```
```python
me, me_info = load_motion_energy(session_dir, nframes)
me_binned = bin_average(me)
```

iii. CONVERSION_NOTES.md Step 5 decision 6: "Dropped camera frames: reconstructed from `interframe_int` gaps and linearly interpolated, as suggested by the data README; only done when `len(motion_energy) < n_frames`." Step 2 documents the verification that underpins it: across all 41 sessions the deficit `nframes - len(me)` is *exactly* equal to the number of extra intervals inferred from `interframe_int` in 38/41 sessions; in 3 jm046 sessions a long interval exists but the length already matches ("a timestamp hiccup without data loss"), "handled by only inserting frames when there is an actual length deficit". Step 10 Check 5 records the observed range of 1–116 dropped frames per affected session and the existence of the `fallback_uniform` safety path. The float64 cast is listed in Step 12 "Issues Found and Resolved": "uint64 motion energy cast to float64 before arithmetic".

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into 5 classes, using **rank-based** quintiles computed **per session** on the binned trace (restricted to the bins that end up inside complete trials). `quantile_bin` computes ordinal ranks `r ∈ [1, n]` and assigns `label = ((r-1) * 5) // n`, which yields exactly ⌊n/5⌋ samples per class regardless of ties — the verification log shows exactly 0.200 in each of the 5 classes in every one of the 41 sessions. Labels are stored as `int64` with names `['very low (0-20%)', 'low (20-40%)', 'medium (40-60%)', 'high (60-80%)', 'very high (80-100%)']`. Discretisation is performed *after* binning, never before.

ii.
```python
NCLASSES = 5            # motion energy quintiles (task specification)

def quantile_bin(x, nclasses=NCLASSES):
    """Discretise into nclasses equal-percentile bins (exactly equal counts).

    Rank based, so ties are split evenly and each class holds 1/nclasses of the
    samples of the session.
    """
    r = rankdata(x, method='ordinal')          # 1..n
    lab = ((r - 1) * nclasses) // len(x)
    return lab.astype(np.int64)
```
```python
# ---- output: 5 equal-percentile bins of motion energy, per session
me_used = me_binned[:nused]
labels = quantile_bin(me_used, NCLASSES)
```

iii. CONVERSION_NOTES.md Step 5 decision 5: "Motion energy quintiles per session: required by the task ('discretized into five equal-percentile bins, selected per session'). Percentiles are computed on the binned (3 Hz) motion-energy trace of the whole session, so each session contributes ~20% of samples per class." Step 10 Check 5 justifies the rank-based method over `np.percentile`+`np.digitize`: "Quantile ties: rank-based (`method='ordinal'`) discretisation guarantees exactly equal class counts even if motion-energy values repeat." Step 10 Check 2 validated the result two ways against the raw files: the class means of motion energy increase monotonically with class (1.21, 1.44, 1.83, 3.22, 19.7 ×10⁶) and "class boundaries equal the session's 20/40/60/80 percentiles" (`np.allclose`, rtol 2%). Step 12 Check 3 records that computing thresholds per trial instead of per session "was rejected because the task explicitly requires percentiles to be 'selected per session'".

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame-for-frame on the imaging clock. The behaviour camera is hardware-triggered by the 2-photon acquisition, so camera frame *i* corresponds to imaging frame *i*; the only departure is dropped camera frames, which are re-inserted at their inferred imaging-frame positions (4-b) so that the repaired trace has exactly `nframes` samples. The repaired trace is then binned with the identical `bin_average`, giving the same number of bins as `neural`, which is enforced by `assert me_binned.shape[0] == nbins`. Labels are finally sliced with the same per-trial `slice` used for `neural` and `input`, so `output[s][k][0, j]` describes the same 333.33 ms window as `neural[s][k][:, j]`. No lead/lag shift is introduced between neural activity and behaviour.

ii.
```python
"""Motion energy, one value per imaging frame.

The camera is hardware-triggered by the microscope, so video frame i
corresponds to imaging frame i.  When camera frames were dropped
(len(motion_energy) < nframes) their positions are recovered from the
inter-frame intervals and the trace is linearly interpolated over them
(as suggested in data/README.md).
"""
...
me, me_info = load_motion_energy(session_dir, nframes)
me_binned = bin_average(me)

nbins = neural.shape[1]
assert me_binned.shape[0] == nbins
...
output_trials.append(labels[sl][None, :].astype(np.int64))
```

iii. CONVERSION_NOTES.md Step 3 Processing Details: "Temporal alignment: camera is hardware-triggered by the 2-photon acquisition, so video frame i corresponds to imaging frame i (1:1), except for dropped camera frames which must be re-inserted", citing the Methods ("with the microscope acquisition acting as a trigger for camera frame acquisition, also allowing for simple synchronisation across the two modalities"). The alignment was validated functionally, not just structurally: Step 4's `cache/check_pc1.py` computes corr(PC1 of binned z-scored dF/F, binned motion energy) per session and reproduces Fig. 7D (jm031: 0.024 → 0.424 across P8–P14; jm046: 0.047 … 0.776), and Step 10 Check 4 reproduces Fig. 7C by running the paper's own protocol (ridge on 50 PCs, 5-fold CV on consecutive 2-min blocks): R² ≈ 0 on early days, 0.402–0.754 on the latest days. Both would collapse under a temporal shift.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Five defect classes are handled explicitly:
1. **Dropped camera frames** (0–148 per session; present in 7 sessions): detected by a length deficit against `nframes`, positions inferred from `interframe_int`, values linearly interpolated (see 4-b), with a `fallback_uniform` path if the inferred gaps do not account for the deficit, and a guard for the never-observed `len(me) > nframes` case. Counts are recorded per session in `metadata['session_info']['n_dropped_camera_frames']`.
2. **Timestamp hiccups without data loss** (3 jm046 sessions have a long interframe interval but no length deficit): left untouched, because repair is triggered only by an actual deficit.
3. **Neurons with an identically-zero F trace** (5 neurons): dropped from all sessions of the affected mouse, preventing 0/0 NaNs in the z-score and keeping rows matched across days (see 2-c).
4. **uint64 overflow/truncation risk** in the motion-energy arithmetic: the trace is cast to float64 on load.
5. **Trailing partial trials**: `nbins // bins_per_trial` silently drops any incomplete 60 s block; on this dataset the remainder is always zero, so no data is lost.

Beyond repair, the script asserts its invariants: `iscell[:,0] == 1` everywhere, `me_binned.shape[0] == nbins`, and a full post-assembly sweep checking per-session/per-trial counts, shapes, finiteness and output range.

ii.
```python
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in released data'
...
assert me_binned.shape[0] == nbins
```
```python
if len(me) > nframes:                      # more camera than imaging frames
    info['gap_positions'] = []
    return me[:nframes], info
...
if idx[-1] != nframes - 1:
    # inferred gaps do not exactly explain the deficit: fall back to a
    # uniform stretch of the acquired samples over the imaging timeline
    idx = np.linspace(0, nframes - 1, len(me))
    info['fallback_uniform'] = True
idx = np.clip(idx, 0, nframes - 1)
me_full = np.interp(np.arange(nframes), idx, me)
```
```python
for s in range(ns):
    assert len(data['input'][s]) == len(data['neural'][s]) == len(data['output'][s])
    nn = data['neural'][s][0].shape[0]
    for k in range(len(data['neural'][s])):
        nk, tk = data['neural'][s][k].shape
        assert nk == nn == len(data['brain_region_idx'][s])
        assert data['input'][s][k].shape[1] == tk
        assert data['output'][s][k].shape[1] == tk
        assert np.isfinite(data['neural'][s][k]).all()
        assert np.isfinite(data['input'][s][k]).all()
        assert data['output'][s][k].min() >= 0 and data['output'][s][k].max() < NCLASSES
```

iii. CONVERSION_NOTES.md Step 10 Check 5 ("Check for edge cases") enumerates exactly these cases and the verification behind each: the gap-count identity checked on all 41 sessions (38/41 exact, 3 with no deficit), the off-by-one check at trial/session boundaries, the zero-trace neuron rationale, the tie-robust quantile method, and the uint64 cast. Step 12 "Issues Found and Resolved" lists the zero-trace NaN and the uint64 cast as defects the AI found during development and fixed. The AI's stated principle is to repair rather than discard where the repair is well-determined (dropped frames are a known, documented artefact the data README itself suggests interpolating over), and to fail loudly via assertions where an unexpected condition would otherwise produce silently misaligned data.

## 6-a. What are the most time-consuming steps of the code?

i. The script instruments itself, printing `load`, `neural` and `beh` times per session. The dominant cost is the **`compute_dff` maximin baseline filtering** (the `gaussian_filter` + 60 s `minimum_filter1d`/`maximum_filter1d` chain over `(n_neurons, 36000–54000)` float arrays), together with the binning and z-scoring in the same timer: 0.3 s/session for 220 neurons × 36000 frames up to 1.4 s/session for 746 neurons × 54000 frames. File loading is 0.0–0.1 s/session (warm page cache; it is ~150 MB per session of `F.npy` + `Fneu.npy` + `ops.npy`, so it would dominate on a cold cache), and the behaviour branch is <0.05 s. Whole-dataset wall time is 41.1 s for 41 sessions, of which roughly 35 s is per-session processing and the remainder is pickling the 413.5 MB output.

ii.
```python
t0 = time.time()
ops = load_ops(session_dir)
fs = float(ops['fs'])
F, Fneu, iscell = load_traces(session_dir)
nframes = F.shape[1]
t_load = time.time() - t0
...
t0 = time.time()
dff = compute_dff(...)
dff_binned = bin_average(dff)
neural = zscore_rows(dff_binned).astype(np.float32)
t_neural = time.time() - t0
...
t0 = time.time()
me, me_info = load_motion_energy(session_dir, nframes)
me_binned = bin_average(me)
t_beh = time.time() - t0
...
print(f'    {os.path.basename(session_dir)}: {neural.shape[0]} neurons, '
      f'{nframes} frames, {nbins} bins, {ntrials} trials  '
      f'(load {t_load:.1f}s, neural {t_neural:.1f}s, beh {t_beh:.1f}s), ...')
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: none significant - the dominant cost is the maximin filtering (~0.3-1.2 s/session) and reading ~130 MB of npy per session." Step 7's Run Time Estimates table gives 1.9 s/session and estimates "< 3 min for 41 sessions", against the instructions' 15-minute budget; the actual full run took 41.1 s, i.e. the estimate was conservative and no further optimisation was warranted.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's own assessment is that no loop is a bottleneck, and the code bears this out: the heavy steps are all array-level operations (`gaussian_filter`, `minimum_filter1d`, `maximum_filter1d`, `reshape(...).mean(-1)`, `rankdata`, `np.interp`). The three remaining Python loops are (1) the per-trial assembly loop (`for k in range(ntrials)`), which only creates slice views and per-trial containers and is unavoidable given the target format is a *list* of trials; (2) `find_zero_neurons`' loop over a subject's sessions, which is I/O-bound, not compute-bound; and (3) the top-level loops over subjects and sessions. Notably the AI **avoided** the one loop that is natural to write here: dropped-frame repair is done in a single vectorised `np.interp` over inferred indices rather than by inserting frames one at a time (which would reallocate the array on every insertion).

ii.
```python
nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
idx = np.arange(len(me)) + np.concatenate([[0], np.cumsum(nmiss)])
me_full = np.interp(np.arange(nframes), idx, me)      # vectorised gap repair
```
```python
def bin_average(x, bin_size=BIN_SIZE):
    n = (x.shape[-1] // bin_size) * bin_size
    newshape = x.shape[:-1] + (n // bin_size, bin_size)
    return x[..., :n].reshape(newshape).mean(axis=-1)  # vectorised binning

def zscore_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
```
```python
for k in range(ntrials):                               # cheap; list output required
    sl = slice(k * bins_per_trial, (k + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
```

iii. CONVERSION_NOTES.md Step 6: "Code speedups added: `F.npy` is read fully only once per session; `find_zero_neurons` uses `mmap_mode='r'`; no `ops.npy` (85 MB) loading in the zero-neuron scan; all operations vectorised." Step 7's speed-up table lists the same items. Since the measured full-dataset runtime (41 s) is far inside the instructions' 15-minute threshold, the AI concluded no further vectorisation or parallelisation was needed.

## 6-c. What processing does the code repeat multiple times?

i. The AI's notes claim "`F.npy` is read fully only once per session", but the code in fact reads each `F.npy` **twice**: once in `find_zero_neurons` (opened with `mmap_mode='r'`, then materialised by `np.asarray(F).any(axis=1)`, which touches every byte) and again in `load_traces`. This is by design rather than waste, however: the drop set is the *union* over all of a mouse's sessions, so it must be known before any session of that mouse is assembled, and holding every session's `F` in memory to avoid the re-read would cost several GB. The second genuine repeat is that `ops.npy` (85 MB, mostly registration images) is re-loaded for every session purely to recover five scalars that are constant across the whole dataset (`fs=30`, `neucoeff=0.7`, `baseline='maximin'`, `sig_baseline=10`, `win_baseline=60`). Nothing else is recomputed: dF/F, binning, z-scoring, gap repair and quantile assignment each run exactly once per session, and the pickle is written once.

ii.
```python
def find_zero_neurons(subject_sessions):
    bad = set()
    for sd in subject_sessions:
        F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'), mmap_mode='r')
        z = np.where(~np.asarray(F).any(axis=1))[0]   # first full read of F.npy
        bad.update(z.tolist())
    return sorted(bad)
```
```python
def load_traces(session_dir):
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))             # second full read of F.npy
    ...

def load_ops(session_dir):                            # 85 MB read for 5 scalars
    return np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'),
                   allow_pickle=True).item()
```

iii. CONVERSION_NOTES.md Step 6 presents the `mmap_mode='r'` scan and the exclusion of `ops.npy` from that scan as deliberate speed-ups, i.e. the AI's justification is that the extra pass is the cheap way to obtain a mouse-level quantity (the union of zero-trace neurons) without buffering all sessions. It does not acknowledge that the mmap pass still reads the whole file, nor the per-session `ops.npy` cost; both are consistent with its overall position that "code inefficiencies identified: none significant" given the 41 s total runtime.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four small items, none of which affects correctness and all of which are dwarfed by the 41 s total runtime:
1. **`ops.npy` (85 MB/session) is fully unpickled** to read five scalars that are identical across all 41 sessions; the mean images, registration offsets and other large fields it contains are never used.
2. **The `_raw` dictionary is always built**, even when `--show-processing` is not passed. It keeps `F`, `dff` (full float64, e.g. 746 × 54000 × 8 B ≈ 320 MB), `dff_binned`, `neural`, `me`, `me_binned`, `labels` and `tvec` alive until the caller's `del sess`, so peak memory is far higher than required for a non-plotting run and the arrays are then discarded.
3. **`bin_average` upcasts float32 data to float64** (`np.asarray(x, dtype=np.float64)`), doubling the memory and compute of the binning and z-scoring, after which the result is cast back to `float32`.
4. **`iscell.npy` is loaded** only to be asserted on, and `find_zero_neurons`' `gap_positions`/`n_camera_samples` bookkeeping is retained in metadata but not used by the decoder.

Offsetting these, the AI correctly avoided the biggest available waste: `spks.npy` and `stat.npy` are never read, `ops.npy` is skipped in the zero-neuron scan, and no discarded intermediate is recomputed.

ii.
```python
ops = load_ops(session_dir)                 # 85 MB unpickled for fs/neucoeff/baseline/...
fs = float(ops['fs'])
```
```python
return dict(neural=neural_trials, input=input_trials, output=output_trials,
            ...
            # kept for plotting / sanity checks
            _raw=dict(F=F, dff=dff, dff_binned=dff_binned, neural=neural,
                      me=me, me_binned=me_binned, labels=labels, tvec=tvec))
```
```python
def bin_average(x, bin_size=BIN_SIZE):
    x = np.asarray(x, dtype=np.float64)     # float32 -> float64 upcast
    ...
```
```python
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in released data'
```

iii. CONVERSION_NOTES.md Step 6 states "Code inefficiencies identified: none significant" and Step 7 concludes the conversion is fast enough ("< 3 min for 41 sessions"; actual 41.1 s) that no restructuring was warranted. The `_raw` dictionary is documented in the code itself as "kept for plotting / sanity checks" — the AI's justification is that it enables `--show-processing` figures and the Step 10 spot-checks against raw files from the same in-memory state; the accompanying `del sess` in `main()` bounds the memory to one session at a time. The `ops.npy` load is justified in Step 5 decision 1 as a deliberate choice to take dF/F parameters from "the authors' own `ops.npy`" rather than hard-coding suite2p defaults, so the cost buys provenance.
