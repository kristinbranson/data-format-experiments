# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the data tree with `os.listdir` in two levels: every directory directly under `/app/data` is treated as a subject, and every directory inside a subject folder is treated as a session. Both levels are sorted, so sessions come out in chronological order (folder names are `YYYY-MM-DD_a`). Non-directory entries (`README.md`, `load_data.ipynb`, and the per-mouse `ground_truth.csv`) are skipped automatically. This yields 6 subjects and 41 sessions. For each session the AI loads only four arrays: `suite2p/plane0/F.npy` (raw fluorescence of the track2p-matched cells), `suite2p/plane0/iscell.npy` (used only to assert that curation was already applied), `move_deve/motion_energy_glob.npy`, and `move_deve/interframe_int.npy`. `Fneu.npy`, `spks.npy`, `ops.npy` and `stat.npy` are deliberately **not** loaded: with the reference `neucoeff=0.0` the neuropil trace is unused, and the frame rate is taken as the documented nominal 30 Hz.

ii.
```python
DATA_ROOT = '/app/data'

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

iii. From CONVERSION_NOTES.md Step 1/2: the loading path mirrors `load_data.ipynb::load_traces` (`np.load(session/suite2p/plane0/F.npy)`) and the data README's description of the folder hierarchy ("For each subject there is a folder corresponding to the subject id … Each subject folder contains a number of session folders, each corresponding to one recording day"). Step 6/Step 10 Check 3 justify the omitted files: "`Fneu` not loaded because the reference dF/F call uses `neucoeff=0.0` (neuropil unused); `ops`/`stat` are not needed (fs is the documented 30 Hz; ROI coordinates unused)", and this saves ~85 MB of I/O per session. The AI cross-checked the resulting counts (6 mice, 41 sessions, 2998 tracked neurons) against the paper's "full dataset of 6 mice imaged daily for a minimum of 6 consecutive days".

## 1-b. How are the data split into subjects?

i. One subject per top-level directory in `/app/data`, sorted alphabetically, giving `['jm031','jm032','jm038','jm039','jm040','jm046']`. The `subjects` list is built from the sorted set of subject names actually present in the processed session list, and `subject_idx` for each session is the position of that session's subject in this list. Note that the AI selects directories by `os.path.isdir` rather than by a `jm` name prefix; in this dataset the two are equivalent.

ii.
```python
subjects = sorted({s[0] for s in sessions})
...
data['subject_idx'].append(subjects.index(subj))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 5 cites the data README: "For cross-referencing with the paper the subjects are named in alphabetically increasing order (e.g. jm031 - mouse A, jm032 - mouse B … jm046 - mouse F)", so alphabetical sorting reproduces the paper's mouse A–F ordering. Step 9 checks the count against the paper's 6 mice.

## 1-c. How are the data split into sessions?

i. One session per subdirectory of a subject folder (e.g. `jm031/2023-10-18_a`), sorted alphabetically, which is chronological given the `YYYY-MM-DD_a` naming. Each session is one daily recording and becomes one element of `neural`/`input`/`output`. 41 sessions total (7,7,7,7,6,7). Sessions of the same mouse are kept as separate sessions even though the neuron rows are matched across days by track2p.

ii.
```python
for sess in sorted(d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))):
    out.append((subj, sess, os.path.join(subj_dir, sess)))
...
for i, (subj, sess, sdir) in enumerate(sessions):
    neural, inp, out, info = process_session(subj, sess, sdir, show_processing=show)
    data['neural'].append(neural)
    data['input'].append(inp)
    data['output'].append(out)
```

iii. CONVERSION_NOTES.md Step 2 quotes the data README: "Each subject folder contains a number of session folders, each corresponding to one recording day … the name of the folder corresponds to the recording date in the YYYY-MM-DD format (the '_a' in the end of the folder name can be ignored)". Step 9 verifies 41 sessions with 7/7/7/7/6/7 per mouse against the paper's "minimum of 6 consecutive days".

## 1-d. How are the data split into trials?

i. There is no trial structure in this dataset (continuous spontaneous recordings), so the AI follows the Decoder Task instruction and cuts each session into consecutive, non-overlapping 60 s blocks: 60 s x 30 Hz = 1800 frames = 180 bins of 10 frames. Sessions are 36000 frames (20 min, jm031/jm032) or 54000 frames (30 min, the other four mice), which are exact multiples of 1800, so the AI asserts that nothing is left over rather than discarding a partial trial. Total 14x20 + 27x30 = 1090 trials.

ii.
```python
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FS))          # 1800
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES                # 180
...
ntrials = nbins // TRIAL_BINS
assert ntrials * TRIAL_BINS == nbins, (
    f'{subj}/{sess}: {nbins} bins is not a multiple of {TRIAL_BINS}')
neural_trials, input_trials, output_trials = [], [], []
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(t_bins[sl].astype(np.float32)[None, :])
    output_trials.append(me_labels[sl].astype(np.int64)[None, :])
```

iii. CONVERSION_NOTES.md Step 3: "the paper has no trials (continuous spontaneous recordings). For this decoder task sessions are cut into 60 s trials." Step 5 Decision 3: "Trial definition: 60 s = 1800 frames = 180 bins, non-overlapping, tiling the session from t=0. Sessions are exact multiples (36000 or 54000 frames) so no partial trials are discarded." Step 9 records "No data lost: for every session `n_trials * 1800 == n_frames`". The AI also notes the paper's own decoding used consecutive 2-minute blocks for cross-validation, so contiguous block segmentation is in the spirit of the reference analysis.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All 1090 trials of all 41 sessions of all 6 mice are kept. The AI explicitly checked that there is nothing to filter on: suite2p's `ops['badframes']` is all `False` in every session, so no imaging frames are flagged bad, and no session is excluded.

ii.
```python
# no filtering code; every session contributes every 60 s block
for k in range(ntrials):
    ...
```
Supporting assertions in `main()`:
```python
for s in range(len(data['neural'])):
    nn = data['neural'][s][0].shape[0]
    for k in range(len(data['neural'][s])):
        assert data['neural'][s][k].shape == (nn, TRIAL_BINS)
        assert np.isfinite(data['neural'][s][k]).all()
        assert np.isfinite(data['input'][s][k]).all()
```

iii. CONVERSION_NOTES.md Step 5 Decision 7: "No further neuron/session curation: all released cells are already suite2p-curated (iscell prob>0.5) and track2p-matched across all days; `badframes` is empty in every session." Step 10 Check 5 adds that `ground_truth.csv` files are manual-tracking benchmark files, not data to exclude. The paper describes no trial rejection because the recordings are continuous and task-free.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Only `suite2p/plane0/F.npy` — the raw fluorescence traces of the cells that track2p matched across all days of that mouse (the released suite2p-format export). `Fneu.npy` is deliberately not used, because the reference `F_processing` in the track2p GUI is called with its default `neucoeff=0.0`, i.e. the neuropil trace does not enter the dF/F used for analysis. `iscell.npy` is loaded only to verify that curation has already been applied; it does not enter the neural signal.

ii.
```python
F = np.load(os.path.join(plane, 'F.npy'))
iscell = np.load(os.path.join(plane, 'iscell.npy'))
...
dF = F_processing(F, None, fs=FS)
```
```python
def F_processing(F, Fneu=None, fs=FS, neucoeff=0.0, ...):
    if neucoeff:
        Fc = F - neucoeff * Fneu
    else:
        Fc = F.astype(np.float32, copy=True)
```

iii. CONVERSION_NOTES.md Step 4 records this as an explicitly investigated discrepancy: "Neuropil coefficient | GUI `F_processing` default `neucoeff=0.0` | ops `neucoeff=0.7` (used by suite2p only for deconvolution) | 'baseline corrected fluorescence traces as our dF/F (default Suite2p parameters)' | Follow the reference code: neuropil is NOT subtracted for the dF/F used in analyses". Step 10 "Issues Found and Resolved" repeats: "Considered neuropil subtraction with ops `neucoeff=0.7` - rejected: the reference `F_processing` call uses `neucoeff=0.0`, and ops' value is only used internally by suite2p for deconvolution." The AI also rejected `spks.npy` because "the paper's decoding uses dF/F".

## 2-b. How is the `neural` data processed?

i. Two steps. (1) Baseline correction identical to the reference implementation `track2p/gui/data_management.py::DataManagement.F_processing`, copied verbatim into the conversion script: maximin baseline (Gaussian smoothing along time with sigma = 10 frames, then a 60 s rolling minimum filter, then a 60 s rolling maximum filter) subtracted from the (neuropil-uncorrected) fluorescence. (2) Averaging in non-overlapping bins of 10 consecutive frames, the paper's decoding pre-processing, giving 3 Hz traces stored as `float32`. No z-scoring, no per-neuron normalization, no deconvolution.

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
def bin_frames(x, nbin=BIN_FRAMES):
    """Average non-overlapping bins of nbin frames along the last axis."""
    x = np.asarray(x)
    T = x.shape[-1]
    nb = T // nbin
    x = x[..., :nb * nbin]
    return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)

dF = F_processing(F, None, fs=FS)
dF_binned = bin_frames(dF).astype(np.float32)          # (nneurons, nbins)
```

iii. CONVERSION_NOTES.md Step 5 Decision 1: "dF/F definition: use the reference implementation `F_processing` with its defaults (no neuropil subtraction, maximin baseline, gaussian sigma=10 frames, 60 s min/max window). Matches the paper's 'baseline corrected fluorescence traces … default Suite2p parameters'." Decision 2 covers binning: "10 frames averaged -> 3 Hz (333.33 ms bins), exactly as in the paper's decoding analysis; applied to both neural and behaviour", citing the Methods sentence "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps". Step 12 reports that a per-neuron z-scored variant was tested and *rejected* because "z-scoring is not part of the reference processing" despite a marginal +0.02 accuracy gain — a deliberate choice of fidelity over score.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering is applied by the conversion script, because the released data is already curated twice over: suite2p's classifier (`iscell[:,0]==1`, probability > 0.5) and track2p's across-day matching. Instead of re-filtering, the AI *asserts* that the curation holds, i.e. that every ROI in every session is flagged as a cell and that `iscell` has one row per trace. This assertion passes for all 41 sessions (2998 unique neurons; 20445 neuron-sessions).

ii.
```python
# --- curation: the released data already contains only suite2p-classified cells
#     (iscell[:,0]==1) that were tracked across all days by track2p. Assert this.
assert iscell.shape[0] == nneurons
assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'
```

iii. CONVERSION_NOTES.md Step 1: "`DataManagement.load_data` … selects cells with `iscell[:,0]==1` (or prob > `iscell_thr`) then reindexes with the track2p match matrix so rows are matched across days. **The released data is already the output of this step** (suite2p-format export of track2p), so no further selection is needed." Step 10 Check 3 row (b): "already applied in the released data; verified by asserting `iscell[:,0]==1` for all ROIs in all 41 sessions | identical". This matches the Methods statement "We considered all ROIs above the default threshold of 0.5 as true cells" and the data README ("the data only includes traces for the cells present across all days").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural alignment event — the recordings are continuous — so trials are aligned to the **start of the imaging session**. Trial k covers bins `[k*180, (k+1)*180)`, i.e. frames `[k*1800, (k+1)*1800)`, counted from the first imaging frame; trials tile the session with no gaps or overlap. The metadata records the alignment event as the session start, with `off_start = 0.0` and `off_end = 60.0` (the within-trial span).

ii.
```python
ntrials = nbins // TRIAL_BINS
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
```
```python
'temporal_alignment_event': (
    'Start of the imaging session; recordings are continuous and are cut into '
    'consecutive non-overlapping 60 s trials.'),
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. CONVERSION_NOTES.md Step 3/Step 5: the paper has no trial events ("the paper has no trials (continuous spontaneous recordings)"), and the Decoder Task only asks for 60 s segmentation. Panel 8 of the `--show-processing` figure was designed to prove the split is lossless: "the trial-concatenated output is identical to the session output" (Step 7), and Step 10 Check 2 verifies by independent recomputation from the raw `.npy` files that trial `k`, neuron `n`, bin `b` equals the mean of raw frames `[k*1800+b*10, +10)` (`np.allclose`, rtol 1e-4).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning from the 30 Hz acquisition rate to 3 Hz. Both the neural dF trace and the motion-energy trace are averaged in non-overlapping bins of 10 consecutive frames, giving a bin size of 10/30 s = 333.33 ms, recorded in `metadata['time_bin_size']` as 333.33 (ms). Each 60 s trial therefore has 180 time points. Binning is applied to the motion energy **before** discretization, so class labels are never averaged. The same `bin_frames` helper is used for both streams so they cannot drift apart in length.

ii.
```python
FS = 30.0               # imaging (and camera) frame rate, Hz (ops['fs'])
BIN_FRAMES = 10         # paper: "averaging in bins of 10 consecutive timestamps"
...
dF_binned = bin_frames(dF).astype(np.float32)          # (nneurons, nbins)
me_binned = bin_frames(me_full)                        # (nbins,)
me_labels, edges = quantile_discretize(me_binned)
...
'time_bin_size': 1000.0 * BIN_FRAMES / FS,     # 333.33 ms
'binning': f'{BIN_FRAMES} frames averaged (paper decoding preprocessing) -> {FS/BIN_FRAMES:.1f} Hz',
```

iii. CONVERSION_NOTES.md Step 3 quotes the Methods directly: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" -> 3 Hz, and notes the same 10-frame binning is used in the paper's calcium-event-rate analysis. Step 4 justifies using the nominal 30 Hz rather than the 29.77 Hz measured from `tstamps`: "Use nominal 30 Hz (as the reference code does, `win=int(60*fs)`); 0.8% timing difference is irrelevant for 60 s trials and is documented."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from any stored variable. The AI computes it arithmetically from the time-bin index and the nominal frame rate, because the acquisition is at a fixed 30 Hz and no per-frame imaging timestamps are stored (the camera `tstamps.npy` exists but is in odd units and belongs to the video stream). The value is the **centre** of each bin, in seconds from the first imaging frame, and it runs continuously across trials within a session (trial k starts at 60k + 0.15 s). Ranges are [0.15, 1199.82] s for 20 min sessions and [0.15, 1799.82] s for 30 min sessions. The input is named `time_from_session_start_s`.

ii.
```python
# --- time from session start (bin centres), seconds
nbins = dF_binned.shape[1]
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
...
input_trials.append(t_bins[sl].astype(np.float32)[None, :])
...
'input_names': ['time_from_session_start_s'],
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "frame index / 30 Hz -> `input[0]` = `time_from_session_start_s`; bin centre time = (trial*1800 + (b+0.5)*10)/30 s", with Decision 8: "Input: single time-varying input, elapsed time from session start in seconds (per Decoder Task)." Step 4 justifies the nominal 30 Hz. Step 10 Check 5 flags the consequence explicitly: "the first time bin is 0.15 s (bin centre), not 0 - documented."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. None beyond the arithmetic above: a `np.arange` over bins scaled by the bin duration and shifted by half a bin, cast to `float32`, then sliced per trial. No smoothing, normalization, or resetting per trial — time is monotonically increasing across the whole session with an exact 1/3 s step.

ii.
```python
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
input_trials.append(t_bins[sl].astype(np.float32)[None, :])
```

iii. CONVERSION_NOTES.md Step 10 Check 2 verifies this independently against the closed-form expectation: "INPUT | expected bin-centre times `(arange(k*180,(k+1)*180)*10+4.5)/30` | all True; t[0]=0.15 s, last trial ends 1199.8167 / 1799.8167 s" and "INPUT continuity | concatenated trials strictly increasing with exact 1/3 s steps in **all 41 sessions** | True".

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: `t_bins` is built with exactly `nbins = dF_binned.shape[1]` entries from the same bin grid, and the identical slice object `sl` is used to cut the neural matrix, the time vector, and the output labels into trial k. So input bin j of trial k is the same 333.33 ms window as neural bin j of trial k. Shape assertions confirm `(1, 180)` for input and `(n_neurons, 180)` for neural in every trial.

ii.
```python
nbins = dF_binned.shape[1]
t_bins = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / FS
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    input_trials.append(t_bins[sl].astype(np.float32)[None, :])
    output_trials.append(me_labels[sl].astype(np.int64)[None, :])
...
assert data['neural'][s][k].shape == (nn, TRIAL_BINS)
assert data['input'][s][k].shape == (1, TRIAL_BINS)
```

iii. CONVERSION_NOTES.md Step 7 describes panel 8 of the processing figure, which plots the concatenated per-trial inputs against the concatenated neural and output traces to show "the trial-concatenated output is identical to the session output". Step 10 Check 2 confirms input continuity in all 41 sessions.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `move_deve/motion_energy_glob.npy` (the pre-computed global motion-energy signal from the behaviour video, one `uint64` value per camera frame) together with `move_deve/interframe_int.npy`, which supplies the camera inter-frame intervals used to recover the true imaging-frame index of each video sample when camera frames were dropped. `tstamps.npy` was examined during exploration but is not used by the converter (it is redundant with the cumulative inter-frame intervals).

ii.
```python
md = os.path.join(sdir, 'move_deve')
me_raw = np.load(os.path.join(md, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(md, 'interframe_int.npy'))
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
```

iii. CONVERSION_NOTES.md Step 2 documents both files ("`motion_energy_glob.npy`: uint64, one value per **camera** frame (camera triggered by the microscope, 30 Hz)"; "`tstamps.npy` … and `interframe_int.npy` … give camera frame times"). Step 3 maps it to the Methods description of motion energy as the summed squared pixelwise difference of consecutive video frames, and the data README's note that "the indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy'".

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four steps. (1) Place the camera samples on the imaging-frame grid: if `len(me) == nframes` the mapping is the identity; otherwise the frame index of each sample is reconstructed as the cumulative sum of `round(ifi / median(ifi))`, which is asserted to land exactly on `nframes - 1`. (2) Fill the resulting gaps by linear interpolation (`np.interp`), and additionally treat `me[0]` as missing, since the first video frame has no predecessor to difference against and its motion energy is identically 0 in every session. (3) Average in the same non-overlapping 10-frame bins as the neural data (3 Hz). (4) Discretize into 5 equal-percentile bins with thresholds computed within that session. No smoothing, log transform, or cross-session normalization is applied to the continuous trace.

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
        assert idx[-1] == nframes - 1, (
            f'frame-index reconstruction failed: {idx[-1]} != {nframes - 1}')
        assert len(idx) == len(me)

    full = np.full(nframes, np.nan)
    full[idx] = me
    full[0] = np.nan                      # first-frame artifact (me[0] == 0)
    bad = np.isnan(full)
    good = ~bad
    full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
    return full, int(bad.sum()), idx
```
```python
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
me_binned = bin_frames(me_full)                        # (nbins,)
me_labels, edges = quantile_discretize(me_binned)
```

iii. CONVERSION_NOTES.md Step 5 Decisions 4–6 and Step 2: "in sessions where `len(motion_energy) < n_frames`, the missing frames are exactly identified by `interframe_int` being an integer multiple of the median interval. Sanity check performed: `cumsum(round(ifi/median(ifi)))[-1] == n_frames-1` holds for **all** sessions with a length mismatch (jm031 x3, jm032 x3, jm039 x1, jm040 x1, jm046 x1)." Decision 5: "me[0]=0 artifact: the first motion-energy sample cannot be computed (no previous frame); it is treated as missing and interpolated". The binning follows the Methods sentence about denoising behaviour traces in bins of 10 timestamps; discretization follows the Decoder Task. Step 10 Check 2 re-derives the labels with independent code for 5 sessions (`np.array_equal` on labels, `np.allclose` on edges — all pass) and checks that mean raw motion energy increases monotonically with label.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into 5 equal-percentile (quintile) bins, with the four interior thresholds computed by `np.quantile` at 20/40/60/80% **on the 3 Hz binned trace of that session only** (not pooled across sessions or mice). Labels are assigned with `np.searchsorted(..., side='right')`, giving integers 0–4 where 0 is the least and 4 the most motion; `output_values` names them `quintile_1 … quintile_5`. Because the thresholds are session-local, each class holds exactly 20% of the time points of every session, which the verification log confirms for all 41 sessions. The per-session threshold values are stored in `metadata['session_info']`.

ii.
```python
NQUANTILES = 5          # decoder task: five equal-percentile bins

def quantile_discretize(x, nq=NQUANTILES):
    """Discretize x into nq equal-percentile bins (thresholds from this session)."""
    edges = np.quantile(x, np.arange(1, nq) / nq)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges

me_labels, edges = quantile_discretize(me_binned)
...
'output_names': ['motion_energy_quintile'],
'output_values': [[f'quintile_{i+1}' for i in range(NQUANTILES)]],
'motion_energy_quantile_edges': edges.tolist(),
```

iii. CONVERSION_NOTES.md Step 5 Decision 6: "Output discretization: quintiles (5 equal-percentile bins) computed **per session** on the 3 Hz binned motion-energy trace, as specified in the Decoder Task." The order (binning before discretization) is justified because averaging class labels would be meaningless. Step 10 Check 2 verifies label ordering ("mean raw motion energy per label in session 0: [777355, 787301, 800618, 812805, 1064356] -> monotonically increasing") and Step 12 uses the same numbers to explain why 5-way quintile accuracy is intrinsically limited: "classes 0-3 are all 'quiescence' and differ mostly by camera/photon noise; only class 4 corresponds to real movement."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera is hardware-triggered by the microscope (Methods), so video frame i corresponds to imaging frame i. The AI relies on this 1:1 correspondence and repairs it where camera frames were dropped: the true imaging-frame index of each surviving video sample is recovered from the inter-frame intervals (which are integer multiples of the median interval), the reconstruction is asserted to end exactly at `nframes - 1`, and the holes are linearly interpolated, producing an array of exactly `nframes` values. This array is then binned with the identical `bin_frames` call used for the neural data and sliced with the identical trial slice, so output bin j of trial k is the same 333.33 ms window as neural bin j. Sessions where the video length already equals `nframes` (including jm046 sessions that show a single long inter-frame interval without a length mismatch) keep the identity mapping.

ii.
```python
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)
me_binned = bin_frames(me_full)
me_labels, edges = quantile_discretize(me_binned)
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
    output_trials.append(me_labels[sl].astype(np.int64)[None, :])
```
```python
    idx = np.concatenate([[0], np.cumsum(steps)])
    assert idx[-1] == nframes - 1, (
        f'frame-index reconstruction failed: {idx[-1]} != {nframes - 1}')
```

iii. CONVERSION_NOTES.md Step 5 Decision 4: "camera and microscope are hardware synchronised; camera frame i == imaging frame i. Where camera frames were dropped (len(me) < nframes), the true frame index of each camera sample is reconstructed from `interframe_int` … and missing values are linearly interpolated. Where len(me)==nframes, mapping is 1:1 (a few jm046 sessions show one timestamp gap but no length mismatch -> treated as a timestamp glitch, per the data README which defines missing frames by the length mismatch)." The strongest end-to-end evidence of correct alignment is Step 12 Check 2: the AI re-ran the paper's own analysis (ridge regression from the converted 3 Hz dF to the continuous binned motion energy, 5-fold CV on consecutive 2-minute blocks) and reproduced Fig. 7C — R^2 near 0 at P7–P11 rising to 0.43–0.79 on the last day of every mouse — concluding "which would be impossible if the neural traces, the motion-energy stream, or their temporal alignment were wrong."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four issues are identified and handled, each with a hard check rather than a silent fallback. (1) **Dropped camera frames** (9 of 41 sessions, 1–148 frames): indices reconstructed from `interframe_int` and linearly interpolated, with an assertion that the reconstruction lands on `nframes - 1`. (2) **The `me[0] == 0` artifact** present in every session: treated as missing and interpolated. (3) **Timestamp glitches without a length mismatch** (some jm046 sessions): deliberately left alone, since the README defines missing frames by the length mismatch. (4) **Partial trials**: rather than silently truncating, the script asserts that the bin count is an exact multiple of 180, which holds for all sessions (36000 and 54000 frames are both multiples of 1800), so no data is discarded. In addition, `iscell` is asserted to be all-ones, shapes are asserted per trial, and all neural and input values are asserted finite before pickling.

ii.
```python
assert idx[-1] == nframes - 1, (
    f'frame-index reconstruction failed: {idx[-1]} != {nframes - 1}')
assert len(idx) == len(me)
...
full[0] = np.nan                      # first-frame artifact (me[0] == 0)
bad = np.isnan(full)
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
```
```python
assert iscell.shape[0] == nneurons
assert np.all(iscell[:, 0] == 1), f'{subj}/{sess}: unexpected non-cell ROIs'
...
assert ntrials * TRIAL_BINS == nbins, (
    f'{subj}/{sess}: {nbins} bins is not a multiple of {TRIAL_BINS}')
...
assert np.isfinite(data['neural'][s][k]).all()
assert np.isfinite(data['input'][s][k]).all()
assert len(data['brain_region_idx'][s]) == nn
```
Per-session bookkeeping is retained in the metadata:
```python
'n_missing_camera_frames': int(n_missing),
'motion_energy_quantile_edges': edges.tolist(),
```

iii. CONVERSION_NOTES.md Step 10 Check 5 enumerates the edge cases: "First bin of a session: `me[0]` is a known artifact (0) and is interpolated … Session lengths are exact multiples of 1800 frames, so no partial trial is dropped and no data is lost (asserted) … Sessions with dropped camera frames (9 of 41, 1-148 frames): index reconstruction asserted to land exactly on `n_frames-1` … jm046 sessions with a single long inter-frame interval but **no** length mismatch: the camera sample count equals the imaging frame count, so the mapping stays 1:1 … `ground_truth.csv` files (3 mice) are manual-tracking ground truth for the tracking benchmark, not neural data - correctly ignored." The data README's guidance ("treated as missing values for motion energy or they can be interpolated over") is cited as the authority for interpolating.

## 6-a. What are the most time-consuming steps of the code?

i. The baseline-correction step dominates: `gaussian_filter` plus the 1800-frame `minimum_filter1d`/`maximum_filter1d` over the full 30 Hz arrays, reported per session in the run log as `dff 0.2 s` (221 neurons x 36000 frames) up to `dff 1.1 s` (685–746 neurons x 54000 frames) — roughly 85–95% of the per-session time. Everything else is small: file loading is ~0.03 s warm (up to ~1 s cold), and binning, motion-energy alignment, trial slicing and assembly together are `bin 0.0–0.1 s`. The whole 41-session conversion took 33.9 s (0.8 s/session). Pickling the 414.5 MB result and, in `--show-processing` mode, rendering the 8-panel figures are the other non-trivial costs.

ii.
```python
t1 = time.time()
dF = F_processing(F, None, fs=FS)
t_dff = time.time() - t1
t1 = time.time()
dF_binned = bin_frames(dF).astype(np.float32)          # (nneurons, nbins)
t_bin = time.time() - t1
...
print(f'  {subj}/{sess}: ... [load {t_load:.1f}s dff {t_dff:.1f}s bin {t_bin:.1f}s '
      f'total {time.time()-t0:.1f}s]')
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: loading `Fneu.npy` and `ops.npy` (85 MB) is unnecessary -> not loaded. `gaussian_filter`/min/max filters dominate runtime (~0.2-0.7 s/session)." Step 7 gives the per-step timing table and the extrapolation to "~2 min" for the full dataset; Step 9 records the actual 33.9 s, far inside the 15-minute budget, so no further optimization was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Very little is left to vectorize. The motion-energy gap filling — the one place where a naive implementation would loop and repeatedly `np.insert` — is already fully vectorized as a scatter into a NaN-filled array followed by a single `np.interp`. Binning is a reshape-and-mean, and discretization is a single `np.searchsorted`. The remaining Python loops are (a) the per-trial slicing loop in `process_session` (20 or 30 iterations per session, 1090 total, each doing only a slice and a copy), (b) the per-session loop in `main`, and (c) the plotting loops over 3 example neurons. (a) could in principle be replaced by a single reshape/`np.split`, but each iteration is O(n_neurons x 180) memory traffic that must happen anyway, so the gain would be negligible; (b) is where multiprocessing could help but is unnecessary at 0.8 s/session. A genuine (small) waste is `np.ascontiguousarray`, which copies each trial's neural block.

ii.
```python
# vectorized gap filling (no per-frame insertion loop)
full = np.full(nframes, np.nan)
full[idx] = me
bad = np.isnan(full)
good = ~bad
full[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), full[good])
```
```python
# vectorized binning and discretization
return x.reshape(*x.shape[:-1], nb, nbin).mean(axis=-1)
labels = np.searchsorted(edges, x, side='right').astype(np.int64)
```
```python
# the remaining (cheap) per-trial loop
for k in range(ntrials):
    sl = slice(k * TRIAL_BINS, (k + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))
```

iii. CONVERSION_NOTES.md Step 6: "Code speedups added: memory-light loading (only `F.npy`, `iscell.npy`, 3 small behaviour arrays), vectorised binning and discretization, no per-trial Python loops over timepoints. Result: ~2 s/session." Step 7's table lists "vectorised binning/discretization | negligible loops". Since the measured full run is 33.9 s, the AI judged further vectorization or parallelization unnecessary.

## 6-c. What processing does the code repeat multiple times?

i. Essentially nothing substantive is recomputed: each session is loaded once, dF is computed once, and both streams are binned once. The repetitions that do exist are cheap and diagnostic: the end-of-run sanity block makes several separate passes over the assembled data (concatenating all outputs, all inputs, and a strided subsample of all neural arrays to print class fractions, input range and neural statistics), the per-session summary recomputes `np.bincount` on labels that were just built, and in `--show-processing` mode `np.nanpercentile(F, 99)` and `np.nanpercentile(dF, 99)` are recomputed inside the 3-neuron plotting loops instead of being hoisted. Separately, the exploratory work outside the converter (`/app/cache/sanity_checks.py`) intentionally recomputes dF and the motion-energy pipeline with independent code — that duplication is the point of the check, not waste.

ii.
```python
frac = np.bincount(me_labels, minlength=NQUANTILES) / len(me_labels)   # per session, for printing
...
allout = np.concatenate([o[0] for s in data['output'] for o in s])
fr = np.bincount(allout, minlength=NQUANTILES) / len(allout)
allin = np.concatenate([i[0] for s in data['input'] for i in s])
allneu = np.concatenate([n.ravel()[::101] for s in data['neural'] for n in s])
```
```python
for i in range(n_show):
    ax.plot(tf, F[i] + i * np.nanpercentile(F, 99), lw=.4)   # recomputed each iteration
```

iii. CONVERSION_NOTES.md does not flag these as problems, and treats the repeated passes as validation rather than computation: Step 6 lists the printed "class fractions/input range/neural stats" as part of the script's built-in assertions, and Step 10 Check 2 describes the deliberately independent re-implementation used for spot checks ("single-neuron `gaussian_filter1d` + min/max filters instead of the 2-D filters used in the converter"), which is precisely the duplication that makes the sanity check meaningful.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, and all of it minor. (1) `iscell.npy` is loaded in every session purely to assert curation; its contents never reach the output. (2) `align_motion_energy` returns `sample_idx`, which is used only by the plotting path and is computed/returned even when plotting is off. (3) A dead local `ops = None` is assigned and never used. (4) `np.ascontiguousarray` copies every trial's neural block although the slice would already be usable. (5) The full 30 Hz `dF` array (up to 746 x 54000 float32) is materialized and then discarded after binning — unavoidable given that maximin baselining must run at full resolution. (6) Rich `session_info` metadata (per-session quantile edges, missing-frame counts, durations) is stored in the pickle and ignored by `train_decoder.py`, though it is useful provenance. (7) The end-of-run summary statistics and, in `--show-processing` mode, the 8-panel figures are pure diagnostics. Nothing computed for the decoder itself is thrown away — notably, `Fneu.npy`, `spks.npy`, `ops.npy` and `stat.npy` are never read at all.

ii.
```python
iscell = np.load(os.path.join(plane, 'iscell.npy'))
ops = None                                      # assigned, never used
...
me_full, n_missing, sample_idx = align_motion_energy(me_raw, ifi, nframes)   # sample_idx only for plots
...
neural_trials.append(np.ascontiguousarray(dF_binned[:, sl]))                 # extra copy
...
'session_info': session_info,   # not consumed by the decoder
```

iii. CONVERSION_NOTES.md Step 6 shows the AI's stance is to *remove* unnecessary work: "loading `Fneu.npy` and `ops.npy` (85 MB) is unnecessary -> not loaded"; "memory-light loading (only `F.npy`, `iscell.npy`, 3 small behaviour arrays)". The retained items are justified as verification and provenance rather than computation: Step 10 Check 3 explains that reading `iscell` is how the claim "curation already applied" is verified in all 41 sessions, and Step 7/Step 10 treat the plots and printed statistics as the evidence that the conversion is correct.
