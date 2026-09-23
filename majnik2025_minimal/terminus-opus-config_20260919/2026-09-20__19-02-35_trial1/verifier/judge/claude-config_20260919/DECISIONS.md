# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the directory tree under `/app/data`. Every sub-directory of `/app/data` is treated as a subject (the two non-directory entries, `README.md` and `load_data.ipynb`, are skipped by the `isdir` test), and every sub-directory of a subject folder is treated as a session (`ground_truth.csv` in `jm038`/`jm039`/`jm046` is skipped for the same reason). Both levels are sorted alphabetically. For each session the AI loads, from `suite2p/plane0/`: `ops.npy` (only to read `fs`), `F.npy`, `Fneu.npy` and `iscell.npy`; and from `move_deve/`: `motion_energy_glob.npy` and `tstamps.npy`. This yields all 6 mice × 6–7 daily recordings = 41 sessions; trials are created afterwards by cutting each session into 60 s blocks (no trial-level files exist).

ii.
```python
DATA_DIR = '/app/data'
...
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
...
for si, subject in enumerate(subjects):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        sdir = os.path.join(subj_dir, sess)
        ops = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'ops.npy'),
                      allow_pickle=True).item()
        fs = float(ops['fs'])
        F = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'))
        Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
        iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
        ...
        me = load_motion_energy(sdir, nframes)
```

iii. From the trajectory (steps 2–13): the AI first ran `ls -R /app/data`, read `/app/data/README.md` and `load_data.ipynb`, and then ran two survey scripts over every mouse/session printing `F.shape`, `ops['fs']`, `ops['nframes']`, motion-energy length and interframe-interval statistics. After the first survey crashed on `ground_truth.csv` it added the `isdir` guard ("*non-directory entries exist (ground_truth.csv) … Fix survey to skip non-dirs*"). It concluded: "*41 sessions across 6 mice; 30 Hz, 20 or 30 min*". The README states the directory convention explicitly (subject folder → session folder → `suite2p/` and `move_deve/`), so the AI simply followed it and kept every session.

## 1-b. How are the data split into subjects?

i. One subject per top-level directory of `/app/data`, sorted alphabetically: `['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`. The subject list is stored in `data['subjects']` and each session records the index of its mouse in `data['subject_idx']` (final value: 7,7,7,7,6,7 sessions for mice 0–5). No mouse is excluded.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
data = { ... 'subjects': subjects, 'subject_idx': [], ... }
...
data['subject_idx'].append(si)
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The AI relied on the dataset README, which states "*For each subject there is a folder corresponding to the subject id*" and that the six folders `jm031 … jm046` map to mice A–F in the paper. The methods likewise describe "*a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days*", so all six folders are kept and none are filtered.

## 1-c. How are the data split into sessions?

i. One session per sub-directory of a subject folder (one daily recording), sorted alphabetically, which for `YYYY-MM-DD_a` names is also chronological order. All 41 session folders are used; each becomes one entry in `data['neural']` / `data['input']` / `data['output']`. Session identity (`subject`, `session`, `n_neurons`, `n_trials`, `fs_imaging_hz`) is recorded in `metadata['session_info']`.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))])
for sess in sessions:
    sdir = os.path.join(subj_dir, sess)
    ...
    session_info.append({'subject': subject, 'session': sess,
                         'n_neurons': int(neural.shape[0]),
                         'n_trials': int(ntrials),
                         'fs_imaging_hz': fs})
```

iii. Per the README, "*Each subject folder contains a number of session folders, each corresponding to one recording day*". The AI's survey confirmed every session folder contains a complete `suite2p/plane0` and `move_deve` set, so no session is dropped. Sorting gives a deterministic, chronological ordering, which matters because neurons are row-matched across days by Track2p.

## 1-d. How are the data split into trials?

i. The recordings are continuous spontaneous-activity sessions with no task structure, so trials are artificial: each session is cut into consecutive, non-overlapping 60 s blocks. After 10-frame binning the bin size is 1/3 s, so `bins_per_trial = round(60.0 / (10/30)) = 180`. A 20 min session (36 000 frames → 3600 bins) gives 20 trials and a 30 min session (54 000 frames → 5400 bins) gives 30 trials; 1090 trials total. Any partial trial at the end is dropped (here there is no remainder: 3600 and 5400 are both exact multiples of 180).

ii.
```python
TRIAL_SEC = 60.0
...
bin_size = BIN_FRAMES / fs
...
bins_per_trial = int(round(TRIAL_SEC / bin_size))
ntrials = T // bins_per_trial

neural_trials, input_trials, output_trials = [], [], []
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
    output_trials.append(me_cat[sl][None, :])
```

with `metadata['trial_definition'] = 'recordings split into consecutive non-overlapping 60 s trials; incomplete trials at the end are discarded'`.

iii. The instructions say directly "*Split sessions into 60-second trials*", and the paper's own decoding analysis splits the recording into consecutive blocks ("*splits were done on consecutive 2 minute blocks of the recording*"). The AI adopted the instructed 60 s block length, computing the number of bins from the actual `fs` in `ops.npy` rather than hard-coding 180, and noted in step 22 that this gives "*41 sessions with 20 or 30 trials each of 180 bins (60 s at 3 Hz)*" — comfortably more than the required two trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. Every 60 s block of every session of every mouse is kept. The only data discarded at the trial level is a trailing partial trial (which does not occur for this dataset).

ii.
```python
ntrials = T // bins_per_trial          # trailing partial trial dropped
for tr in range(ntrials):
    ...
# no trial rejection criterion anywhere in the file
```

iii. The AI never discusses trial rejection in the trajectory. This is consistent with the source material: because the trials are arbitrary 60 s cuts of a continuous spontaneous-activity recording rather than behavioural trials, there is no per-trial quality metric in the dataset and neither the paper nor the track2p repository defines one. The paper's decoding analysis likewise uses the entire recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy` (raw fluorescence of the Track2p-tracked ROIs). `Fneu.npy` (neuropil) and `iscell.npy` are also loaded and enter the computation formally, but with `neucoeff = 0.0` the neuropil term contributes nothing, and `iscell[:,0] > 0.5` is true for every ROI in this dataset, so the effective source is `F.npy` alone. `ops.npy` supplies the sampling rate `fs` used to set the maximin window and the bin duration.

ii.
```python
F    = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'Fneu.npy'))
iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
keep = iscell[:, 0] > 0.5
F, Fneu = F[keep], Fneu[keep]
nframes = F.shape[1]
dff = maximin_dff(F.astype(np.float64), Fneu.astype(np.float64), ops)
```

iii. The AI's header comment states the choice: "*Neural data: Suite2p F.npy of the Track2p-tracked cells … dF/F = baseline-corrected fluorescence obtained with the Suite2p 'maximin' baseline*". This follows the methods ("*We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses*") and the README, which says the supplied suite2p folders contain "*the neural data for the successfully tracked neurons*". In step 9 the AI explicitly inspected `spks.npy` as an alternative and did not use it, because the paper's decoding is done on dF/F rather than deconvolved spikes.

## 2-b. How is the `neural` data processed?

i. Two steps. (1) Baseline correction with the Suite2p "maximin" method, reimplemented in `maximin_dff()` exactly as in the track2p repository's `F_processing`: Gaussian smoothing along time with `sig_baseline = 10` frames, then a running minimum filter and a running maximum filter with `win = win_baseline * fs = 60 s × 30 Hz = 1800` frames, and subtraction of that baseline. Crucially the AI uses `neucoeff = 0.0`, i.e. **no neuropil subtraction**, copying track2p's default rather than Suite2p's pipeline value of 0.7. (2) Denoising by averaging non-overlapping bins of 10 consecutive frames (30 Hz → 3 Hz), cast to `float32`. No z-scoring, no normalisation by F0, no PCA.

ii.
```python
def maximin_dff(F, Fneu, ops):
    """Baseline-corrected fluorescence (dF/F), as in track2p F_processing /
    Suite2p defaults: maximin baseline."""
    fs = float(ops['fs'])
    neucoeff = 0.0  # track2p F_processing is called without neuropil subtraction
    sig_baseline = float(ops.get('sig_baseline', 10.0))
    win_baseline = float(ops.get('win_baseline', 60.0))
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow

def bin_average(x, binsize):
    x = np.asarray(x, dtype=np.float64)
    n = (x.shape[-1] // binsize) * binsize
    x = x[..., :n]
    newshape = x.shape[:-1] + (n // binsize, binsize)
    return x.reshape(newshape).mean(axis=-1)
...
dff = maximin_dff(F.astype(np.float64), Fneu.astype(np.float64), ops)
neural = bin_average(dff, BIN_FRAMES).astype(np.float32)   # (n_neurons, T)
```

iii. The AI grepped the repository for the dF/F computation (steps 14–15) and found `track2p/gui/data_management.py :: F_processing(self, F, Fneu, fs, neucoeff=0.0, baseline='maximin', sig_baseline=10.0, win_baseline=60.0, ...)`, called at line 90 as `self.F_processing(F=..., Fneu=..., fs=...)` — i.e. with the default `neucoeff = 0.0`. Its reasoning in step 15: "*track2p GUI has F_processing implementing suite2p maximin baseline correction (neucoeff default 0.0). Need to see how it is called (whether neuropil subtraction is applied)*". Rather than settle it by reading alone, in steps 25–28 the AI built an alternative dataset with Suite2p's default `neucoeff = 0.7` and trained the decoder on it: "*Variant with neuropil subtraction gives essentially the same accuracy (0.299 vs 0.306). I will keep the track2p-repo-consistent processing (neucoeff=0)*". The 10-frame binning is taken verbatim from the methods: "*For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps*".

## 2-c. How is the `neural` data filtered based on quality controls?

i. The Suite2p cell-classifier criterion from the paper is applied — ROIs are kept only if `iscell[:, 0] > 0.5` — but in this dataset it is a no-op: the supplied suite2p folders already contain only the Track2p-tracked, curated cells, and `iscell[:, 0]` equals 1 for every ROI in all 41 sessions. No further filtering (SNR, activity level, cross-day consistency) is applied, and no neurons are dropped: 221/370/685/746/541/435 neurons for mice A–F in every session.

ii.
```python
iscell = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'iscell.npy'))
keep = iscell[:, 0] > 0.5
F, Fneu = F[keep], Fneu[keep]
```

iii. The AI verified the filter was redundant before deciding to keep it. Step 10: "*iscell all 1 (already tracked cells only)*", and the step-13 survey printed `iscell[:,0].min()` for every session. Its header comment records the reasoning: "*already curated, iscell == 1 for all provided ROIs; we still apply the iscell > 0.5 criterion used in the paper*" — i.e. the criterion is stated in the methods ("*We considered all ROIs above the default threshold of 0.5 as true cells*") and is kept as an explicit, self-documenting guard even though it removes nothing here.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural alignment event in this dataset — the recordings are continuous spontaneous activity. The AI aligns everything to the **start of the imaging session** (the first 2-photon frame): bin 0 of a session is imaging frames 0–9, and trial *k* covers bins `180k … 180k+179`, i.e. seconds `[60k, 60(k+1))` of the session. Trials are therefore contiguous and non-overlapping, and neural, input and output are cut with the identical slice, so they are aligned by construction. In `metadata` the AI declares `temporal_alignment_event = 'start of the imaging session (first 2-photon frame)'` with `off_start = 0.0` and `off_end = 60.0`.

ii.
```python
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural_trials.append(np.ascontiguousarray(neural[:, sl]))
input_trials.append(tvec[sl][None, :].astype(np.float32))
output_trials.append(me_cat[sl][None, :])
...
'temporal_alignment_event': 'start of the imaging session (first 2-photon frame)',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The AI's step-21 plan states the scheme: "*split into 60 s trials (180 bins) … time from session start as input*". Since the instructions specify "*Time elapsed from the beginning of the session*" as the decoder input, the session start is the only meaningful temporal anchor; the methods confirm the imaging is spontaneous, "*in the dark, under sensory-minimised conditions*", with no stimuli to align to. The `off_start`/`off_end` pair is written relative to the start of each trial (0 s to 60 s of trial extent) rather than relative to the declared session-start event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning is applied. The raw imaging and videography are at 30 Hz (33.3 ms). Both the dF/F traces and the motion-energy trace are averaged over non-overlapping bins of 10 consecutive frames, giving 3 Hz, i.e. a **333.33 ms** time bin, identical for every trial and session. The rebinning is done once per session on the full continuous trace, *before* the trace is cut into trials and *before* motion energy is discretized. `metadata['time_bin_size'] = 1000.0 * 10 / 30.0 = 333.333…` ms. Each trial is therefore 180 bins × 333.33 ms = 60 s.

ii.
```python
BIN_FRAMES = 10          # denoising bin (paper: average of 10 consecutive timestamps)
...
neural = bin_average(dff, BIN_FRAMES).astype(np.float32)
me_b   = bin_average(me, BIN_FRAMES)
T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]
...
bin_size = BIN_FRAMES / fs                     # 1/3 s
...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,   # ms (10 frames at 30 Hz)
```

iii. Straight from the methods: "*For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps*" (the same 10-frame denoising is used for the paper's calcium-event-rate analysis). The AI notes in step 21 "*denoising by averaging 10 consecutive frames (30 Hz -> 3 Hz)*". Binning the two streams with the same helper and then trimming to `T = min(...)` keeps them the same length; binning before discretization is necessary because averaging categorical labels would be meaningless.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from any stored raw variable. It is computed analytically from the bin index and the imaging sampling rate `fs` read from `ops.npy` (30 Hz for all 41 sessions). The camera timestamps in `tstamps.npy` are used only for aligning motion energy, not for building the time axis. The input is named `'time from session start (s)'` and has shape `(1, 180)` per trial.

ii.
```python
fs = float(ops['fs'])
...
bin_size = BIN_FRAMES / fs
# time (s) at the centre of each bin, from the start of the session
tvec = (np.arange(T) + 0.5) * bin_size
...
input_trials.append(tvec[sl][None, :].astype(np.float32))
...
'input_names': ['time from session start (s)'],
```

iii. The imaging is a fixed-rate resonant-scanner acquisition at 30 Hz (methods: "*Imaging rate was 30 Hz (resonant scanner)*"), and `ops['fs']` confirms 30 Hz for every session, so the frame index is an exact proxy for elapsed time; there is no per-frame 2-photon timestamp file to read. The AI took `fs` from each session's `ops.npy` instead of hard-coding it so the conversion stays correct if a session were acquired at a different rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Minimal. A single continuous time vector is built for the whole session as the **centre** of each 333.33 ms bin — `t_i = (i + 0.5)/3` s — then sliced per trial. It therefore runs 0.167, 0.5, 0.833, … within trial 0 and continues 60.167, 60.5, … in trial 1, i.e. it counts from the start of the *session*, not from the start of each trial, and increases monotonically across the whole 20 or 30 min recording. Values are cast to `float32`. No normalisation, centring, or scaling is applied.

ii.
```python
bin_size = BIN_FRAMES / fs
tvec = (np.arange(T) + 0.5) * bin_size
...
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    input_trials.append(tvec[sl][None, :].astype(np.float32))
```
Verified in the saved pickle: session 0 trial 0 input starts `[0.1667, 0.5, 0.8333]`, trial 1 starts `[60.167, 60.5, 60.833]`.

iii. The instruction asks for "*Time elapsed from the beginning of the session in seconds*", so the AI kept raw seconds from session start rather than time within the trial. The bin centre (rather than the leading edge) is the natural representative time for a bin that averages 10 frames. In steps 18–20 the AI checked how `decoder.py` consumes the input stream to decide whether the time variable needed rescaling, and concluded no normalisation was required.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: `tvec` is built with one entry per binned neural time point of the session (`np.arange(T)`, where `T` is the common length after trimming neural and motion energy), and the input is cut with exactly the same `slice` object used for the neural matrix. So `input[s][k][0, j]` is the timestamp of the bin in `neural[s][k][:, j]`, with no interpolation or offset.

ii.
```python
T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]
...
tvec = (np.arange(T) + 0.5) * bin_size
...
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_trials.append(np.ascontiguousarray(neural[:, sl]))
    input_trials.append(tvec[sl][None, :].astype(np.float32))
```

iii. The AI never states this explicitly beyond the inline comment "*time (s) at the centre of each bin, from the start of the session*"; the alignment is implicit in deriving the time axis from the same bin index as the neural data and slicing all three streams with one shared `sl`, which makes misalignment structurally impossible.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. From `move_deve/motion_energy_glob.npy`, the pre-computed global motion-energy trace from the behaviour video, together with `move_deve/tstamps.npy` (camera frame timestamps) which is used only to locate dropped camera frames. The AI does **not** use `interframe_int.npy` (it uses the equivalent `np.diff(tstamps)`). The number of imaging frames `nframes = F.shape[1]` defines the target length.

ii.
```python
def load_motion_energy(session_dir, nframes):
    """Motion energy aligned to the imaging frames (length nframes)."""
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
    ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy')).astype(np.float64)
    ...
...
me = load_motion_energy(sdir, nframes)
```

iii. The README says `move_deve` "*Contains the processed behavioural data (motion energy extracted from videography of spontaneous behaviour 'motion_energy_glob.npy')*", and that missing camera frames "*can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy'*". The methods define the quantity: "*we quantified these by looking at the pixel-wise difference of consecutive frames … squared all individual pixel-wise values and summed across pixels*" — already done for us, so the AI loads it rather than recomputing (the raw `.avi` files are not distributed). Its header comment repeats this derivation. In step 9 the AI inspected all three `move_deve` arrays and their interval statistics before choosing `tstamps.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three stages. (1) **Dropped-frame repair at 30 Hz**: an array of length `nframes` is filled with NaN, the surviving camera samples are scattered into their true imaging-frame positions (positions reconstructed from the rounded ratio of each inter-timestamp interval to the median interval), and the remaining NaNs are linearly interpolated by `interp_nans`. Sample 0 is deliberately forced to NaN (and hence filled from its neighbour) because a frame-difference measure is undefined for the first frame. (2) **Binning**: averaged over the same non-overlapping 10-frame bins as the neural data, then trimmed to the common length `T`. (3) **Discretization**: converted to 5 equal-percentile categories per session (see 4-c).

ii.
```python
def interp_nans(x):
    """Linearly interpolate NaNs (and extrapolate at edges with nearest value)."""
    x = np.asarray(x, dtype=np.float64).copy()
    nans = np.isnan(x)
    if nans.all():
        raise ValueError('all values missing')
    if nans.any():
        idx = np.arange(x.size)
        x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x

def load_motion_energy(session_dir, nframes):
    me = np.load(...'motion_energy_glob.npy').astype(np.float64)
    ts = np.load(...'tstamps.npy').astype(np.float64)
    full = np.full(nframes, np.nan)
    if me.size == nframes:
        full[:] = me
    else:
        # locate dropped camera frames from the interframe intervals
        d = np.diff(ts)
        med = np.median(d)
        steps = np.round(d / med).astype(int)
        steps[steps < 1] = 1
        idx = np.concatenate([[0], np.cumsum(steps)])
        keep = idx < nframes
        full[idx[keep]] = me[keep]
    # first sample has no preceding frame -> not a valid motion energy value
    full[0] = np.nan
    return interp_nans(full)
...
me = load_motion_energy(sdir, nframes)
me_b = bin_average(me, BIN_FRAMES)
```

iii. The AI's header comment: "*When camera frames are dropped (len(motion_energy) < n imaging frames) the dropped frames are located using the interframe intervals in tstamps.npy and linearly interpolated, as suggested in the dataset README*" — the README indeed offers exactly this choice ("*treated as missing values for motion energy or they can be interpolated over*"). The 10-frame averaging is the paper's stated denoising of "*the behaviour traces*". In step 27 the AI cross-checked the reconstruction globally — "*Check total duration of tstamps vs nframes/fs for all sessions*" — and reported in step 28 "*Timestamps confirm missing camera frames are recoverable and alignment is correct (span/median ≈ nframes for every session)*".

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into **5 equal-percentile (quintile) bins whose edges are computed separately within each session**, on the already-binned 3 Hz trace. The four interior cut points are the 20th, 40th, 60th and 80th percentiles of that session's motion energy (`np.quantile(me_b, [0.2, 0.4, 0.6, 0.8])`), and `np.digitize` maps each bin to an integer 0–4. Because the edges are session-specific quantiles, each session is almost exactly balanced — the saved data has 720/720/720/720/720 samples per class in session 0. The five categories are labelled `['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']`.

ii.
```python
N_OUT_BINS = 5           # number of equal-percentile motion-energy bins
...
# discretize motion energy into 5 equal-percentile bins (per session)
edges = np.quantile(me_b, np.arange(1, N_OUT_BINS) / N_OUT_BINS)
me_cat = np.digitize(me_b, edges).astype(np.int64)
...
'output_names': ['motion energy'],
'output_values': [['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']],
'output_discretization': ('motion energy binned in the same 10-frame bins and '
                          'discretized into 5 equal-percentile bins using the '
                          'quantiles of that session'),
```

iii. This is verbatim from the instructions: "*Motion energy, discretized into five equal-percentile bins, selected per session*". Per-session quantiles also handle the fact that motion energy is an uncalibrated, camera- and age-dependent pixel sum that is not comparable across mice or across days. The AI confirmed the result empirically in step 23: "*Verification passed with balanced 20% per class*".

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. One camera frame per imaging frame. The methods state the microscope triggered the camera, so the correspondence is 1:1 at 30 Hz; the AI's job is only to restore that correspondence when the camera dropped frames. It does so positionally: each surviving motion-energy sample is written to the imaging-frame index implied by the cumulative sum of `round(Δt / median(Δt))`, so a doubled interval leaves exactly one empty slot, which is then filled by linear interpolation. The result always has length `nframes`, and is then binned with the same 10-frame helper as the neural data and trimmed to the shared length `T` before being cut with the identical per-trial `slice`. I checked this reconstruction against the raw files: 6 of the 41 sessions have missing camera frames (2, 3, 116, 2, 2, 148, 1, 1, 1 frames), every gap is a single dropped frame, and the AI's timestamp-ratio method recovers exactly the same drop positions and counts as the reference solution's `interframe_int * 1000 > 0.04` threshold.

ii.
```python
    d = np.diff(ts)
    med = np.median(d)
    steps = np.round(d / med).astype(int)
    steps[steps < 1] = 1
    idx = np.concatenate([[0], np.cumsum(steps)])
    keep = idx < nframes
    full[idx[keep]] = me[keep]
...
me = load_motion_energy(sdir, nframes)          # length == nframes
me_b = bin_average(me, BIN_FRAMES)
T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]
...
output_trials.append(me_cat[sl][None, :])       # same slice as neural
```

iii. The AI's header comment: "*The camera was triggered by the 2p microscope, so there is a 1:1 correspondence between camera and imaging frames*", which is the methods' own statement ("*with the microscope acquisition acting as a trigger for camera frame acquisition, also allowing for simple synchronisation across the two modalities*"). It chose a median-ratio reconstruction rather than an absolute interval threshold, which is unit-agnostic (the timestamps are in an unusual unit, ~3.36e-5 per frame) and would also handle multi-frame drops. Step 28: "*Timestamps confirm missing camera frames are recoverable and alignment is correct (span/median ≈ nframes for every session)*".

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms. (1) Dropped camera frames are reconstructed and linearly interpolated as described in 4-b/4-d, so the motion-energy trace is always brought to exactly `nframes`. (2) The first motion-energy sample is treated as invalid (no preceding video frame) and replaced by interpolation from its neighbour. (3) `interp_nans` raises `ValueError('all values missing')` if a trace is entirely NaN. (4) Length mismatches are absorbed defensively: `bin_average` drops any tail shorter than a full 10-frame bin, `T = min(neural.shape[1], me_b.size)` trims both streams to the shorter one, and `ntrials = T // bins_per_trial` discards any trailing partial trial. Non-directory entries (`README.md`, `load_data.ipynb`, `ground_truth.csv`) are skipped by `isdir` checks. There is no assertion that the repaired motion-energy length equals `nframes`, and no session, trial or neuron is ever rejected for quality.

ii.
```python
    if nans.all():
        raise ValueError('all values missing')
...
    # first sample has no preceding frame -> not a valid motion energy value
    full[0] = np.nan
    return interp_nans(full)
...
def bin_average(x, binsize):
    n = (x.shape[-1] // binsize) * binsize
    x = x[..., :n]
    ...
T = min(neural.shape[1], me_b.size)
neural, me_b = neural[:, :T], me_b[:T]
ntrials = T // bins_per_trial
```

iii. The README explicitly warns "*In some recordings there might be some missing frames from the camera*" and sanctions interpolating over them; the AI's survey scripts (steps 10–13) quantified how many sessions were affected and how many frames each was missing before the conversion was written, and step 28 records the global consistency check on timestamp span. The `min(...)` trim and the floor division are belt-and-braces guards: for this dataset they remove nothing, since 36 000 and 54 000 frames give 3600 and 5400 bins, both exact multiples of 180.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is `maximin_dff` — `gaussian_filter` plus `minimum_filter1d` and `maximum_filter1d` with a 1800-frame window over a `(n_neurons, 36000–54000)` float64 array, on the CPU via SciPy. Measured on the largest session (685 neurons × 54 000 frames) this takes ~1.1 s, versus ~0.07 s to load `F.npy` + `Fneu.npy` and ~0.04 s to unpickle `ops.npy`; over 41 sessions the baseline correction is therefore roughly 40 s of a ~1 min run. Everything else — binning, quantiles, `np.digitize`, per-trial slicing, pickling — is negligible. A secondary, avoidable cost is loading `ops.npy` with `allow_pickle=True` for every session (85 MB on disk each, ~3.5 GB read in total) purely to obtain the scalar `fs`, and the float64 upcast of `F`/`Fneu`, which doubles the working-set memory of the filters.

ii.
```python
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow
...
    ops = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'ops.npy'),
                  allow_pickle=True).item()
    fs = float(ops['fs'])
```

iii. The AI does not discuss runtime anywhere in the trajectory or in the code comments; the conversion was fast enough (well under a minute) that it never became an issue. Its choice to reimplement the maximin baseline with SciPy rather than call `suite2p.extraction.dcnv.preprocess` was driven by fidelity to track2p's `F_processing` — which uses exactly these three SciPy calls — even though the AI had confirmed in step 13 that `suite2p` was installed and its `dcnv.preprocess` offers a batched GPU path for the same computation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two, both minor. (1) The per-trial loop `for tr in range(ntrials)` slices and copies each trial one at a time; the entire split is a pure reshape (`neural[:, :ntrials*bins_per_trial].reshape(n_neurons, ntrials, bins_per_trial)`), so all 20–30 trials of a session could be produced at once. Note `np.ascontiguousarray` on each neural slice forces a genuine copy of every trial. (2) The outer subject/session loop is serial and trivially parallelisable across the 41 independent sessions (e.g. `multiprocessing`), which would cut the ~40 s of baseline filtering to a few seconds. Neither matters at this data size. Notably, the AI's dropped-frame repair is already fully vectorized (`np.cumsum` + fancy indexing + `np.interp`) rather than looping over drops.

ii.
```python
    for tr in range(ntrials):
        sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
        neural_trials.append(np.ascontiguousarray(neural[:, sl]))
        input_trials.append(tvec[sl][None, :].astype(np.float32))
        output_trials.append(me_cat[sl][None, :])
```

iii. Not discussed by the AI. The target format explicitly requires a Python list of per-trial arrays, so the loop is in part dictated by the output structure; and `bin_average` and `load_motion_energy` — the places where a naive implementation would loop over frames — were written in vectorized form from the outset.

## 6-c. What processing does the code repeat multiple times?

i. Very little is recomputed. The one clear redundancy is loading and unpickling the 85 MB `ops.npy` once per session solely to read `fs`, which is 30 Hz for all 41 sessions; `sig_baseline`/`win_baseline` are then fetched with `ops.get(..., default)` and in practice always fall back to the defaults. `float(ops['fs'])` is read in the main loop and read again inside `maximin_dff`. `BIN_FRAMES / fs` is recomputed per session, and the literal `30.0` is hard-coded in `metadata['time_bin_size']` while the same quantity is derived from `ops['fs']` for `tvec` — two independent sources for one number. The heavy operations (baseline correction, binning, quantiles) are each performed exactly once per session, and the conversion is a single pass over the data with no re-reading.

ii.
```python
    ops = np.load(os.path.join(sdir, 'suite2p', 'plane0', 'ops.npy'),
                  allow_pickle=True).item()
    fs = float(ops['fs'])
...
def maximin_dff(F, Fneu, ops):
    fs = float(ops['fs'])          # read again
...
    'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,   # hard-coded 30 Hz
```

iii. Not discussed. Reading `fs` per session instead of assuming 30 Hz is a deliberate robustness choice (see 3-a) and the AI passes the whole `ops` dict into `maximin_dff` so that the baseline parameters come from one place; the duplication is a side effect of that.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items, all inconsequential to the result. (1) **`Fneu` is loaded, `iscell`-indexed and upcast to float64 for every session, then multiplied by `neucoeff = 0.0`** — about 1.3 GB read and an array-sized multiply-and-subtract whose result is exactly `F`. (2) **The `iscell` filter is a no-op**: `iscell[:, 0]` is 1 for all ROIs in all 41 sessions, so `keep` is all-True and the fancy-indexing copies of `F` and `Fneu` are pure overhead (the AI knew this and kept the filter deliberately, for documentation). (3) **`ops.npy` (85 MB) is unpickled per session for one scalar**, and `ops.get('sig_baseline')` / `ops.get('win_baseline')` always resolve to the hard-coded defaults. Additionally, `full[0] = np.nan` followed by `interp_nans` is executed even when no frames were dropped, and `np.ascontiguousarray` copies neural slices that a reshape would produce contiguously anyway. Nothing computed is stored but unused in the saved pickle itself: every key in the output dictionary is part of the required format.

ii.
```python
    keep = iscell[:, 0] > 0.5
    F, Fneu = F[keep], Fneu[keep]          # all-True mask: full copies for nothing
...
    neucoeff = 0.0
    Fc = F - neucoeff * Fneu               # Fneu contributes nothing
...
    ops = np.load(..., allow_pickle=True).item()   # 85 MB, used for ops['fs']
```

iii. The AI's own header comment flags item (2) as intentional: "*already curated, iscell == 1 for all provided ROIs; we still apply the iscell > 0.5 criterion used in the paper*" — keeping the paper's stated criterion visible in the code is worth a no-op. Item (1) is the residue of following `track2p`'s `F_processing` signature literally, which takes `Fneu` and a `neucoeff` that defaults to 0; retaining the parameter is what let the AI flip it to 0.7 in step 25 to test the alternative, and it documents that the choice was made rather than overlooked.
