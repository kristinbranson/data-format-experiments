# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. One MATLAB file per session, `data_structure_<anm>_<date>.mat`, read directly with `h5py` (all of these files are MATLAB v7.3/HDF5), plus a companion `motionEnergy_<anm>_<date>.mat` read with `scipy.io.loadmat`. The session list is **not** globbed: 25 `(animal, date, probe)` triples are hard-coded in `SESSIONS`, transcribed from the authors' `DataLoadingScripts/'Recording and video'/load<ANM>_ALMVideo.m` scripts (I verified the probe ids against those scripts — e.g. `EKH1 2021-08-07 → probe 2`, `JEB15 2022-07-26 → probes [1 2]`, `JEB15 2022-07-29 → probe 2`). `DATA_DIR` is hard-coded to `/app/data/Ephys_Behavior`, so **only the fixed-delay dataset is loaded**; the 19 sessions of `RandomizedDelay_Ephys_Behavior` (4 mice, 845 units in the paper) are deliberately excluded, as are the two optogenetics-only behaviour folders. `main()` loops over `SESSIONS`, calls `convert_session`, and appends each result.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
    ('JEB13', '2022-09-13', [2]),
    ...
    ('JGR3',  '2021-11-18', [1]),
]

def convert_session(anm, date, probes, verbose=True):
    path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
    f = h5py.File(path, 'r')
    obj = f['obj']
    b = load_behavior(f, obj)
```
```python
def main():
    sessions = []
    for anm, date, probes in SESSIONS:
        s = convert_session(anm, date, probes)
        if s is None:
            print(f'skipping {anm}_{date}')
            continue
        sessions.append(s)
```

iii. From the module docstring: `sessions / probes -> DataLoadingScripts/'Recording and video'/load<ANM>_ALMVideo.m`. From the agent's final report: *"the 25 fixed-delay ALM ephys+video sessions in `Ephys_Behavior`, with the exact per-session ALM probe ids transcribed from `load<ANM>_ALMVideo.m`. This is the paper's full DR dataset (12 two-context sessions + 13 DR-only, matching the Methods' '25 sessions, 1,651 units'; I get 1,531 after the >1 Hz criterion). I **excluded** the randomized-delay dataset: the delay varies 0.3–3.6 s, so the pre-go-cue window would contain a different mixture of task epochs across trials than in the fixed-delay sessions, and those mice never experienced the WC context. The optogenetic directories have no neural data."* The trajectory also shows that the agent's survey script crashed with `OSError: file signature not found` when it reached a randomized-delay file (those files are MATLAB v5, not v7.3), and the final loader only ever uses `h5py`.

## 1-b. How are the data split into subjects?

i. The animal id is the `anm` field of each `SESSIONS` entry (i.e. the filename prefix), carried through into `session_info['subject']`. At assembly, `subjects` is the sorted set of unique animal ids and `subject_idx` indexes into it, one entry per session in session order. This yields 10 subjects over the 25 sessions.

ii.
```python
'session_info': {
    'subject': anm,
    'date': date,
    ...
```
```python
subjects = sorted({s['session_info']['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['session_info']['subject'])
                        for s in sessions], dtype=np.int64)
```

iii. Not stated explicitly. The animal id is taken from the authors' load scripts rather than from `obj.meta`, which is absent/inconsistent in several files (the agent's probing in the trajectory shows `obj` key sets varying between sessions, e.g. `ex.probe` missing in EKH1/JEB7/JGR2/JGR3).

## 1-c. How are the data split into sessions?

i. One entry of `SESSIONS` = one `data_structure_*.mat` file = one session = one element of `neural`, `input`, `output`, `brain_region_idx` and `metadata['session_info']`. Sessions are emitted in the order listed in `SESSIONS`, and the two-probe sessions (JEB15 2022-07-26/27/28) are concatenated into a single population rather than split into two sessions. A session is dropped only if it would have fewer than two usable trials or no surviving units.

ii.
```python
    keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
    trials = np.where(keep)[0] + 1                  # 1-based MATLAB trial ids
    if len(trials) < 2:
        f.close()
        return None
    ...
    rates, nquality, keep_ids = neural_matrix(f, obj, probes, trials, align)
    if rates is None or rates.shape[2] == 0:
        f.close()
        return None
```
```python
    data = {
        'neural': [s['neural'] for s in sessions],
        'input':  [s['input']  for s in sessions],
        'output': [s['output'] for s in sessions],
```

iii. Implicit in the docstring's mapping of sessions to `load<ANM>_ALMVideo.m`. No session-level quality criterion (e.g. the paper's "at least 10 units") is applied; it would not have bound anyway, since the smallest retained session has 27 units.

## 1-d. How are the data split into trials?

i. A trial is one row of the Bpod table `obj.bp`: `bp.Ntrials` gives the count and every per-trial field is truncated to that length (several fields are stored longer). Trial identity is carried as 1-based MATLAB trial numbers through `trials`; spikes carry their own `clu.trial` index and camera data are stored per trial in `obj.traj{view}(trial)`, so no trial boundaries have to be reconstructed. There is exactly one `bp.ev.goCue` per trial.

ii.
```python
def load_behavior(f, obj):
    bp = obj['bp']
    ev = bp['ev']
    n = int(vec(bp, 'Ntrials')[0])
    b = {
        'n': n,
        'R': np.nan_to_num(vec(bp, 'R')[:n]) > 0,
        'L': np.nan_to_num(vec(bp, 'L')[:n]) > 0,
        'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
        'miss': np.nan_to_num(vec(bp, 'miss')[:n]) > 0,
        'no': np.nan_to_num(vec(bp, 'no')[:n]) > 0,
        'early': np.nan_to_num(vec(bp, 'early')[:n]) > 0,
        'autowater': np.nan_to_num(vec(bp, 'autowater')[:n]) > 0,
        'stim': np.nan_to_num(vec(bp['stim'], 'enable')[:n]) > 0,
        'goCue': vec(ev, ALIGN_EVENT)[:n],
        ...
```
```python
    # map 1-based MATLAB trial ids onto rows of the output matrix (-1 = trial dropped)
    trial_pos = -np.ones(int(len(align)) + 2, dtype=np.int64)
    trial_pos[trials] = np.arange(len(trials))
```

iii. Not stated explicitly; it follows the structure of `obj.bp` as explored in the trajectory.

## 1-e. How are trials filtered based on quality controls?

i. Three masks, applied once up front and used for every stream: drop early-lick trials (`bp.early`), drop photostimulation trials (`bp.stim.enable`), and drop trials with a non-finite go cue. Hit, miss and ignore trials are all kept. Across the 25 sessions this keeps 7,426 of 8,829 trials. No cut is made for trials that run past the end of the recording; the agent spot-checked three sessions for this (I re-ran the check over all 25 sessions and confirmed the last spiking trial equals `Ntrials` in every one, so the cut would have been a no-op here).

ii.
```python
    # ---- trial curation -------------------------------------------------- #
    # `early` (lickport contact before the response epoch) and photostimulation
    # trials are excluded from every analysis in the paper.  hit / miss / no
    # (ignore) trials are all kept because outcome and lick direction are decoded.
    keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
    trials = np.where(keep)[0] + 1                  # 1-based MATLAB trial ids
```

iii. Docstring: `trial curation -> the '~early & ~stim.enable' masks that every analysis condition in the paper's scripts uses`. Final report: *"dropped early-lick and photostimulation trials (`~early & ~stim.enable`, the mask behind every condition in the paper's scripts); kept hit/miss/ignore, since outcome and lick direction are decoded."* The trajectory shows the agent verifying with `python -c` that after this filter the delay is exactly 0.9 s and the sample 1.3 s on every retained trial of every fixed-delay session, so no time warping is needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) named by the authors' load script: per cluster, `quality` (curation label), `trial` (1-based trial of each spike) and `trialtm` (spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment times. Units from the two probes of a two-probe session are concatenated into one population. All units are labelled `ALM` (a single brain region).

ii.
```python
def clu_groups(f, obj):
    """Return a list (indexed by probe-1) of the per-probe cluster structs."""
    clu = obj['clu']
    if isinstance(clu, h5py.Group):          # single probe stored as a plain struct
        return [clu]
    out = []
    for i in range(clu.shape[0]):
        g = f[clu[i, 0]]
        out.append(g if isinstance(g, h5py.Group) else None)
    return out
```
```python
    for p in probes:
        g = groups[p - 1]
        if g is None:
            continue
        qualities = [mat_str(f, r) for r in np.array(g['quality']).flatten()]
        for icell, q in enumerate(qualities):
            ...
            tm = np.array(f[g['trialtm'][icell, 0]]).flatten()
            tr = np.array(f[g['trial'][icell, 0]]).flatten().astype(np.int64)
```

iii. Docstring: `alignment -> params.alignEvent = 'goCue' (alignSpikes.m)`; `brain_regions` comment: *"every analysed probe is the ALM probe designated by the paper's `load<ANM>_ALMVideo.m` scripts."*

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins per trial (`bincount` on a flattened `trial × bin` index), divided by the bin width to give spikes/s, and smoothed along time with the authors' **causal** kernel: `gausswin(15)` with the acausal half zeroed and renormalised, applied with a `'reflect'` boundary — a direct port of `mySmooth.m`/`getSeq.m`. No normalisation, baseline subtraction or z-scoring. Output is `float32` Hz. The agent verified the port numerically against `np.convolve(..., 'same')` with the MATLAB kernel (max difference 1.1e-16).

ii.
```python
def _causal_gauss_kernel(n=SMOOTH_N):
    """gausswin(n) with the acausal half zeroed and renormalised (mySmooth.m)."""
    k = np.arange(n)
    alpha = 2.5
    w = np.exp(-0.5 * (alpha * (2 * k - (n - 1)) / (n - 1)) ** 2)
    w[: n // 2] = 0.0          # kern(1:floor(numel(kern)/2)) = 0  -> causal
    return w / w.sum()

def my_smooth(x):
    """mySmooth(x, 15, 'reflect') applied along axis 0 of a 2-D array."""
    if BCTYPE == 'reflect':
        pad = x[:SMOOTH_N]
        xf = np.concatenate([pad, x], axis=0)
        return lfilter(_TAPS, [1.0], xf, axis=0)[SMOOTH_N:]
    return lfilter(_TAPS, [1.0], x, axis=0)
```
```python
            counts = np.bincount(row * NBINS + b,
                                 minlength=len(trials) * NBINS).reshape(len(trials), NBINS)
    ...
    counts = np.stack(keep_rates, axis=-1)                     # (ntrials, nbins, nunits)
    rates = counts.astype(np.float64) / DT
    flat = rates.transpose(1, 0, 2).reshape(NBINS, -1)         # time first for smoothing
    flat = my_smooth(flat)
    rates = flat.reshape(NBINS, len(trials), -1).transpose(1, 0, 2)
    return rates.astype(np.float32), nquality, keep_ids
```

iii. Docstring: `binning + smoothing -> getSeq.m (tmin=-2.5, tmax=2.5, dt=1/100, causal Gaussian kernel of 15 bins, 'reflect' boundary)`. Final report: *"ported `getSeq.m`/`mySmooth.m` exactly — 10 ms spike counts → spikes/s → causal 15-bin Gaussian with the `'reflect'` boundary (verified numerically against the MATLAB `conv(...,'same')` semantics)."* Metadata field: `neural_units: 'spikes/s (spike counts in 10 ms bins, smoothed with a causal Gaussian kernel of 15 bins as in getSeq.m/mySmooth.m)'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Manual curation label: the cluster's `quality` string, lower-cased, is dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the exclusion list in `findClusters.m` under `params.quality = {'all'}`. Everything else survives, including multi-units, `poor`/`fair` units, and the 203 unlabelled clusters. (2) Firing rate: the all-trials PSTH (summed counts / n trials / dt, then smoothed) must have a mean above 1 Hz, mirroring `removeLowFRClusters.m`. This leaves 1,531 units over 25 sessions (27–141 per session).

ii.
```python
# cluster qualities that findClusters.m rejects when params.quality = {'all'}
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0         # params.lowFR (Hz)
...
        for icell, q in enumerate(qualities):
            if q.lower() in BAD_QUALITY:
                continue
            nquality += 1
            ...
            # condition-1 PSTH ("all trials") used by removeLowFRClusters.m
            psth = my_smooth((counts.sum(axis=0) / len(trials) / DT)[:, None])[:, 0]
            if psth.mean() > LOW_FR:
                keep_rates.append(counts)
                keep_ids.append((p, icell))
```

iii. Docstring: `unit curation -> findClusters.m ('all' = every quality except garbage/noisy/real?) + removeLowFRClusters.m (>1 Hz)`. Metadata: *"cluster qualities other than garbage/noisy/real? (findClusters.m with params.quality = 'all'), then units with mean firing rate <= 1 Hz removed (removeLowFRClusters.m with params.lowFR = 1)."* The agent enumerated every quality string in the dataset first (`Counter({'garbage': 8790, 'multi': 574, 'poor': 307, 'fair': 304, 'Poor': 258, '': 203, ...})`), which is why the comparison is done lower-cased.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction: `clu.trialtm` is already on the behaviour clock and relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm - goCue[trial]` gives seconds from the go cue. Spikes outside `[-2.5, 2.5)` are dropped by the mask; the bin index is then `floor((aligned - TMIN)/DT)`. No interpolation or extra offset. (The video streams get a separate clock correction — see 7-d.)

ii.
```python
            inrange = (tr >= 1) & (tr <= len(align))
            tm, tr = tm[inrange], tr[inrange]
            row = trial_pos[tr]
            aligned = tm - align[tr - 1]
            ok = (row >= 0) & (aligned >= TMIN) & (aligned < TMAX)
            row = row[ok]
            b = ((aligned[ok] - TMIN) / DT).astype(np.int64)
```

iii. Docstring: `alignment -> params.alignEvent = 'goCue' (alignSpikes.m)`. The agent's metadata entry adds a caveat: `temporal_alignment_event: 'go cue onset (DR trials); on WC trials the same event field marks the water-drop presentation, which is the analogous movement-releasing event'`. Alignment was sanity-checked in the trajectory: population rate steps up at bin 250 (t = 0) and left-vs-right selectivity ramps through the delay and peaks after the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins**, 500 bins spanning −2.5 s to +2.5 s around the go cue, identical for every trial and session. The grid is built once at module level exactly as `getSeq.m` builds `obj.time` (`edges = tmin:dt:tmax; time = edges + dt/2; time = time(1:end-1)`). No rebinning is applied anywhere: spikes are histogrammed once at 10 ms, and the video streams (400 Hz) are linearly interpolated straight onto the same 500-point axis, so all streams share one time base.

ii.
```python
TMIN = -2.5          # params.tmin
TMAX = 2.5           # params.tmax
DT = 1.0 / 100.0     # params.dt  (10 ms bins)
...
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)          # 501 edges
TAXIS = EDGES[:-1] + DT / 2                          # obj.time, 500 bin centres
NBINS = len(TAXIS)
```
```python
            'time_bin_size': DT * 1000.0,
            'off_start': TMIN,
            'off_end': TMAX,
```

iii. Declared as `params.dt`. Note on provenance: `DataLoadingScripts/getDefaultParams.m` sets `params.dt = 1/200` (5 ms), but `params.dt = 1/100` is what the repo's actual analysis scripts use — `WorkingWithDataObjs.m`, `ParallelAnalysis/dimensionality.m`, `Scripts/EDFigure 3/EDFigure3.m` and all the `Behavior/*.m` scripts all set `params.dt = 1/100` (together with `params.lowFR = 1` and `params.bctype = 'reflect'`, both of which the agent also adopted). The agent did not spell this out; the docstring simply attributes `dt=1/100` to `getSeq.m`, which does not itself define `dt`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files — it is the analysis time axis itself, the centres of the 500 bins of the go-cue-aligned window. It is identical for every trial and every session, and it is the only input (`d_input = 1`).

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)          # 501 edges
TAXIS = EDGES[:-1] + DT / 2                          # obj.time, 500 bin centres
```
```python
    'input_names': ['time_from_go_cue'],
```

iii. Implicit: this is `obj.time` from `getSeq.m`, the same axis the neural data is binned onto.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. `TAXIS` is cast to `float32`, given a leading singleton dimension to make it `(1, 500)`, and copied once per trial.

ii.
```python
    time_input = TAXIS.astype(np.float32)[None, :]
    for i in range(len(trials)):
        ...
        inputs.append(time_input.copy())
```

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural grid. Spike times are expressed relative to the go cue and indexed into bin `floor((t − TMIN)/DT)`, and the input is the centre of that same bin, so bin *k* denotes the same 10 ms interval in `neural`, `input` and all six `output` channels. Range is [−2.495, 2.495] s.

ii.
```python
            b = ((aligned[ok] - TMIN) / DT).astype(np.int64)
```
```python
TAXIS = EDGES[:-1] + DT / 2                          # obj.time, 500 bin centres
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial Bpod flags: `bp.R` and `bp.L` (the instructed/rewarded port) together with `bp.hit` and `bp.miss`. The direction actually licked is not stored, so it is inferred from the pair.

ii.
```python
        'R': np.nan_to_num(vec(bp, 'R')[:n]) > 0,
        'L': np.nan_to_num(vec(bp, 'L')[:n]) > 0,
        'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
        'miss': np.nan_to_num(vec(bp, 'miss')[:n]) > 0,
```

iii. Not stated in prose, but the trajectory shows the inference being validated directly against the lick-port contact times: the agent decoded `bp.ev.lickL`/`lickR`, took the first lick after the go cue, and cross-tabulated it against the hit/miss-derived label — the two agreed on every trial with a post-go-cue lick (`('DR','L','L') 85, ('DR','R','R') 43, ('WC','L','L') 35, ('WC','R','R') 37`, with only 2 disagreements out of ~284, both on `none`-labelled trials).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port, a miss means it licked the other one, anything else means it did not lick. Codes: left 0, right 1, none 2, with `none` as the default. The value is constant within a trial and is broadcast across all 500 bins.

ii.
```python
    # lick direction: R&hit or L&miss -> right, L&hit or R&miss -> left, ignore -> none
    lick = np.full(len(idx), 2, dtype=np.int8)                     # 2 = none
    licked_right = (b['R'][idx] & b['hit'][idx]) | (b['L'][idx] & b['miss'][idx])
    licked_left = (b['L'][idx] & b['hit'][idx]) | (b['R'][idx] & b['miss'][idx])
    lick[licked_left] = 0
    lick[licked_right] = 1
```
```python
        out = np.empty((6, NBINS), dtype=np.int8)
        out[0] = lick[i]
```
```python
        'output_values': [
            ['left', 'right', 'none'],
```

iii. Implicit in the inline comment. The ordering left/right/none follows the prompt's "(left, right, none)".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `bp.autowater`, which marks the water-cued block.

ii.
```python
        'autowater': np.nan_to_num(vec(bp, 'autowater')[:n]) > 0,
```

iii. Implicit; the metadata `task_description` explains that the WC context is the "autowater" context in which all auditory cues are omitted and water is presented at a random port.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: autowater → WC (0), everything else → DR (1), constant within a trial and broadcast over the 500 bins. 8 of the 25 sessions were DR-only, and three of those contain no WC trials at all (their `context` channel is constant 1).

ii.
```python
    context = np.where(b['autowater'][idx], 0, 1).astype(np.int8)  # 0 = WC, 1 = DR
...
        out[1] = context[i]
```
```python
            ['WC', 'DR'],
```

iii. Codes follow the prompt's "(WC, DR)". The per-session counts are recorded in `session_info` (`nWC_trials`, `nDR_trials`, `two_context_session`).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags, `bp.hit` and `bp.miss`. `bp.no` is loaded into the behaviour dict but never used — a trial that is neither hit nor miss is an ignore by construction.

ii.
```python
        'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
        'miss': np.nan_to_num(vec(bp, 'miss')[:n]) > 0,
        'no': np.nan_to_num(vec(bp, 'no')[:n]) > 0,      # loaded, never used
```

iii. Implicit. The agent's trial-curation comment notes that ignore trials are deliberately retained: *"hit / miss / no (ignore) trials are all kept because outcome and lick direction are decoded"* (the paper omits ignore trials from its own analyses).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Three classes: default ignore (2), then miss → incorrect (0) and hit → correct (1). Constant within a trial, broadcast over the 500 bins.

ii.
```python
    outcome = np.full(len(idx), 2, dtype=np.int8)                  # 2 = ignore
    outcome[b['miss'][idx]] = 0                                    # 0 = incorrect
    outcome[b['hit'][idx]] = 1                                     # 1 = correct
...
        out[2] = outcome[i]
```
```python
            ['incorrect', 'correct', 'ignore'],
```

iii. Codes follow the prompt's "(incorrect, correct, ignore)".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`. Only the **side camera** (view 1) and only its `tongue` feature are used: `traj{1}(trial).ts[featIndex, 0:2, :]` (x, y) plus `traj{1}(trial).frameTimes`. The bottom camera's `top_tongue`/`bottom_tongue` markers are not used. Alignment additionally needs `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, `obj.bp.ev.bitStart` and `bp.ev.goCue` (see 7-d), and `traj{1}(trial).NdroppedFrames` as a validity flag.

ii.
```python
# DeepLabCut features used for the two kinematic outputs.
# view 1 = side camera, view 2 = bottom camera (only the bottom camera sees the paws).
TONGUE_FEATURE = ('tongue', 1)
```
```python
        for name, view in wanted:
            v = views[view - 1]
            k = featnames[view - 1].index(name)
            ts = np.array(f[v['ts'][j, 0]])          # (nfeat, 3, nframes)
            xy = ts[k, :2, :].T                      # (nframes, 2)
```

iii. The docstring attributes the kinematics to `findPosition.m` + `findVelocity.m`. Choosing the side view alone is not argued for explicitly; the final report just says *"Tongue speed from the side-view tongue marker."* Note that `ts[k, 2, :]` (DeepLabCut likelihood) is never read — the agent's probing established that the authors already set x and y to NaN exactly where likelihood < 0.9 (I re-verified this: `nanfrac == (lk<0.9).frac` to three decimals for every tracked feature), so a separate likelihood cut is unnecessary.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, a port of the authors' pipeline. **(1)** The raw (x, y) are linearly interpolated from camera frame times onto the 500-bin 10 ms axis with MATLAB `interp1` semantics: NaN outside the support and NaN-propagating, so a bin adjacent to an untracked frame is itself NaN. No smoothing is applied (`findPosition.m` smooths non-tongue features only, and with `N = 1`, i.e. not at all). **(2)** A visibility mask is taken from the interpolated positions *before* any filling. **(3)** The NaNs are then replaced with the session's mean lick-onset position — the mean x and y at the first frame of every visible run, pooled over all trials — exactly `setTongueBaselinePosition()` in `getKinematicsFromVideo.m`. **(4)** Speed is `hypot(gradient(x), gradient(y))` on that filled trace, with no baseline-drift subtraction (`findVelocity.m` skips `basederiv` for tongue features). Units are pixels per 10 ms bin, which is irrelevant because the threshold is a percentile.

ii.
```python
def tongue_speed(pos):
    visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
    starts_x, starts_y = [], []
    for i in range(pos.shape[0]):
        for s in runs_of_true(visible[i]):
            starts_x.append(pos[i, s, 0])
            starts_y.append(pos[i, s, 1])
    mux = np.nanmean(starts_x) if starts_x else 0.0
    muy = np.nanmean(starts_y) if starts_y else 0.0

    filled = pos.copy()
    filled[..., 0] = np.where(np.isfinite(filled[..., 0]), filled[..., 0], mux)
    filled[..., 1] = np.where(np.isfinite(filled[..., 1]), filled[..., 1], muy)

    vx = np.gradient(filled[..., 0], axis=1)
    vy = np.gradient(filled[..., 1], axis=1)
    return np.sqrt(vx ** 2 + vy ** 2), visible
```
```python
def interp_to_taxis(t_src, y_src, taxis):
    """MATLAB-style interp1 (linear, NaN outside the support, NaN-propagating)."""
```
```python
def runs_of_true(mask):
    """Start indices of each run of True values (utils/ZeroOnesCount.m)."""
```

iii. Docstring: `kinematics -> findPosition.m + findVelocity.m (tongue NaNs filled with the session's mean lick-onset position, all other features filled 'nearest')`. The `tongue_speed` docstring says *"NaNs (tongue not visible) are replaced by the session's mean lick-onset position, exactly as `setTongueBaselinePosition()` in `getKinematicsFromVideo.m`, before differentiating."*

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the 50th percentile of the speed over **all valid (visible and finite) bins pooled across every trial and timepoint of that session**. Bins below → 0, at or above → 1, bins where the tongue was not visible → 2. Because the tongue is out of view in ~92% of bins, classes 0 and 1 each get ~3.9% of the data.

ii.
```python
def discretize(values, valid, missing_code=2):
    """
    0 : below the session's 50th percentile, 1 : at or above it, 2 : not measurable.
    The percentile is taken over every valid sample in the session.
    """
    out = np.full(values.shape, missing_code, dtype=np.int8)
    v = values[valid]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return out
    thresh = np.percentile(v, 50)
    finite = valid & np.isfinite(values)
    out[finite & (values < thresh)] = 0
    out[finite & (values >= thresh)] = 1
    return out
```
```python
    tvis &= has_video[:, None]
    tongue_code = discretize(tspeed, tvis)
```
```python
            ['below_50th_pctile', 'at_or_above_50th_pctile', 'not_visible'],
```

iii. Directly from the prompt's specification ("discretized with per-session threshold: 0 < 50th percentile, 1 >= 50th percentile, 2 not visible"). Final report: *"Class 2 = feature untracked at that timepoint (tongue in the mouth) / no video."*

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a session-constant offset is computed once from the bitcode pulse recorded on both streams — `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`, a direct port of `findVideoOffset.m` (≈0.99 s in the session the agent checked). Frame times become `frameTimes − vidshift − goCue[trial]`, and the x/y traces are interpolated onto the same `TAXIS` grid as the spikes, so bin *k* is the same interval in both streams. A fallback of 0.5 s is used if `sglx.bitcode` is unreadable (never triggered — all 25 sessions have a bitcode).

ii.
```python
def video_offset(f, obj):
    """findVideoOffset.m: offset between the ephys and the video clocks (s)."""
    try:
        bitstart = vec(obj['sglx']['bitcode'], 'bitstart')
        fs = vec(obj['sglx'], 'fs')[0]
        return mode_(bitstart) / fs - mode_(vec(obj['bp']['ev'], 'bitStart'))
    except Exception:
        return 0.5          # fallback used by the paper's code when bitcode is absent
```
```python
        t_src = ft - vidshift - align[j]
        has_video[i] = True
        for name, view in wanted:
            ...
            pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```

iii. Docstring: `video alignment -> findVideoOffset.m / findPosition.m / loadMotionEnergy.m`. Final report: *"clock offset from `findVideoOffset.m` (bitcode vs. `bp.ev.bitStart`), MATLAB-style NaN-propagating `interp1` onto the same time axis."* The agent sanity-checked the result: tongue visibility is ~1% throughout the delay and jumps at t = 0, reaching 16% at +0.25 s.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The **bottom camera** (view 2) only, and **both** of its paw markers, `top_paw` and `bottom_paw` — x and y from `traj{2}(trial).ts`. Frame times, as for the tongue, come from `traj{1}(trial).frameTimes` plus the session video offset.

ii.
```python
PAW_FEATURES = [('top_paw', 2), ('bottom_paw', 2)]
...
    pspeed, pvis = paw_speed([pos[n] for n, _ in PAW_FEATURES])
```

iii. Comment: *"view 1 = side camera, view 2 = bottom camera (only the bottom camera sees the paws)."* The `paw_speed` docstring says *"Mean speed of the two bottom-view paw markers."* Both markers are in the authors' `params.traj_features` list for view 2. No explicit argument is given for averaging the two rather than picking one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Per marker and per trial: interpolate (x, y) onto `TAXIS`, then `fillmissing(..., 'nearest')` within the trial (`findPosition.m` fills every non-tongue feature this way), then `gradient` minus the per-trial median frame-to-frame displacement to remove slow drift (`findVelocity.m`'s `basederiv`), then speed = `hypot(vx, vy)`. The two markers' speeds are then averaged bin-by-bin over whichever markers were visible. Visibility is again taken from the interpolated positions before filling. Units are pixels per bin. One small departure from the MATLAB: `findVelocity.m` subtracts `basederiv(1)` (the *x* median) from both vx and vy, whereas this code subtracts each axis's own median.

ii.
```python
def paw_speed(pos_list):
    speeds, visibles = [], []
    for pos in pos_list:
        visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
        sp = np.zeros(pos.shape[:2])
        for i in range(pos.shape[0]):
            if not visible[i].any():
                continue
            x = fill_nearest(pos[i, :, 0])
            y = fill_nearest(pos[i, :, 1])
            vx = np.gradient(x) - np.nanmedian(np.diff(x))
            vy = np.gradient(y) - np.nanmedian(np.diff(y))
            sp[i] = np.sqrt(vx ** 2 + vy ** 2)
        speeds.append(sp)
        visibles.append(visible)
    speeds = np.stack(speeds)
    visibles = np.stack(visibles)
    any_visible = visibles.any(axis=0)
    with np.errstate(invalid='ignore'):
        mean_speed = np.nansum(np.where(visibles, speeds, np.nan), axis=0) / \
                     np.maximum(visibles.sum(axis=0), 1)
    return mean_speed, any_visible
```
```python
def fill_nearest(y):
    """fillmissing(y, 'nearest') along axis 0 (column-wise)."""
```

iii. Docstring: *"Positions are filled 'nearest' within each trial and the per-trial median frame-to-frame displacement is removed, following `findPosition.m` / `findVelocity.m`."* No normalisation is applied between the two markers, since both come from the same camera and so share a pixel scale.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: the same `discretize()` with the session's 50th percentile over all valid bins, and class 2 where neither paw marker was visible (or the trial has no video). Because the paws are tracked in ~97% of frames, classes 0 and 1 each get ~47.8% and class 2 only ~4.4% of bins — noticeably less "not visible" than the expert solution's 18.6%.

ii.
```python
    pvis &= has_video[:, None]
    paw_code = discretize(pspeed, pvis)
```
```python
            ['below_50th_pctile', 'at_or_above_50th_pctile', 'not_visible'],
```

iii. From the prompt's specification, same as 7-c.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same video offset and the same 500-bin grid as every other stream. One detail: the frame times used for the bottom-camera paw markers are taken from the **side** camera (`views[0]['frameTimes']`), not from the bottom camera's own `frameTimes` as `findPosition.m` does; if the two views report different frame counts for a trial, both are truncated to the shorter. (I checked this directly: the two views' `frameTimes` are byte-identical in the trials I sampled, so the shortcut is harmless here.)

ii.
```python
        v0 = views[0]
        try:
            ft = np.array(f[v0['frameTimes'][j, 0]]).flatten()
        except Exception:
            ft = np.array([])
        ...
        t_src = ft - vidshift - align[j]
        for name, view in wanted:
            v = views[view - 1]
            k = featnames[view - 1].index(name)
            ts = np.array(f[v['ts'][j, 0]])          # (nfeat, 3, nframes)
            xy = ts[k, :2, :].T                      # (nframes, 2)
            if xy.shape[0] != t_src.shape[0]:
                m = min(xy.shape[0], t_src.shape[0])
                pos[name][i] = interp_to_taxis(t_src[:m], xy[:m], TAXIS)
            else:
                pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```

iii. Not commented on beyond the general `findVideoOffset.m` / `findPosition.m` attribution; the two cameras are frame-synchronised, so one frame-time vector serves both.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure, read with `scipy.io.loadmat`: `me.data{trial}` is one value per side-camera frame. `obj.me` (present in only some sessions) is not used. Frame times come from `traj{1}(trial).frameTimes`; `me.moveThresh` is ignored, since the prompt specifies a 50th-percentile split rather than the authors' manual bimodality threshold.

ii.
```python
def load_motion_energy(anm, date, f, obj, trials, align, vidshift):
    """loadMotionEnergy.m: interpolate motion energy onto TAXIS."""
    fn = os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat')
    if not os.path.exists(fn):
        return np.full((len(trials), NBINS), np.nan)
    m = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=False)
    me = m['me'][0, 0]
    data = me.data
    # loadMotionEnergy.m: some files wrap the cell array in another struct
    while hasattr(data, '_fieldnames') or (data.dtype == object and data.size == 1
                                           and hasattr(data.flatten()[0], '_fieldnames')):
        data = data.data if hasattr(data, '_fieldnames') else data.flatten()[0].data
```

iii. The unwrapping loop mirrors `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`. The loop (rather than a single unwrap) was added after the first full run crashed on `JEB15_2022-07-26` with `TypeError: float() argument must be ... not 'mat_struct'` — the agent then inspected that file specifically and generalised the guard.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling: the value is already one scalar per frame (the paper reduces each frame to the 99th percentile of its per-pixel motion energy). The trace is linearly interpolated onto `TAXIS` and then `fillmissing(..., 'nearest')` is applied within each trial, exactly as `loadMotionEnergy.m` does. Consequence: because the nearest-fill removes every NaN in any trial that has any motion-energy samples, the third class ("no video") is **never produced** — the delivered dataset has only 2 observed classes for this output, even though `output_values` declares 3.

ii.
```python
        out[i] = interp_to_taxis(t_src, y, TAXIS)
        out[i] = fill_nearest(out[i])
```
```python
    me_code = discretize(me, np.isfinite(me))
```

iii. Docstring: `video alignment -> ... loadMotionEnergy.m`. The agent explicitly flagged the consequence in its final report: *"One thing worth flagging: `motion_energy` class 2 ('no video') never occurs — every retained trial has video — so the decoder sees 2 classes there while `output_values` lists 3. I kept the third label because the task spec defines that coding."*

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize()`: 50th percentile of all finite values pooled over the session's trials and bins; below → 0, at or above → 1, NaN → 2. Observed split is 0.500/0.500/0.000.

ii.
```python
    me_code = discretize(me, np.isfinite(me))
...
            ['below_50th_pctile', 'at_or_above_50th_pctile', 'no_video'],
```

iii. From the prompt's specification. The authors' own `me.moveThresh` (a manually set, per-session bimodality threshold) was deliberately not used, because the prompt asks for a median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same session video offset and same 500-bin grid. Motion energy has exactly one value per side-camera frame, so `traj{1}(trial).frameTimes − vidshift − goCue[trial]` is the time base; if the frame-time vector is missing or its length disagrees with the motion-energy trace, the code falls back to `loadMotionEnergy.m`'s catch branch (assume 400 Hz frames and a fixed 0.5 s offset). The agent checked all 25 sessions for length mismatches and found zero, so the fallback never fires.

ii.
```python
        if ft.size != y.size or not np.isfinite(ft).any():
            # loadMotionEnergy.m's catch branch: assume 400 Hz frames and a 0.5 s offset
            t_src = np.arange(1, y.size + 1) / 400.0 - 0.5 - align[j]
        else:
            t_src = ft - vidshift - align[j]
        out[i] = interp_to_taxis(t_src, y, TAXIS)
```

iii. A direct port of `loadMotionEnergy.m`, including its `try/catch`. Sanity-checked in the trajectory: motion energy rises at the go cue and shows a second bump at the sample tone (−2.2 s).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven cases, all handled by keeping the trial and marking the gap rather than by dropping or inventing data.
1. **NaN behaviour flags** — every boolean Bpod field is passed through `np.nan_to_num` before thresholding, so a NaN becomes False.
2. **Over-long per-trial fields** — every field is truncated to `bp.Ntrials`.
3. **Non-finite go cue** — the trial is dropped (it cannot be aligned).
4. **Missing / degenerate frame times** — a trial with fewer than 10 frame times, all-NaN frame times, or a read error is marked `has_video = False`; its tongue and paw channels become 500 bins of class 2.
5. **`NdroppedFrames` = NaN** — the trial's video is skipped, following the explicit guard in `findPosition.m`.
6. **Mismatched frame/feature counts** — both arrays are truncated to the shorter.
7. **Motion-energy trial index beyond the file, or an empty trace** — left NaN → class 2. Untracked DeepLabCut frames (already NaN in the source) propagate through `interp1` to NaN bins, which become class 2 for tongue/paw.
Additionally, out-of-range spike trial ids are filtered, clusters with no spikes are skipped, sessions with fewer than 2 usable trials or 0 surviving units return `None` and are reported as skipped (none were).

ii.
```python
        'hit': np.nan_to_num(vec(bp, 'hit')[:n]) > 0,
```
```python
    keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
```
```python
        try:
            ft = np.array(f[v0['frameTimes'][j, 0]]).flatten()
        except Exception:
            ft = np.array([])
        if ft.size < 10 or not np.isfinite(ft).any():
            continue
        # findPosition.m: trials flagged with NdroppedFrames = NaN are skipped
        try:
            nd = np.array(f[v0['NdroppedFrames'][j, 0]]).flatten()
            if nd.size and np.isnan(nd[0]):
                continue
        except Exception:
            pass
```
```python
    tvis &= has_video[:, None]
    pvis &= has_video[:, None]
```
```python
            inrange = (tr >= 1) & (tr <= len(align))
            tm, tr = tm[inrange], tr[inrange]
```

iii. The `NdroppedFrames` and 400 Hz/0.5 s fallbacks are attributed in comments to `findPosition.m` and `loadMotionEnergy.m`. The general philosophy — keep the trial, mark the bin as class 2 — is stated in the final report: *"Class 2 = feature untracked at that timepoint (tongue in the mouth) / no video."* Nothing is interpolated across a genuine gap except the paw/motion-energy `fillmissing('nearest')`, which is itself what the authors' code does. In the delivered dataset only 1 trial out of 7,426 (in JEB19_2023-04-19) lost its video this way.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 files, and within that the per-trial DeepLabCut reads. The whole conversion takes ~75 s for 25 sessions (~3 s/session). Profiling one representative session (`JEB14_2022-08-24`, 314 trials, 134 units) gives: `load_traj` 2.18 s, `neural_matrix` 0.84 s, `load_motion_energy` 0.18 s, the speed/discretisation maths 0.09 s, `load_behavior` and `video_offset` 0.01 s each. So ~65% of the runtime is the trajectory reads and ~25% is spike binning plus smoothing; everything downstream of loading is negligible. Pickling the 0.99 GB result adds a few seconds at the end.

ii.
```python
    for i, trial in enumerate(trials):
        ...
        for name, view in wanted:
            v = views[view - 1]
            k = featnames[view - 1].index(name)
            ts = np.array(f[v['ts'][j, 0]])          # (nfeat, 3, nframes)
```
```python
            tm = np.array(f[g['trialtm'][icell, 0]]).flatten()
            tr = np.array(f[g['trial'][icell, 0]]).flatten().astype(np.int64)
```

iii. Not discussed by the agent; it only measured wall-clock totals (`time python convert_data.py` → 44 s on the partial run, 75 s on the complete one) and checked available RAM/GPU before committing to the full run.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) `load_traj` loops over trials × 3 features; this is hard to vectorise because each trial is a separate HDF5 object reference with its own frame count, though the three features of a trial could share one read (see 11-c). (2) `load_motion_energy` loops over trials for the same reason. (3) `neural_matrix` loops over clusters, doing one `bincount` per cluster; this could have been a single `histogram2d`/`bincount` over the concatenated spike times of all clusters at once, as the expert solution does. Within each of these, the work is already vectorised across trials and bins. `paw_speed`'s inner per-trial loop over `fill_nearest`/`gradient` could be vectorised for the trials that are fully tracked. Given that HDF5 reading dominates the runtime, none of these would change the total much.

ii.
```python
            counts = np.bincount(row * NBINS + b,
                                 minlength=len(trials) * NBINS).reshape(len(trials), NBINS)
```
```python
    flat = rates.transpose(1, 0, 2).reshape(NBINS, -1)         # time first for smoothing
    flat = my_smooth(flat)
```

iii. Not discussed. The batched smoothing of all units × trials in a single `lfilter` call, and the single `bincount` per cluster across all trials, show that vectorisation was applied where it was easy.

## 11-c. What processing does the code repeat multiple times?

i. Four repeats, all cheap in absolute terms but avoidable.
1. **The `ts` array is re-read once per wanted feature.** `np.array(f[v['ts'][j, 0]])` pulls the *whole* `(nfeat, 3, nframes)` array and then indexes one feature out of it; since `top_paw` and `bottom_paw` are both in view 2, view 2's array is read twice per trial. This is the single largest avoidable cost in the script.
2. **Side-camera `frameTimes` are read twice per trial** — once in `load_traj`, then again in `load_motion_energy`.
3. **Smoothing is applied twice to every retained unit** — once as the all-trials PSTH used for the >1 Hz test, then again on the single-trial rates.
4. `featnames[view-1].index(name)` is recomputed inside the per-trial loop, and `video_offset` is (correctly) computed once per session, not per trial.

ii.
```python
        for name, view in wanted:
            v = views[view - 1]
            k = featnames[view - 1].index(name)
            ts = np.array(f[v['ts'][j, 0]])          # re-read per feature
```
```python
            psth = my_smooth((counts.sum(axis=0) / len(trials) / DT)[:, None])[:, 0]
            if psth.mean() > LOW_FR:
```

iii. Not discussed. The per-session constants that matter most — the video offset, the bin grid, the smoothing kernel — are each computed exactly once (`_KERN`/`_TAPS` and `EDGES`/`TAXIS` at module level, `vidshift` once per session).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts.
- `neural_matrix` builds and returns `keep_ids` (the `(probe, cluster)` identity of every retained unit), which `convert_session` unpacks and then never uses.
- `load_behavior` reads `bp.no`, `bp.ev.sample` and `bp.ev.delay`; none of the three is used anywhere (`sample`/`delay` were used during the agent's interactive verification of the epoch structure, but not in the final pipeline).
- Whole `ts` arrays are materialised to extract one feature's x and y, so 5/7 (view 1) and 9/10 (view 2) of the tracked features, plus all the likelihood rows, are read and thrown away.
- `nquality` is computed only to populate a metadata field.
- The smoothed PSTH per cluster is used solely as a scalar threshold test and then discarded (the counts are re-smoothed later).
Everything else computed — rates, the six output channels, the time axis — ends up in the pickle.

ii.
```python
    rates, nquality, keep_ids = neural_matrix(f, obj, probes, trials, align)   # keep_ids unused
```
```python
        'no': np.nan_to_num(vec(bp, 'no')[:n]) > 0,
        'sample': vec(ev, 'sample')[:n],
        'delay': vec(ev, 'delay')[:n],
```

iii. Not discussed. Storage is handled economically: rates are cast to `float32` and the six outputs to `int8`, and each session's HDF5 handle is closed before the kinematics are computed, giving a 0.99 GB pickle.
